"""Ollama embedding client with retry + cache + dimension guard.

Audit round 10 fixes M1+M2:
  * **M2 dimension guard.** ``embeddings`` table is declared
    ``FLOAT[768]`` (see ``db.py``). If the configured Ollama model is
    swapped to one producing a different dimension (e.g. nomic-embed-
    text returns 768, mxbai-embed-large returns 1024), every insert
    silently corrupts ``vss`` and downstream search returns garbage —
    or worse, sqlite-vss segfaults on dimension mismatch. We now reject
    mismatched vectors at the API boundary with a loud debug log so the
    operator notices on first call rather than after a corrupted index.
  * **M1 retry + cache + typed errors.** Embedding requests hit Ollama
    over Tailscale; a transient timeout or 5xx used to silently drop
    the vector (bare ``except Exception: return None``) which meant the
    summary lived in FTS but never made it into the vector index — the
    user got worse recall for days until the next re-flush. Now we
    retry with bounded exponential backoff on connection/timeout/5xx
    and only surface ``None`` after retries are exhausted. We also
    cache by (model, hash) so the daily diary flush, which embeds the
    SAME 24h summary multiple times when partial flushes overlap,
    doesn't hammer the GPU box.

The cache deliberately bounds memory (LRU, default 256 entries) — a
text-keyed dict would balloon. The hash is sha256-truncated to 16
bytes; collision probability is irrelevant for 256 live entries.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import Lock

import requests

from ..debug import debug_log

# Expected embedding dimension. Pinned to the ``embeddings`` table
# declaration in ``db.py`` — bump both together when changing the model.
EXPECTED_DIM = 768

# Cache size (number of distinct text→vector pairs to keep). The diary
# rarely sees more than a few dozen distinct summaries per session;
# 256 covers bursty re-flushes with headroom.
_CACHE_MAX = 256
_cache: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
_cache_lock = Lock()


def _cache_key(model: str, text: str) -> tuple[str, str]:
    # sha256 truncated to 16 bytes — small dict footprint, collision
    # space large enough that 256 live entries will never alias.
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]
    return model, digest


def _cache_get(key: tuple[str, str]) -> list[float] | None:
    with _cache_lock:
        vec = _cache.get(key)
        if vec is None:
            return None
        # LRU bump
        _cache.move_to_end(key)
        return list(vec)  # defensive copy — caller may mutate


def _cache_put(key: tuple[str, str], vec: list[float]) -> None:
    with _cache_lock:
        _cache[key] = list(vec)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


def get_embedding(
    text: str,
    base_url: str,
    model: str,
    timeout_sec: float = 15.0,
    *,
    expected_dim: int = EXPECTED_DIM,
    max_attempts: int = 3,
) -> list[float] | None:
    """Fetch a single embedding from Ollama with retry/cache/dim check.

    Returns ``None`` on:
      * empty/whitespace input — Ollama would 400 anyway
      * dimension mismatch — caller MUST treat this as a configuration
        error, not a transient failure (do not retry indefinitely)
      * exhausted retries on transient errors

    Never raises; callers expect ``Optional[list[float]]`` everywhere
    (see ``db.py::upsert_summary_embedding``).
    """
    # Empty text would generate a zero-ish vector at best, a 400 at
    # worst. Either way it pollutes vss with non-discriminating rows.
    if not text or not text.strip():
        return None

    # Audit round 21 fix (F13): the cache key is the RAW input text;
    # we only run the (relatively expensive) ``scrub_secrets`` 18-regex
    # pass on cache miss. Same memory-leak protection (scrub before
    # sending to Ollama) but free for repeated lookups of the same
    # text (typical pattern: tool-router embedding the same N
    # descriptions per turn).
    cache_key = _cache_key(model, text)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Audit round 13 fix: defensive PII scrub before sending the text
    # to Ollama. Most callers (memory/conversation, tools/selection)
    # already redact upstream, but a future caller that hands raw
    # voice text in (and a base_url pointed at a remote Ollama box
    # over Tailscale, which is the default in this project) would
    # leak. Redact is idempotent on already-scrubbed strings, so
    # double-application is harmless.
    try:
        from ..utils.redact import scrub_secrets as _scrub
        text = _scrub(text)
    except Exception:
        # Never block embedding on a redact failure — fall back to
        # original text (the caller may already have redacted).
        pass

    url = f"{base_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}

    last_err: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout_sec)
            # 5xx is transient (Ollama loading/reloading the model);
            # 4xx is a config error (bad model name, oversized prompt)
            # — no point retrying.
            if 500 <= resp.status_code < 600:
                last_err = f"http {resp.status_code}"
                raise requests.HTTPError(last_err)
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding")
            if not isinstance(vec, list) or not vec:
                debug_log(f"get_embedding: missing/empty 'embedding' field from {model}", "jarvis")
                return None
            vec_f = [float(x) for x in vec]
            if expected_dim and len(vec_f) != expected_dim:
                # M2: do NOT cache, do NOT retry — config error.
                debug_log(
                    f"get_embedding: DIMENSION MISMATCH model={model} got={len(vec_f)} expected={expected_dim} — refusing to corrupt vss index",
                    "jarvis",
                )
                return None
            _cache_put(cache_key, vec_f)
            return vec_f
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last_err = str(e) or e.__class__.__name__
            if attempt < max_attempts:
                # Exponential backoff: 0.5s, 1s, 2s — total ≤3.5s of
                # waiting, well below caller timeouts.
                sleep_for = 0.5 * (2 ** (attempt - 1))
                debug_log(
                    f"get_embedding: attempt {attempt}/{max_attempts} failed ({last_err}); retrying in {sleep_for:.1f}s",
                    "jarvis",
                )
                time.sleep(sleep_for)
                continue
            break
        except (ValueError, KeyError, TypeError) as e:
            # JSON decode / shape errors — non-transient.
            debug_log(f"get_embedding: malformed response from {model}: {e}", "jarvis")
            return None
        # Note: we intentionally do NOT catch the broad ``Exception`` so
        # KeyboardInterrupt, MemoryError, etc. propagate.

    debug_log(f"get_embedding: gave up after {max_attempts} attempts (last error: {last_err})", "jarvis")
    return None

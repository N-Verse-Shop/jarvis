"""NVIDIA NIM API — OpenAI-compatible client for tier-1 GPU inference.

R35-S23. NVIDIA NIM (https://build.nvidia.com) exposes 100+ frontier
LLMs (Llama 3.3 70B, Mistral Nemotron, Qwen 2.5 72B, etc.) on H200/B200
GPUs through an OpenAI-compatible REST API at
``https://integrate.api.nvidia.com/v1/chat/completions``.

This module wraps the streaming chat endpoint with the same interface
shape as ``jarvis.llm.call_llm_streaming`` so Jarvis can swap it in
transparently. It does NOT touch the Ollama path — that stays as the
fallback tier in ``call_llm_tiered`` below.

Activation:
  Set ``JARVIS_NVIDIA_NIM_KEY=nvapi-...`` in ``~/.config/jarvis/.env``
  and ``nvidia_nim_chat_model_enabled: true`` in the Jarvis config.
  No code change needed downstream — ``call_llm_tiered`` reads the env
  and routes automatically.

Privacy note:
  Voice transcripts traverse NVIDIA's cloud. For client-sensitive
  conversations (Werkvertrag negotiations, IBONS internal details,
  etc.) the privacy keyword gate in ``call_llm_tiered`` forces the
  Hetzner Ollama tier instead. The keyword list lives in config under
  ``privacy_keywords`` (default includes "IBONS", "Werkvertrag",
  "Rechnung", "клієнт", "client", "контракт").
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from .debug import debug_log


# OpenAI-compatible chat completions endpoint. Verified public host
# allowlisted in ``jarvis.llm._HOSTNAME_ALLOW_LIST`` so the SSRF guard
# in ``call_llm_streaming`` lets it through.
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Default chat model. ``meta/llama-3.3-70b-instruct`` is one of the
# strongest available on NIM as of R35-S23 — top-tier multilingual
# (incl. UA/RU/DE), ~50+ tok/s on H200. If NVIDIA deprecates it the
# fallback is ``meta/llama-3.1-70b-instruct``.
DEFAULT_NIM_CHAT_MODEL = "meta/llama-3.3-70b-instruct"

# Default privacy keywords — if any appear in the user prompt, force
# the Hetzner Ollama tier so the transcript doesn't leave EU infra.
DEFAULT_PRIVACY_KEYWORDS: tuple[str, ...] = (
    # Nexus Studio client names
    "IBONS", "ibons", "Ibons",
    "founder cockpit", "Founder Cockpit",
    # German contractual / financial terms
    "Werkvertrag", "Anhang", "Anhänge", "Anhangsprotokoll",
    "Rechnung", "Steuer", "Steuern", "GoBD",
    # General sensitive terms (multilingual)
    "клієнт", "клиент", "client",
    "контракт", "контракта", "contract",
    "договір", "договор", "agreement",
    "пароль", "password", "Passwort",
    # API / secrets
    "api_key", "api-key", "secret", "token",
)


class _NIMError(Exception):
    """Raised when NIM call should not retry (auth, quota, model 404)."""
    pass


class _NIMTransient(Exception):
    """Raised when NIM call is retryable (timeout, 5xx, connection reset)."""
    pass


def _get_api_key() -> Optional[str]:
    """Read ``JARVIS_NVIDIA_NIM_KEY`` from env. None if missing.

    The env var name uses the ``JARVIS_`` prefix to match the project's
    other secrets (``JARVIS_N8N_API_KEY``, ``JARVIS_DASHBOARD_TOKEN``).
    Loaded by ``config.load_settings`` from ``~/.config/jarvis/.env``.
    """
    key = os.environ.get("JARVIS_NVIDIA_NIM_KEY", "").strip()
    return key or None


def is_nim_enabled(cfg: Any) -> bool:
    """True only if BOTH the config flag AND the env key are set.

    This double-gate avoids the ``"oh I left it on by accident"``
    failure mode — a tampered config alone can't route traffic, and a
    key in env alone doesn't change behaviour either.
    """
    if not getattr(cfg, "nvidia_nim_chat_model_enabled", False):
        return False
    return _get_api_key() is not None


def _is_privacy_sensitive(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True if ``text`` contains any privacy-keyword substring.

    Case-sensitive on the literal substring — preserves the explicit
    keyword list semantics. Run on every user turn before tier routing.
    """
    if not text or not keywords:
        return False
    return any(k in text for k in keywords)


def call_llm_nim_streaming(
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    abort_event: Optional[threading.Event] = None,
    messages_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """Stream a chat completion from NVIDIA NIM. OpenAI-compatible.

    Returns the full assembled response text, or ``None`` on error /
    abort. ``on_token`` is called for each delta as it arrives so the
    TTS layer can start synthesising before generation completes.

    Args:
        chat_model: NIM model id (e.g. ``meta/llama-3.3-70b-instruct``)
        system_prompt: System message content
        user_content: User message content
        on_token: Optional per-token callback. Receives delta strings.
        timeout_sec: Total read timeout (connect clamped to 5 s)
        temperature: Optional sampling temperature
        max_tokens: Optional output cap
        abort_event: Optional Event — if set mid-stream, returns None
            immediately. Mirrors ``call_llm_streaming`` semantics.
    """
    api_key = _get_api_key()
    if not api_key:
        debug_log("nim: JARVIS_NVIDIA_NIM_KEY missing — refusing call", "llm")
        return None

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # R35-S25: voice path uses last 2-4 message pairs for context. Insert
    # between system and current user so it reads "system, hist..., user".
    if messages_history:
        messages.extend(messages_history)
    messages.append({"role": "user", "content": user_content})

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    # Note: the SSRF guard in ``call_llm_streaming`` is intentionally
    # NOT reused here — that guard rejects public hosts by default,
    # and NIM's public host (``integrate.api.nvidia.com``) is added
    # to the explicit allow-list there. We post directly.
    url = f"{NVIDIA_NIM_BASE_URL}/chat/completions"
    _connect_timeout = min(5.0, max(2.0, float(timeout_sec)))

    chunks: List[str] = []
    started = time.monotonic()
    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(_connect_timeout, timeout_sec),
            stream=True,
        ) as resp:
            if resp.status_code == 401:
                debug_log("nim: 401 — JARVIS_NVIDIA_NIM_KEY invalid", "llm")
                return None
            if resp.status_code == 429:
                debug_log("nim: 429 — rate limited; caller should fall back", "llm")
                raise _NIMTransient("rate limited")
            if 500 <= resp.status_code < 600:
                debug_log(f"nim: {resp.status_code} — server error", "llm")
                raise _NIMTransient(f"server {resp.status_code}")
            if resp.status_code != 200:
                debug_log(f"nim: HTTP {resp.status_code} {resp.text[:200]}", "llm")
                raise _NIMError(f"HTTP {resp.status_code}")

            for line in resp.iter_lines(decode_unicode=False):
                if abort_event is not None and abort_event.is_set():
                    debug_log("nim: aborted mid-stream", "llm")
                    return None
                if not line:
                    continue
                # SSE format: ``data: {json}\n\n`` or ``data: [DONE]``
                if line.startswith(b"data: "):
                    chunk = line[6:].decode("utf-8", errors="replace")
                elif line.startswith(b":"):
                    # SSE keepalive comment — ignore.
                    continue
                else:
                    continue
                if chunk.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    debug_log(f"nim: malformed SSE chunk: {chunk[:80]!r}", "llm")
                    continue
                # OpenAI-compatible delta shape:
                # {"choices":[{"delta":{"content":"..."}}]}
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                tok = delta.get("content")
                if tok:
                    chunks.append(tok)
                    if on_token is not None:
                        try:
                            on_token(tok)
                        except Exception as exc:
                            debug_log(f"nim: on_token raised — {exc!r}", "llm")
    except _NIMTransient:
        raise
    except _NIMError:
        raise
    except requests.Timeout:
        elapsed = time.monotonic() - started
        debug_log(f"nim: timeout after {elapsed:.1f}s", "llm")
        return None
    except requests.RequestException as exc:
        debug_log(f"nim: request error {type(exc).__name__}: {exc}", "llm")
        return None

    full = "".join(chunks).strip()
    if not full:
        debug_log("nim: empty response", "llm")
        return None
    return full


def call_llm_tiered(
    cfg: Any,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    abort_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """Try NVIDIA NIM first, fall back to Hetzner Ollama on failure.

    Routing logic:
        1. If user_content matches a privacy keyword → Ollama only.
        2. Else if NIM is enabled (env key + config flag) → try NIM.
            On 429 / 5xx / connection error → fall back to Ollama.
        3. Else → Ollama (existing call_llm_streaming).

    This is the SINGLE entry point Jarvis's voice path should call
    once R35-S23.1 wires it into ``pipecat_loop._make_direct_chat_processor``.
    Until that wire-up lands, the existing ``call_llm_streaming``
    continues to handle voice replies via Ollama unchanged.
    """
    keywords = tuple(getattr(cfg, "privacy_keywords", DEFAULT_PRIVACY_KEYWORDS))
    if _is_privacy_sensitive(user_content, keywords):
        debug_log("tiered: privacy keyword matched → forcing Ollama tier", "llm")
        return _call_ollama(cfg, system_prompt, user_content, on_token, timeout_sec, abort_event)

    if is_nim_enabled(cfg):
        nim_model = getattr(cfg, "nvidia_nim_chat_model", DEFAULT_NIM_CHAT_MODEL)
        try:
            result = call_llm_nim_streaming(
                chat_model=nim_model,
                system_prompt=system_prompt,
                user_content=user_content,
                on_token=on_token,
                timeout_sec=timeout_sec,
                temperature=float(getattr(cfg, "ollama_chat_temperature", 0.3) or 0.3),
                max_tokens=int(getattr(cfg, "ollama_chat_num_predict", 100) or 100),
                abort_event=abort_event,
            )
            if result is not None:
                return result
            debug_log("tiered: NIM returned None — falling back to Ollama", "llm")
        except _NIMTransient as exc:
            debug_log(f"tiered: NIM transient ({exc}) — falling back to Ollama", "llm")
        except _NIMError as exc:
            debug_log(f"tiered: NIM hard error ({exc}) — falling back to Ollama", "llm")

    return _call_ollama(cfg, system_prompt, user_content, on_token, timeout_sec, abort_event)


def _call_ollama(
    cfg: Any,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]],
    timeout_sec: float,
    abort_event: Optional[threading.Event],
) -> Optional[str]:
    """Delegate to the existing Ollama path. Imported lazily to avoid
    circular import with ``jarvis.llm``."""
    from .llm import call_llm_streaming
    return call_llm_streaming(
        base_url=getattr(cfg, "ollama_base_url"),
        chat_model=getattr(cfg, "ollama_chat_model"),
        system_prompt=system_prompt,
        user_content=user_content,
        on_token=on_token,
        timeout_sec=timeout_sec,
        thinking=False,
        abort_event=abort_event,
    )

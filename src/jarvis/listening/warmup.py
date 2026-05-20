"""Warmup utilities — pre-prime the voice pipeline at daemon boot.

The voice loop has three independent cold-cache costs that hit on
the FIRST user turn after a daemon restart:

  Ollama (chat)   ~150-300 ms — KV cache for the system prompt
  Whisper (STT)   ~200-400 ms — first MLX inference
  Piper (TTS)     ~40-80 ms  — first synthesis

A single user phrase ("Привіт, Jarvis") would otherwise eat
~500-700 ms of cold-cache pain. Pre-warming runs these in
background threads at pipeline build time so by the time the user
speaks, all three caches are hot.

All warmup work happens on daemon threads — failures are logged
but never propagate. We never block the voice pipeline waiting on
warmup; if it doesn't finish before the user speaks, the worst
case is "first turn is still slow" — exactly the current state.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.warmup")

# Serialise stderr writes so warmup threads don't interleave with
# each other or with other daemon log writers.
_print_lock = threading.Lock()


def _log(msg: str, level: str = "info") -> None:
    """Emit a warmup log line.

    Warmup messages are LOW-VOLUME (a handful per daemon boot) but
    HIGH-VALUE — they tell us whether the cold-cache prep actually
    ran. We bypass the voice-debug gate (which is off by default)
    and write straight to stderr with flush=True so they always
    show up in ``jarvis-assistant.err.log``.

    Also fans out to:
      - stdlib logging — tests, future log aggregators.
      - debug_log — when voice_debug is on, the line shows up
        twice in stderr (once raw, once with [warmup] prefix), but
        that's harmless and keeps a single source of truth for
        debug-mode consumers.
    """
    # Always-on stderr write — this is what shows up in the daemon log.
    try:
        with _print_lock:
            print(f"[warmup] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass
    # debug_log honour, gated on voice_debug.
    try:
        from ..debug import debug_log
        debug_log(msg, "warmup")
    except Exception:
        pass
    # stdlib logging fan-out for tests / aggregators.
    getattr(log, level, log.info)(msg)


def _warmup_ollama(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    timeout_s: float = 90.0,
    # R34-S40 — bumped 45→90s. qwen3:8b cold load on Hetzner GPU is
    # 30-60s under contention, occasionally 70s+. 45s was too tight and
    # we saw repeated read-timeout failures on boot. 90s covers the
    # realistic worst case while still failing fast if the host is
    # genuinely unreachable (DNS fail / Tailscale down hit in <3s via
    # connect timeout). Once warm the model stays for 24h.
) -> None:
    """Prime Ollama's KV cache with the actual system prompt.

    A single short user message ("ok") runs through the full chat
    flow so:
      1. The model is loaded into VRAM (no-op if keep_alive is set).
      2. The system prompt is tokenised + evaluated → KV cache populated.
      3. Subsequent real turns reuse the cache, paying only for the
         new user message tokens.

    Uses requests sync (cheap, single shot). Pipecat's OLLamaLLMService
    talks to the same endpoint so the cache survives.
    """
    try:
        import requests
    except ImportError:
        _log("requests not installed — skipping Ollama warmup")
        return

    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "ok"},
        ],
        "stream": False,
        "options": {
            "num_predict": 1,    # we throw the reply away
            "temperature": 0.0,
        },
        "keep_alive": "24h",   # match the daemon's chat policy
    }
    t0 = time.monotonic()
    try:
        r = requests.post(url, json=body, timeout=timeout_s)
        dt = time.monotonic() - t0
        if r.status_code == 200:
            _log(f"Ollama warmup OK ({model}, {dt:.2f}s)")
        else:
            _log(
                f"Ollama warmup HTTP {r.status_code}: {r.text[:200]}",
                "warning",
            )
    except Exception as exc:
        _log(f"Ollama warmup failed: {exc}", "warning")


def _warmup_whisper_mlx(model_repo: Optional[str]) -> None:
    """Run a 0.1 s silent transcription to populate MLX caches.

    MLX Whisper's first inference compiles graphs on the GPU which
    takes ~250 ms on M-Pro. A no-content warmup with low-amplitude
    noise avoids both ENV-FILE warnings ("audio is empty") and
    decoder fast-paths that skip the heavy compile.

    R34-S26 PROBLEM: MLX maintains a per-thread GPU stream. Warming up
    in this background thread initialises the stream HERE — but Pipecat
    runs the real transcription via ``asyncio.to_thread`` on a DIFFERENT
    thread (the default ThreadPoolExecutor). That thread has no GPU
    stream → ``RuntimeError: There is no Stream(gpu, 1) in current
    thread.`` We now skip this warmup by default and let the first
    real utterance pay the ~250 ms graph-compile on the executor
    thread. Set ``JARVIS_WHISPER_WARMUP=true`` to re-enable for
    benchmarking only (will break runtime transcription).
    """
    if not model_repo:
        return
    import os
    if os.environ.get("JARVIS_WHISPER_WARMUP", "false").lower() not in (
        "1", "true", "yes", "on"
    ):
        _log(
            f"Whisper warmup SKIPPED ({model_repo}) — MLX per-thread "
            "GPU streams prevent cross-thread reuse. First real "
            "transcription pays the graph-compile cost (~250 ms)."
        )
        return
    try:
        import mlx_whisper
        import numpy as np
    except ImportError:
        _log("mlx_whisper / numpy missing — skipping Whisper warmup")
        return
    t0 = time.monotonic()
    try:
        # 0.1 s of low-amplitude pink-noise (avoid the empty-input
        # bail-out path).
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(1600).astype("float32") * 0.001)
        _ = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_repo,
            language=None,
            verbose=False,
        )
        _log(
            f"Whisper warmup OK ({model_repo}, {time.monotonic() - t0:.2f}s)"
        )
    except Exception as exc:
        _log(f"Whisper warmup failed: {exc}", "warning")


def _warmup_piper(voice_id: str, piper_dir: Optional[Path] = None) -> None:
    """Synthesise a single phoneme to load Piper's ONNX runtime.

    Piper's first synthesis pays the ONNX session-create cost
    (~80 ms on M-Pro) plus model file load. Subsequent synthesis is
    ~20-40 ms. A 1-character warmup pays the upfront cost once at
    daemon boot.
    """
    try:
        from piper import PiperVoice
    except ImportError:
        _log("piper missing — skipping Piper warmup")
        return
    try:
        target_dir = piper_dir or (
            Path.home() / ".local/share/jarvis/piper"
        )
        target = target_dir / f"{voice_id}.onnx"
        if not target.exists():
            # Pipecat downloads lazily on first use; we'd race with
            # that path. Skip and let Pipecat handle it.
            _log("Piper voice not cached locally — skipping warmup")
            return
        t0 = time.monotonic()
        voice = PiperVoice.load(str(target))
        # The ONNX-session create happens inside ``load`` — that
        # IS the warmup. Different Piper versions expose different
        # streaming methods; we don't actually need to synthesise
        # anything, just having the session in memory is enough to
        # cut first-real-synthesis time. Try a few known method
        # names for full warm; ignore if none match.
        for attr in ("synthesize_stream_raw", "synthesize_stream", "synthesize"):
            fn = getattr(voice, attr, None)
            if fn is None:
                continue
            try:
                result = fn("а")
                if hasattr(result, "__iter__"):
                    for _ in result:
                        break
                break
            except Exception:
                continue
        _log(
            f"Piper warmup OK ({voice_id}, {time.monotonic() - t0:.2f}s)"
        )
    except Exception as exc:
        _log(f"Piper warmup failed: {exc}", "warning")


def warmup_voice_stack(
    *,
    ollama_base_url: str,
    chat_model: str,
    system_prompt: str,
    whisper_mlx_repo: Optional[str] = None,
    piper_voice_id: Optional[str] = None,
    piper_dir: Optional[Path] = None,
) -> None:
    """Kick off warmup in three parallel background threads.

    Returns immediately; threads are daemon-mode so they die with
    the process. Caller doesn't get a "done" signal — that's
    intentional, callers shouldn't depend on warmup completing
    before their work starts.
    """
    threads: list[threading.Thread] = []

    threads.append(threading.Thread(
        target=_warmup_ollama,
        kwargs={
            "base_url": ollama_base_url,
            "model": chat_model,
            "system_prompt": system_prompt,
        },
        name="warmup-ollama",
        daemon=True,
    ))
    if whisper_mlx_repo:
        threads.append(threading.Thread(
            target=_warmup_whisper_mlx,
            args=(whisper_mlx_repo,),
            name="warmup-whisper",
            daemon=True,
        ))
    if piper_voice_id:
        threads.append(threading.Thread(
            target=_warmup_piper,
            args=(piper_voice_id, piper_dir),
            name="warmup-piper",
            daemon=True,
        ))

    for t in threads:
        t.start()
    _log(f"Voice stack warmup kicked off — {len(threads)} thread(s)")

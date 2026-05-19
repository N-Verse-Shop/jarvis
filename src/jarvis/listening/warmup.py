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
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.warmup")


def _warmup_ollama(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    timeout_s: float = 10.0,
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
        log.debug("requests not installed — skipping Ollama warmup")
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
            log.info("Ollama warmup OK (%s, %.2fs)", model, dt)
        else:
            log.warning(
                "Ollama warmup HTTP %s: %s",
                r.status_code,
                r.text[:200],
            )
    except Exception as exc:
        log.warning("Ollama warmup failed: %s", exc)


def _warmup_whisper_mlx(model_repo: Optional[str]) -> None:
    """Run a 0.1 s silent transcription to populate MLX caches.

    MLX Whisper's first inference compiles graphs on the GPU which
    takes ~250 ms on M-Pro. A no-content warmup with low-amplitude
    noise avoids both ENV-FILE warnings ("audio is empty") and
    decoder fast-paths that skip the heavy compile.
    """
    if not model_repo:
        return
    try:
        import mlx_whisper
        import numpy as np
    except ImportError:
        log.debug("mlx_whisper / numpy missing — skipping Whisper warmup")
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
        log.info(
            "Whisper warmup OK (%s, %.2fs)",
            model_repo,
            time.monotonic() - t0,
        )
    except Exception as exc:
        log.warning("Whisper warmup failed: %s", exc)


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
        log.debug("piper missing — skipping Piper warmup")
        return
    try:
        target_dir = piper_dir or (
            Path.home() / ".local/share/jarvis/piper"
        )
        target = target_dir / f"{voice_id}.onnx"
        if not target.exists():
            # Pipecat downloads lazily on first use; we'd race with
            # that path. Skip and let Pipecat handle it.
            log.info("Piper voice not cached locally — skipping warmup")
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
        log.info(
            "Piper warmup OK (%s, %.2fs)",
            voice_id,
            time.monotonic() - t0,
        )
    except Exception as exc:
        log.warning("Piper warmup failed: %s", exc)


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
    log.info(
        "Voice stack warmup kicked off — %d thread(s)",
        len(threads),
    )

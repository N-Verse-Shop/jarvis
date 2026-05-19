"""Pipecat-based voice loop for Jarvis (Round 31).

This module is the second-generation voice loop, a drop-in alternative
to the legacy ``listener.Listener``. It is selected at runtime via the
``voice_engine`` config flag (``"pipecat"`` vs ``"legacy"``).

Why we are migrating
====================

The legacy ``listener.py`` is ~6800 lines and hand-rolls a state
machine that Pipecat (https://docs.pipecat.ai) ships out of the box:

* turn-taking (VAD + interruption frames)
* full-duplex echo isolation (TTS frames never re-enter STT)
* streaming sentence flush + per-sentence TTS queueing
* function-calling protocol shared with OpenAI tools schema
* clean interrupt epoch (cancel-aware frame propagation)

This file scaffolds the Pipecat pipeline. The wiring is done in
incremental stages:

  Stage 1 (this file) — module skeleton, config import, public entry
                         points. NO pipeline yet.
  Stage 2 — core pipeline (Audio→VAD→STT→LLM→TTS→Audio).
  Stage 3 — JarvisEventStreamProcessor / JarvisStateProcessor adapters
            so the existing Electron HUD keeps working unchanged.
  Stage 4 — mac_control tool bridge + USER_COMMAND_PATTERNS fast-path
            pre-LLM filter (~50ms direct execution path).
  Stage 5 — wake-word gate (reuses whisper-wake-detect heuristics)
            and echo filter wrapping the proven echo_detection logic.
  Stage 6 — feature-flag wiring in daemon.py + end-to-end test.

Stage flags
-----------

While the pipeline grows, ``run()`` raises ``NotImplementedError`` on
purpose so that anyone calling it before the stage completes gets a
clear failure instead of a half-working loop. ``daemon.py`` only
selects this engine when ``cfg.voice_engine == "pipecat"`` — the
default remains ``"legacy"`` so production behaviour is unchanged
until the migration is fully tested.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..debug import debug_log


# Stage gate — bumped as each stage lands. ``run()`` refuses to start
# if it cannot satisfy the minimum stage required for a working voice
# loop (currently 5 = core + adapters + tools + wake/echo).
_MIN_STAGE_FOR_RUN = 5
_CURRENT_STAGE = 1


@dataclass
class PipecatLoopConfig:
    """Subset of jarvis config the Pipecat loop cares about.

    Built from the global ``Settings`` in ``jarvis.config`` via
    :func:`from_settings`. We isolate the fields here so the loop is
    decoupled from the larger config schema — easier to test and to
    evolve without ripple effects.
    """

    # Audio I/O ----------------------------------------------------------------
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    sample_rate: int = 16_000  # Pipecat's standard for Whisper

    # STT ----------------------------------------------------------------------
    # Either an MLX model enum name ("LARGE_V3_TURBO") for on-device,
    # or empty string to use the remote whisper-service (Hetzner).
    stt_mlx_model: str = "LARGE_V3_TURBO"
    stt_language: str = "ru"  # ISO-639-1; "" = auto-detect

    # LLM ----------------------------------------------------------------------
    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:8b"
    chat_temperature: float = 0.4
    chat_num_predict: int = 220
    chat_num_ctx: int = 2048
    chat_keep_alive: str = "24h"

    # TTS ----------------------------------------------------------------------
    piper_voice_id: str = "uk_UA-ukrainian_tts-medium"
    piper_download_dir: Optional[Path] = None

    # Wake-word / VAD ----------------------------------------------------------
    wake_words: tuple[str, ...] = ("jarvis", "джарвіс", "джарвис")
    vad_threshold: float = 0.5  # Silero confidence; tuned in Stage 5
    vad_min_silence_ms: int = 700

    # Behaviour ----------------------------------------------------------------
    # Active language seeds the language-aware Piper transliteration
    # (RU vs UA) — same logic as legacy ``_sanitize_for_piper_uk``.
    active_language: str = "ru"

    # Misc fields filled at runtime by ``from_settings``.
    extra: dict[str, Any] = field(default_factory=dict)


def from_settings(cfg) -> PipecatLoopConfig:
    """Build the loop config from the global ``Settings`` object.

    Falls back to safe defaults if a field is missing — the legacy
    listener and Pipecat loop are supposed to consume the same
    ``config.json``, but we are tolerant of older versions of the
    config schema while users still have unmigrated installs.
    """
    return PipecatLoopConfig(
        input_device_index=getattr(cfg, "input_device_index", None),
        output_device_index=getattr(cfg, "output_device_index", None),
        stt_mlx_model=getattr(cfg, "whisper_model", "LARGE_V3_TURBO"),
        stt_language=getattr(cfg, "active_language", "ru"),
        ollama_base_url=getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434"),
        chat_model=str(getattr(cfg, "ollama_chat_model", "qwen3:8b")),
        chat_temperature=float(getattr(cfg, "ollama_chat_temperature", 0.4)),
        chat_num_predict=int(getattr(cfg, "ollama_chat_num_predict", 220)),
        chat_num_ctx=int(getattr(cfg, "ollama_chat_num_ctx", 2048)),
        piper_voice_id=str(getattr(cfg, "piper_voice", "uk_UA-ukrainian_tts-medium")),
        piper_download_dir=(
            Path(p) if (p := getattr(cfg, "piper_download_dir", None)) else None
        ),
        active_language=str(getattr(cfg, "active_language", "ru")),
        extra={
            "voice_engine": getattr(cfg, "voice_engine", "legacy"),
            "remote_whisper_url": getattr(cfg, "remote_whisper_url", ""),
            "remote_whisper_token": getattr(cfg, "remote_whisper_token", ""),
        },
    )


class PipecatLoop:
    """Lifecycle wrapper around the Pipecat pipeline.

    The legacy ``Listener`` is also a long-lived object that
    ``daemon.py`` owns and drives via ``listener.run()``. We mirror
    that contract so the daemon doesn't care which engine is active —
    both expose ``run()``, ``stop()``, and a couple of properties.
    """

    def __init__(self, cfg: PipecatLoopConfig) -> None:
        self.cfg = cfg
        self._task: Optional[asyncio.Task] = None
        self._runner: Optional["asyncio.AbstractEventLoop"] = None
        self._stop_flag = False
        debug_log(
            f"PipecatLoop initialised (stage={_CURRENT_STAGE}, "
            f"chat={cfg.chat_model}, stt={cfg.stt_mlx_model}, "
            f"piper={cfg.piper_voice_id})",
            "pipecat",
        )

    # ------------------------------------------------------------------ run --
    def run(self) -> None:
        """Blocking entry point — daemon.py calls this in a thread.

        At Stage 1 the pipeline is not wired yet; we deliberately
        refuse to start so the user gets a clear error if they flip
        ``voice_engine=pipecat`` too early. Each subsequent stage
        replaces this body with real wiring.
        """
        if _CURRENT_STAGE < _MIN_STAGE_FOR_RUN:
            raise NotImplementedError(
                f"Pipecat loop is at stage {_CURRENT_STAGE}/"
                f"{_MIN_STAGE_FOR_RUN}. Set voice_engine=legacy in "
                "config.json until the migration completes."
            )
        # Real implementation lands in Stage 2.
        raise NotImplementedError("Stage 2 wiring not yet present.")

    # ---------------------------------------------------------------- stop ---
    def stop(self) -> None:
        """Co-operative shutdown — sets a flag and cancels the task.

        ``daemon.py`` calls this from the signal handler. We rely on
        Pipecat's ``PipelineTask.cancel()`` to drain frames cleanly
        once it's wired in Stage 2.
        """
        self._stop_flag = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            debug_log("PipecatLoop stop() — task cancellation requested", "pipecat")


__all__ = [
    "PipecatLoop",
    "PipecatLoopConfig",
    "from_settings",
]

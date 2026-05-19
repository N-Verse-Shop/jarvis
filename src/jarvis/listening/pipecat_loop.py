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
_CURRENT_STAGE = 2  # Stage 2: core pipeline wired


# ──────────────────────── F94 Russian system prompt ────────────────────
# Minimal voice-focused system prompt — same direction as the legacy
# listener (~132 tokens, RU, terse). The full Jarvis persona prompt is
# heavy (~3000 tokens, expensive cold-cache eval); for the streaming
# voice loop we use the slim version proven in legacy listener
# ``VOICE_STATIC_SYSTEM_PROMPT`` (line 200-ish of listener.py).
_VOICE_SYSTEM_PROMPT_RU = (
    "Ты — Джарвис, голосовой ассистент Данило. Отвечай "
    "ОЧЕНЬ КРАТКО (1-2 предложения, обычно ≤25 слов), на "
    "русском языке. Никаких префиксов вроде «Хорошо», "
    "«Конечно», «Понял». Никаких эмодзи. Никаких списков. "
    "Никаких блоков кода. Если открываешь приложение или "
    "URL — формулируй: «Сейчас открою X. Подтверди моё "
    "действие, пожалуйста.» Имя Данило склоняй "
    "грамматически правильно (Данила/Данилу/Даниле). "
    "Никогда не сокращай до «Нило» или «Даня»."
)
_VOICE_SYSTEM_PROMPT_UK = (
    "Ти — Джарвіс, голосовий асистент Данила. Відповідай "
    "ДУЖЕ КОРОТКО (1-2 речення, зазвичай ≤25 слів), "
    "українською. Без префіксів типу «Добре», «Звичайно», "
    "«Зрозумів». Без емодзі, списків, блоків коду. Якщо "
    "відкриваєш додаток або URL — формулюй: «Зараз відкрию "
    "X. Підтверди мою дію, будь ласка.»"
)


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


def _system_prompt_for(lang: str) -> str:
    """Return the slim voice system prompt for the active language."""
    return _VOICE_SYSTEM_PROMPT_UK if lang == "uk" else _VOICE_SYSTEM_PROMPT_RU


# ─────────────────────────── pipeline factory ────────────────────────────


def _build_pipeline(cfg: PipecatLoopConfig):
    """Construct the core voice pipeline.

    Returns a tuple ``(pipeline, context, task)`` ready for the
    PipelineRunner. Kept as a pure factory so we can unit-test the
    wiring without actually opening the microphone.

    Pipeline topology
    -----------------

    ::

      LocalAudioTransport.input
          → SileroVADAnalyzer
          → WhisperSTTServiceMLX
          → ContextAggregator.user
          → OLLamaLLMService
          → ContextAggregator.assistant
          → PiperTTSService
          → LocalAudioTransport.output

    The context aggregators sandwich the LLM so user transcripts and
    assistant replies are appended to the shared ``LLMContext`` —
    that gives us multi-turn dialog memory inside the pipeline
    without us hand-rolling a dialog_history list like in legacy.
    """
    # Imports are local so the module can be imported and inspected on
    # a machine that doesn't have pipecat installed (e.g. CI without
    # ML extras). The hard import only happens when we actually try
    # to run the loop.
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.services.whisper.stt import (
        WhisperSTTServiceMLX,
        WhisperMLXSTTSettings,
        MLXModel,
    )
    from pipecat.services.ollama.llm import (
        OLLamaLLMService,
        OllamaLLMSettings,
    )
    from pipecat.services.piper.tts import (
        PiperTTSService,
        PiperTTSSettings,
    )
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
    )

    # ── transport (mic + speakers) ───────────────────────────────────
    vad_params = VADParams(
        confidence=cfg.vad_threshold,
        stop_secs=cfg.vad_min_silence_ms / 1000.0,
    )
    transport_params = LocalAudioTransportParams(
        input_device_index=cfg.input_device_index,
        output_device_index=cfg.output_device_index,
        audio_in_enabled=True,
        audio_out_enabled=True,
        # Sample-rate alignment: Whisper expects 16k; Piper UA model is
        # 22050 Hz. Pipecat resamples internally between processors.
        audio_in_sample_rate=cfg.sample_rate,
        vad_analyzer=SileroVADAnalyzer(params=vad_params),
    )
    transport = LocalAudioTransport(params=transport_params)

    # ── STT ─────────────────────────────────────────────────────────
    # MLX path (on-device Apple Silicon). For Hetzner remote STT we
    # would swap in a custom RemoteWhisperSTTService — that's a
    # Stage-5 follow-up; Stage 2 boots local-only.
    try:
        mlx_model = MLXModel[cfg.stt_mlx_model]
    except KeyError:
        debug_log(
            f"unknown MLX whisper model '{cfg.stt_mlx_model}', "
            "falling back to LARGE_V3_TURBO",
            "pipecat",
        )
        mlx_model = MLXModel.LARGE_V3_TURBO
    # F95: use the new Settings-based API to silence DeprecationWarning.
    # The deprecated kwargs (``model=...``, ``voice_id=...``) still
    # work but pipecat plans to remove them; future-proof now.
    stt = WhisperSTTServiceMLX(
        settings=WhisperMLXSTTSettings(model=mlx_model),
    )

    # ── LLM ─────────────────────────────────────────────────────────
    # OLLamaLLMService inherits from OpenAI — it talks to Ollama's
    # OpenAI-compatible /v1/chat/completions endpoint. Hetzner runs
    # Ollama on the same host:port the legacy listener uses, so the
    # base_url config maps directly.
    llm = OLLamaLLMService(
        settings=OllamaLLMSettings(model=cfg.chat_model),
        # Ollama's OpenAI shim is at /v1, not /api.
        base_url=cfg.ollama_base_url.rstrip("/") + "/v1",
    )

    # ── TTS ─────────────────────────────────────────────────────────
    # Piper voice files live under ``~/.local/share/jarvis/piper`` by
    # convention. PiperTTSService downloads on first run if absent.
    # F95: ensure the directory exists — Piper's own download helper
    # only mkdir's the parent, so a fresh install crashes with a
    # FileNotFoundError opening the .onnx for write.
    piper_dir = cfg.piper_download_dir or (
        Path.home() / ".local" / "share" / "jarvis" / "piper"
    )
    piper_dir.mkdir(parents=True, exist_ok=True)
    tts = PiperTTSService(
        settings=PiperTTSSettings(voice=cfg.piper_voice_id),
        download_dir=piper_dir,
    )

    # ── context aggregators ────────────────────────────────────────
    system_prompt = _system_prompt_for(cfg.active_language)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
    )
    aggregators = LLMContextAggregatorPair(context)

    # ── pipeline ───────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            aggregators.assistant(),
            tts,
            transport.output(),
        ]
    )

    task_params = PipelineParams(
        # Allow interruptions — user can talk over TTS and we drop
        # the in-flight reply (the equivalent of the legacy
        # ``_stream_abort`` flag, but baked into Pipecat frames).
        allow_interruptions=True,
        enable_metrics=False,
        # Same idle policy as legacy: 5min of silence before we
        # consider the loop idle.
        idle_timeout_secs=300.0,
    )
    task = PipelineTask(pipeline, params=task_params)

    return pipeline, context, task, transport


# ─────────────────────────────── loop class ──────────────────────────────


class PipecatLoop:
    """Lifecycle wrapper around the Pipecat pipeline.

    The legacy ``Listener`` is also a long-lived object that
    ``daemon.py`` owns and drives via ``listener.run()``. We mirror
    that contract so the daemon doesn't care which engine is active —
    both expose ``run()``, ``stop()``, and a couple of properties.
    """

    def __init__(self, cfg: PipecatLoopConfig) -> None:
        self.cfg = cfg
        self._task: Optional[Any] = None  # PipelineTask once built
        self._runner: Optional[Any] = None  # PipelineRunner instance
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

        Builds the pipeline lazily so any import errors surface only
        when the loop is actually selected (and so a missing optional
        dependency doesn't crash the daemon at startup for legacy
        users).
        """
        # Stage 2 lifts the gate to 2; the loop can technically boot
        # and process audio. Stages 3-5 layer adapters / tools / wake
        # gate on top; until those land we still don't recommend
        # running it in production.
        if _CURRENT_STAGE < _MIN_STAGE_FOR_RUN:
            debug_log(
                f"PipecatLoop running at stage {_CURRENT_STAGE}/"
                f"{_MIN_STAGE_FOR_RUN} — incomplete; expect "
                "missing HUD/tool wiring",
                "pipecat",
            )

        # Run the Pipecat task on its own event loop in this thread.
        # daemon.py owns the thread, so we own the loop. We import
        # PipelineRunner lazily for the same reason as the factory.
        from pipecat.pipeline.runner import PipelineRunner

        async def _amain() -> None:
            _, _, task, _ = _build_pipeline(self.cfg)
            self._task = task
            runner = PipelineRunner(handle_sigint=False)
            self._runner = runner
            debug_log("PipecatLoop starting PipelineRunner", "pipecat")
            try:
                await runner.run(task)
            except asyncio.CancelledError:
                debug_log("PipecatLoop runner cancelled", "pipecat")
            except Exception as exc:  # pragma: no cover — surface to logs
                debug_log(f"PipecatLoop runner crashed: {exc!r}", "pipecat")
                raise
            finally:
                debug_log("PipecatLoop runner stopped", "pipecat")

        asyncio.run(_amain())

    # ---------------------------------------------------------------- stop ---
    def stop(self) -> None:
        """Co-operative shutdown — cancels the PipelineTask.

        Pipecat's task.cancel() drains frames cleanly so any in-flight
        TTS audio finishes playing instead of cutting off mid-word.
        """
        self._stop_flag = True
        task = self._task
        if task is not None:
            try:
                # PipelineTask.cancel() is an async method; we can't
                # await here because stop() is called synchronously
                # from the signal handler. We schedule the cancel on
                # the runner's event loop if it's still running.
                loop = getattr(self._runner, "_loop", None)
                if loop is not None and not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(task.cancel(), loop)
                debug_log("PipecatLoop stop() — cancel scheduled", "pipecat")
            except Exception as exc:
                debug_log(f"PipecatLoop stop() error: {exc!r}", "pipecat")


__all__ = [
    "PipecatLoop",
    "PipecatLoopConfig",
    "from_settings",
    "_build_pipeline",  # exposed for unit tests / introspection
]

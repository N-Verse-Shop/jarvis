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
_CURRENT_STAGE = 3  # Stage 3: HUD adapters (events.jsonl + state.json)


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


# ───────────────────────── Stage-3 HUD adapters ──────────────────────────
#
# The Pipecat pipeline is brand-new but the Electron HUD has been polished
# across 30 audit rounds — it watches two files:
#
#   1. ``~/Library/Application Support/jarvis/events.jsonl``
#      Typed append-only JSONL of pipeline events. Schema is documented
#      in ``jarvis.ipc.stream``. The HUD streams partial transcripts,
#      LLM tokens, tool badges from this file.
#
#   2. ``~/Library/Application Support/jarvis/state.json``
#      Single-line JSON of ``{state, ts, level}``. The HUD's Three.js
#      coin reads this 4×/sec to pick the active animation
#      (IDLE / LISTENING / THINKING / SPEAKING).
#
# Stage 3 keeps those contracts 100% backwards-compatible — we map
# Pipecat frames to the same event shape the legacy listener was
# already producing, so the HUD code base needs ZERO changes.
#
# Two processors are inserted into the pipeline:
#
#   ``JarvisEventStreamProcessor`` — emits typed events. Inserted twice,
#       once on the user side (after STT, before LLM) so we capture
#       transcripts BEFORE the LLM consumes the frame, and once on the
#       assistant side (after LLM, before TTS) so we capture LLM tokens
#       and downstream TTS lifecycle frames.
#
#   ``JarvisStateProcessor`` — maintains the IDLE/LISTENING/THINKING/
#       SPEAKING state machine and writes state.json atomically. Sits
#       at the END of the pipeline so it sees every frame in its final
#       form (post-aggregation, post-TTS).
#
# Both are pure observers: they call ``super().process_frame()`` and
# ``self.push_frame()`` unmodified, never blocking or dropping a frame.


def _write_hud_state_atomic(state_value: str, level: float = 0.0) -> None:
    """Atomically write ``state.json`` for the Electron HUD coin.

    Mirrors :func:`desktop_app.face_widget._write_hud_state` byte-for-byte
    (pid + random suffix on tmp, ``os.replace`` rename, 0o600 perms,
    same payload schema). Duplicated locally instead of imported because
    ``desktop_app`` pulls in PyQt6 which we don't want as a runtime dep
    of the daemon's voice loop — keeping the layering clean.
    """
    import json as _json
    import os as _os
    import time as _time
    import uuid as _uuid

    base = Path.home() / "Library" / "Application Support" / "jarvis"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return  # best-effort — HUD overlay is non-critical
    target = base / "state.json"
    tmp = base / f"state.json.tmp.{_os.getpid()}.{_uuid.uuid4().hex[:8]}"
    payload = {
        "state": state_value.upper(),
        "ts": _time.time(),
        "level": float(level),
    }
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(_json.dumps(payload))
        try:
            _os.chmod(tmp, 0o600)
        except OSError:
            pass
        _os.replace(tmp, target)
    except OSError:
        # Clean up orphan tmp if rename failed
        try:
            _os.unlink(tmp)
        except OSError:
            pass


def _lang_to_iso(lang) -> str:
    """Best-effort mapping of pipecat ``Language`` enum to ISO-639-1.

    Falls back to the string repr if it's not an enum we know. The HUD
    accepts any short tag so this never needs to be exhaustive.
    """
    if lang is None:
        return ""
    # pipecat.transcriptions.language.Language is a string-valued Enum.
    name = getattr(lang, "value", None) or str(lang)
    name = str(name).lower()
    if name.startswith("uk"):
        return "uk"
    if name.startswith("ru"):
        return "ru"
    if name.startswith("en"):
        return "en"
    # last resort — strip dashes/underscores, take first 2 letters
    return name.split("-")[0].split("_")[0][:2]


def _make_event_stream_processor():
    """Build the events.jsonl bridging FrameProcessor.

    Built lazily inside :func:`_build_pipeline` because importing
    ``pipecat.processors.frame_processor`` at module top is expensive
    (it pulls in the whole pipecat runtime). We want this module to
    stay cheap to import for callers that only inspect the config /
    ``_CURRENT_STAGE`` (e.g. ``daemon.py`` deciding which engine to
    use).
    """
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        CancelFrame,
        EndFrame,
        Frame,
        InterimTranscriptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        StartFrame,
        TranscriptionFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream

    class JarvisEventStreamProcessor(FrameProcessor):
        """Pure observer — translates Pipecat frames → events.jsonl.

        Pass-through: every frame is forwarded unmodified after the
        observation hook runs. We never raise, never drop frames, never
        await blocking I/O (``get_stream().emit`` is non-blocking and
        line-buffered; max ~80 µs under contention).

        Sentence boundary detection
        ---------------------------

        Pipecat does not emit a ``SentenceFrame`` by default — the LLM
        streams ``LLMTextFrame`` chunks that the TTS service splits
        client-side. The HUD's caption renderer wants per-sentence
        flushes (one bubble per sentence), so we accumulate tokens and
        flush a ``sentence`` event whenever we see ``.``/``?``/``!``/
        ``…`` followed by whitespace or end-of-response.
        """

        # NOTE: ``__init__`` overrides MUST forward kwargs — pipecat's
        # base FrameProcessor pulls a couple of internal-only kwargs
        # like ``name=`` and ``sync=`` from the pipeline builder.
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._stream = get_stream()
            self._sentence_buf = ""  # accumulates LLMTextFrame chunks
            self._sentence_idx = 0   # monotonic per-utterance index
            self._tts_t0 = 0.0       # for tts_done duration calc
            self._utterance_t0 = 0.0  # for stt_final duration calc

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            # CRITICAL: super().process_frame is required for the
            # pipeline metrics / clock subsystem to track this
            # processor. Skipping it leaves Pipecat in an inconsistent
            # state and ``PipelineTask`` will hang on shutdown.
            await super().process_frame(frame, direction)
            try:
                self._observe(frame)
            except Exception as exc:  # pragma: no cover — observability path
                # Never crash the pipeline on an IPC failure — the
                # voice loop must keep working even if the HUD is dead.
                try:
                    from ..debug import debug_log
                    debug_log(
                        f"JarvisEventStreamProcessor observe failed: {exc!r}",
                        "pipecat",
                    )
                except Exception:
                    pass
            await self.push_frame(frame, direction)

        # ---- frame → event mapping ---------------------------------
        def _observe(self, frame: "Frame") -> None:
            import time as _time

            # ── user-side STT ──────────────────────────────────────
            if isinstance(frame, InterimTranscriptionFrame):
                if frame.text.strip():
                    self._stream.emit(
                        "stt_partial",
                        text=frame.text,
                        lang=_lang_to_iso(frame.language),
                    )
                return

            if isinstance(frame, TranscriptionFrame):
                # ``finalized`` distinguishes the truly-final cut from
                # mid-utterance fixups (rare on MLX whisper but the
                # streaming STT API can emit both).
                if not frame.text.strip():
                    return
                duration_ms = 0
                if self._utterance_t0:
                    duration_ms = int(
                        (_time.time() - self._utterance_t0) * 1000
                    )
                self._utterance_t0 = 0.0
                self._stream.emit(
                    "stt_final",
                    text=frame.text,
                    lang=_lang_to_iso(frame.language),
                    confidence=1.0,  # MLX whisper doesn't expose conf
                    duration_ms=duration_ms,
                )
                # Reset sentence accumulator at the start of every
                # new utterance — the LLM's reply is one full turn.
                self._sentence_buf = ""
                self._sentence_idx = 0
                return

            # ── start of user speech → utterance clock ─────────────
            if isinstance(frame, UserStartedSpeakingFrame):
                self._utterance_t0 = _time.time()
                self._stream.emit("vad", speaking=True, level=1.0)
                return

            if isinstance(frame, UserStoppedSpeakingFrame):
                self._stream.emit("vad", speaking=False, level=0.0)
                return

            # ── LLM streaming tokens ───────────────────────────────
            if isinstance(frame, LLMFullResponseStartFrame):
                self._sentence_buf = ""
                self._sentence_idx = 0
                self._stream.emit(
                    "llm_request",
                    model="",  # OLLamaLLMService doesn't expose it in the frame
                    num_messages=0,
                    num_tools=0,
                )
                return

            if isinstance(frame, LLMTextFrame):
                # Per-token event (used by HUD for streaming caption).
                self._stream.emit(
                    "token",
                    content=frame.text,
                    sentence_idx=self._sentence_idx,
                    total_chars=len(self._sentence_buf) + len(frame.text),
                )
                self._sentence_buf += frame.text
                # Flush a sentence on punctuation followed by space or
                # at the natural end of a clause. We're deliberately
                # lenient — false-positive sentence boundaries are
                # cosmetic (just more bubbles) but false-negatives mean
                # the HUD shows nothing until the whole response is
                # done streaming, which feels laggy.
                buf = self._sentence_buf
                if any(buf.rstrip().endswith(p) for p in (".", "!", "?", "…", "。")):
                    text = buf.strip()
                    if text:
                        self._stream.emit(
                            "sentence",
                            text=text,
                            idx=self._sentence_idx,
                        )
                        self._sentence_idx += 1
                    self._sentence_buf = ""
                return

            if isinstance(frame, LLMFullResponseEndFrame):
                # Flush any trailing fragment (response didn't end on
                # punctuation, e.g. a single-clause reply).
                tail = self._sentence_buf.strip()
                if tail:
                    self._stream.emit(
                        "sentence", text=tail, idx=self._sentence_idx
                    )
                    self._sentence_idx += 1
                self._sentence_buf = ""
                return

            # ── TTS lifecycle ──────────────────────────────────────
            if isinstance(frame, TTSStartedFrame):
                self._tts_t0 = _time.time()
                # We don't know the text-to-be-spoken here (Pipecat
                # passes only context_id); fall back to the last
                # sentence we observed.
                self._stream.emit(
                    "tts_start",
                    text="",
                    estimated_ms=0,
                )
                return

            if isinstance(frame, TTSStoppedFrame):
                duration_ms = 0
                if self._tts_t0:
                    duration_ms = int(
                        (_time.time() - self._tts_t0) * 1000
                    )
                self._tts_t0 = 0.0
                self._stream.emit("tts_done", duration_ms=duration_ms)
                return

    return JarvisEventStreamProcessor


def _make_state_processor():
    """Build the state.json bridging FrameProcessor.

    State machine
    -------------

    ::

        UserStartedSpeaking   → LISTENING
        UserStoppedSpeaking   → THINKING        (between EoS and LLM)
        LLMFullResponseStart  → THINKING        (LLM token stream begins)
        TTSStarted            → SPEAKING
        TTSStopped            → IDLE
        Cancel / End          → IDLE

    Multiple frames can map to the same state — we de-duplicate writes
    by tracking the last-written value. Only state transitions hit
    disk, so a typical turn produces 4 state.json writes
    (IDLE→LISTENING→THINKING→SPEAKING→IDLE), well under the HUD's
    250 ms poll interval.
    """
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        CancelFrame,
        EndFrame,
        Frame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        StartFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream

    class JarvisStateProcessor(FrameProcessor):
        """Maintains JarvisState and pushes state.json updates.

        Also mirrors the state to events.jsonl via a ``state`` event
        so any consumer that prefers the typed stream over a separate
        file watcher (e.g. the Tauri shell mentioned in stream.py
        docstring) sees the same transitions in real-time.
        """

        # Stable strings — matches the JarvisState enum in face_widget.
        S_IDLE = "IDLE"
        S_LISTENING = "LISTENING"
        S_THINKING = "THINKING"
        S_SPEAKING = "SPEAKING"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._current = self.S_IDLE
            self._stream = get_stream()
            # Seed the state file at construction time so the HUD has
            # a valid state before the first frame arrives.
            _write_hud_state_atomic(self.S_IDLE, level=0.0)

        def _transition(self, new_state: str, level: float = 0.0) -> None:
            if new_state == self._current:
                return
            self._current = new_state
            _write_hud_state_atomic(new_state, level=level)
            try:
                self._stream.emit("state", state=new_state, level=level)
            except Exception:
                pass

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)
            try:
                self._observe(frame)
            except Exception as exc:  # pragma: no cover — observability path
                try:
                    from ..debug import debug_log
                    debug_log(
                        f"JarvisStateProcessor observe failed: {exc!r}",
                        "pipecat",
                    )
                except Exception:
                    pass
            await self.push_frame(frame, direction)

        def _observe(self, frame: "Frame") -> None:
            # Order matters here because some frames coincide (a TTS
            # chunk can arrive while the user is already starting their
            # next utterance — interruption case). We always honour the
            # most-recent transition.
            if isinstance(frame, UserStartedSpeakingFrame):
                self._transition(self.S_LISTENING, level=1.0)
                return
            if isinstance(frame, UserStoppedSpeakingFrame):
                self._transition(self.S_THINKING, level=0.5)
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                self._transition(self.S_THINKING, level=0.5)
                return
            if isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame)):
                self._transition(self.S_SPEAKING, level=1.0)
                return
            if isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
                self._transition(self.S_IDLE, level=0.0)
                return
            if isinstance(frame, (CancelFrame, EndFrame)):
                self._transition(self.S_IDLE, level=0.0)
                return

    return JarvisStateProcessor


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

    # ── Stage-3 HUD adapters ───────────────────────────────────────
    # Two separate event-stream observer instances (one per pipeline
    # half) plus a single state processor at the tail.
    #
    # We instantiate the processor CLASSES lazily here because the
    # factories import pipecat — if pipecat isn't installed we want
    # ``from_settings`` / config helpers to still work for callers
    # that just want to introspect the loop without booting it.
    EventStreamProc = _make_event_stream_processor()
    StateProc = _make_state_processor()
    events_user = EventStreamProc()        # observes STT side
    events_assistant = EventStreamProc()   # observes LLM/TTS side
    state_proc = StateProc()               # tail — last word on state

    # ── pipeline ───────────────────────────────────────────────────
    # Topology:
    #   transport.input → STT → events_user → user-aggregator → LLM
    #     → assistant-aggregator → events_assistant → TTS
    #     → state_proc → transport.output
    #
    # Why this ordering:
    #
    # * ``events_user`` sits right after STT so we capture
    #   InterimTranscriptionFrame / TranscriptionFrame at the point
    #   they are emitted — before the LLM consumes/transforms them.
    #
    # * ``events_assistant`` sits between the assistant-aggregator and
    #   TTS so we see LLMTextFrame chunks AND the TTSStarted/Stopped
    #   bracket that wraps them. (UserStarted/StoppedSpeakingFrames
    #   from the transport propagate downstream too, so the state
    #   processor at the tail sees them all.)
    #
    # * ``state_proc`` is last so it observes every frame in its
    #   final, post-aggregation form. The single observer at the end
    #   removes any risk of two halves of the pipeline racing on
    #   conflicting state writes.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            events_user,
            aggregators.user(),
            llm,
            aggregators.assistant(),
            events_assistant,
            tts,
            state_proc,
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

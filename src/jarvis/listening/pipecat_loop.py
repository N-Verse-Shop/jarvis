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
_CURRENT_STAGE = 5  # Stage 5: wake-word gate + echo filter + feature flag


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
            "remote_whisper_url": getattr(cfg, "whisper_remote_url", ""),
            "remote_whisper_token": getattr(cfg, "whisper_remote_token", ""),
            # Stage-5 wake-word + hot-window knobs — reuse the same
            # config keys the legacy listener already consumes so
            # users don't have to re-tune their config.json.
            "wake_word": getattr(cfg, "wake_word", "jarvis"),
            "wake_aliases": list(getattr(cfg, "wake_aliases", []) or []),
            "wake_fuzzy_ratio": float(getattr(cfg, "wake_fuzzy_ratio", 0.78)),
            "hot_window_seconds": float(
                getattr(cfg, "hot_window_seconds", 30.0)
            ),
        },
    )


def _system_prompt_for(lang: str) -> str:
    """Build the full voice system prompt for the active language.

    Composition (in order, each section optional + best-effort):

      1. Persona block from ~/.config/jarvis/persona.md (R33-S1)
      2. Facts block — top decay-ranked user traits (R33-S2)
      3. Base voice prompt (RU or UK, hard-coded fallback)
      4. L1 skill catalog from ~/.config/jarvis/skills/ (R32-1)

    The persona section comes FIRST because it shapes the model's
    voice for the whole turn; facts come second because they're
    user-specific context the model should weave into responses;
    base prompt + skill catalog are operational instructions.

    Total budget: ~350 tokens. Persona ~120, facts ~80 (cap 600
    chars), base ~130, catalog ~50 at 5 skills. Cold-cache eval on
    qwen3:8b stays under ~250 ms.

    Lazy-imports are intentional: ``skills``, ``persona``, ``facts``
    are optional modules and must NEVER break the voice loop on
    import failure.
    """
    base = _VOICE_SYSTEM_PROMPT_UK if lang == "uk" else _VOICE_SYSTEM_PROMPT_RU
    parts: list[str] = []

    # 1. Persona (R33-S1)
    try:
        from ..persona import get_persona_store
        persona_block = get_persona_store().get().render_prompt_block(lang=lang)
        if persona_block:
            parts.append(persona_block)
    except Exception as exc:
        debug_log(f"persona block unavailable: {exc!r}", "pipecat")

    # 2. Facts (R33-S2)
    try:
        from ..memory.facts import render_facts_for_prompt
        facts_block = render_facts_for_prompt(key_prefix="user.", limit=8, max_chars=400)
        if facts_block:
            parts.append(facts_block)
    except Exception as exc:
        debug_log(f"facts block unavailable: {exc!r}", "pipecat")

    # 3. Base voice prompt
    parts.append(base)

    # 4. L1 skill catalog (R32-1)
    try:
        from ..skills import get_skill_store
        catalog = get_skill_store().catalog_block(active_locale=lang)
        if catalog:
            parts.append(catalog.lstrip("\n"))
    except Exception as exc:
        debug_log(f"skills catalog unavailable: {exc!r}", "pipecat")

    return "\n\n".join(p for p in parts if p)


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


# ──────────────────────── Stage-4 fast-path filter ───────────────────────
#
# The legacy listener used a ~50ms regex-only fast path that fired
# direct user commands without involving the LLM at all
# (``parse_user_command`` in ``action_dispatcher.py``). That path
# handled ~80% of voice commands ("відкрий Safari", "play music",
# "set volume 50") with near-zero latency.
#
# Stage 4 ports that to a Pipecat ``FrameProcessor`` that intercepts
# ``TranscriptionFrame`` between STT and the user-aggregator. On a
# match it:
#
#   1. Suppresses the transcript so the LLM never sees it.
#   2. Runs the action in a thread (synchronous ``subprocess.run``).
#   3. Emits ``tool_call`` events (starting → completed/failed).
#   4. Pushes a synthetic ``TTSSpeakFrame`` downstream so TTS speaks
#      the action's confirmation phrase ("Зараз відкрию Safari").
#
# On no match the frame is passed through unchanged — the LLM gets
# its normal chance to respond.


def _make_fast_path_processor():
    """Build the regex fast-path FrameProcessor.

    Wraps ``action_dispatcher.parse_user_command`` in a frame-aware
    filter. Action execution happens in the default thread executor
    because the underlying ops use blocking ``subprocess.run``
    (AppleScript / ``open -a``) that would stall the asyncio loop
    and disrupt audio frame timing.
    """
    import asyncio as _asyncio

    from pipecat.frames.frames import (
        Frame,
        TextFrame,
        TranscriptionFrame,
        TTSSpeakFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream
    from .action_dispatcher import Action, parse_user_command

    class JarvisFastPathProcessor(FrameProcessor):
        """Direct-execution path for matched regex user commands."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._stream = get_stream()
            # R33-S3 audit — lazy import so the legacy listener path
            # doesn't pay audit init cost when only Pipecat uses it.
            from ..audit import get_audit_store
            self._audit = get_audit_store()

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)

            # We only care about user-direction final transcripts.
            if not isinstance(frame, TranscriptionFrame):
                await self.push_frame(frame, direction)
                return
            text = (frame.text or "").strip()
            if not text:
                await self.push_frame(frame, direction)
                return

            try:
                action = parse_user_command(text)
            except Exception as exc:
                # Regex failure must never break the pipeline — fall
                # through to the LLM path so the user still gets a
                # response.
                try:
                    from ..debug import debug_log
                    debug_log(
                        f"fast-path parse_user_command crashed: {exc!r}",
                        "pipecat",
                    )
                except Exception:
                    pass
                await self.push_frame(frame, direction)
                return

            if action is None:
                # No fast-path match → normal LLM flow.
                await self.push_frame(frame, direction)
                return

            # R33-S4: capability gate check. If a fast-path action
            # corresponds to a closed-gate tool, fall through to the
            # LLM — the LLM will see only allowed tools in its schema
            # and respond conversationally instead of executing.
            try:
                from ..capabilities import is_tool_allowed
                if not is_tool_allowed(action.name):
                    self._audit.emit(
                        kind="gate_blocked",
                        tool=action.name,
                        status="blocked",
                        args={"text": text},
                    )
                    await self.push_frame(frame, direction)
                    return
            except Exception:
                pass

            # ── MATCH ─────────────────────────────────────────────
            # Suppress the original transcript (do not call push_frame
            # for it). Run the action in a worker thread so blocking
            # subprocess calls don't stall the event loop. Speak the
            # acknowledgement via a TTSSpeakFrame so the user hears
            # immediate feedback.
            self._stream.emit(
                "tool_call",
                tool=action.name,
                args={},
                status="starting",
            )
            self._audit.emit(
                kind="fast_path",
                tool=action.name,
                status="starting",
                args={"text": text},
            )
            import time as _time
            t0 = _time.monotonic()
            try:
                ok, msg = await _asyncio.to_thread(action.fn)
            except Exception as exc:
                ok, msg = False, f"fast-path exec error: {exc!r}"
            duration_ms = int((_time.monotonic() - t0) * 1000)
            self._stream.emit(
                "tool_call",
                tool=action.name,
                args={},
                status="completed" if ok else "failed",
                result=msg if ok else None,
                error=None if ok else msg,
            )
            self._audit.emit(
                kind="fast_path",
                tool=action.name,
                status="completed" if ok else "failed",
                args={"text": text},
                result=msg if ok else None,
                error=None if ok else msg,
                duration_ms=duration_ms,
            )

            # Speak the human-friendly description ("Зараз відкрию
            # Safari") regardless of success — the user still wants
            # confirmation that the daemon heard them. We pass it
            # downstream so the assistant-aggregator + TTS both see
            # it: ``append_to_context=True`` (default) means it ends
            # up in dialog history as if the LLM had said it.
            ack = action.description or ("Готово." if ok else "Не вдалося.")
            await self.push_frame(
                TTSSpeakFrame(text=ack),
                direction,
            )
            # DO NOT push the original TranscriptionFrame — that's
            # the whole point of the fast path.

    return JarvisFastPathProcessor


# ─────────────────── Stage-4 mac_control LLM tool bridge ─────────────────
#
# A curated subset of ``mac_control._OPS`` exposed to the LLM as
# OpenAI-compatible function-calls. We deliberately do NOT register
# all 25 ops — too many tools confuse a small 8B model. The fast-path
# above already covers the high-frequency commands; the LLM tools
# below are for compositional / context-sensitive cases the regex
# can't match (e.g. "remind me to call mom tomorrow at 4pm" — needs
# free-form text arg).


def _mac_control_tools_schema():
    """Return a :class:`ToolsSchema` for the curated mac_control ops.

    R33-S4: each tool is checked against
    :func:`jarvis.capabilities.is_tool_allowed` and DROPPED from the
    schema if its gate is closed. The LLM never sees the function in
    its tools list, so it can't be invoked even if hallucinated.
    """
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    from ..capabilities import is_tool_allowed

    # Keep this list small + well-described. The LLM picks by name +
    # description; a vague description gets the wrong tool picked.
    schemas = [
        FunctionSchema(
            name="focus_app",
            description=(
                "Bring a macOS application to the foreground. Use the "
                "canonical English app name (e.g. 'Safari', 'Mail', "
                "'Calendar'). For a known app the user named in a "
                "non-English language, translate first."
            ),
            properties={
                "app": {
                    "type": "string",
                    "description": "Canonical app name, e.g. 'Safari'",
                },
            },
            required=["app"],
        ),
        FunctionSchema(
            name="open_url",
            description=(
                "Open a URL in the user's default browser. Always "
                "include the scheme (https://...)."
            ),
            properties={
                "url": {
                    "type": "string",
                    "description": "Full URL with scheme",
                },
            },
            required=["url"],
        ),
        FunctionSchema(
            name="new_note",
            description=(
                "Create a new Note in Apple Notes with the given "
                "title and body. Use for free-form text the user "
                "wants saved (e.g. 'note: meeting agenda ...')."
            ),
            properties={
                "title": {"type": "string", "description": "Note title"},
                "body": {"type": "string", "description": "Note body text"},
            },
            required=["title"],
        ),
        FunctionSchema(
            name="new_reminder",
            description=(
                "Create a reminder in Apple Reminders. The text is "
                "what the user wants to be reminded of. Specifying a "
                "list_name is optional — defaults to the main list."
            ),
            properties={
                "text": {"type": "string", "description": "Reminder body"},
                "list_name": {
                    "type": "string",
                    "description": "Optional: target list name",
                },
            },
            required=["text"],
        ),
        FunctionSchema(
            name="query_calendar",
            description=(
                "List the user's upcoming calendar events for the "
                "next N days. Use when the user asks 'what's on my "
                "calendar' / 'what's my next meeting'."
            ),
            properties={
                "days": {
                    "type": "integer",
                    "description": "Look-ahead window, days (default 7)",
                },
            },
            required=[],
        ),
        FunctionSchema(
            name="send_message",
            description=(
                "Send an iMessage / SMS via Messages.app. ``to`` is "
                "a phone number or contact name."
            ),
            properties={
                "to": {
                    "type": "string",
                    "description": "Phone number or contact name",
                },
                "body": {
                    "type": "string",
                    "description": "Message body",
                },
            },
            required=["to", "body"],
        ),
        FunctionSchema(
            name="run_shortcut",
            description=(
                "Run a macOS Shortcut by name. Use when the user "
                "asks 'run the X shortcut' or refers to an automation "
                "they've set up in Shortcuts.app."
            ),
            properties={
                "name": {
                    "type": "string",
                    "description": "Exact shortcut name",
                },
                "input_text": {
                    "type": "string",
                    "description": "Optional text input passed to the shortcut",
                },
            },
            required=["name"],
        ),
        FunctionSchema(
            name="list_shortcuts",
            description=(
                "List the names of all macOS Shortcuts available on "
                "this Mac. Use when the user asks 'what shortcuts do "
                "I have' / 'which shortcuts can you run'."
            ),
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="system_info",
            description=(
                "Return a single piece of system information. ``field`` "
                "is one of: ``battery``, ``volume``, ``focus_mode``, "
                "``uptime``, ``all``."
            ),
            properties={
                "field": {
                    "type": "string",
                    "description": "battery | volume | focus_mode | uptime | all",
                },
            },
            required=["field"],
        ),
        FunctionSchema(
            name="set_volume",
            description=(
                "Set the macOS output volume. ``level`` is 0-100."
            ),
            properties={
                "level": {
                    "type": "integer",
                    "description": "0-100",
                },
            },
            required=["level"],
        ),
        FunctionSchema(
            name="set_mute",
            description=(
                "Mute or unmute the macOS output. ``state`` is "
                "'on' (mute) or 'off' (unmute)."
            ),
            properties={
                "state": {
                    "type": "string",
                    "description": "on | off",
                },
            },
            required=["state"],
        ),
        FunctionSchema(
            name="clipboard_set",
            description=(
                "Replace the macOS clipboard contents with the given "
                "text. Use when the user asks to 'copy X to the "
                "clipboard'."
            ),
            properties={
                "text": {
                    "type": "string",
                    "description": "Text to place on the clipboard",
                },
            },
            required=["text"],
        ),
        # ── R32 skill loaders ──────────────────────────────────
        # See ``jarvis.skills.store`` for the L1/L2/L3 model. The L1
        # catalog appears in the system prompt; these tools let the
        # LLM expand to L2/L3 when a skill matches the task.
        FunctionSchema(
            name="list_skills",
            description=(
                "List the names + one-line descriptions of all "
                "skills available in this Jarvis workspace. Use when "
                "the L1 catalog in the system prompt didn't show the "
                "skill you need, or before deciding which skill fits "
                "the user's request."
            ),
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="load_skill",
            description=(
                "Load the full SKILL.md protocol for the given skill "
                "name. Call this BEFORE attempting a complex task "
                "that matches an L1 catalog entry — the SKILL.md "
                "tells you the step-by-step protocol, tools to use, "
                "and expected output shape. Returns the full "
                "markdown content."
            ),
            properties={
                "name": {
                    "type": "string",
                    "description": (
                        "Exact skill name from the L1 catalog "
                        "(e.g. 'research-brief')."
                    ),
                },
            },
            required=["name"],
        ),
        FunctionSchema(
            name="load_skill_reference",
            description=(
                "Load a supporting reference file from a skill's "
                "``references/`` directory. Use only after "
                "``load_skill`` has been called and that SKILL.md "
                "explicitly pointed at the reference."
            ),
            properties={
                "name": {
                    "type": "string",
                    "description": "Skill name",
                },
                "reference": {
                    "type": "string",
                    "description": (
                        "Reference file stem (without .md). E.g. "
                        "'EXAMPLE' for 'references/EXAMPLE.md'."
                    ),
                },
            },
            required=["name", "reference"],
        ),
    ]
    # R33-S4: filter the schema by gate state. Closed gates don't
    # appear in the LLM's function list at all.
    filtered = [s for s in schemas if is_tool_allowed(s.name)]
    if len(filtered) != len(schemas):
        dropped = sorted(
            s.name for s in schemas if not is_tool_allowed(s.name)
        )
        debug_log(
            f"capability gates dropped {len(schemas) - len(filtered)} "
            f"tool(s) from LLM schema: {dropped}",
            "pipecat",
        )
    return ToolsSchema(standard_tools=filtered)


def _register_mac_control_handlers(llm):
    """Register one async handler per curated tool on the LLM service.

    Each handler bridges to :func:`mac_control._dispatch_op` and
    pipes the result through the function-call ``result_callback``.
    Execution runs in a thread because the ops are synchronous
    blocking subprocess calls — same reason as the fast path.

    Also emits ``tool_call`` events for HUD observability so the
    user can see which tool the LLM chose and whether it worked.
    """
    import asyncio as _asyncio
    import time as _time

    from ..ipc import get_stream
    from ..tools.builtin.mac_control import _dispatch_op, _OPS

    stream = get_stream()

    # The names we expose — must match the schemas above. Restrict to
    # the curated set + sanity-check against the underlying op
    # registry so a typo in the schema (or a removed op in
    # mac_control) fails loudly at registration time.
    # R33-S4: also filter by capability gate so closed-gate handlers
    # don't get registered at all (otherwise a hallucinated function
    # call could find them).
    from ..capabilities import is_tool_allowed
    candidate_exposed = [
        "focus_app", "open_url", "new_note", "new_reminder",
        "query_calendar", "send_message", "run_shortcut",
        "list_shortcuts", "system_info", "set_volume",
        "set_mute", "clipboard_set",
    ]
    exposed = [n for n in candidate_exposed if is_tool_allowed(n)]
    for op_name in exposed:
        if op_name not in _OPS:
            # Don't silently skip — surface the misconfiguration so
            # the dev sees it the first time the loop is built.
            raise RuntimeError(
                f"Cannot register Pipecat function {op_name!r}: not in "
                f"mac_control._OPS (registry has {len(_OPS)} ops). "
                "Update _mac_control_tools_schema or fix mac_control."
            )

    # R33-S3: audit store also receives each tool_call so the
    # dashboard can query "failed focus_app in last 24h" etc.
    from ..audit import get_audit_store
    audit = get_audit_store()

    def _make_handler(op_name: str):
        async def _handler(params) -> None:
            args = dict(params.arguments or {})
            stream.emit(
                "tool_call",
                tool=op_name,
                args=args,
                status="starting",
            )
            audit.emit(
                kind="tool_call",
                tool=op_name,
                status="starting",
                args=args,
            )
            t0 = _time.monotonic()
            try:
                ok, msg = await _asyncio.to_thread(
                    _dispatch_op, op_name, args
                )
            except Exception as exc:
                ok, msg = False, f"dispatch error: {exc!r}"
            duration_ms = int((_time.monotonic() - t0) * 1000)
            stream.emit(
                "tool_call",
                tool=op_name,
                args=args,
                status="completed" if ok else "failed",
                result=msg if ok else None,
                error=None if ok else msg,
            )
            audit.emit(
                kind="tool_call",
                tool=op_name,
                status="completed" if ok else "failed",
                args=args,
                result=msg if ok else None,
                error=None if ok else msg,
                duration_ms=duration_ms,
            )
            # Pipecat callback delivers the result back to the LLM
            # so it can continue the response ("Зробив. Що далі?").
            try:
                await params.result_callback(
                    {"ok": ok, "message": msg}
                )
            except Exception as exc:
                try:
                    from ..debug import debug_log
                    debug_log(
                        f"result_callback failed for {op_name}: {exc!r}",
                        "pipecat",
                    )
                except Exception:
                    pass

        return _handler

    for op_name in exposed:
        llm.register_function(op_name, _make_handler(op_name))


def _register_skill_handlers(llm) -> None:
    """Register the three skill-loader function-calls.

    Separate from the mac_control bridge because skills don't go
    through ``_dispatch_op`` — they're a pure read-side accessor
    on the local skill store. No subprocess, no AppleScript, no
    blocking I/O beyond a small file read. We still hop to a
    thread to keep the asyncio loop responsive in case the
    SKILL.md is unusually large.
    """
    import asyncio as _asyncio

    from ..ipc import get_stream
    from ..skills import get_skill_store

    stream = get_stream()
    store = get_skill_store()

    async def _h_list(params) -> None:
        stream.emit(
            "tool_call", tool="list_skills", args={}, status="starting"
        )
        try:
            skills = await _asyncio.to_thread(store.list_skills)
            payload = [
                {
                    "name": s.name,
                    "description": s.description,
                    "tags": s.tags,
                    "risk": s.risk,
                    "has_references": bool(s.references),
                }
                for s in skills
            ]
            stream.emit(
                "tool_call",
                tool="list_skills",
                args={},
                status="completed",
                result={"count": len(payload)},
            )
            await params.result_callback({"skills": payload})
        except Exception as exc:
            stream.emit(
                "tool_call",
                tool="list_skills",
                args={},
                status="failed",
                error=str(exc),
            )
            await params.result_callback({"ok": False, "error": str(exc)})

    async def _h_load(params) -> None:
        args = dict(params.arguments or {})
        name = str(args.get("name", "")).strip()
        stream.emit(
            "tool_call", tool="load_skill", args=args, status="starting"
        )
        try:
            skill = await _asyncio.to_thread(store.get_skill, name)
            if skill is None:
                msg = (
                    f"Unknown skill {name!r}. Call list_skills() to "
                    "see available names."
                )
                stream.emit(
                    "tool_call",
                    tool="load_skill",
                    args=args,
                    status="failed",
                    error=msg,
                )
                await params.result_callback({"ok": False, "error": msg})
                return
            stream.emit(
                "tool_call",
                tool="load_skill",
                args=args,
                status="completed",
                result={"chars": len(skill.content)},
            )
            await params.result_callback(
                {
                    "ok": True,
                    "name": skill.name,
                    "description": skill.description,
                    "content": skill.content,
                    "references": sorted(skill.references.keys()),
                    "risk": skill.risk,
                    "tools": skill.tools,
                }
            )
        except Exception as exc:
            stream.emit(
                "tool_call",
                tool="load_skill",
                args=args,
                status="failed",
                error=str(exc),
            )
            await params.result_callback({"ok": False, "error": str(exc)})

    async def _h_load_ref(params) -> None:
        args = dict(params.arguments or {})
        name = str(args.get("name", "")).strip()
        ref = str(args.get("reference", "")).strip()
        stream.emit(
            "tool_call",
            tool="load_skill_reference",
            args=args,
            status="starting",
        )
        try:
            text = await _asyncio.to_thread(
                store.load_reference, name, ref
            )
            if text is None:
                msg = (
                    f"No reference {ref!r} found on skill {name!r}. "
                    "Call load_skill first to see its 'references' "
                    "list."
                )
                stream.emit(
                    "tool_call",
                    tool="load_skill_reference",
                    args=args,
                    status="failed",
                    error=msg,
                )
                await params.result_callback({"ok": False, "error": msg})
                return
            stream.emit(
                "tool_call",
                tool="load_skill_reference",
                args=args,
                status="completed",
                result={"chars": len(text)},
            )
            await params.result_callback(
                {"ok": True, "name": name, "reference": ref, "content": text}
            )
        except Exception as exc:
            stream.emit(
                "tool_call",
                tool="load_skill_reference",
                args=args,
                status="failed",
                error=str(exc),
            )
            await params.result_callback({"ok": False, "error": str(exc)})

    llm.register_function("list_skills", _h_list)
    llm.register_function("load_skill", _h_load)
    llm.register_function("load_skill_reference", _h_load_ref)


# ──────────────── Stage-5 wake-word gate + echo filter ───────────────────
#
# Two more pre-LLM filters added in Stage 5:
#
#   * ``JarvisWakeWordGateProcessor`` — requires either a wake word in
#     the utterance OR an active hot window before any transcript
#     reaches the fast-path / LLM. Outside the hot window ambient
#     speech is silently dropped (we don't reply unless explicitly
#     addressed). A successful interaction extends the window so the
#     user can follow up without saying "jarvis" again.
#
#   * ``JarvisEchoFilterProcessor`` — suppresses transcripts that
#     arrive while TTS is actively playing (plus a short tail after
#     TTS stops). Pipecat's transport-level interruption handling
#     covers most echo cases, but the OS speaker/mic cross-talk path
#     occasionally leaks a partial transcript of our own bot. We
#     defence-in-depth by dropping any TranscriptionFrame that
#     coincides with TTS playback.
#
# Both sit BEFORE the fast-path so they short-circuit the highest
# layers — no point parsing a regex on a transcript we're going to
# discard.


def _make_wake_word_processor(cfg: "PipecatLoopConfig"):
    """Build the wake-word gate FrameProcessor.

    Behaviour:

    * Outside the hot window: drop ``TranscriptionFrame`` unless the
      transcript contains a wake-word. On wake-word, open a hot
      window of ``hot_window_seconds`` and forward the text AFTER
      the wake word.
    * Inside the hot window: forward all transcripts unchanged.
    * On every assistant TTS turn (``TTSStoppedFrame``): refresh the
      hot window so follow-ups don't need a fresh wake word.

    Reuses :func:`wake_detection.is_wake_word_detected` and
    :func:`wake_detection.extract_query_after_wake` so the heuristics
    (fuzzy ratio, prefix tolerance, alias list) stay identical to the
    legacy listener.
    """
    import time as _time

    from pipecat.frames.frames import (
        Frame,
        TranscriptionFrame,
        TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream
    from .wake_detection import (
        extract_query_after_wake,
        is_wake_word_detected,
    )

    # Pull legacy settings from config.extra so we honour the same
    # wake-word knobs the legacy listener used (wake_word, wake_aliases,
    # wake_fuzzy_ratio, hot_window_seconds).
    wake_word = str(cfg.extra.get("wake_word") or cfg.wake_words[0])
    wake_aliases = list(cfg.extra.get("wake_aliases") or list(cfg.wake_words[1:]))
    fuzzy_ratio = float(cfg.extra.get("wake_fuzzy_ratio", 0.78))
    hot_window_seconds = float(cfg.extra.get("hot_window_seconds", 30.0))

    class JarvisWakeWordGateProcessor(FrameProcessor):
        """Drop ambient transcripts; require wake word or hot window."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._hot_until: float = 0.0  # unix ts; 0 = closed
            self._stream = get_stream()

        def _open_hot_window(self) -> None:
            self._hot_until = _time.time() + hot_window_seconds
            try:
                self._stream.emit(
                    "hot_window",
                    active=True,
                    expires_in_ms=int(hot_window_seconds * 1000),
                )
            except Exception:
                pass

        def _is_hot(self) -> bool:
            return _time.time() < self._hot_until

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)

            # Bot just finished speaking — extend the hot window so
            # the user can follow up without re-saying the wake word.
            if isinstance(frame, TTSStoppedFrame):
                self._open_hot_window()
                await self.push_frame(frame, direction)
                return

            # Non-transcript frames pass through.
            if not isinstance(frame, TranscriptionFrame):
                await self.push_frame(frame, direction)
                return

            text = (frame.text or "").strip()
            if not text:
                await self.push_frame(frame, direction)
                return

            text_lower = text.lower()
            wake_hit = is_wake_word_detected(
                text_lower, wake_word, wake_aliases, fuzzy_ratio
            )

            # Hot window open → pass everything.
            if self._is_hot():
                # If the wake word IS present in a follow-up, strip
                # it so the LLM sees the bare command.
                if wake_hit:
                    query = extract_query_after_wake(
                        text_lower, wake_word, wake_aliases
                    )
                    if query:
                        # Build a new frame with the trimmed text;
                        # frozen dataclasses → use replace().
                        from dataclasses import replace as _replace
                        try:
                            frame = _replace(frame, text=query)
                        except Exception:
                            pass
                self._open_hot_window()  # refresh
                await self.push_frame(frame, direction)
                return

            # Hot window closed → require wake word.
            if not wake_hit:
                # Silent drop — emit a log event so the HUD can show
                # "(ignored: no wake word)" if it wants.
                try:
                    self._stream.emit(
                        "log",
                        level="DEBUG",
                        component="wake-gate",
                        message="dropped (no wake word, hot window closed)",
                    )
                except Exception:
                    pass
                # Do NOT push the frame — gate is closed.
                return

            # Wake hit while cold → open window + forward bare query.
            self._stream.emit(
                "wake_word",
                word=wake_word,
                confidence=1.0,
            )
            self._open_hot_window()
            query = extract_query_after_wake(
                text_lower, wake_word, wake_aliases
            )
            if query:
                from dataclasses import replace as _replace
                try:
                    frame = _replace(frame, text=query)
                except Exception:
                    pass
            else:
                # Wake word alone with no follow-up command — drop
                # the frame but keep the window open so the user
                # can speak the actual command next.
                return
            await self.push_frame(frame, direction)

    return JarvisWakeWordGateProcessor


def _make_echo_filter_processor():
    """Build the echo-filter FrameProcessor.

    Tracks ``BotStartedSpeakingFrame``/``BotStoppedSpeakingFrame``
    boundaries (and ``TTSStartedFrame``/``TTSStoppedFrame`` as
    secondary signals). A ``TranscriptionFrame`` that arrives while
    the bot is speaking is treated as echo and dropped. We also keep
    a small tail (``_TAIL_SEC``) after TTS stops because hardware
    audio latency means the speaker is still outputting samples for
    ~200-500 ms after the frame finishes.
    """
    import time as _time

    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        Frame,
        TranscriptionFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream

    # Tail period — audio hardware finishes draining the last buffer
    # well after the BotStoppedSpeaking frame fires. Tuned empirically
    # on macOS CoreAudio + Piper TTS; raise if cross-talk leaks
    # through, lower if the user feels they can't interrupt.
    _TAIL_SEC = 0.5

    class JarvisEchoFilterProcessor(FrameProcessor):
        """Drop transcripts that arrive during bot's own speech."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._speaking = False
            self._speak_end_ts: float = 0.0
            self._stream = get_stream()

        def _is_blocking_window(self) -> bool:
            if self._speaking:
                return True
            if self._speak_end_ts and _time.time() < self._speak_end_ts:
                return True
            return False

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)

            if isinstance(frame, (BotStartedSpeakingFrame, TTSStartedFrame)):
                self._speaking = True
                self._speak_end_ts = 0.0
            elif isinstance(frame, (BotStoppedSpeakingFrame, TTSStoppedFrame)):
                self._speaking = False
                self._speak_end_ts = _time.time() + _TAIL_SEC

            if isinstance(frame, TranscriptionFrame):
                if self._is_blocking_window():
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="echo-filter",
                            message=(
                                f"dropped transcript during TTS "
                                f"(speaking={self._speaking}, "
                                f"tail={self._speak_end_ts - _time.time():.2f}s)"
                            ),
                        )
                    except Exception:
                        pass
                    return  # drop

            await self.push_frame(frame, direction)

    return JarvisEchoFilterProcessor


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

    # ── Stage-4 tool schema + function-call registration ────────────
    # Build the tools schema BEFORE the LLMContext so we can pass it
    # into the context (the LLM reads tools from the context at every
    # turn). Register handlers on the LLM service itself.
    # R33-S4: log capability gate state once at pipeline build so the
    # user can see which categories of tools are currently enabled.
    try:
        from ..capabilities import log_gate_state
        log_gate_state()
    except Exception:
        pass
    tools_schema = _mac_control_tools_schema()
    _register_mac_control_handlers(llm)
    # R32 — skill loaders (list_skills / load_skill / load_skill_reference)
    # are already in the schema; register their handlers too.
    try:
        _register_skill_handlers(llm)
    except Exception as exc:
        debug_log(
            f"skill handlers unavailable: {exc!r} (catalog still works)",
            "pipecat",
        )

    # ── context aggregators ────────────────────────────────────────
    system_prompt = _system_prompt_for(cfg.active_language)
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=tools_schema,
    )
    aggregators = LLMContextAggregatorPair(context)

    # ── Stage-3 HUD adapters + Stage-5 gates ───────────────────────
    # We instantiate the processor CLASSES lazily here because the
    # factories import pipecat — if pipecat isn't installed we want
    # ``from_settings`` / config helpers to still work for callers
    # that just want to introspect the loop without booting it.
    EventStreamProc = _make_event_stream_processor()
    StateProc = _make_state_processor()
    FastPathProc = _make_fast_path_processor()
    WakeWordProc = _make_wake_word_processor(cfg)
    EchoFilterProc = _make_echo_filter_processor()
    events_user = EventStreamProc()        # observes STT side
    events_assistant = EventStreamProc()   # observes LLM/TTS side
    echo_filter = EchoFilterProc()         # drops bot-echo transcripts
    wake_gate = WakeWordProc()             # requires wake word / hot window
    fast_path = FastPathProc()             # regex shortcut, pre-LLM
    state_proc = StateProc()               # tail — last word on state

    # ── pipeline ───────────────────────────────────────────────────
    # Topology:
    #   transport.input → STT → events_user → fast_path → user-aggregator
    #     → LLM → assistant-aggregator → events_assistant → TTS
    #     → state_proc → transport.output
    #
    # Why this ordering:
    #
    # * ``events_user`` sits right after STT so we capture
    #   InterimTranscriptionFrame / TranscriptionFrame at the point
    #   they are emitted — before the LLM consumes/transforms them.
    #
    # * ``fast_path`` sits between events_user and the user-aggregator
    #   so it can intercept ``TranscriptionFrame`` and short-circuit
    #   the LLM path entirely on regex match (~50ms latency vs
    #   ~500ms LLM round-trip). The events_user observer ABOVE has
    #   already emitted stt_final for the HUD, so the user sees the
    #   transcript even though the LLM never gets it.
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
    # Final topology (Stage 5):
    #   transport.input → STT → events_user → echo_filter → wake_gate
    #     → fast_path → user-aggregator → LLM → assistant-aggregator
    #     → events_assistant → TTS → state_proc → transport.output
    #
    # Ordering rationale:
    #
    # * ``echo_filter`` is BEFORE ``wake_gate`` so transcripts created
    #   from our own TTS audio never even reach the wake-word check
    #   (otherwise the bot's own "Зараз відкрию Safari" could
    #   self-trigger if it contained the wake word "джарвіс").
    # * ``wake_gate`` is BEFORE ``fast_path`` because executing an
    #   action without the user actually addressing us is a much
    #   worse failure than not executing one they did. Conservative.
    # * Both gates observe ``TTSStartedFrame``/``TTSStoppedFrame``/
    #   ``BotStartedSpeakingFrame``/``BotStoppedSpeakingFrame`` that
    #   propagate from the transport's interruption logic, so the
    #   ordering above still lets them see TTS lifecycle frames.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            events_user,
            echo_filter,
            wake_gate,
            fast_path,
            aggregators.user(),
            llm,
            aggregators.assistant(),
            events_assistant,
            tts,
            state_proc,
            transport.output(),
        ]
    )

    # ── R33-S7: pre-warm the voice stack in background threads ────
    # Primes Ollama KV cache for the exact system prompt + Whisper
    # MLX graph + Piper ONNX session so the FIRST user utterance
    # after a daemon restart doesn't eat 500-700 ms of cold-cache
    # pain. Threads are daemon-mode so they die with the process.
    # Failures log + continue — never block pipeline startup.
    try:
        from .warmup import warmup_voice_stack
        whisper_repo = None
        try:
            from .listener import _get_mlx_model_repo
            normalised = cfg.stt_mlx_model.lower().replace("_", "-")
            whisper_repo = _get_mlx_model_repo(normalised)
        except Exception:
            pass
        warmup_voice_stack(
            ollama_base_url=cfg.ollama_base_url,
            chat_model=cfg.chat_model,
            system_prompt=system_prompt,
            whisper_mlx_repo=whisper_repo,
            piper_voice_id=cfg.piper_voice_id,
            piper_dir=piper_dir,
        )
    except Exception as _warm_exc:
        debug_log(f"warmup kick-off failed: {_warm_exc!r}", "pipecat")

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


# ─────────────────────────── Thread wrapper ──────────────────────────────


class PipecatVoiceThread:
    """``threading.Thread``-shaped adapter so daemon.py can hot-swap.

    Mirrors the public surface of ``listening.listener.VoiceListener``:

    * Construct with ``(db, cfg, tts, dialogue_memory)`` — extra args
      are accepted and ignored (Pipecat manages its own audio + TTS
      + memory via the LLMContext).
    * ``.start()`` spawns a thread that drives the Pipecat loop.
    * ``.join(timeout)`` and ``.is_alive()`` work like ``Thread``.
    * ``._should_stop`` / ``._dictation_active`` attributes are set
      by daemon.py — we honour ``_should_stop`` to break out.
    """

    def __init__(self, db, cfg, tts, dialogue_memory) -> None:
        # We don't actually use db/tts/memory — Pipecat owns those.
        # Stored for diagnostic access by tests.
        self._db = db
        self._cfg = cfg
        self._tts = tts
        self._dialogue_memory = dialogue_memory
        self._should_stop = False
        self._dictation_active = False
        self._loop_cfg = from_settings(cfg)
        self._loop = PipecatLoop(self._loop_cfg)
        import threading as _threading
        # DictationEngine wants a shared ``transcribe_lock`` so its
        # MLX transcribe call serialises against the voice listener's.
        # Pipecat owns its own internal WhisperSTTServiceMLX instance
        # which doesn't actually share state with mlx_whisper.transcribe
        # used by dictation, BUT exposing the same attribute keeps
        # ``daemon.py`` engine-agnostic — and a process-wide lock costs
        # nothing in the common case where dictation isn't recording.
        self.transcribe_lock = _threading.Lock()
        self._thread = _threading.Thread(
            target=self._run_thread,
            name="PipecatVoiceThread",
            daemon=True,
        )

    def _run_thread(self) -> None:
        try:
            debug_log(
                "PipecatVoiceThread starting (engine=pipecat, stage="
                f"{_CURRENT_STAGE}/{_MIN_STAGE_FOR_RUN})",
                "pipecat",
            )
            self._loop.run()
        except Exception as exc:
            debug_log(f"PipecatVoiceThread crashed: {exc!r}", "pipecat")
        finally:
            debug_log("PipecatVoiceThread exited", "pipecat")

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._should_stop = True
        self._loop.stop()

    # The DictationEngine calls these refs every time the
    # hold-to-dictate hotkey fires. Even though Pipecat owns its own
    # audio loop, dictation does NOT — it captures separately and
    # transcribes via the shared MLX repo. We expose ``"mlx"`` as
    # the backend and the canonical MLX-community repo for the
    # configured whisper model, matching exactly what the legacy
    # listener would have returned. ``model`` stays None because
    # MLX transcription is stateless (loads from repo each call).
    @property
    def model(self):  # pragma: no cover — diagnostic accessor
        return None

    @property
    def _whisper_backend(self):  # pragma: no cover
        # Always "mlx" on Apple Silicon — Pipecat's WhisperSTTServiceMLX
        # uses the same MLX backend dictation needs.
        return "mlx"

    @property
    def _mlx_model_repo(self):  # pragma: no cover
        # Reuse the legacy MLX-repo resolver so dictation gets the
        # same repo Pipecat is using. ``whisper_model`` config field
        # (e.g. "large-v3-turbo") drives the lookup.
        try:
            from .listener import _get_mlx_model_repo
            model_name = str(getattr(self._cfg, "whisper_model", "large-v3-turbo"))
            # Normalise the Pipecat-flavoured enum name back to the
            # legacy short form. Pipecat: "LARGE_V3_TURBO" → legacy:
            # "large-v3-turbo".
            normalised = model_name.lower().replace("_", "-")
            return _get_mlx_model_repo(normalised)
        except Exception:
            return None


__all__ = [
    "PipecatLoop",
    "PipecatLoopConfig",
    "PipecatVoiceThread",
    "from_settings",
    "_build_pipeline",  # exposed for unit tests / introspection
]

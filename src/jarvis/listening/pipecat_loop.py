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
import re as _re
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
    "Никаких блоков кода. "
    # R34-S43 — strict confirmation gate.
    "НИКОГДА не выполняй действия без явного согласия пользователя. "
    "Любое действие на машине (открыть/закрыть приложение, написать "
    "файл, отправить письмо, изменить настройки) сначала ОПИШИ "
    "одним коротким предложением и закончи вопросом «Подтверждаешь?». "
    "ЖДИ его «да» прежде чем что-то делать. Если он сказал «нет» "
    "или «стоп» — спокойно подтверди отмену. "
    "Имя Данило склоняй грамматически правильно (Данила/Данилу/Даниле). "
    "Никогда не сокращай до «Нило» или «Даня»."
)
# R34-S52 C-9: _VOICE_SYSTEM_PROMPT_UK removed. R34-S48/S51 made
# Russian the only outbound language; this constant was dead under the
# RU pin in _system_prompt_for() but kept tempting future regressions
# every time someone touched language gating. Voice pipeline is now
# RU-only end-to-end.


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
    # Silero VAD confidence cutoff. R34-S14: recalibrated to 0.42 after
    # live probe on Danylo's MBP M1 mic showed real speech only peaks
    # at ~0.73 and lives in 0.40-0.55 range — not the 0.99+ that wider
    # benchmark recordings would suggest. 0.62 (R34-S11) was caught
    # only 1 % of speech frames; 0.42 catches the meaty middle of real
    # utterances and leaves Whisper-side hallucination filtering to
    # drop the false-positives that slip through. Lift via
    # vad_aggressiveness bridge in :func:`from_settings`.
    vad_threshold: float = 0.42
    vad_min_silence_ms: int = 700

    # Behaviour ----------------------------------------------------------------
    # Active language seeds the language-aware Piper transliteration
    # (RU vs UA) — same logic as legacy ``_sanitize_for_piper_uk``.
    active_language: str = "ru"

    # Misc fields filled at runtime by ``from_settings``.
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_input_device_index(
    explicit_index: Optional[int],
    name_substring: Optional[str],
) -> Optional[int]:
    """Resolve a PyAudio input device index from config.

    Resolution order (first match wins):

    1. ``input_device_index`` numeric — pass through (legacy/exact path)
    2. ``voice_device`` substring — case-insensitive match against
       ``pa.get_device_info_by_index(i)['name']`` for any device with
       ``maxInputChannels > 0``
    3. ``None`` — Pipecat falls back to PortAudio default

    Why this matters: on macOS, Continuity Camera routes the iPhone
    microphone as input[0] and quietly takes over the system default.
    A legacy comment in this user's config.json captured the bug
    perfectly — "daemon listened to silent iPhone mic → wake-word
    never fired". The Pipecat path bypassed that bridge for two
    rounds (R31..R34-S15) because we only read ``input_device_index``
    and never the user-friendly ``voice_device`` substring.
    """
    if explicit_index is not None:
        try:
            return int(explicit_index)
        except (TypeError, ValueError):
            pass
    if not name_substring or not str(name_substring).strip():
        return None
    name_lc = str(name_substring).strip().lower()
    try:
        import pyaudio  # lazy import — heavy module
        pa = pyaudio.PyAudio()
        try:
            best_idx: Optional[int] = None
            best_default = False
            try:
                default_idx = pa.get_default_input_device_info()["index"]
            except (OSError, KeyError):
                default_idx = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                if name_lc not in str(info.get("name", "")).lower():
                    continue
                # Prefer the device that's already the system default
                # if multiple matches (saves a re-open).
                is_default = (i == default_idx)
                if best_idx is None or (is_default and not best_default):
                    best_idx = i
                    best_default = is_default
            return best_idx
        finally:
            pa.terminate()
    except Exception:
        # Probe failures must not block daemon boot — fall back to
        # PortAudio default and let the user see the noisy log.
        return None


def _vad_threshold_from_legacy(
    aggressiveness: Optional[int], explicit: Optional[float]
) -> float:
    """Translate legacy ``vad_aggressiveness`` (0..3) → Silero confidence.

    The legacy WebRTC VAD took an integer 0-3 (lenient → aggressive).
    Silero takes a float confidence cutoff. The user-tuned config.json
    still carries the legacy integer; without translation Pipecat
    sees a default that's wildly inappropriate for the user's mic.

    Mapping (R34-S14 — RECALIBRATED on Danylo's MBP M1 mic @ 30-50 cm):

    Probe data (10 s of real Ukrainian speech via Daria TTS + room
    ambient) showed:
      * max Silero confidence on speech segments: ~0.73
      * ambient floor (HVAC + room): conf < 0.25
      * speech mean confidence: 0.40-0.55 (NOT the 0.99+ I expected;
        Silero is more conservative on close-mic ASMR-style audio
        than on far-field demo recordings)

    The R34-S11 mapping (0.45/0.55/0.62/0.72) only passed 1 % of real
    speech frames. Recalibrated to be realistic for the actual mic:
      0 → 0.30  (whispers, far-field)
      1 → 0.35  (lenient)
      2 → 0.42  (R34-S14 default — passes normal speech, drops fan/typing)
      3 → 0.55  (aggressive — for noisy environments only)

    An explicit ``silero_vad_threshold`` config key overrides everything.
    Whisper-side hallucination filtering (see ``_HALLUCINATION_PHRASES``)
    catches the false-positives that the looser threshold lets through.
    """
    if explicit is not None:
        try:
            return max(0.10, min(0.95, float(explicit)))
        except (TypeError, ValueError):
            pass
    table = {0: 0.30, 1: 0.35, 2: 0.42, 3: 0.55}
    try:
        return table.get(int(aggressiveness), 0.42)
    except (TypeError, ValueError):
        return 0.42


def from_settings(cfg) -> PipecatLoopConfig:
    """Build the loop config from the global ``Settings`` object.

    Falls back to safe defaults if a field is missing — the legacy
    listener and Pipecat loop are supposed to consume the same
    ``config.json``, but we are tolerant of older versions of the
    config schema while users still have unmigrated installs.

    R34-S11: Bridge legacy ``vad_aggressiveness`` / ``voice_min_energy`` /
    ``endpoint_silence_ms`` into Silero-shaped params so months of
    careful user tuning actually takes effect under Pipecat.

    R34-S11: Also fall back ``whisper_language`` → ``active_language``
    so the STT language always tracks the user's transcription
    preference (legacy listener uses ``whisper_language``).
    """
    # Language fallback chain: active_language (Pipecat-native)
    # → whisper_language (legacy) → "ru" (project default for May 16+).
    lang = getattr(cfg, "active_language", None)
    if not lang:
        lang = getattr(cfg, "whisper_language", None) or "ru"
    lang = str(lang)

    vad_threshold = _vad_threshold_from_legacy(
        getattr(cfg, "vad_aggressiveness", None),
        getattr(cfg, "silero_vad_threshold", None),
    )
    # endpoint_silence_ms ⇆ vad_min_silence_ms — same semantic.
    try:
        silence_ms = int(getattr(cfg, "endpoint_silence_ms", 700) or 700)
    except (TypeError, ValueError):
        silence_ms = 700
    # R34-S43 — bumped ceiling 2000→5000ms. User explicitly requested
    # 2-3s pause-tolerance so natural thinking pauses don't cut off
    # the utterance. 5000ms gives headroom for thoughtful speech;
    # `max_utterance_ms` still caps the absolute window from the
    # other side so a stuck VAD can't lock us up forever.
    silence_ms = max(150, min(5000, silence_ms))

    # R34-S16: resolve voice_device substring → numeric PortAudio
    # index BEFORE handing it to Pipecat. The legacy listener
    # honoured a string match ("MacBook" matches "Мікрофон MacBook
    # Pro"); Pipecat only accepts an integer index, so we bridge it
    # here. Falls back to None → PortAudio default.
    resolved_in_idx = _resolve_input_device_index(
        getattr(cfg, "input_device_index", None),
        getattr(cfg, "voice_device", None),
    )
    # R34-S16: ``Settings`` is a strict frozen dataclass — keys we add
    # via config.json (``piper_voice``, future tuning knobs) don't
    # surface via getattr. Read them from the raw merged dict so power
    # users can override without touching ``config.py``. Cheap: just a
    # JSON re-read at boot.
    try:
        from ..config import load_config as _load_raw_config
        _raw_cfg = _load_raw_config() or {}
    except Exception:
        _raw_cfg = {}
    _piper_voice_override = _raw_cfg.get("piper_voice")
    return PipecatLoopConfig(
        input_device_index=resolved_in_idx,
        output_device_index=getattr(cfg, "output_device_index", None),
        stt_mlx_model=getattr(cfg, "whisper_model", "LARGE_V3_TURBO"),
        stt_language=lang,
        ollama_base_url=getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434"),
        chat_model=str(getattr(cfg, "ollama_chat_model", "qwen3:8b")),
        # R34-S42 К4 — Settings dataclass doesn't declare
        # ollama_chat_{num_predict,num_ctx,temperature}, so getattr(cfg,…)
        # silently returned the fallback ALWAYS. Read the live values
        # from the raw merged config dict (_raw_cfg is already populated
        # above from load_config()). Falls back to dataclass defaults
        # only if the key is genuinely absent.
        chat_temperature=float(_raw_cfg.get("ollama_chat_temperature", 0.4)),
        chat_num_predict=int(_raw_cfg.get("ollama_chat_num_predict", 220)),
        chat_num_ctx=int(_raw_cfg.get("ollama_chat_num_ctx", 2048)),
        piper_voice_id=str(
            _piper_voice_override
            or getattr(cfg, "piper_voice", None)
            or "uk_UA-ukrainian_tts-medium"
        ),
        piper_download_dir=(
            Path(p) if (p := getattr(cfg, "piper_download_dir", None)) else None
        ),
        vad_threshold=vad_threshold,
        vad_min_silence_ms=silence_ms,
        active_language=lang,
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
            # R34-S11 diagnostic: surface min mic-energy hint so the
            # event stream can correlate ambient floor vs Silero output.
            "voice_min_energy": float(
                getattr(cfg, "voice_min_energy", 0.0025) or 0.0025
            ),
            "max_utterance_ms": int(
                getattr(cfg, "max_utterance_ms", 7000) or 7000
            ),
        },
    )


def _system_prompt_for(lang: str, *, slim: bool = False) -> str:
    """Build the voice system prompt for the active language.

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

    R34-S42 К4 — ``slim`` flag (used by JarvisDirectChatProcessor) omits
    the L1 skills catalog. The direct_chat path bypasses the LLM tool
    aggregator entirely, so listing skills wastes ~1100 chars of
    prompt-eval and pushes num_ctx tight on qwen3:8b. The catalog
    adds nothing to a pure-chat reply but multiplies cold-cache latency.
    """
    # R34-S52 C-8 / C-9: hard-pin Russian regardless of ``lang``. The
    # user enforced RU-only output project-wide in R34-S48/S51. Keeping
    # the UA branch active meant any user who once flipped
    # ``active_language`` to "uk" (config edit, voice command, test
    # fixture) got 100% Ukrainian system prompt — which we then
    # post-hoc normalised at TTS time. That left the LLM context
    # itself Ukrainian-tainted across turns. By overriding here we
    # keep the entire prompt chain in Russian. The ``lang`` argument
    # is preserved in the signature for callers but no longer
    # influences the prompt selection.
    lang = "ru"
    base = _VOICE_SYSTEM_PROMPT_RU
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

    # R34-S48 — hard language pin. The Piper voice is RU; UA TTS
    # sounds awful. Tell the model in unmistakable terms that input
    # may be Ukrainian but output MUST be Russian. We prepend this
    # to the language-lock block built later in ``_call_ollama``;
    # keeping it here ensures even cached prompts carry the rule.
    ru_only_rule = (
        "ВАЖНО: пользователь иногда говорит по-украински, но ты ВСЕГДА "
        "отвечаешь ТОЛЬКО по-русски. Никогда не используй украинские "
        "слова, буквы (і, ї, є, ґ) или окончания. Понимай украинский — "
        "отвечай по-русски."
    )
    parts.append(ru_only_rule)

    # 3b. R34-S47 — Nexus Brain awareness. We don't dump the whole
    # vault into context (that's 5+ MB of markdown), but we DO tell
    # the LLM the brain exists, where it lives, and how to ask for a
    # search. Without this hint, the model never volunteers "хочеш
    # щоб я подивився в Brain?" — it just hallucinates from
    # parametric training. A 250-char nudge is enough to teach the
    # behaviour without breaking the prompt budget.
    # R34-S52 C-8: was UA-conditional. R34-S48/S51 made RU the only
    # outbound language, so the UA branch became dead code that still
    # tainted the prompt chain if anyone toggled ``active_language`` to
    # ``"uk"``. Collapsed to the RU branch unconditionally.
    brain_hint = (
        "Пользователь ведёт базу знаний Nexus Studio в "
        "~/Documents/Nexus-Brain (Obsidian vault: 01-CLIENTS, "
        "02-PROJECTS, 04-KNOWLEDGE, 08-PARTNERS …). Если вопрос "
        "касается клиентов, проектов, партнёров или внутренних "
        "процессов — попроси пользователя сказать «найди в мозге …» "
        "чтобы ты прочитал реальный файл, а не выдумывал. Новые "
        "факты о Nexus Studio пользователь может зафиксировать "
        "фразой «запиши в мозг: …»."
    )
    parts.append(brain_hint)

    # 4. L1 skill catalog (R32-1) — skipped in slim mode (R34-S42 К4)
    if not slim:
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


# R34-S32: shared module-level registry of bot utterances. Populated
# by fast_path._speak_ack and direct_chat._speak; consumed by the
# echo filter to drop late mic-captured bot echoes that arrived past
# the time-window gate. Kept module-level so all three live without
# explicit cross-references.
import threading as _threading
_BOT_HISTORY: list = []  # list of (ts, fingerprint)
_BOT_HISTORY_TTL_SEC = 30.0
_BOT_HISTORY_MAX = 16  # R34-S41 — bumped from 8 to survive 30s burst
# R34-S41 — protects _BOT_HISTORY mutations + reads across the
# asyncio loop thread and the executor threads used by _run_say /
# _piper_synthesize. CPython GIL guarantees individual list ops but
# slice-replace + del-prefix are multi-op and can tear under load.
_BOT_HISTORY_LOCK = _threading.Lock()
# R34-S41 — same race on the playback handle. _stop_current_playback
# can run from either the interrupt-poller (async) or the wake-gate
# universal-stop branch (also async, different task); meanwhile
# _play_audio_interruptable's finally clears it from the executor.
_PLAYBACK_LOCK = _threading.Lock()

# R34-S45 — persistent standby. When the HUD's Close (✕) button is
# pressed, the renderer writes ``control.json {"action":"end_session"}``
# plus ``interrupt.flag``. The flag cancels the current turn, but the
# pipeline keeps the mic open and would speak again on the next
# transcript that survives the wake-gate (background voices, ambient
# Whisper hallucinations). Persistent standby blocks ALL bot speech
# (fast-path acks + direct_chat replies + tool dispatch) until the
# user explicitly says the wake word — at which point the wake-word
# gate calls ``_clear_standby()`` and the daemon resumes.
#
# This is module-level (not per-processor) so every speaking site
# can cheaply check ``_is_standby()`` before generating audio. Uses
# a plain bool + lock — no monotonic clock needed because there's no
# TTL; the only exit path is an explicit wake.
_STANDBY = False
_STANDBY_LOCK = _threading.Lock()

# R34-S47 — callbacks invoked whenever standby flips to True. Used by
# JarvisFastPathProcessor to wipe its ``_pending_action`` so a HUD
# Close mid-confirmation doesn't leave the next wake-word turn stuck
# in the "pending-lock: dropped non-yes/no transcript" branch.
_STANDBY_CALLBACKS: "list[Any]" = []
_STANDBY_CALLBACKS_LOCK = _threading.Lock()


def register_standby_callback(fn: "Any") -> None:
    """Register a function to be called the moment standby engages.

    Idempotent — registering the same callable twice is a no-op so
    processors that get recreated in tests don't pile up duplicates.
    """
    with _STANDBY_CALLBACKS_LOCK:
        if fn not in _STANDBY_CALLBACKS:
            _STANDBY_CALLBACKS.append(fn)


def _set_standby(value: bool) -> None:
    """Atomically flip the standby flag.

    When transitioning OFF → ON, also fan out to every registered
    callback so subsystems (fast-path confirmation gate, repetition
    guard, etc.) can clear ephemeral state. Callbacks must be
    fast + exception-safe; we swallow any errors so a single broken
    handler can't block standby from engaging.
    """
    global _STANDBY
    with _STANDBY_LOCK:
        was, _STANDBY = _STANDBY, value
    if value and not was:
        with _STANDBY_CALLBACKS_LOCK:
            handlers = list(_STANDBY_CALLBACKS)
        for fn in handlers:
            try:
                fn()
            except Exception:
                pass


def _is_standby() -> bool:
    """True if the HUD has put the daemon into Close-to-standby mode."""
    with _STANDBY_LOCK:
        return _STANDBY


def _bot_history_fingerprint(text: str) -> str:
    """Normalise text for echo comparison (lower, strip punct, collapse ws)."""
    import re as _re
    if not text:
        return ""
    s = _re.sub(r"[^\w\s]", " ", text.lower())
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _register_bot_utterance(text: str) -> None:
    """Record a bot phrase + timestamp so the echo filter can match late echo.

    Called by ``fast_path._speak_ack`` and ``direct_chat._speak``
    immediately before invoking the macOS ``say`` subprocess. Best-
    effort — never raises, never blocks.
    """
    import time as _time
    if not text:
        return
    fp = _bot_history_fingerprint(text)
    if not fp:
        return
    now = _time.monotonic()  # R34-S41: monotonic — clock-jump-proof
    # R34-S41 — lock the entire read-modify-write; called from both
    # the asyncio loop thread AND _run_say executor threads.
    with _BOT_HISTORY_LOCK:
        _BOT_HISTORY.append((now, fp))
        cutoff = now - _BOT_HISTORY_TTL_SEC
        _BOT_HISTORY[:] = [(t, f) for (t, f) in _BOT_HISTORY if t > cutoff]
        if len(_BOT_HISTORY) > _BOT_HISTORY_MAX:
            del _BOT_HISTORY[: len(_BOT_HISTORY) - _BOT_HISTORY_MAX]


# R34-S48 — Russian-only reply guard.
#
# User feedback: "відключи українську розмовну!! нехай відповідає
# завжди російською но українську просто розуміє коли я говорю".
#
# The Piper voice we ship (``ru_RU-ruslan-medium``) is a Russian
# model trained on Russian phonemes. When given Ukrainian text it
# falls back to grapheme-by-grapheme pronunciation that mangles the
# UA-specific letters (і / ї / є / ґ → odd vowel substitutions),
# producing a jarring half-Russian half-Polish accent the user
# explicitly called "ужасну українську мову розмовну".
#
# Solution: keep STT multilingual (Whisper happily transcribes UA),
# keep the user free to SPEAK Ukrainian, but EVERY outbound TTS
# string is normalised through a UA→RU lexical map before reaching
# Piper. The LLM is also locked to ``lang="ru"`` regardless of the
# detected input language, so chat replies never come back in UA.

# Compact UA→RU table covering all the verbs / nouns the fast-path
# actions, confirmation prompts, and direct-chat scaffolding actually
# emit. Order matters — multi-word phrases must precede their
# single-word components (e.g. "не вдалося" → "не получилось" runs
# before any bare "не" substitution).
_UA_TO_RU_TRANSLATIONS: list[tuple[str, str]] = [
    # ── Multi-word phrases (must precede single-word components) ──
    ("Підтверджуєш дію", "Подтверждаешь действие"),
    ("Підтверджуєш", "Подтверждаешь"),
    ("підтверджуєш", "подтверждаешь"),
    ("Скажи «так» щоб виконати", "Скажи «да» чтобы выполнить"),
    ("«так» щоб виконати", "«да» чтобы выполнить"),
    ("«ні» щоб скасувати", "«нет» чтобы отменить"),
    ("щоб виконати", "чтобы выполнить"),
    ("щоб скасувати", "чтобы отменить"),
    ("щоб не", "чтобы не"),
    ("щоб ", "чтобы "),
    ("повтори, будь ласка", "повтори, пожалуйста"),
    ("Я не зрозумів, повтори, будь ласка", "Я не понял, повтори, пожалуйста"),
    ("не зрозумів", "не понял"),
    ("не зрозуміла", "не поняла"),
    ("не зрозуміло", "не понятно"),
    ("зрозумів", "понял"),
    ("зрозумив", "понял"),  # post-glyph mutation
    ("зрозуміла", "поняла"),
    ("зрозуміло", "понятно"),
    ("Зрозумів", "Понял"),
    ("Зрозумив", "Понял"),
    ("Зрозуміло", "Понятно"),
    # New / nouns (нова/нове/новий)
    ("Нова ", "Новая "),
    ("Нову ", "Новую "),
    ("нова ", "новая "),
    ("нову ", "новую "),
    ("Нове ", "Новое "),
    ("нове ", "новое "),
    ("будь ласка", "пожалуйста"),
    ("Будь ласка", "Пожалуйста"),
    ("Я вже відповідав на це", "Я уже отвечал на это"),
    ("вже відповідав", "уже отвечал"),
    ("вже ", "уже "),
    ("Не вдалося", "Не получилось"),
    ("не вдалося", "не получилось"),
    ("Скасовано", "Отменено"),
    ("скасовано", "отменено"),
    # ── R34-S51 high-frequency LLM emissions ──
    # The model (qwen3:8b) drifts into UA tokens mid-reply even after
    # R34-S48's RU-only rule. Each entry below is a token observed
    # leaking through to TTS during R34-S50 live testing. Order:
    # longer / capitalised forms first so they win over partial matches.
    ("допомогти", "помочь"),
    ("Допомогти", "Помочь"),
    ("допоможу", "помогу"),
    ("Допоможу", "Помогу"),
    ("допоможи", "помоги"),
    ("Допоможи", "Помоги"),
    ("допомога", "помощь"),
    ("Допомога", "Помощь"),
    ("можу", "могу"),
    ("Можу", "Могу"),
    ("треба", "надо"),
    ("Треба", "Надо"),
    ("щось", "что-то"),
    ("Щось", "Что-то"),
    ("нічого", "ничего"),
    ("Нічого", "Ничего"),
    ("тільки", "только"),
    ("Тільки", "Только"),
    ("трохи", "немного"),
    ("Трохи", "Немного"),
    ("гаразд", "хорошо"),
    ("Гаразд", "Хорошо"),
    ("добре", "хорошо"),
    ("Добре", "Хорошо"),
    ("вибач", "извини"),
    ("Вибач", "Извини"),
    ("вибачте", "извините"),
    ("Вибачте", "Извините"),
    ("навіщо", "зачем"),
    ("Навіщо", "Зачем"),
    ("чому", "почему"),
    ("Чому", "Почему"),
    ("де ", "где "),
    ("Де ", "Где "),
    ("куди", "куда"),
    ("Куди", "Куда"),
    ("коли ", "когда "),
    ("Коли ", "Когда "),
    ("швидко", "быстро"),
    ("Швидко", "Быстро"),
    ("багато", "много"),
    ("Багато", "Много"),
    ("забув", "забыл"),
    ("забула", "забыла"),
    ("Забув", "Забыл"),
    ("Забула", "Забыла"),
    ("повтори", "повтори"),  # identical, kept for explicit table coverage
    ("перепитай", "переспроси"),
    ("Перепитай", "Переспроси"),
    ("кажу", "говорю"),
    ("Кажу", "Говорю"),
    ("скажу", "скажу"),  # identical
    ("вікно", "окно"),
    ("Вікно", "Окно"),
    ("сьогодні", "сегодня"),
    ("Сьогодні", "Сегодня"),
    ("учора", "вчера"),
    ("Учора", "Вчера"),
    ("вчора", "вчера"),
    ("завтра", "завтра"),  # identical
    ("Завтра", "Завтра"),
    # UA past-tense be-verb (huge source of leaks)
    ("був ", "был "),
    ("була ", "была "),
    ("було ", "было "),
    ("Був ", "Был "),
    ("Була ", "Была "),
    ("Було ", "Было "),
    # ── Fast-path action descriptions (existing R34-S48 set) ──
    ("Відкриваю", "Открываю"),
    ("відкриваю", "открываю"),
    ("Закриваю", "Закрываю"),
    ("закриваю", "закрываю"),
    ("Дивлюся час", "Смотрю время"),
    ("Дивлюся", "Смотрю"),
    ("Дивлюсь", "Смотрю"),
    ("Записую у Brain", "Записываю в Brain"),
    ("Записую", "Записываю"),
    ("Записав нотатку", "Записал заметку"),
    ("Записав", "Записал"),
    ("Знайшов у Brain", "Нашёл в Brain"),
    ("Знайшов", "Нашёл"),
    ("Шукаю у Brain", "Ищу в Brain"),
    ("Шукаю", "Ищу"),
    ("Перевіряю", "Проверяю"),
    ("перевіряю", "проверяю"),
    ("Ховаю", "Скрываю"),
    ("ховаю", "скрываю"),
    ("Перемикаю вікно", "Переключаю окно"),
    ("Перемикаю", "Переключаю"),
    ("перемикаю", "переключаю"),
    ("Показую робочий стіл", "Показываю рабочий стол"),
    ("Показую", "Показываю"),
    ("Дивлюся пошту", "Смотрю почту"),
    ("Слухаю", "Слушаю"),
    ("слухаю", "слушаю"),
    ("Записую ідею", "Записываю идею"),
    ("ідею", "идею"),
    ("ідея", "идея"),
    ("Не зміг", "Не смог"),
    ("не зміг", "не смог"),
    ("додатка", "приложения"),
    ("додаток", "приложение"),
    # ── R34-S51 action_dispatcher description coverage ──
    ("Зроблю скріншот екрана", "Сделаю скриншот экрана"),
    ("Зроблю скріншот", "Сделаю скриншот"),
    ("Роблю скріншот", "Делаю скриншот"),
    ("скріншот", "скриншот"),
    ("Поставлю гучність", "Поставлю громкость"),
    ("гучність", "громкость"),
    ("Гучність", "Громкость"),
    ("Вимкну звук", "Выключу звук"),
    ("Увімкну звук", "Включу звук"),
    ("Вимикаю звук", "Выключаю звук"),
    ("Увімкаю звук", "Включаю звук"),
    ("Увімкаю", "Включаю"),
    ("Вимикаю", "Выключаю"),
    ("Заблокую екран", "Заблокирую экран"),
    ("Блокую екран", "Блокирую экран"),
    ("екран", "экран"),
    ("Перемкну відтворення", "Переключу воспроизведение"),
    ("відтворення", "воспроизведение"),
    ("Наступний трек", "Следующий трек"),
    ("Попередній трек", "Предыдущий трек"),
    ("Скопіюю в буфер", "Скопирую в буфер"),
    ("Дивлюся в буфер обміну", "Смотрю в буфер обмена"),
    ("Дивлюсь буфер", "Смотрю буфер"),
    ("Дивлюся буфер", "Смотрю буфер"),
    ("буфер обміну", "буфер обмена"),
    ("Шукаю на YouTube", "Ищу на YouTube"),
    ("Шукаю в Google", "Ищу в Google"),
    ("Шукаю в Spotlight", "Ищу в Spotlight"),
    ("у Spotlight", "в Spotlight"),
    ("у Brain", "в Brain"),
    ("у Google", "в Google"),
    ("у YouTube", "в YouTube"),
    ("Відкрию нового листа до", "Открою новое письмо для"),
    ("Відкрию", "Открою"),
    ("листа", "письмо"),
    ("Подивлюся котра година", "Посмотрю который час"),
    ("Подивлюся котра", "Посмотрю который"),
    ("Подивлюся", "Посмотрю"),
    ("Подивлюсь", "Посмотрю"),
    ("котра година", "который час"),
    ("Перевіряю рівень батареї", "Проверяю уровень батареи"),
    ("Перевіряю рівень", "Проверяю уровень"),
    ("Перевіряю батарею", "Проверяю батарею"),
    ("рівень батареї", "уровень батареи"),
    ("батареї", "батареи"),
    ("Створюю Linear-задачу", "Создаю задачу в Linear"),
    ("Створюю GitHub issue", "Создаю GitHub issue"),
    ("Створюю GitLab issue", "Создаю GitLab issue"),
    ("Створюю", "Создаю"),
    ("Створив", "Создал"),
    ("Створено", "Создано"),
    ("Додаю до Notion", "Добавляю в Notion"),
    ("Додаю до", "Добавляю в"),
    ("Додаю", "Добавляю"),
    ("Додано", "Добавлено"),
    ("Дивлюсь дату", "Смотрю дату"),
    ("Записую нотатку", "Записываю заметку"),
    ("нотатку", "заметку"),
    ("Закриваю вкладку", "Закрываю вкладку"),
    ("Нова вкладка у", "Новая вкладка в"),
    ("Нова вкладка", "Новая вкладка"),
    ("Пауза", "Пауза"),  # identical
    ("Вмикаю музику", "Включаю музыку"),
    ("Вмикаю", "Включаю"),
    # ── R34-S51 action_dispatcher description fillers ──
    ("Зараз відкрию", "Сейчас открою"),
    ("відкрию", "открою"),
    ("Відкрию", "Открою"),
    ("Відкрив", "Открыл"),
    ("відкрив", "открыл"),
    ("Відкрила", "Открыла"),
    ("закрию", "закрою"),
    ("Закрию", "Закрою"),
    ("Закрив", "Закрыл"),
    ("закрив", "закрыл"),
    ("Зараз закрию", "Сейчас закрою"),
    # Past-tense action returns
    ("Зробив", "Сделал"),
    ("Зробила", "Сделала"),
    ("Зробив скріншот", "Сделал скриншот"),
    ("Перемикнув", "Переключил"),
    ("Перемикнула", "Переключила"),
    ("збережено", "сохранено"),
    ("Збережено", "Сохранено"),
    ("заснув", "заснул"),
    ("Дисплей заснув", "Дисплей заснул"),
    ("без блокування", "без блокировки"),
    ("увімкни", "включи"),
    ("Увімкни", "Включи"),
    ("у налаштуваннях", "в настройках"),
    ("налаштуваннях", "настройках"),
    ("відсотків", "процентов"),
    ("відсотка", "процента"),
    ("відсоток", "процент"),
    # Brain / write feedback
    ("Записав у Brain", "Записал в Brain"),
    ("Записав", "Записал"),
    ("Знайшов 1 файл", "Нашёл 1 файл"),
    ("Нашёл 1 файл", "Нашёл 1 файл"),  # identity
    ("Нашёл", "Нашёл"),  # identity, but blocks accidental UA re-leak
    ("файлів", "файлов"),  # genitive
    ("файлов", "файлов"),  # identity
    # "Відкриваю X у Y" — UA "у" before consonant → RU "в"; we
    # blanket-translate " у " (with spaces) to " в " so we don't
    # accidentally mangle word-internal "у". Standalone " у " almost
    # never appears in legitimate Russian — RU uses " в " or " у "
    # only as "near" which is rare in Jarvis spoken output.
    (" у Safari", " в Safari"),
    (" у Chrome", " в Chrome"),
    (" у Firefox", " в Firefox"),
    (" у Arc", " в Arc"),
    (" у Edge", " в Edge"),
    (" у Notion", " в Notion"),
    (" у Linear", " в Linear"),
    (" у GitHub", " в GitHub"),
    (" у GitLab", " в GitLab"),
    (" у браузер", " в браузер"),
    (" у файл", " в файл"),
    (" у папк", " в папк"),
    (" у директ", " в директ"),
    # ── R34-S51 system_prompt + extra LLM leaks (live-spotted) ──
    ("Поточний час", "Текущее время"),
    ("Поточний", "Текущий"),
    ("Поточне", "Текущее"),
    ("Поточна", "Текущая"),
    ("поточний", "текущий"),
    ("поточне", "текущее"),
    ("поточна", "текущая"),
    ("Це єдина правда", "Это единственная правда"),
    ("єдина правда", "единственная правда"),
    ("НЕ вигадуй", "НЕ выдумывай"),
    ("вигадуй", "выдумывай"),
    ("вигадав", "выдумал"),
    ("визначити", "определить"),
    ("Налаштувати", "Настроить"),
    ("налаштувати", "настроить"),
    ("Налаштую", "Настрою"),
    ("налаштую", "настрою"),
    ("Налаштування", "Настройки"),
    ("сторінка", "страница"),
    ("сторінку", "страницу"),
    ("сторінки", "страницы"),
    ("Сторінка", "Страница"),
    ("Сторінку", "Страницу"),
    ("купити", "купить"),
    ("Купити", "Купить"),
    ("каву", "кофе"),
    # UA preposition "з" before consonant → RU "с"; "із" before vowel → RU "из"
    (" з проектами", " с проектами"),
    (" з клієнтами", " с клиентами"),
    (" з тобою", " с тобой"),
    (" з мене", " с меня"),
    (" з тебе", " с тебя"),
    (" з тим ", " с тем "),
    (" із ", " из "),
    (" клієнт", " клиент"),
    ("клієнт", "клиент"),
    ("Клієнт", "Клиент"),
    ("проектами", "проектами"),  # identical, kept for visibility
    # ── R34-S51 action_dispatcher errors ──
    ("Помилка", "Ошибка"),
    ("помилка", "ошибка"),
    ("Не вказано назву додатка", "Не указано название приложения"),
    ("Не вказано назву", "Не указано название"),
    ("Не вказано що шукати", "Не указано что искать"),
    ("Не вказано", "Не указано"),
    ("назву", "название"),
    ("не схоже на назву додатка", "не похоже на название приложения"),
    ("не схоже на назву", "не похоже на название"),
    ("не схоже", "не похоже"),
    ("Не зміг відкрити", "Не смог открыть"),
    ("Не зміг закрити", "Не смог закрыть"),
    ("Не зміг записати", "Не смог записать"),
    ("Не зміг прочитати", "Не смог прочитать"),
    ("Не зміг визначити", "Не смог определить"),
    ("Не зміг змінити", "Не смог изменить"),
    ("Nexus-Brain не знайдено локально", "Nexus-Brain не найден локально"),
    ("не знайдено локально", "не найден локально"),
    ("не знайдено", "не найдено"),
    ("Не знайшов запущений", "Не нашёл запущенный"),
    ("Не знайшов музичний плеєр", "Не нашёл музыкальный плеер"),
    ("Не знайшов", "Не нашёл"),
    ("Порожній запит до Nexus-Brain", "Пустой запрос к Nexus-Brain"),
    ("Порожній запит", "Пустой запрос"),
    ("Порожній зміст для запису", "Пустое содержимое для записи"),
    ("Порожній", "Пустой"),
    ("прочитати vault", "прочитать vault"),
    ("записати в Brain", "записать в Brain"),
    ("Підключи Linear токен", "Подключи Linear токен"),
    ("Підключи GitHub токен", "Подключи GitHub токен"),
    ("Підключи GitLab токен", "Подключи GitLab токен"),
    ("Підключи Notion токен", "Подключи Notion токен"),
    ("Підключи", "Подключи"),
    ("Постав JARVIS_LINEAR_TEAM_ID", "Установи JARVIS_LINEAR_TEAM_ID"),
    ("Невалідний email", "Невалидный email"),
    ("містить заборонені символи", "содержит запрещённые символы"),
    ("містить", "содержит"),
    ("заборонені", "запрещённые"),
    ("Звук вимкнено", "Звук выключен"),
    ("Звук увімкнено", "Звук включен"),
    ("вимкнено", "выключено"),
    ("увімкнено", "включено"),
    ("Екран заблоковано", "Экран заблокирован"),
    ("заблоковано", "заблокировано"),
    # ── Common weekday names ──
    ("понеділок", "понедельник"),
    ("вівторок", "вторник"),
    ("середа", "среда"),
    ("четвер", "четверг"),
    ("п'ятниця", "пятница"),
    ("субота", "суббота"),
    ("неділя", "воскресенье"),
    # ── Common months ──
    ("січня", "января"),
    ("лютого", "февраля"),
    ("березня", "марта"),
    ("квітня", "апреля"),
    ("травня", "мая"),
    ("червня", "июня"),
    ("липня", "июля"),
    ("серпня", "августа"),
    ("вересня", "сентября"),
    ("жовтня", "октября"),
    ("листопада", "ноября"),
    ("грудня", "декабря"),
    # ── Connectives + small words ──
    ("або", "или"),
    ("дію", "действие"),
    ("дія", "действие"),
    ("Готово", "Готово"),  # identical, kept for visibility
    # ── Final glyph guards (RUN LAST after lexical map) ──
    # Ukrainian-only glyphs as last-resort phoneme guards. These never
    # appear in legit Russian text. The lexical map above should catch
    # the high-frequency tokens; these rules salvage anything else
    # so Piper at least pronounces Russian-ish vowels.
    ("і", "и"),
    ("ї", "и"),
    ("є", "е"),
    ("ґ", "г"),
    ("І", "И"),
    ("Ї", "И"),
    ("Є", "Е"),
    ("Ґ", "Г"),
]


def _ru_normalise(text: str) -> str:
    """Best-effort UA→RU translation for outbound TTS.

    Idempotent on already-RU strings (every replacement either
    targets a UA-unique form or is a no-op identity). Designed to
    run on every ``_speak`` / ``_speak_ack`` call so a Ukrainian
    fast-path description (``Закриваю Safari``) reaches Piper as
    pure Russian text (``Закрываю Safari``) regardless of the
    action factory's authoring language.

    NOT a general translator — only handles the vocabulary the
    codebase actually emits. New UA strings need their pair added
    to ``_UA_TO_RU_TRANSLATIONS``.
    """
    if not text:
        return text
    out = text
    for ua, ru in _UA_TO_RU_TRANSLATIONS:
        if ua and ua in out:
            out = out.replace(ua, ru)
    return out


# R34-S47 — repetition guard.
#
# The user reported "Jarvis повторюється в розмові" — the LLM
# (qwen3:8b) sometimes regenerates the same reply for two consecutive
# turns, or echoes a phrase verbatim across turn boundaries. The user
# explicitly asked: "ніколи не повторювався (може тільки якщо я
# попрошу повторити!!)".
#
# Strategy: before TTS-ing any reply, fingerprint it and compare
# against the last few entries in ``_BOT_HISTORY``. If the new reply
# is byte-identical to anything spoken in the last 90 s AND the user
# DID NOT explicitly ask for a repeat, swap the reply for a short
# "вже відповідав на це" line. The user knows the answer is the
# same; we save 4 s of cringe-worthy verbatim repetition.
#
# We pick 90 s (vs the 30 s echo TTL) because the legitimate echo
# window only needs to catch mic-captured echo of just-spoken audio,
# whereas repetition can happen 60+ s apart when the LLM forgets it
# already answered.

_REPEAT_GUARD_TTL_SEC = 90.0

# Phrases that signal "yes I want a repeat" — when the user says any
# of these, ``would_be_repeat`` is suppressed so the bot DOES repeat.
_REPEAT_REQUEST_PHRASES = (
    "повтори", "повторите", "повторіть",
    "ще раз", "еще раз", "знову", "снова",
    "again", "repeat",
)


def _is_repeat_request(user_text: str | None) -> bool:
    """True if the user explicitly asked to hear the previous reply again."""
    if not user_text:
        return False
    low = user_text.lower()
    return any(p in low for p in _REPEAT_REQUEST_PHRASES)


def would_be_repeat(text: str) -> bool:
    """True if ``text`` matches a phrase the bot already spoke in the last 90s.

    Uses the same ``_bot_history_fingerprint`` normalisation as the
    echo filter so wording-level whitespace / punctuation differences
    don't accidentally bypass the guard.
    """
    import time as _time
    if not text:
        return False
    fp = _bot_history_fingerprint(text)
    if not fp:
        return False
    now = _time.monotonic()
    cutoff = now - _REPEAT_GUARD_TTL_SEC
    with _BOT_HISTORY_LOCK:
        for ts, hist_fp in _BOT_HISTORY:
            if ts < cutoff:
                continue
            if hist_fp == fp:
                return True
    return False


# R34-S34: Piper voice cache + interruptable playback state.
#
# We deliberately avoid macOS `say` because the only Russian voice
# available natively is Milena (female + robotic), and `say` ignores
# SIGINT — once started, the audio MUST play to completion. The
# dashboard Stop button can't interrupt it.
#
# Piper gives:
#  • The user-requested male voice (ru_RU-ruslan-medium — "billionaire
#    CEO tone" per the project config comment).
#  • Generation in ~300ms for a typical Jarvis reply (~80 chars).
#  • Output is a WAV file → ``afplay`` subprocess plays it, and afplay
#    DOES respond to SIGTERM/SIGKILL, so the Stop button works.
#
# We keep the loaded ``PiperVoice`` in module-level cache so we don't
# pay the 1.6s ONNX load on every reply.
_PIPER_VOICE_CACHE: dict = {}
# R34-S42 — _PLAYBACK_LOCK is defined ABOVE at line 462 (threading.Lock).
# The earlier `_PLAYBACK_LOCK: object | None = None` annotation here used
# to overwrite it with None at module load, which made every `with
# _PLAYBACK_LOCK:` raise "NoneType has no context manager". Removed.
_CURRENT_PLAYBACK_PROC: object | None = None  # subprocess.Popen of afplay


def _get_piper_voice(voice_id: str = "ru_RU-ruslan-medium"):
    """Load + cache a Piper voice ONNX model.

    Returns the ``PiperVoice`` instance, or ``None`` if the model file
    isn't on disk (caller should then fall back to macOS ``say``).
    """
    if voice_id in _PIPER_VOICE_CACHE:
        return _PIPER_VOICE_CACHE[voice_id]
    try:
        from piper import PiperVoice
        # Two possible locations — daemon vs warmup pre-stage.
        for base in (
            Path.home() / ".local/share/jarvis/piper",
            Path.home() / ".local/share/jarvis/models/piper",
        ):
            onnx = base / f"{voice_id}.onnx"
            if onnx.is_file():
                voice = PiperVoice.load(str(onnx))
                _PIPER_VOICE_CACHE[voice_id] = voice
                return voice
    except Exception as exc:
        try:
            debug_log(f"piper load failed for {voice_id}: {exc!r}", "pipecat")
        except Exception:
            pass
    return None


def _piper_synthesize(text: str, voice_id: str = "ru_RU-ruslan-medium") -> Path | None:
    """Synthesise ``text`` via Piper, return path to WAV file.

    Returns ``None`` if Piper isn't available (caller falls back to
    ``say``). The WAV is written to a per-pid tmp file so concurrent
    daemons don't clobber each other.

    R34-S51 — defense-in-depth UA→RU normaliser. This is the LAST
    line of defence before Piper sees the text. Even if a caller
    forgets to call ``_ru_normalise``, mutates ``text`` after the
    upstream normalise call, or constructs a fallback string the
    map doesn't cover, the call here guarantees Piper never sees
    raw Ukrainian glyphs. Idempotent on already-RU strings.
    """
    voice = _get_piper_voice(voice_id)
    if voice is None:
        return None
    try:
        text = _ru_normalise(text)
    except Exception:
        pass
    import os as _os
    import time as _time
    import wave as _wave
    out = Path(f"/tmp/jarvis_tts_{_os.getpid()}_{int(_time.time() * 1000) % 100000}.wav")
    try:
        with _wave.open(str(out), "wb") as wav:
            voice.synthesize_wav(text, wav)
        return out
    except Exception as exc:
        try:
            debug_log(f"piper synth failed: {exc!r}", "pipecat")
        except Exception:
            pass
        return None


async def _play_audio_interruptable(wav_path: Path) -> None:
    """Play ``wav_path`` via ``afplay`` in a way that can be cancelled.

    Stores the subprocess globally so an external Stop signal (the
    interrupt flag from the dashboard) can ``terminate()`` it. The
    file is deleted afterwards.
    """
    global _CURRENT_PLAYBACK_PROC
    import asyncio as _asyncio
    import subprocess as _sp
    proc = None
    try:
        proc = _sp.Popen(
            ["afplay", str(wav_path)],
            stdin=_sp.DEVNULL,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )
        # R34-S41 — atomically publish the handle so _stop_current_playback
        # always sees a consistent view (set+test from another thread).
        with _PLAYBACK_LOCK:
            _CURRENT_PLAYBACK_PROC = proc
        loop = _asyncio.get_event_loop()
        await loop.run_in_executor(None, proc.wait)
    except _asyncio.CancelledError:
        # R34-S52 H: ``CancelledError`` is NOT an ``Exception`` in
        # Python 3.8+, so the ``except Exception`` below would let it
        # propagate WITHOUT terminating the subprocess. Result: after
        # a pipeline-level cancel (Pipecat task.cancel(), daemon
        # shutdown), the user kept hearing afplay until the WAV
        # ended naturally. Now: kill the subprocess explicitly and
        # re-raise so cancellation semantics are preserved.
        # R34-S53.1 (Phase 6b — C-9): dropped the sync ``proc.wait(timeout=1.0)``
        # that ran inside the CancelledError handler. We're already on
        # the asyncio event loop with cooperative cancellation pending —
        # a synchronous 1 s wait blocks the loop and defeats the point
        # of being interruptible. ``terminate()`` is enough: afplay
        # honours SIGTERM immediately, and the finally-clause clears
        # ``_CURRENT_PLAYBACK_PROC`` so subsequent stop attempts no-op.
        # The orphaned subprocess (if any) is reaped by the OS or by
        # ``_stop_current_playback``'s background reaper thread.
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        raise
    except Exception as exc:
        try:
            debug_log(f"afplay failed: {exc!r}", "pipecat")
        except Exception:
            pass
    finally:
        # Only clear if WE still own the slot. _stop_current_playback may
        # have already swapped in a fresh None, OR another playback might
        # have raced ahead — either way, don't clobber a stranger's handle.
        with _PLAYBACK_LOCK:
            if _CURRENT_PLAYBACK_PROC is proc:
                _CURRENT_PLAYBACK_PROC = None
        try:
            wav_path.unlink()
        except Exception:
            pass


def _stop_current_playback() -> bool:
    """Best-effort: kill the currently playing afplay (if any).

    Returns True if we asked something to stop, False otherwise.
    Called by the interrupt-flag poller when the Stop button is pressed.

    R34-S39 — clears _CURRENT_PLAYBACK_PROC immediately so a fast
    follow-up Stop press doesn't try to kill a stale handle. Also
    schedules a non-blocking 2s wait in the default executor to reap
    the zombie (without this, terminated afplay can linger as a
    zombie until the parent exits because no .wait() was called).
    """
    global _CURRENT_PLAYBACK_PROC
    # R34-S41 — atomic read+clear under lock so a concurrent
    # _play_audio_interruptable.finally doesn't race-clear the slot
    # then we hit a None handle here.
    with _PLAYBACK_LOCK:
        proc = _CURRENT_PLAYBACK_PROC
        if proc is None:
            return False
        _CURRENT_PLAYBACK_PROC = None
    # Skip if the process already finished — terminate() on a dead
    # PID is harmless but the reaper thread is unnecessary.
    try:
        if proc.poll() is not None:
            return False
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        try:
            proc.kill()
        except Exception:
            return False
    # Reap in background so the zombie clears even if the original
    # _play_audio_interruptable executor task was cancelled.
    def _reap():
        try:
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
    try:
        import threading as _th
        _th.Thread(target=_reap, daemon=True, name="afplay-reap").start()
    except Exception:
        pass
    return True


def _cleanup_orphan_tts_wavs() -> int:
    """Sweep /tmp for orphan jarvis_tts_*.wav files.

    R34-S39 — called once at daemon startup. Each turn writes a temp
    WAV; if the daemon crashes mid-turn the file leaks. With many
    crash cycles these accumulate (each ~50-200 KB). Cheap sweep
    once per boot. Files matching the current pid are skipped — they
    may belong to the freshly-started process.
    """
    import os as _os
    import time as _time
    cleared = 0
    try:
        pid = _os.getpid()
        for entry in Path("/tmp").glob("jarvis_tts_*.wav"):
            try:
                # Skip files that belong to a process still alive.
                # Filename is jarvis_tts_<pid>_<ms>.wav
                parts = entry.stem.split("_")
                if len(parts) >= 3:
                    fpid = int(parts[2])
                    if fpid == pid:
                        continue
                    # Check if the owning PID is still alive.
                    try:
                        _os.kill(fpid, 0)
                        continue  # alive, leave alone
                    except OSError:
                        pass  # process gone, file is orphan
                # Also skip very recent files (<5s) to avoid races.
                if _time.time() - entry.stat().st_mtime < 5.0:
                    continue
                entry.unlink()
                cleared += 1
            except Exception:
                continue
    except Exception:
        pass
    return cleared


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
        InputAudioRawFrame,
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

    # ── R34-S26: AudioRMSProbe — diagnostic processor right after
    # LocalAudioInputTransport. Wake-word was not firing despite mic
    # capturing real audio (mic_probe events showed RMS 0.013-0.076)
    # but ZERO vad/stt events flowed for 30+ minutes. This probe
    # confirms whether InputAudioRawFrame is reaching the pipeline.
    # Emits ``pipecat_audio_rms`` every ~1 s with peak RMS over the
    # window so the dashboard can see if Pipecat's stream is alive.
    # Pure observer: never modifies the frame, never blocks.
    class JarvisAudioProbeProcessor(FrameProcessor):
        """Emit pipecat_audio_rms heartbeat on inbound audio frames."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._stream = get_stream()
            self._last_emit_ts = 0.0
            self._window_peak = 0.0
            self._window_frame_count = 0
            self._window_sample_rate = 0

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)
            try:
                if isinstance(frame, InputAudioRawFrame):
                    import time as _t
                    # Compute RMS of this frame's int16 audio.
                    try:
                        import numpy as _np
                        arr = _np.frombuffer(frame.audio, dtype=_np.int16)
                        if arr.size:
                            arr_f = arr.astype(_np.float32) / 32768.0
                            rms = float(_np.sqrt((arr_f * arr_f).mean()))
                            if rms > self._window_peak:
                                self._window_peak = rms
                            self._window_frame_count += 1
                            self._window_sample_rate = int(
                                getattr(frame, "sample_rate", 0) or 0
                            )
                    except Exception:
                        pass
                    now = _t.time()
                    if now - self._last_emit_ts >= 1.0:
                        self._stream.emit(
                            "pipecat_audio_rms",
                            peak_rms=round(self._window_peak, 5),
                            frames_in_window=self._window_frame_count,
                            sample_rate=self._window_sample_rate,
                        )
                        self._last_emit_ts = now
                        self._window_peak = 0.0
                        self._window_frame_count = 0
            except Exception:
                pass
            await self.push_frame(frame, direction)

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
                    # R34-S54.1 Phase 7b: monotonic — see _speak_end_ts
                    # rationale in JarvisEchoFilterProcessor.__init__.
                    duration_ms = int(
                        (_time.monotonic() - self._utterance_t0) * 1000
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
                # R34-S54.1 Phase 7b: monotonic, paired with the
                # _utterance_t0 read above.
                self._utterance_t0 = _time.monotonic()
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
                # R34-S54.1 Phase 7b: monotonic, paired with the
                # TTSStoppedFrame read below.
                self._tts_t0 = _time.monotonic()
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
                    # R34-S54.1 Phase 7b: monotonic, see TTSStartedFrame.
                    duration_ms = int(
                        (_time.monotonic() - self._tts_t0) * 1000
                    )
                self._tts_t0 = 0.0
                self._stream.emit("tts_done", duration_ms=duration_ms)
                return

    return JarvisEventStreamProcessor, JarvisAudioProbeProcessor


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
                # R34-S44 — when the bot finishes speaking, the VAD is
                # immediately ready for the next user utterance. The
                # HUD should reflect that (LISTENING with a visible
                # wave indicator) instead of falling to IDLE=0% which
                # made the user think Jarvis had switched off after
                # asking a confirmation question. The pipeline keeps
                # the mic open in either state; this is purely a UX
                # signal. IDLE is only reached via CancelFrame /
                # EndFrame / explicit end_session below.
                self._transition(self.S_LISTENING, level=1.0)
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


def _make_direct_chat_processor(cfg: "PipecatLoopConfig"):
    """Build the DirectChat FrameProcessor — bypasses Pipecat LLM+TTS.

    Why we need it
    --------------
    Pipecat 1.2.1's OLLamaLLMService + LLMAssistantAggregator chain
    becomes "wedged" after the first few minutes of uptime: the
    OpenAI streaming client opens, Generating chat is logged, but
    no chunks ever reach LLMAssistantAggregator — so no
    LLMTextFrame, no TTSStartedFrame, no audio. We confirmed
    via direct curl that Ollama itself is healthy and answers in
    <2s. The bug is internal to Pipecat's streaming bridge.

    Rather than patch Pipecat, we bypass it. Every TranscriptionFrame
    that makes it past the wake gate is:
      1. Converted to an Ollama `/api/chat` call (think:false,
         the native endpoint, which we already proved works
         reliably for the /api/chat dashboard route).
      2. The reply is spoken via macOS `say` (which doesn't
         depend on Pipecat at all).
      3. State events are emitted so the dashboard SPEAKING ring
         lights up while we're talking.
      4. The frame is CONSUMED — never forwarded to the broken
         LLM chain.
    """
    import asyncio as _asyncio

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

    ollama_url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    model = cfg.chat_model
    num_predict = int(cfg.chat_num_predict)
    num_ctx = int(cfg.chat_num_ctx)

    # System prompt assembled the same way as the regular Pipecat
    # path so behaviour stays consistent. We import the helper
    # locally so a missing personas dir doesn't break module load.
    #
    # R34-S37: this is now the BASE prompt only — the per-turn
    # language lock + recent-conversation digest are appended inside
    # process_frame() so we always re-state them in the freshest
    # detected language.
    try:
        from ..system_prompt import build_system_prompt
        _base_system_prompt = build_system_prompt("Jarvis")
    except Exception:
        _base_system_prompt = (
            "Ты — Джарвис, голосовой ассистент. Отвечай 1-2 фразами. "
            "Никаких префиксов, никаких эмодзи, никаких списков."
        )

    class JarvisDirectChatProcessor(FrameProcessor):
        """Direct LLM+TTS bypass for the user-facing voice turn."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._stream = get_stream()
            self._busy = False  # crude reentrancy guard
            # Reusable aiohttp session — single connection pool,
            # finite keepalive so stale-NAT problems are bounded.
            self._session = None
            # R34-S47 — remember the latest user turn so the
            # repetition guard in ``_speak`` can tell whether the
            # user explicitly asked for a repeat ("повтори", "ще раз")
            # vs the LLM spontaneously regenerating the same answer.
            self._last_user_text: str | None = None
            # R34-S47 — rolling conversation history fed to Ollama
            # alongside the system prompt. Audit finding: previously
            # each turn was sent as ``[system, user]`` only, so the
            # LLM had no memory of the previous user turn and
            # frequently re-answered the same question or contradicted
            # itself. We carry the last 8 messages (≈4 user/assistant
            # pairs) which is enough for natural follow-ups without
            # blowing past num_ctx=2048 on qwen3:8b.
            self._messages_history: list[dict[str, str]] = []
            # R34-S50 LATENCY FIX — depth slashed 8 → 4 (2 user/assistant
            # pairs). Every extra history message INVALIDATES Ollama's
            # KV-cache prefix reuse: with 8 messages each turn the cache
            # is busted and prompt_eval pays the full 6 s; with just the
            # system prompt + last 2 messages the prefix matches the
            # previous turn's tail in the common case and prompt_eval
            # drops to ~150 ms (40× speed-up, measured live R34-S50).
            self._messages_history_max: int = 4

        async def _ensure_session(self):
            if self._session is None or self._session.closed:
                import aiohttp as _ah
                # R34-S50 LATENCY FIX — bumped sock_read 60→120s,
                # total 75→135s. Live evidence (R34-S50): keepwarm
                # calls take 71-96s on Hetzner CPU when KV-cache is
                # cold, and a user call arriving DURING a keepwarm
                # queues behind it → 100-120s wall time before
                # Ollama returns the first byte. 60s sock_read was
                # too tight for that case, producing repeated
                # SocketTimeoutErrors and empty replies.
                # Connect timeout stays 3s — Tailscale fails fast
                # if it's actually down, no need for headroom there.
                timeout = _ah.ClientTimeout(
                    total=135, connect=3, sock_connect=3, sock_read=120,
                )
                # R34-S42 К4 ROOT CAUSE — Tailscale links silently drop
                # idle TCP keep-alive connections after ~60-90s, but the
                # client-side socket stays in ESTABLISHED. aiohttp's
                # default connection pool reuses these dead sockets;
                # the next POST hangs waiting for bytes that never come,
                # producing exactly the SocketTimeoutError we observed.
                # ``force_close=True`` opens a fresh TCP connection per
                # request (one extra handshake ≈ 30-80ms over Tailscale),
                # eliminating the dead-socket reuse class of failure.
                # ``enable_cleanup_closed`` reaps connections whose
                # transport already died on the kernel side. Together
                # they bring real-daemon latency parity with raw curl
                # (~2-3s warm vs ∞ hang on stale pool).
                connector = _ah.TCPConnector(
                    force_close=True,
                    enable_cleanup_closed=True,
                )
                self._session = _ah.ClientSession(
                    timeout=timeout, connector=connector,
                )
            return self._session

        async def cleanup(self) -> None:
            # R34-S39 — graceful shutdown so the daemon doesn't leak
            # FIN_WAIT TCP sockets to Ollama across launchctl restarts.
            # Pipecat calls FrameProcessor.cleanup() during pipeline
            # teardown; we close the aiohttp session here.
            try:
                if self._session is not None and not self._session.closed:
                    await self._session.close()
                    # aiohttp connector close grace — without this
                    # SSLContext is reaped abruptly and the connector
                    # logs a "Unclosed connector" warning.
                    await _asyncio.sleep(0.25)
            except Exception as exc:
                debug_log(
                    f"direct_chat: session cleanup failed: {exc!r}",
                    "pipecat",
                )
            try:
                await super().cleanup()
            except Exception:
                pass

        async def _call_ollama(self, text: str, lang: str = "ru") -> str:
            import time as _time
            session = await self._ensure_session()
            # R34-S39/40 — system prompt cached per-language. Re-validated
            # against ~/.config/jarvis/persona.md mtime so dashboard
            # persona edits propagate to the next turn without daemon
            # restart. Without invalidation, the processor would serve a
            # stale persona until the process was killed.
            cache = getattr(self, "_sys_prompt_cache", None)
            cache_mtime = getattr(self, "_sys_prompt_cache_mtime", 0.0)
            if cache is None:
                cache = {}
                self._sys_prompt_cache = cache
            # R34-S42 — invalidate cache when ANY of persona.md, facts.db,
            # or 30s TTL expires. Without facts.db tracking, newly-learned
            # facts wouldn't appear in the prompt until next daemon
            # restart. The 30s TTL is a backstop for skill/config edits.
            try:
                from pathlib import Path as _Path
                import time as _t_cache
                persona_path = _Path.home() / ".config/jarvis/persona.md"
                facts_path = _Path.home() / "Library/Application Support/jarvis/facts.db"
                stamps = []
                if persona_path.exists():
                    stamps.append(persona_path.stat().st_mtime)
                if facts_path.exists():
                    stamps.append(facts_path.stat().st_mtime)
                new_mtime = max(stamps) if stamps else 0.0
                last_built = getattr(self, "_sys_prompt_cache_built_at", 0.0)
                age = _t_cache.monotonic() - last_built
                if new_mtime != cache_mtime or age > 30.0:
                    cache.clear()
                    self._sys_prompt_cache_mtime = new_mtime
                    self._sys_prompt_cache_built_at = _t_cache.monotonic()
            except Exception:
                pass
            sys_prompt = cache.get(lang)
            if sys_prompt is None:
                # Use the FULL system prompt builder (persona + facts +
                # base + time block) — NOT just _base_system_prompt
                # which omits the persona block (R34-S37 bug). Then append
                # the per-turn language lock.
                # R34-S42 К4 — ``slim=True`` drops the L1 skill catalog
                # (~1100 chars). direct_chat bypasses the LLM tool
                # aggregator, so the catalog only inflates prompt eval
                # cost (~5-8s on cold qwen3:8b) without adding value.
                try:
                    full_prompt = _system_prompt_for(lang, slim=True)
                except Exception:
                    full_prompt = _base_system_prompt
                try:
                    from ..memory.self_learning import language_lock_block
                    lock = language_lock_block(lang)
                except Exception:
                    lock = ""
                # R34-S40 — clarification instruction so the model asks
                # "переспроси" instead of confidently inventing an answer.
                # R34-S51 — RU-only. Removed the UA branch entirely;
                # the model is now NEVER told to clarify in Ukrainian,
                # which previously caused it to OUTPUT clarifications
                # like "Я не зрозумів, перепитай..." in Ukrainian and
                # those phrases reached TTS even with normalise running.
                clarify = (
                    "Если запрос неясный, нечёткий или похож на "
                    "бессмыслицу — переспроси одним коротким "
                    "предложением по-русски, НЕ выдумывай ответ."
                )
                parts = [full_prompt, clarify]
                if lock:
                    parts.append(lock)
                sys_prompt = "\n\n".join(p for p in parts if p)
                cache[lang] = sys_prompt
            # R34-S47 — build the messages list with rolling history.
            # ``_messages_history`` holds the last few user/assistant
            # pairs; appending the current user turn happens HERE so the
            # LLM sees `[system, …prior pairs…, current_user]`. The
            # assistant reply is appended in process_frame after Ollama
            # returns so the next turn has the context.
            history_msgs: list[dict[str, str]] = list(self._messages_history)
            # R34-S50 LATENCY FIX — adaptive num_predict.
            #
            # Default 140 tokens × ~150ms/token on qwen3:8b CPU = 21s
            # eval time for EVERY reply, even one-word "Привет!". A
            # measured per-turn baseline showed:
            #   short user (≤30ch)  → reply ~10-25 tokens
            #   normal user (≤80ch) → reply ~25-50 tokens
            #   complex user        → reply ~80-140 tokens
            # Capping num_predict to the realistic ceiling per turn
            # cuts mean reply latency 2-3x with NO visible quality
            # loss — the model stops at the natural sentence boundary
            # via the stop sequence below in 95% of turns anyway.
            user_len = len(text or "")
            if user_len <= 30:
                effective_num_predict = min(num_predict, 60)
            elif user_len <= 80:
                effective_num_predict = min(num_predict, 100)
            else:
                effective_num_predict = num_predict
            payload = {
                "model": model,
                "messages": (
                    [{"role": "system", "content": sys_prompt}]
                    + history_msgs
                    + [{"role": "user", "content": text}]
                ),
                # R34-S50 — kept stream=False here; the architectural
                # win (stream + chunked TTS) is being staged in a
                # separate change. The latency gains from adaptive
                # num_predict + stop sequences + smaller history below
                # cover the 80/20 case (most replies are ≤60 tokens
                # anyway).
                "stream": False,
                "think": False,
                # R34-S39 — pass keep_alive so the model stays in VRAM
                # between turns. Without this, after ~5min idle Ollama
                # unloads qwen3:8b → next turn pays 30-60s reload.
                "keep_alive": "24h",
                "options": {
                    "num_predict": effective_num_predict,
                    "num_ctx": num_ctx,
                    "temperature": float(getattr(cfg, "chat_temperature", 0.3)),
                    # R34-S50 — explicit stop sequences. qwen3:8b
                    # sometimes drifts past the first paragraph into
                    # markdown lists / dialogue hallucinations; these
                    # tokens are clear "this turn is done" markers and
                    # let the model finish naturally even if
                    # num_predict gave it more headroom.
                    "stop": ["\n\n", "</s>", "<|im_end|>"],
                },
            }
            # R34-S42 К4 — diagnostic so future timeouts can be triaged
            # quickly. Logs sys-prompt size, user-text len, model + ctx
            # settings + start time. Visible in jarvis-assistant.err.log
            # when JARVIS_VOICE_DEBUG=true.
            try:
                debug_log(
                    f"direct_chat→ollama: sys={len(sys_prompt)}ch "
                    f"user={len(text)}ch hist={len(history_msgs)} model={model} "
                    f"num_predict={effective_num_predict}/{num_predict} "
                    f"num_ctx={num_ctx} lang={lang}",
                    "pipecat",
                )
            except Exception:
                pass
            t0_call = _time.monotonic()
            try:
                async with session.post(ollama_url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        debug_log(
                            f"direct_chat: ollama {resp.status} after "
                            f"{_time.monotonic()-t0_call:.2f}s: {body[:200]}",
                            "pipecat",
                        )
                        return ""
                    data = await resp.json()
                    reply = str(data.get("message", {}).get("content", "") or "")
                    debug_log(
                        f"direct_chat: ollama OK in "
                        f"{_time.monotonic()-t0_call:.2f}s reply_len={len(reply)}",
                        "pipecat",
                    )
                    return reply
            except Exception as exc:
                debug_log(
                    f"direct_chat: ollama call failed after "
                    f"{_time.monotonic()-t0_call:.2f}s: {exc!r}",
                    "pipecat",
                )
                return ""

        async def _speak(self, text: str, direction: "FrameDirection") -> None:
            """Speak the text via macOS `say`. Blocking subprocess in
            executor so we don't stall the event loop. We pick a
            Russian voice (Yuri or Milena) and a slightly faster
            rate so responses feel snappy.

            CRITICAL (R34-S30): we push REAL Pipecat
            ``TTSStartedFrame``/``BotStartedSpeakingFrame`` (and their
            stopped counterparts) downstream so the
            ``JarvisStateProcessor`` at the pipeline tail can update
            ``state.json`` to "SPEAKING", which is what the Electron
            HUD overlay (the coin) watches. Without these frames the
            coin stays stuck on LISTENING even though Jarvis is
            actively talking.

            R34-S45 — standby gate. If the HUD's Close button placed
            the daemon into standby, abort the speech entirely. Same
            policy as ``JarvisFastPathProcessor._speak_ack``.

            R34-S47 — repetition guard. If the LLM regenerated a
            phrase we already spoke in the last 90 s AND the user
            didn't say "повтори"/"again", swap to a tiny "вже
            відповідав" line instead of replaying the full reply.

            R34-S48 — UA→RU normalise. The Piper voice is Russian,
            so any leftover Ukrainian glyph would be mispronounced.
            We run ``_ru_normalise`` on EVERY outbound TTS string;
            already-Russian text is a no-op so this is safe.
            """
            if not text:
                return
            # R34-S48 — lock outbound speech to Russian phonemes BEFORE
            # any other processing. Done up front so the repetition
            # fingerprint compares apples-to-apples (both candidate and
            # history are normalised).
            text = _ru_normalise(text)
            if _is_standby():
                try:
                    self._stream.emit(
                        "log",
                        level="DEBUG",
                        component="direct-chat",
                        message=f"STANDBY: skipped speak {text[:60]!r}",
                    )
                except Exception:
                    pass
                return
            # R34-S47 — repetition guard. ``self._last_user_text`` is
            # set in ``process_frame`` from the incoming TranscriptionFrame
            # before we call ``_speak``. If the user explicitly asked
            # for a repeat, allow it; otherwise mutate.
            try:
                if (
                    would_be_repeat(text)
                    and not _is_repeat_request(getattr(self, "_last_user_text", None))
                ):
                    self._stream.emit(
                        "log",
                        level="INFO",
                        component="direct-chat",
                        message=(
                            f"repetition guard: would repeat "
                            f"{text[:60]!r} — substituting"
                        ),
                    )
                    # R34-S51 — hard-code Russian directly. Previously
                    # this assigned the UA literal AFTER ``_ru_normalise``
                    # had already run on the original text, so the
                    # substitution leaked to TTS verbatim. The literal
                    # below skips the table entirely.
                    text = "Я уже отвечал на это."
            except Exception:
                pass
            # R34-S32: register phrase before speaking for echo filter.
            try:
                _register_bot_utterance(text)
            except Exception:
                pass
            # Belt-and-suspenders: write state.json directly so the HUD
            # coin flips to SPEAKING even if Pipecat's task is sluggish.
            try:
                _write_hud_state_atomic("SPEAKING", level=1.0)
            except Exception:
                pass
            try:
                self._stream.emit("bot_started_speaking", text=text[:200])
                self._stream.emit("state", state="SPEAKING", level=1.0)
                await self.push_frame(TTSStartedFrame(), direction)
                await self.push_frame(BotStartedSpeakingFrame(), direction)
            except Exception as exc:
                debug_log(
                    f"direct_chat: TTS-start frames failed: {exc!r}",
                    "pipecat",
                )

            # R34-S34: Piper male voice (Ruslan) + interruptable
            # ``afplay``. Falls back to ``say`` if Piper unavailable.
            wav_path = await _asyncio.get_event_loop().run_in_executor(
                None, _piper_synthesize, text,
            )
            if wav_path is not None:
                await _play_audio_interruptable(wav_path)
            else:
                # R34-S51 — final normalise just before ``say``. Piper
                # path already gets the same guard via _piper_synthesize.
                _say_text = _ru_normalise(text)
                def _run_say() -> None:
                    import subprocess as _sp
                    try:
                        # R34-S40 — close stdio so launchd-detached child
                        # can't accidentally inherit terminal (would cause
                        # SIGTTIN under some shells).
                        _sp.run(
                            ["say", "-v", "Milena", "-r", "210", _say_text],
                            check=False,
                            timeout=60,
                            stdin=_sp.DEVNULL,
                            stdout=_sp.DEVNULL,
                            stderr=_sp.DEVNULL,
                        )
                    except Exception as exc:
                        debug_log(
                            f"say subprocess failed: {exc!r}", "pipecat",
                        )
                loop = _asyncio.get_event_loop()
                await loop.run_in_executor(None, _run_say)
            # R34-S39 — drop the post-playback drain sleep. The
            # fingerprint-based echo filter catches late mic-captured
            # echo for up to _BOT_HISTORY_TTL_SEC (30s), so we don't need
            # the artificial dead-time anymore. Saves ~1s before the
            # user can start the next turn.
            #
            # R34-S40 — RE-register bot phrase AFTER playback so the
            # fingerprint's 30s TTL window starts when the user can
            # actually hear it (and the mic can capture echo).
            # Without this, a 10-second reply uses up most of the TTL
            # on its own playback time, leaving <20s for late echo.
            try:
                _register_bot_utterance(text)
            except Exception:
                pass
            try:
                await self.push_frame(BotStoppedSpeakingFrame(), direction)
                await self.push_frame(TTSStoppedFrame(), direction)
                self._stream.emit("bot_stopped_speaking")
                self._stream.emit("state", state="LISTENING", level=1.0)
            except Exception:
                pass
            try:
                _write_hud_state_atomic("LISTENING", level=1.0)
            except Exception:
                pass

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)
            # Only TranscriptionFrames go through the bypass.
            if not isinstance(frame, TranscriptionFrame):
                await self.push_frame(frame, direction)
                return
            text = (frame.text or "").strip()
            if not text:
                await self.push_frame(frame, direction)
                return
            # R34-S47 — remember the latest user turn for the
            # repetition guard (see ``_speak``). Set this BEFORE the
            # standby drop so the next post-resume turn correctly
            # tracks "повтори" requests.
            self._last_user_text = text
            # R34-S45 — standby gate. Drop transcript entirely instead
            # of wasting an Ollama call and TTS round-trip we'd just
            # silence in _speak() anyway. Wake-gate upstream already
            # filters non-wake transcripts during standby, but this
            # belt-and-braces check guards against dashboard-injected
            # frames that bypass the wake-gate (user_id="dashboard").
            if _is_standby():
                try:
                    self._stream.emit(
                        "log",
                        level="DEBUG",
                        component="direct-chat",
                        message=f"STANDBY: dropped frame {text[:60]!r}",
                    )
                except Exception:
                    pass
                return
            # R34-S38: drop wake-word-only utterances BEFORE LLM call.
            # When STT transcribes only the wake-word (often mis-heard:
            # "Джарвіс" → "Джагвіз", "Джаглись", "Jarvis", ...), there's
            # no question for the LLM to answer. Smaller models then
            # HALLUCINATE responses (observed: "сейчас 00:25" from a
            # bare "джаглись" input — model invents a time query).
            #
            # Strategy: strip punctuation, lowercase, check word count.
            # If the remaining text is just a wake-word variant or ≤1
            # short token, speak a short "Слухаю" ack and return —
            # don't waste an LLM call or risk a hallucination loop.
            _normalised = _re.sub(r"[\s\.\!\?,\-—…\"']+", " ", text).strip().lower()
            _wake_variants = {
                "джарвіс", "джарвис", "джервіс", "джервис", "jarvis",
                "джагвіз", "джагвис", "джагвіс", "джаглись", "джаквіс",
                "джервиз", "джервиз", "джарвиз", "джарвиз",
            }
            _tokens = _normalised.split()
            _is_wake_only = (
                _normalised in _wake_variants
                or (len(_tokens) == 1 and any(
                    _tokens[0].startswith(w[:5]) for w in _wake_variants
                ))
            )
            if _is_wake_only:
                debug_log(
                    f"direct_chat: skipping LLM for wake-only input: {text!r}",
                    "pipecat",
                )
                # R34-S41 — debounce repeated wake-only acks. Without
                # this, mashing "Джарвіс. Джарвіс. Джарвіс." queues
                # multiple "Слухаю" replies that chain after each
                # plays. 5s gap suppresses follow-on acks; the wake
                # window stays open so the user can still issue a
                # command after the first ack.
                import time as _t_ack
                _last_ack = getattr(self, "_last_ack_monotonic", 0.0)
                _now_mon = _t_ack.monotonic()
                if _now_mon - _last_ack < 5.0:
                    debug_log(
                        "direct_chat: ack debounced — already acked recently",
                        "pipecat",
                    )
                    return
                # Just emit a short audible ack so user knows mic is hot.
                if not self._busy:
                    self._busy = True
                    self._last_ack_monotonic = _now_mon
                    try:
                        # R34-S51 — RU-only output policy. Wake ack is
                        # always Russian; the user explicitly demanded
                        # "відключи українську розмовну". STT still
                        # transcribes UA, so the user can SAY anything,
                        # but TTS never echoes UA back.
                        ack = "Слушаю."
                        self._stream.emit("sentence", text=ack, lang="ru")
                        await self._speak(ack, direction)
                    finally:
                        self._busy = False
                return
            if self._busy:
                # Another direct chat call in flight — drop the new one
                # to avoid speaker overlap.
                return
            self._busy = True
            try:
                # Flip state to THINKING so the HUD coin's orange glow
                # comes up while we wait on Ollama.
                try:
                    _write_hud_state_atomic("THINKING", level=0.5)
                except Exception:
                    pass
                self._stream.emit(
                    "state", state="THINKING", level=0.5,
                )
                self._stream.emit(
                    "direct_chat_user", text=text[:300],
                )
                # R34-S37 — detect input language per-turn so the
                # system prompt's language lock matches what the user
                # actually said. Falls back to cfg.active_language on
                # any detector failure.
                #
                # R34-S48 — FORCE reply language to Russian regardless
                # of detected input. User explicitly asked: "нехай
                # відповідає завжди російською но українську просто
                # розуміє". STT still transcribes UA correctly via
                # Whisper, so understanding works; only the reply path
                # is pinned. We detect for diagnostic logging but pass
                # ``"ru"`` into ``_call_ollama``.
                try:
                    from ..memory.self_learning import detect_user_language
                    detected_lang = detect_user_language(text)
                except Exception:
                    detected_lang = cfg.active_language or "ru"
                # Always reply in Russian.
                reply_lang = "ru"
                if detected_lang != reply_lang:
                    debug_log(
                        f"direct_chat: detected={detected_lang} but "
                        f"forcing reply_lang={reply_lang} (R34-S48)",
                        "pipecat",
                    )
                reply = await self._call_ollama(text, lang=reply_lang)
                debug_log(
                    f"direct_chat: reply len={len(reply)}: {reply[:80]!r}",
                    "pipecat",
                )
                # R34-S47 — append BOTH user + assistant to the rolling
                # history so the NEXT turn sees the dialogue. We only
                # append if the reply is non-empty (Ollama timeout =
                # empty string → we don't want to teach the LLM that
                # silence is a valid assistant reply).
                if reply and reply.strip():
                    try:
                        # R34-S52 C-2: normalise the assistant text BEFORE
                        # storing into history. Without this, the LLM sees
                        # its own UA-tainted prior reply on the next turn
                        # and treats UA tokens as in-distribution → keeps
                        # answering UA despite the language_lock_block.
                        # The TTS-side normaliser only fixes audio, not
                        # context. Both sides must be RU-pinned.
                        try:
                            _hist_reply = _ru_normalise(reply)[:800]
                        except Exception:
                            _hist_reply = reply[:800]
                        self._messages_history.append(
                            {"role": "user", "content": text[:800]}
                        )
                        self._messages_history.append(
                            {"role": "assistant", "content": _hist_reply}
                        )
                        # Trim from the front so we never exceed the
                        # max — keeps the window O(8) regardless of
                        # session length.
                        excess = (
                            len(self._messages_history)
                            - self._messages_history_max
                        )
                        if excess > 0:
                            del self._messages_history[:excess]
                    except Exception:
                        pass
                self._stream.emit(
                    "direct_chat_reply", text=reply[:500], model=model,
                )
                # Also emit a `sentence` event — that's the canonical
                # event the HUD's captions panel watches to show the
                # streaming reply line under the floating coin. Stock
                # Pipecat's TTS chain emits these per-sentence; since
                # we bypass that chain, we emit one whole-reply
                # ``sentence`` here so the user sees the text.
                try:
                    self._stream.emit(
                        "sentence",
                        text=(reply or "").strip()[:1000],
                        lang="ru",
                    )
                except Exception:
                    pass
                # Also emit ``stt_final``-shaped event with the user's
                # query text so the caption "user" row populates even
                # when the path didn't come from real STT (i.e. when
                # the request was injected via /api/voice/inject for
                # tests). Real STT injects its own stt_final upstream
                # via the event stream processor.
                try:
                    self._stream.emit(
                        "stt_final",
                        text=text[:500],
                        lang="ru",
                        confidence=1.0,
                        duration_ms=0,
                    )
                except Exception:
                    pass
                if reply.strip():
                    await self._speak(reply.strip(), direction)
                else:
                    # Empty reply — speak a polite fallback so the user
                    # knows we heard them, not silence.
                    # R34-S51 — RU-only output. Removed the UA branch
                    # entirely; user demanded "нехай відповідає завжди
                    # російською". The phrase below is the EXACT one
                    # he reported hearing in Ukrainian — now it's only
                    # ever pronounced in Russian.
                    fallback = "Я не понял, повтори, пожалуйста."
                    await self._speak(fallback, direction)
                # R34-S37 — self-learning: extract facts + record turn.
                # Fire-and-forget; never delays the next turn.
                try:
                    from ..memory.self_learning import record_turn
                    record_turn(
                        user_text=text,
                        bot_text=reply or "",
                        lang=detected_lang,
                        ollama_url=cfg.ollama_base_url,
                        chat_model=cfg.chat_model,
                        enable_llm_extract=True,
                    )
                except Exception as exc:
                    debug_log(
                        f"self-learn record_turn failed: {exc!r}",
                        "pipecat",
                    )
            finally:
                self._busy = False
            # CONSUME the frame — do NOT forward to the broken LLM chain.

    return JarvisDirectChatProcessor


def _make_fast_path_processor(cfg: "PipecatLoopConfig | None" = None):
    """Build the regex fast-path FrameProcessor.

    Wraps ``action_dispatcher.parse_user_command`` in a frame-aware
    filter. Action execution happens in the default thread executor
    because the underlying ops use blocking ``subprocess.run``
    (AppleScript / ``open -a``) that would stall the asyncio loop
    and disrupt audio frame timing.

    R34-S31: speaks confirmation via macOS ``say`` subprocess
    directly (NOT ``TTSSpeakFrame``) because the Pipecat 1.2.1 TTS
    chain wedges after a few minutes — same root cause that forced
    the ``direct_chat`` bypass. Without this fix the user sees the
    app launch but never hears "Зараз відкрию Safari".
    """
    import asyncio as _asyncio

    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        Frame,
        TextFrame,
        TranscriptionFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream
    from .action_dispatcher import Action, parse_user_command

    # R34-S43 — confirmation gate. Invasive actions (open app, close
    # tab, quit app, lock screen, mute, write clipboard, set volume,
    # web search, send email…) MUST receive explicit verbal user
    # confirmation before executing. The user reported "Jarvis виконує
    # дії без мене" — this gate prevents that class of harm.
    #
    # Read-only / informational actions stay zero-friction (telling
    # the time or reading the clipboard isn't a destructive op).
    _SAFE_ACTIONS_NO_CONFIRM = {
        "say_time",      # purely informational
        "say_date",      # purely informational
        "battery",       # informational
        "read_clipboard",# informational (does NOT write)
        # R34-S47 — Nexus-Brain read is non-destructive: ripgrep over
        # the user's own vault, no side effects beyond Jarvis's RAM.
        # Asking for confirmation on every "знайди у мозку X" would
        # be friction with no safety benefit.
        "nexus_brain_search",
        # R34-S49 REC 2 — Web fact-check is read-only HTTP GET to DDG.
        # No browser tab is opened, no state mutates; the action just
        # returns a short summary string. Treating it as safe means
        # voice "перевір в інтернеті X" gives an immediate spoken
        # answer (the whole point of the skill) without an annoying
        # "Подтверждаешь?" round-trip.
        "web_fact_check",
    }
    # Confirmation vocab — keep generous; STT mishearings happen.
    # All entries lowercased & punctuation-stripped before match.
    _CONFIRM_YES = {
        "так", "да", "yes", "yeah", "ага", "ок", "ok", "okay",
        "підтверджую", "підтверджу", "подтверждаю", "согласен",
        "давай", "ага давай", "так давай", "погоджуюсь",
        "звичайно", "конечно", "sure", "go", "go ahead", "доброго",
        "роби", "сделай", "виконуй", "выполняй",
    }
    _CONFIRM_NO = {
        "ні", "нет", "no", "nope", "відміни", "отмени", "скасуй",
        "cancel", "abort", "не треба", "не надо",
        "стоп", "стой", "stop",
    }

    class JarvisFastPathProcessor(FrameProcessor):
        """Direct-execution path for matched regex user commands."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._stream = get_stream()
            # R33-S3 audit — lazy import so the legacy listener path
            # doesn't pay audit init cost when only Pipecat uses it.
            from ..audit import get_audit_store
            self._audit = get_audit_store()
            # R34-S43 confirmation gate — pending action awaiting
            # user yes/no. (action, t0_monotonic). Reset on confirm,
            # cancel, on TTL-expiry (15s — R34-S47 tightened from 60),
            # on STANDBY engage, or on any non-yes-no transcript that
            # arrives before confirmation.
            self._pending_action: tuple["Action", float] | None = None
            # R34-S47 — TTL slashed 60 → 15 s. Real users say "так"
            # within ~2 s of the prompt; anything longer almost always
            # means they walked away / got distracted / decided no.
            # Holding pending for 60 s caused the symptom the user
            # reported: random video-call cross-talk landing in the
            # confirmation slot ages after the prompt was forgotten.
            self._pending_ttl_sec: float = 15.0
            # R34-S47 — register a standby callback that wipes any
            # pending confirmation. Without this, pressing the HUD
            # Close button mid-prompt left ``_pending_action`` set;
            # the next wake-word turn was then silently dropped by
            # the ambient-lock branch ("pending-lock: dropped non-
            # yes/no transcript 'Джарвіс, котра година'") instead of
            # parsing the fresh command.
            register_standby_callback(self._clear_pending_on_standby)

        def _clear_pending_on_standby(self) -> None:
            """Wipe pending confirmation when standby engages.

            Runs on the standby-fanout thread, so we keep it cheap
            and exception-safe. Emits a log so the user/dashboard can
            see WHY the next turn isn't ambient-locked.

            R34-S54.1 Phase 7c: documented the cross-thread mutation.
            ``_pending_action`` is also touched by ``process_frame`` on
            the asyncio loop (TTL expiry / confirm / cancel / supersede
            / set). CPython reference assignment is atomic, so a single
            ``self._pending_action = None`` write from this thread races
            benignly against the loop's own assignments. The ``try /
            except`` guards the snapshot read against ``None[0]`` if
            the loop cleared between our None-check and our index.
            That's the only window — no torn read possible past this
            method. A proper ``threading.Lock`` would over-engineer for
            zero observable benefit (worst case: a redundant log line).
            """
            if self._pending_action is None:
                return
            try:
                cancelled = self._pending_action[0].name
            except Exception:
                cancelled = "?"
            self._pending_action = None
            try:
                self._stream.emit(
                    "log",
                    level="INFO",
                    component="fast-path",
                    message=(
                        f"STANDBY engaged → cleared pending {cancelled}"
                    ),
                )
            except Exception:
                pass

        async def _speak_ack(
            self, text: str, direction: "FrameDirection"
        ) -> None:
            """Speak confirmation via macOS ``say`` and update HUD.

            Mirrors :meth:`JarvisDirectChatProcessor._speak`. We can't
            push a ``TTSSpeakFrame`` here because the Pipecat TTS
            chain downstream is the very thing the ``direct_chat``
            bypass was created to avoid (wedges after warmup, never
            emits ``TTSStartedFrame``). Direct ``say`` + atomic
            ``state.json`` writes keep the HUD coin in sync.

            R34-S45 — standby gate. If the HUD's Close button placed
            the daemon into standby, abort the speech entirely. The
            wake-gate will let a wake-word through and clear standby
            before any speak call lands here.

            R34-S48 — UA→RU normalise. Fast-path descriptions are
            authored in Ukrainian (``Закриваю Safari`` etc.) but the
            Piper voice is Russian. Normalising here keeps the action
            factories untouched while making sure Piper only ever
            sees Russian.
            """
            if not text:
                return
            text = _ru_normalise(text)
            if _is_standby():
                try:
                    self._stream.emit(
                        "log",
                        level="DEBUG",
                        component="fast-path",
                        message=f"STANDBY: skipped speak {text[:60]!r}",
                    )
                except Exception:
                    pass
                return
            # R34-S32: register the phrase BEFORE speaking so the
            # echo filter has the fingerprint when mic-captured echo
            # arrives ~2-3 s later.
            try:
                _register_bot_utterance(text)
            except Exception:
                pass
            try:
                _write_hud_state_atomic("SPEAKING", level=1.0)
            except Exception:
                pass
            try:
                self._stream.emit("bot_started_speaking", text=text[:200])
                self._stream.emit("state", state="SPEAKING", level=1.0)
                await self.push_frame(TTSStartedFrame(), direction)
                await self.push_frame(BotStartedSpeakingFrame(), direction)
            except Exception as exc:
                debug_log(
                    f"fast_path: TTS-start frames failed: {exc!r}",
                    "pipecat",
                )
            # Also emit a `sentence` event so the HUD captions panel
            # picks up the confirmation line.
            try:
                self._stream.emit(
                    "sentence",
                    text=text[:1000],
                    # R34-S52 H: HUD label must match the actually-spoken
                    # language. After R34-S48/S51 RU-only pin + _ru_normalise
                    # at TTS time, the text reaching the user is always RU.
                    # Mislabelling as UA confused the dashboard locale logic.
                    lang="ru",
                )
            except Exception:
                pass

            # R34-S34: Piper (ru_RU-ruslan-medium = MALE, low tone)
            # + interruptable ``afplay``. Falls back to macOS ``say``
            # if Piper isn't available (e.g. ONNX missing).
            wav_path = await _asyncio.get_event_loop().run_in_executor(
                None, _piper_synthesize, text,
            )
            if wav_path is not None:
                await _play_audio_interruptable(wav_path)
            else:
                # Fallback: macOS `say` with Milena (female RU). Not
                # interruptable — user will have to wait for the
                # current utterance to end.
                # R34-S51 — final normalise guard.
                _say_text = _ru_normalise(text)
                def _run_say() -> None:
                    import subprocess as _sp
                    try:
                        # R34-S42 — DEVNULL all streams (same fix as
                        # DirectChat path) to prevent SIGTTIN under
                        # foreground daemon launches.
                        _sp.run(
                            ["say", "-v", "Milena", "-r", "210", _say_text],
                            check=False,
                            timeout=30,
                            stdin=_sp.DEVNULL,
                            stdout=_sp.DEVNULL,
                            stderr=_sp.DEVNULL,
                        )
                    except Exception as exc:
                        debug_log(
                            f"fast_path: say subprocess failed: {exc!r}",
                            "pipecat",
                        )
                loop = _asyncio.get_event_loop()
                await loop.run_in_executor(None, _run_say)
            # Short drain so the BotStoppedSpeakingFrame fires AFTER
            # the speaker buffer has fully flushed (matters for echo
            # filter — see R34-S32 commentary).
            # R34-S39 — drop the post-playback drain sleep. The
            # fingerprint-based echo filter catches late mic-captured
            # echo for up to _BOT_HISTORY_TTL_SEC (30s), so we don't need
            # the artificial dead-time anymore. Saves ~1s before the
            # user can start the next turn.
            #
            # R34-S40 — RE-register bot phrase AFTER playback so the
            # fingerprint's 30s TTL window starts when the user can
            # actually hear it (and the mic can capture echo).
            # Without this, a 10-second reply uses up most of the TTL
            # on its own playback time, leaving <20s for late echo.
            try:
                _register_bot_utterance(text)
            except Exception:
                pass
            try:
                await self.push_frame(BotStoppedSpeakingFrame(), direction)
                await self.push_frame(TTSStoppedFrame(), direction)
                self._stream.emit("bot_stopped_speaking")
                self._stream.emit("state", state="LISTENING", level=1.0)
            except Exception:
                pass
            try:
                _write_hud_state_atomic("LISTENING", level=1.0)
            except Exception:
                pass

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

            # R34-S43/S44 — confirmation gate.
            # ``action`` is None until we resolve which path applies:
            #   (a) Yes-answer to a pending confirmation → use the
            #       saved Action and fall through to execute below.
            #   (b) No-answer → cancel pending, speak "Скасовано", done.
            #   (c) Non-yes/no while pending → DROP (ambient lock), so
            #       background voices / Whisper hallucinations during
            #       the 60s pending window don't blow away the
            #       confirmation state. User must say wake-word
            #       ("Джарвіс …") to explicitly start a new command.
            #   (d) No pending state → just parse the transcript.
            import time as _t_conf
            action: "Action | None" = None
            if self._pending_action is not None:
                pending, t0 = self._pending_action
                # TTL check — abandon stale prompts so old "так" answers
                # don't accidentally fire an action the user forgot about.
                if (_t_conf.monotonic() - t0) > self._pending_ttl_sec:
                    self._pending_action = None
                else:
                    norm = text.lower().strip().strip(".,!?;:-—–").strip()
                    if norm in _CONFIRM_YES:
                        action = pending
                        self._pending_action = None
                        try:
                            self._stream.emit(
                                "log",
                                level="INFO",
                                component="fast-path",
                                message=f"confirmed → execute {action.name}",
                            )
                        except Exception:
                            pass
                        # Fall through to the execute block below.
                    elif norm in _CONFIRM_NO:
                        cancelled = pending.name
                        self._pending_action = None
                        try:
                            self._stream.emit(
                                "log",
                                level="INFO",
                                component="fast-path",
                                message=f"cancelled pending {cancelled}",
                            )
                        except Exception:
                            pass
                        await self._speak_ack("Скасовано.", direction)
                        return
                    else:
                        # R34-S44/R34-S47 — softened ambient lock.
                        #
                        # Old behaviour locked out ALL non-yes/no turns
                        # for the entire 60 s TTL. In practice the user
                        # often:
                        #   • Walks away mid-prompt → 60 s later wants
                        #     to start a fresh request, but every wake
                        #     word "Джарвіс, котра година" got dropped
                        #     as "pending-lock". Reported by Danylo.
                        #   • Decides on a different command without
                        #     saying "ні" first — the pending stays
                        #     and the new command never reaches the
                        #     parser.
                        #
                        # New behaviour: if the new transcript LOOKS LIKE
                        # an explicit new command — i.e. it parses to a
                        # different fast-path Action — implicitly cancel
                        # the old pending and treat the new utterance as
                        # the live command. Otherwise (random ambient
                        # chatter, Whisper hallucinations) we still
                        # drop, preserving the protective behaviour
                        # against family members talking nearby.
                        new_action = None
                        try:
                            new_action = parse_user_command(text)
                        except Exception:
                            new_action = None
                        if new_action is not None and (
                            new_action.name != pending.name
                            or (new_action.description or "") != (pending.description or "")
                        ):
                            try:
                                self._stream.emit(
                                    "log",
                                    level="INFO",
                                    component="fast-path",
                                    message=(
                                        f"pending {pending.name} superseded by "
                                        f"new command {new_action.name}: "
                                        f"{text[:60]!r}"
                                    ),
                                )
                            except Exception:
                                pass
                            # Drop the old pending, then fall through to
                            # the normal "fresh command" branch below by
                            # leaving ``action`` None — that branch will
                            # re-parse (cheap, deterministic) and then
                            # apply the SAFE/invasive policy correctly.
                            # The new fast-path's confirmation prompt
                            # (if invasive) will be asked just like a
                            # cold-start command, so we never bypass the
                            # confirmation step for destructive ops.
                            self._pending_action = None
                            # Intentionally do NOT set ``action`` here.
                        else:
                            try:
                                self._stream.emit(
                                    "log",
                                    level="INFO",
                                    component="fast-path",
                                    message=(
                                        f"pending-lock: dropped non-yes/no "
                                        f"transcript {text[:80]!r} "
                                        f"(waiting on {pending.name})"
                                    ),
                                )
                            except Exception:
                                pass
                            return  # consume, keep pending

            if action is None:
                # R34-S53.1 (Phase 6b — C-3 side-effect): if there is
                # NO pending action AND the transcript is a bare
                # confirmation token ("так", "да", "нет", "ні") — silently
                # drop. Without this, the C-3 fix that bypassed the
                # hallucination filter for confirmation tokens caused
                # "так"/"да" to fall through to parse_user_command (no
                # match) → push_frame() → LLM, which then earnestly
                # generated a chatty reply to a one-word utterance with
                # no context. The result was random TTS lines like
                # "Да, я тебя слышу" firing whenever the user said "да"
                # during a Zoom call. The guard below is tight: only
                # for ≤2-word transcripts that match the confirm
                # vocabulary, and only when no action is pending.
                _norm_chk = text.lower().strip().strip(".,!?;:-—–").strip()
                if _norm_chk and (
                    _norm_chk in _CONFIRM_YES or _norm_chk in _CONFIRM_NO
                ):
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="fast-path",
                            message=(
                                f"dropped bare confirmation token "
                                f"{text[:40]!r} (no pending action)"
                            ),
                        )
                    except Exception:
                        pass
                    return  # consume — never reaches LLM
                # Path (c) or (d): parse the transcript as a new command.
                try:
                    action = parse_user_command(text)
                except Exception as exc:
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

                # Invasive action → ask for confirmation, do NOT exec.
                if action.name not in _SAFE_ACTIONS_NO_CONFIRM:
                    self._pending_action = (action, _t_conf.monotonic())
                    desc = (action.description or action.name).strip()
                    # R34-S48 — Russian-only confirmation prompts.
                    #
                    # Previously we detected the user's spoken language
                    # and built a UA-or-RU prompt to match. User now
                    # asks for pure-Russian output ("він має ужасну
                    # українську мову розмовну"), so we hard-code RU
                    # for the prompt scaffolding and pipe ``desc``
                    # through ``_ru_normalise`` so even UA-authored
                    # action descriptions ("Закриваю Safari") reach
                    # Piper as Russian ("Закрываю Safari").
                    desc_ru = _ru_normalise(desc)
                    prompt = (
                        f"{desc_ru}. Подтверждаешь? "
                        "Скажи «да» чтобы выполнить или «нет» чтобы отменить."
                    )
                    try:
                        self._audit.emit(
                            kind="fast_path_confirm",
                            tool=action.name,
                            status="awaiting_user_confirm",
                            args={"text": text, "desc": desc},
                        )
                    except Exception:
                        pass
                    await self._speak_ack(prompt, direction)
                    return  # consume the frame — wait for yes/no.

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
            # acknowledgement via macOS ``say`` directly — the Pipecat
            # TTS chain is wedged (the whole reason direct_chat exists).
            try:
                _write_hud_state_atomic("THINKING", level=0.5)
            except Exception:
                pass
            self._stream.emit("state", state="THINKING", level=0.5)
            self._stream.emit(
                "tool_call",
                tool=action.name,
                args={},
                status="starting",
            )
            # Also emit ``stt_final`` so the HUD captions show the
            # user's matched command — real STT events already fired
            # upstream, but injected /api/voice/inject paths need this.
            try:
                self._stream.emit(
                    "stt_final",
                    text=text[:500],
                    # R34-S52 H: was hardcoded "uk" — but R34-S48 made
                    # RU the only outbound language, AND the input text
                    # itself can be either UA or RU (we accept both at
                    # STT). For HUD labelling consistency with the
                    # actual reply, mark as RU. If accurate input-side
                    # language detection is ever needed, plumb
                    # ``detect_user_language(text)`` here.
                    lang="ru",
                    confidence=1.0,
                    duration_ms=0,
                )
            except Exception:
                pass
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

            # R34-S48 — speak the RESULT for informational actions
            # (say_time, say_date, battery, read_clipboard,
            # nexus_brain_search). Previously we only ever spoke the
            # action ``description`` ("Дивлюся час") on success, so
            # the user heard "I'm looking at the time" but never the
            # actual time — the result string was logged but silent.
            # For destructive actions we keep the legacy behaviour
            # (speak the description) but always append a short error
            # hint on failure so the user knows what went wrong.
            _RESULT_IS_THE_ANSWER = {
                "say_time", "say_date", "battery", "read_clipboard",
                "nexus_brain_search",
                # R34-S49 REC 2 — DDG snippet IS the answer the user
                # asked for ("перевір в інтернеті X" → spoken summary).
                "web_fact_check",
                # R34-S49 REC 3 — speak the "Учусь делать X" status
                # message (the action's return value) instead of the
                # generic description, so the user immediately knows
                # the skill add is in progress.
                "self_add_skill",
            }
            if ok and action.name in _RESULT_IS_THE_ANSWER and msg and msg.strip():
                # The action's msg IS what the user asked for. Cap at
                # 600 chars so `nexus_brain_search` (which can return
                # multi-file snippets) doesn't run away.
                ack = msg.strip()
                if len(ack) > 600:
                    ack = ack[:600] + "…"
            else:
                ack = action.description or ("Готово." if ok else "Не получилось.")
                # On failure, append a short error hint so the user
                # knows WHY it didn't work (e.g. "приложение не
                # найдено").
                if not ok and msg and msg.strip():
                    hint = msg.strip()
                    if len(hint) > 80:
                        hint = hint[:80] + "..."
                    ack = f"{ack}. {hint}"
            await self._speak_ack(ack, direction)
            # DO NOT push the original TranscriptionFrame — that's
            # the whole point of the fast path. The frame stops here.

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
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        Frame,
        InterruptionFrame,
        TranscriptionFrame,
        TTSStartedFrame,
        TTSStoppedFrame,
    )
    from pipecat.processors.frame_processor import (
        FrameDirection,
        FrameProcessor,
    )

    from ..ipc import get_stream
    from .wake_detection import (
        extract_query_after_wake,
        is_wake_word_at_start,
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
            # R34-S34: track bot SPEAKING state so a wake-word during
            # active playback triggers barge-in (kill audio + interrupt).
            self._bot_speaking: bool = False

        def _open_hot_window(self) -> None:
            # R34-S52 H: was ``_time.time()`` (wall-clock). An NTP step
            # backwards left _hot_until in the future forever (wake-gate
            # open to ambient speech for hours); an NTP step forwards
            # closed the window early. Other parts of the file already
            # use ``_time.monotonic()`` consistently — match that here.
            self._hot_until = _time.monotonic() + hot_window_seconds
            try:
                self._stream.emit(
                    "hot_window",
                    active=True,
                    expires_in_ms=int(hot_window_seconds * 1000),
                )
            except Exception:
                pass

        def _is_hot(self) -> bool:
            # R34-S52 H: paired with _open_hot_window monotonic switch.
            return _time.monotonic() < self._hot_until

        async def process_frame(
            self, frame: "Frame", direction: "FrameDirection"
        ) -> None:
            await super().process_frame(frame, direction)

            # R34-S34: track SPEAKING state for barge-in logic below.
            if isinstance(frame, (BotStartedSpeakingFrame, TTSStartedFrame)):
                self._bot_speaking = True
            elif isinstance(frame, (BotStoppedSpeakingFrame, TTSStoppedFrame)):
                self._bot_speaking = False

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

            # R34-S45 — standby gate runs FIRST, before any other
            # bypass. If the HUD's Close button put the daemon into
            # standby, ONLY an explicit wake-word resumes it.
            # Everything else (ambient mic speech, hallucinated
            # transcripts, AND dashboard-injected requests) is
            # consumed silently. We check this above the dashboard
            # bypass because otherwise a dashboard inject during
            # standby would skip the gate entirely and the daemon
            # would respond to text the user can't see / didn't
            # authorise verbally.
            #
            # R34-S46 — STRICT wake-word for resume. The loose
            # ``wake_hit`` (substring + fuzzy) fires when ANY ambient
            # transcript mentions "джарвис" — a third party in a
            # video call saying "джарвис как раз управляет
            # компьютером" would unlock standby and unleash the
            # daemon on the rest of the conversation. Standby-resume
            # now requires the wake-word at the START of the
            # utterance (first 3 tokens). Genuine commands always
            # start with the wake-word ("Джарвіс, котра година");
            # mentions of the assistant mid-paragraph never do.
            if _is_standby():
                wake_at_start = is_wake_word_at_start(
                    text_lower, wake_word, wake_aliases, fuzzy_ratio,
                )
                if wake_at_start:
                    _set_standby(False)
                    try:
                        self._stream.emit(
                            "log",
                            level="INFO",
                            component="wake-gate",
                            message="wake-word during STANDBY — resuming",
                        )
                    except Exception:
                        pass
                    # Fall through — wake-word at start of dashboard
                    # or mic frame clears standby and proceeds.
                else:
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="wake-gate",
                            message=(
                                f"STANDBY: dropped non-wake transcript "
                                f"{text[:60]!r}"
                            ),
                        )
                    except Exception:
                        pass
                    return  # consume — daemon stays silent

            # R34-S31: dashboard-injected transcripts bypass the wake
            # gate entirely — the user already explicitly asked via
            # /api/voice/inject or the chat UI, so requiring a wake
            # word would be redundant friction. We ALSO open the hot
            # window so any natural follow-up doesn't need a wake
            # word either.
            #
            # R34-S45 — Order matters: standby is checked above. If
            # we reached here, either standby is off or the inject
            # contained the wake-word (which already cleared it).
            #
            # R34-S47 — strip the wake-word from dashboard injects too
            # (mirroring the hot-window stripping path). Without this,
            # "Джарвіс, котра година" reached the fast-path verbatim
            # and ``parse_user_command`` returned None because the
            # say_time regex anchors on ``^\s*котра``, not on a wake-
            # prefixed phrase. The user-visible symptom: trivial fast-
            # path queries fell through to a 30-60 s Ollama call.
            try:
                if getattr(frame, "user_id", "") == "dashboard":
                    self._open_hot_window()
                    if wake_hit:
                        query = extract_query_after_wake(
                            text_lower, wake_word, wake_aliases,
                        )
                        if query:
                            from dataclasses import replace as _replace
                            try:
                                frame = _replace(frame, text=query)
                            except Exception:
                                pass
                    await self.push_frame(frame, direction)
                    return
            except Exception:
                pass

            # R34-S34: barge-in — if Jarvis is currently speaking AND
            # the new transcript contains the wake-word, kill the
            # current audio and let the new transcript propagate
            # through the pipeline. Without wake-word during SPEAKING
            # we IGNORE the transcript entirely (no interruption on
            # background talk).
            #
            # R34-S41: ALSO barge-in on universal stop keywords. The
            # user complains "perebivayetsya tilki pry imeni" but they
            # ALSO need a fast way to say "стоп"/"тихо" without prefix
            # when Jarvis is rambling. We pre-check a tight set of
            # interrupt words BEFORE the wake-only filter so they get
            # honoured even mid-sentence.
            _STOP_KEYWORDS = {
                "стоп", "стой", "тихо", "хватит",
                "досить", "тиша", "вистачить",
                "stop", "silence", "quiet", "enough",
            }
            _normalised_stop = text_lower.strip().strip(".,!?;:-—–").strip()
            _is_universal_stop = (
                _normalised_stop in _STOP_KEYWORDS
                or any(_normalised_stop == f"{w} джарвіс" for w in _STOP_KEYWORDS)
                or any(_normalised_stop == f"{w} джарвис" for w in _STOP_KEYWORDS)
                or any(_normalised_stop == f"джарвіс {w}" for w in _STOP_KEYWORDS)
                or any(_normalised_stop == f"джарвис {w}" for w in _STOP_KEYWORDS)
            )
            if self._bot_speaking and _is_universal_stop:
                try:
                    _stop_current_playback()
                    self._stream.emit(
                        "log",
                        level="INFO",
                        component="wake-gate",
                        message=(
                            f"universal stop during SPEAKING: {text[:40]!r} "
                            "→ killed playback"
                        ),
                    )
                except Exception:
                    pass
                # Also write interrupt.flag so DirectChat in-flight
                # LLM call gets cancelled by the poller.
                try:
                    from pathlib import Path as _P
                    import time as _t
                    _flag = _P.home() / "Library/Application Support/jarvis/interrupt.flag"
                    _flag.write_text(str(int(_t.time())))
                except Exception:
                    pass
                return  # consume — don't process "стоп" as a query
            if self._bot_speaking:
                if wake_hit:
                    try:
                        _stop_current_playback()
                        self._stream.emit(
                            "log",
                            level="INFO",
                            component="wake-gate",
                            message="barge-in: wake-word during SPEAKING — killing playback",
                        )
                    except Exception:
                        pass
                    # Fall through to standard wake-hit handling below.
                else:
                    # Background talk while Jarvis is speaking → drop.
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="wake-gate",
                            message=(
                                "dropped during SPEAKING (no wake-word in "
                                f"transcript): {text[:60]!r}"
                            ),
                        )
                    except Exception:
                        pass
                    return  # consume frame, do NOT push

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
    """Build the echo-filter + Whisper-hallucination FrameProcessor.

    Two passes on every ``TranscriptionFrame``:

    1. ECHO — tracks ``BotStartedSpeakingFrame``/``BotStoppedSpeaking
       Frame`` boundaries (and ``TTSStartedFrame``/``TTSStoppedFrame``
       as secondary signals). A transcript that arrives while the bot
       is speaking is dropped. We also keep a small tail
       (``_TAIL_SEC``) after TTS stops because hardware audio latency
       means the speaker is still outputting samples for ~200-500 ms
       after the frame finishes.

    2. HALLUCINATION — Whisper large-v3-turbo's well-known artefact
       set: when fed near-silence or low-volume noise it produces a
       confident-but-meaningless phrase from its training tail
       ("Thank you for watching", "what time is it", "[Music]",
       "Дякую за перегляд", "Подписывайтесь на канал" …). R34-S14
       lowered the Silero threshold to 0.42 so real speech actually
       gets transcribed; the trade-off is that occasional ambient
       noise crosses the gate and Whisper hallucinates on it. Catching
       the hallucinations HERE (before the wake-word gate) means the
       LLM never sees them and the user never gets fake responses.

    The hallucination list is intentionally conservative — only
    phrases observed multiple times in production logs, never matching
    a real user command (you don't say "Thank you for watching" at
    your voice assistant). Case + whitespace are normalised before
    comparison.
    """
    import re as _re
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
    #
    # R34-S32: bumped 0.5 → 3.0s. With direct ``say``-based TTS the
    # `say` subprocess returns BEFORE the speaker has finished
    # playing the audio (CoreAudio queues the buffer). Add to that
    # ~500-1000 ms of room reverb + ~1-2 s mic-to-Whisper-to-
    # FrameProcessor latency, and 0.5s wasn't catching ANY of the
    # bot-echo. Live evidence (events.jsonl R34-S32): 6+ consecutive
    # mic stt_finals matched Jarvis own utterances ("Шукаю на YouTube
    # котики", "Сейчас 23 часа 24 минуты") with `Dropped by echo: 0`.
    # 3.0s catches the realistic round-trip; if the user wants to
    # barge in faster they can use the explicit Stop button.
    # R34-S39 — dropped from 3.0s to 0.7s. The bot-utterance
    # fingerprint registry (_BOT_HISTORY, 30s TTL) is the real defence
    # against late echo; the 3.0s window was paranoid overkill that
    # cost the user 3 full seconds of dead-time after every reply.
    # 0.7s still catches the immediate room reverb (typically <250ms)
    # plus STT pipeline latency, fingerprint covers the rest.
    _TAIL_SEC = 0.7

    # R34-S32: bot-utterance fingerprint match against the module-
    # level _BOT_HISTORY registry catches mic-captured echoes that
    # arrived past _TAIL_SEC due to room reverb + STT pipeline lag.

    # Whisper large-v3-turbo hallucination corpus. Sourced from the
    # legacy listener's ``KNOWN_HALLUCINATIONS`` set + new entries
    # observed in R34 production logs (RU/UK/EN). Each entry is the
    # normalised (lower, stripped of trailing punctuation) form;
    # matching is "exact equality after normalisation" so we never
    # eat a real command that happens to *contain* one of these.
    _HALLUCINATION_PHRASES = frozenset({
        # English — YouTube tail
        "thank you for watching",
        "thanks for watching",
        "thank you",
        "thanks",
        "please subscribe",
        "subscribe to my channel",
        # English — model "filler"
        "what time is it",
        "is a cool name",
        "will it rain today",
        "you",
        "bye",
        "okay",
        "ok",
        "uh",
        "um",
        "hmm",
        # English — music/sound tags
        "[music]",
        "[applause]",
        "[laughter]",
        "(music)",
        "(applause)",
        # Ukrainian
        "дякую за перегляд",
        "дякую",
        "підпишіться на канал",
        "продовження буде",
        # Russian
        "спасибо за просмотр",
        "спасибо",
        "подписывайтесь на канал",
        "подписывайтесь",
        "продолжение следует",
        # R34-S32: extra fillers seen in live events.jsonl when mic
        # was capturing room conversation. Whisper transcribed these
        # short Russian/Ukrainian interjections every few seconds and
        # they were polluting the pipeline → wake-gate burned cycles
        # on each. All confirmed safe to drop (not real commands).
        "так",
        "ну",
        "ну ладно",
        "ну хорошо",
        "хорошо",
        "ладно",
        "конечно",
        "понятно",
        "да",
        "нет",
        "не",
        "ага",
        "ага ага",
        "ой",
        "эй",
        "э",
        "ху",
        "хм",
        "м",
        "ммм",
        "а",
        "о",
        "у",
        # Russian variant of "yes/no"
        "да да",
        "нет нет",
        # Misc Whisper-isms
        ".",
        "-",
        "—",
        "…",
        "...",
        # R34-S39 — Jarvis's own fallback phrases (when LLM returned
        # empty / timed-out). These get spoken via TTS, mic re-captures,
        # STT transcribes back, and without permanent drop the user
        # hears an infinite "Я не понял, повтори, пожалуйста" loop
        # because each cycle pushes the 30s _BOT_HISTORY TTL forward.
        # Permanent ban — these are NEVER valid user input.
        "я не понял повтори пожалуйста",
        "я не понял, повтори, пожалуйста",
        "я не зрозумів повтори будь ласка",
        "я не зрозумів, повтори, будь ласка",
        "я не понял",
        "я не зрозумів",
        "слухаю",
        "слушаю",
    })

    # Strip surrounding punctuation/whitespace for normalised match.
    _NORMALISE_RE = _re.compile(r"^[\s\.\!\?,\-—…\"']+|[\s\.\!\?,\-—…\"']+$")

    def _is_hallucination(text: str) -> bool:
        # Empty / whitespace-only input → drop. The outer process_frame
        # already filters frame.text.strip() == '' but we keep the
        # belt-and-braces check here so the helper is safe to unit
        # test in isolation.
        if not text or not text.strip():
            return True
        normalised = _NORMALISE_RE.sub("", text).lower().strip()
        if not normalised:
            return True  # punctuation-only = drop
        # R34-S52 C-3: never drop a bare confirmation/denial token as a
        # hallucination — these are EXACTLY the words the user says to
        # answer a "Подтверждаешь?" prompt. `_HALLUCINATION_PHRASES`
        # contains "да", "нет", "так", "ні", "ага", "ок" because they're
        # also common Whisper-on-silence outputs, but dropping them here
        # silently breaks the confirmation flow. The downstream fast-path
        # already routes these to `_handle_pending_confirmation` when an
        # action is armed, and the EchoFilter handles bot-echo separately.
        if normalised in _CONFIRMATION_RESERVED:
            return False
        if normalised in _HALLUCINATION_PHRASES:
            return True
        # Whisper sometimes repeats a single token N times — "ого ого ого ого".
        # Drop if every word equals every other word AND there's >= 3 of them.
        words = normalised.split()
        if len(words) >= 3 and len(set(words)) == 1:
            return True
        return False

    # R34-S44 — short yes/no answers must NEVER be treated as bot echo,
    # even if they appear as substrings inside a recent bot prompt like
    # "Скажи «так» щоб виконати". Without this carve-out the confirmation
    # gate becomes unanswerable: the bot speaks the word "так" inside the
    # prompt, registers the whole sentence as a bot utterance, and then
    # drops the user's literal "так" reply via substring match.
    _CONFIRMATION_RESERVED = {
        "так", "да", "yes", "ага", "ок", "ok",
        "ні", "нет", "no", "стоп", "stop",
    }

    def _looks_like_bot_echo(text: str) -> bool:
        """True if ``text`` is plausibly a mic-capture of a recent bot
        phrase (registered via ``_register_bot_utterance``).

        Two-way containment: either the bot phrase is a substring of
        the transcript (typical for shorter bot replies + mic noise
        prefix/suffix), or the transcript is a substring of the bot
        phrase (mic truncated the echo). Either match scores high
        enough for a drop. Also: ≥70% word overlap.
        """
        fp = _bot_history_fingerprint(text)
        if not fp or len(fp) < 3:
            return False
        # R34-S44 — never drop confirmation answers as echo. ``fp`` is
        # already normalised (lowercased, punctuation stripped, single-
        # spaced) so this check matches "так" / "ага" / "ок" cleanly.
        if fp in _CONFIRMATION_RESERVED:
            return False
        # R34-S41 — monotonic clock + snapshot under lock so we
        # don't iterate a list being mutated by the producer thread.
        now = _time.monotonic()
        with _BOT_HISTORY_LOCK:
            snapshot = list(_BOT_HISTORY)
        for ts, bot_fp in snapshot:
            if now - ts > _BOT_HISTORY_TTL_SEC:
                continue
            if not bot_fp or len(bot_fp) < 3:
                continue
            # bidirectional containment
            if fp in bot_fp or bot_fp in fp:
                return True
            # word-level overlap — if ≥70% of transcript words match
            # bot-phrase words, treat as echo. Catches partial mic
            # capture / inserted filler words.
            t_words = set(fp.split())
            b_words = set(bot_fp.split())
            if t_words and b_words:
                overlap = len(t_words & b_words) / max(len(t_words), 1)
                if overlap >= 0.7:
                    return True
        return False

    class JarvisEchoFilterProcessor(FrameProcessor):
        """Drop transcripts that are TTS echo OR Whisper hallucinations."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._speaking = False
            # R34-S53.1 (Phase 6b — I-3): ``_speak_end_ts`` switched
            # to ``_time.monotonic()``. Using wall-clock ``_time.time()``
            # meant an NTP step / DST roll-over while the echo gate was
            # armed could either freeze the gate forever (clock jumped
            # forward past the tail) or unblock it instantly (clock
            # jumped back). Monotonic is unaffected by NTP and is the
            # right primitive for "X seconds from now". Mirrors the
            # ``_speak_end_ts`` callsites at lines 4358, 4372, 4427.
            self._speak_end_ts: float = 0.0
            self._stream = get_stream()

        def _is_blocking_window(self) -> bool:
            if self._speaking:
                return True
            if self._speak_end_ts and _time.monotonic() < self._speak_end_ts:
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
                # R34-S53.1 (Phase 6b — I-3): monotonic, see comment in __init__.
                self._speak_end_ts = _time.monotonic() + _TAIL_SEC

            if isinstance(frame, TranscriptionFrame):
                # 0. Dashboard-injected transcripts always pass — the
                # user explicitly asked, so echo-gate / hallucination
                # filters shouldn't second-guess them.
                if getattr(frame, "user_id", "") == "dashboard":
                    await self.push_frame(frame, direction)
                    return
                # R34-S42 — UNIVERSAL STOP must run BEFORE the echo
                # blocking window, otherwise "стоп" said mid-TTS gets
                # dropped here and never reaches wake-gate to barge-in.
                # Check tokenised first-word so "стоп, джарвіс" / "хватит уже"
                # all match without depending on whole-string equality.
                if self._speaking:
                    _STOP_KEYS = {
                        "стоп", "стой", "тихо", "хватит",
                        "досить", "тиша", "вистачить",
                        "stop", "silence", "quiet", "enough",
                    }
                    _ftext = (frame.text or "").lower()
                    # Strip all leading punct and look at first token
                    _tokens = _re.findall(r"[\wа-щьюяїієґА-ЩЬЮЯЇІЄҐ']+", _ftext)
                    if _tokens and _tokens[0] in _STOP_KEYS:
                        try:
                            _stop_current_playback()
                        except Exception:
                            pass
                        try:
                            from pathlib import Path as _P
                            import time as _t_stop
                            _flag = _P.home() / "Library/Application Support/jarvis/interrupt.flag"
                            _flag.write_text(str(int(_t_stop.time())))
                        except Exception:
                            pass
                        try:
                            self._stream.emit(
                                "log",
                                level="INFO",
                                component="echo-filter",
                                message=f"universal stop honoured pre-echo-window: {frame.text!r}",
                            )
                        except Exception:
                            pass
                        return  # consume — never reaches wake-gate
                # 1. Echo gate (time window after BotStoppedSpeaking)
                if self._is_blocking_window():
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="echo-filter",
                            message=(
                                f"dropped transcript during TTS "
                                f"(speaking={self._speaking}, "
                                f"tail={self._speak_end_ts - _time.monotonic():.2f}s)"
                            ),
                        )
                    except Exception:
                        pass
                    return  # drop
                # 2. Bot-utterance fingerprint match (catches late echo
                # that arrived past _TAIL_SEC — room reverb + STT lag).
                if _looks_like_bot_echo(frame.text or ""):
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="echo-filter",
                            message=f"dropped late bot-echo: {frame.text!r}",
                        )
                    except Exception:
                        pass
                    return  # drop
                # 3. Hallucination gate
                if _is_hallucination(frame.text or ""):
                    try:
                        self._stream.emit(
                            "log",
                            level="DEBUG",
                            component="hallucination-filter",
                            message=f"dropped Whisper hallucination: {frame.text!r}",
                        )
                    except Exception:
                        pass
                    return  # drop

            await self.push_frame(frame, direction)

    # Expose for unit testing.
    JarvisEchoFilterProcessor._is_hallucination = staticmethod(_is_hallucination)  # type: ignore[attr-defined]
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
    # R34-S26: VADProcessor is a separate pipeline stage in Pipecat 1.2.x.
    # The legacy ``vad_analyzer=`` kwarg on LocalAudioTransportParams was
    # quietly accepted (Pydantic allows extras) but never wired in — see
    # pipecat/transports/local/audio.py: LocalAudioTransportParams only
    # exposes ``input_device_index`` / ``output_device_index``. Without a
    # VADProcessor, no VADUserStartedSpeakingFrame ever fires, so the
    # downstream STT/aggregator/wake-word stages get audio but no segments.
    from pipecat.processors.audio.vad_processor import VADProcessor
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
        LLMUserAggregatorParams,
    )

    # ── transport (mic + speakers) ───────────────────────────────────
    # R34-S12 (CRITICAL): Pipecat's VADParams default is
    #   confidence=0.7, min_volume=0.6, start_secs=0.2, stop_secs=0.2
    # The ``min_volume`` knob is NORMALISED EBU R128 loudness in [0,1]:
    # 0.0 → −20 LUFS, 1.0 → +80 LUFS. Default 0.6 maps to ~+40 LUFS —
    # louder than a jackhammer and effectively unreachable for any
    # real-world voice mic. Leaving it at default silently rejected
    # 100 % of speech for nine straight hours. We're already gating on
    # Silero confidence, so volume gating is redundant — pin it to 0.0.
    # ``start_secs`` is also lifted from 0.2 → 0.25 to drop transient
    # door-slam / keyboard clatter spikes without delaying real speech.
    vad_params = VADParams(
        confidence=cfg.vad_threshold,
        start_secs=0.25,
        stop_secs=cfg.vad_min_silence_ms / 1000.0,
        min_volume=0.0,
    )
    # R34-S14 startup banner — print once at pipeline build so users
    # can see in jarvis-assistant.err.log which VAD knobs are actually
    # active. Bypasses debug_log's voice-debug gate (always stderr).
    try:
        print(
            f"[voice] VAD active: confidence={cfg.vad_threshold} "
            f"start_secs={vad_params.start_secs} "
            f"stop_secs={vad_params.stop_secs} "
            f"min_volume={vad_params.min_volume} "
            f"lang={cfg.stt_language} "
            f"wake={cfg.wake_words[0]!r}",
            file=__import__("sys").stderr,
            flush=True,
        )
    except Exception:
        pass
    transport_params = LocalAudioTransportParams(
        input_device_index=cfg.input_device_index,
        output_device_index=cfg.output_device_index,
        audio_in_enabled=True,
        audio_out_enabled=True,
        # Sample-rate alignment: Whisper expects 16k; Piper UA model is
        # 22050 Hz. Pipecat resamples internally between processors.
        audio_in_sample_rate=cfg.sample_rate,
    )
    transport = LocalAudioTransport(params=transport_params)

    # R34-S26: explicit VAD stage. In Pipecat 1.2.x, the LocalAudioTransport
    # does NOT run VAD itself — the ``vad_analyzer=`` arg used to be wired
    # into the input callback in older pipecat versions but was removed in
    # the 1.2 transport refactor. We now run Silero VAD as a standalone
    # processor right after the transport, which broadcasts the
    # VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame events the
    # WhisperSTT service and LLM aggregator need to segment user turns.
    # Audio frames continue to pass through unmodified (VADProcessor is
    # pure observer + control-frame emitter).
    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=vad_params),
    )

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
    #
    # R34-S26: pass the ``.value`` (the HF repo string like
    # ``mlx-community/whisper-large-v3-turbo``) NOT the MLXModel enum.
    # mlx_whisper.transcribe() does ``isinstance(model, (str, bytes,
    # os.PathLike))`` and rejects enums with: "expected str, bytes or
    # os.PathLike object, not MLXModel". The error fires after every
    # VAD-stop, killing every transcription attempt silently. The
    # WhisperMLXSTTSettings type hint says ``str | MLXModel | None``
    # but the runtime path doesn't unwrap the enum. Always pass .value.
    #
    # R34-S26 (cont.): force ``language`` so MLX whisper doesn't auto-
    # detect English on every utterance. Without an explicit language,
    # Whisper picked "en" for live transcriptions of Russian/Ukrainian
    # voice and mangled them into English phrases like "I rarely
    # remember…" — the wake-gate then dropped every utterance because
    # "jarvis/джарвіс" never appeared in the English mistranscription.
    stt = WhisperSTTServiceMLX(
        settings=WhisperMLXSTTSettings(
            model=mlx_model.value,
            language=cfg.stt_language or "ru",
        ),
    )

    # ── LLM ─────────────────────────────────────────────────────────
    # OLLamaLLMService inherits from OpenAI — it talks to Ollama's
    # OpenAI-compatible /v1/chat/completions endpoint. Hetzner runs
    # Ollama on the same host:port the legacy listener uses, so the
    # base_url config maps directly.
    #
    # R34-S30 (CRITICAL FIX, third revision):
    #
    #   v1 (failed): tried `extra_body={think:false}` for qwen3's
    #   reasoning-dump — produced a stream Pipecat hung on.
    #   v2 (insufficient): switched to qwen2.5:3b — fixes content but
    #   stream STILL hangs after a few minutes of uptime.
    #   v3 (this fix): the stock Pipecat ``create_client`` uses
    #   ``keepalive_expiry=None`` with ``max_keepalive_connections=
    #   100``. Over Tailscale to Hetzner, NAT mappings expire after
    #   ~3 min idle. Pipecat reuses a stale TCP connection that the
    #   remote already closed — the request hangs until the client's
    #   request_timeout kicks in, which is also unbounded. Symptom:
    #   ``lsof -p <daemon>`` shows ZERO connections to Ollama while
    #   ``state: THINKING`` persists for 60-90 s.
    #
    #   Override ``create_client`` to set a finite keepalive_expiry
    #   (60 s) AND wrap the httpx client with sane connect/read/
    #   write timeouts. This forces stale connections to be dropped
    #   instead of stuck in the pool.
    class _RobustOLLamaLLMService(OLLamaLLMService):  # type: ignore[misc]
        def create_client(self, base_url=None, **kwargs):
            from openai import AsyncOpenAI
            from openai import DefaultAsyncHttpxClient
            import httpx as _httpx
            return AsyncOpenAI(
                api_key=kwargs.get("api_key", "ollama"),
                base_url=base_url or kwargs.get("base_url"),
                http_client=DefaultAsyncHttpxClient(
                    limits=_httpx.Limits(
                        max_keepalive_connections=10,
                        max_connections=20,
                        # Finite expiry — drop idle conns after 60 s
                        # so stale-NAT-mapping reuses never happen.
                        keepalive_expiry=60.0,
                    ),
                    timeout=_httpx.Timeout(
                        # connect: 5s; reading body: 60s; writing: 10s;
                        # acquiring a pool slot: 5s
                        connect=5.0,
                        read=60.0,
                        write=10.0,
                        pool=5.0,
                    ),
                ),
            )

    llm = _RobustOLLamaLLMService(
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
    # R34-S30 (CRITICAL): the default VADUserTurnStartStrategy
    # broadcasts an InterruptionFrame on EVERY VAD speech_started
    # event. With a low-threshold Silero (0.30 to cope with the
    # user's close-mic Russian speech), ambient noise was reliably
    # firing speech_started about every 15-20 s — which CANCELLED
    # the in-flight LLM stream and prevented any TTS audio from
    # ever reaching the speaker. Symptom: Pipecat logs
    # ``Generating chat`` → 17 s of silence → next user_turn_started
    # → no LLMTextFrame, no TTSStartedFrame, no Piper output.
    #
    # Fix: keep VAD as a start-strategy (so we still detect speech
    # for echo-filter + state HUD), but DISABLE its interruption
    # behaviour. The transcription-based start strategy still fires
    # an interruption on real STT output — so the user CAN still
    # interrupt Jarvis mid-sentence by speaking, it just requires a
    # real transcription rather than raw VAD shimmer.
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from pipecat.turns.user_start.vad_user_turn_start_strategy import (
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
        TranscriptionUserTurnStartStrategy,
    )
    # R34-S34: also DISABLE transcription-based interruptions. The
    # user wants Jarvis to be interruptable ONLY when explicitly
    # named ("Джарвіс, стоп" / "Джарвіс, кажи замість цього …").
    # With this disabled, the wake_gate processor handles barge-in
    # explicitly: when SPEAKING + wake-word detected in transcript,
    # it kills the current audio playback and queues an
    # InterruptionFrame. Random talking nearby / background noise
    # CANNOT cancel Jarvis mid-reply any more.
    _user_params = LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            start=[
                VADUserTurnStartStrategy(enable_interruptions=False),
                TranscriptionUserTurnStartStrategy(enable_interruptions=False),
            ],
        ),
    )
    aggregators = LLMContextAggregatorPair(context, user_params=_user_params)

    # ── Stage-3 HUD adapters + Stage-5 gates ───────────────────────
    # We instantiate the processor CLASSES lazily here because the
    # factories import pipecat — if pipecat isn't installed we want
    # ``from_settings`` / config helpers to still work for callers
    # that just want to introspect the loop without booting it.
    EventStreamProc, AudioProbeProc = _make_event_stream_processor()
    StateProc = _make_state_processor()
    FastPathProc = _make_fast_path_processor(cfg)
    WakeWordProc = _make_wake_word_processor(cfg)
    EchoFilterProc = _make_echo_filter_processor()
    audio_probe = AudioProbeProc()         # R34-S26: input-audio RMS heartbeat
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
    # R34-S30: Direct chat bypass. The stock Pipecat LLM→TTS chain
    # gets wedged after a few minutes of uptime — `_RobustOLLamaLLMService`
    # streams the response from Ollama but the LLMAssistantAggregator
    # / TTS pair never emits a TTSStartedFrame, so the user hears
    # silence. Symptom: `state: THINKING` → `state: LISTENING` with
    # no `SPEAKING` in between, no `bot_started_speaking` event.
    #
    # `direct_chat` sits BETWEEN wake_gate and the LLM/TTS chain. On
    # every TranscriptionFrame that survives the wake gate, it:
    #   (1) calls native Ollama `/api/chat` over aiohttp directly
    #       (think:false, options:{num_predict, num_ctx}),
    #   (2) speaks the reply via macOS `say` in a thread,
    #   (3) emits state events so the dashboard reflects SPEAKING,
    #   (4) consumes the frame so the broken Pipecat LLM chain
    #       never sees it.
    direct_chat = _make_direct_chat_processor(cfg)()

    pipeline = Pipeline(
        [
            transport.input(),
            audio_probe,   # R34-S26: emit pipecat_audio_rms ~1 Hz
            vad_processor, # R34-S26: Silero VAD → VADUserStarted/Stopped
            stt,
            events_user,
            echo_filter,
            wake_gate,
            fast_path,
            direct_chat,   # R34-S30: bypass Pipecat LLM/TTS — direct say+ollama
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
        # R34-S42 К4 — pass the SLIM prompt (same as direct_chat uses)
        # plus the language lock & clarify glue, so the warmup eval
        # primes the exact KV-cache prefix the first real turn will
        # reuse. Otherwise the warmup populates one prefix and the
        # first user turn pays a full re-eval. Build the same prompt
        # the JarvisDirectChatProcessor builds at line ~1380.
        try:
            from ..memory.self_learning import language_lock_block
            lock_ = language_lock_block(cfg.active_language)
        except Exception:
            lock_ = ""
        # R34-S52 C-8: was UA-conditional. With R34-S48/S51 RU-only
        # policy the UA branch poisoned the warmup KV-cache prefix when
        # ``active_language == "uk"`` — the first real turn then built
        # a Russian prefix, so the cache reuse missed entirely, hurting
        # latency AND tainting context with UA tokens. Russian-only now.
        clarify_ = (
            "Если запрос неясный, нечёткий или похож на бессмыслицу — "
            "переспроси одним коротким предложением, НЕ выдумывай ответ."
        )
        try:
            slim_prompt_ = _system_prompt_for(cfg.active_language, slim=True)
        except Exception:
            slim_prompt_ = system_prompt
        warmup_sys = "\n\n".join(p for p in (slim_prompt_, clarify_, lock_) if p)
        warmup_voice_stack(
            ollama_base_url=cfg.ollama_base_url,
            chat_model=cfg.chat_model,
            system_prompt=warmup_sys,
            whisper_mlx_repo=whisper_repo,
            piper_voice_id=cfg.piper_voice_id,
            piper_dir=piper_dir,
        )
    except Exception as _warm_exc:
        debug_log(f"warmup kick-off failed: {_warm_exc!r}", "pipecat")

    # R34-S1: Pipecat 1.2.1 moved ``allow_interruptions`` +
    # ``idle_timeout_secs`` off PipelineParams onto PipelineTask
    # itself. Previously we passed them to PipelineParams where
    # they were silently DROPPED — so the default 300s idle-cancel
    # was kicking in after 5 minutes of silence and killing the
    # voice loop until the next daemon restart.
    #
    # Critical fix: ``cancel_on_idle_timeout=False`` — Jarvis is a
    # long-lived background agent, not a session-bounded call.
    # We NEVER want the pipeline to suicide on silence.
    task_params = PipelineParams(
        enable_metrics=False,
    )
    task = PipelineTask(
        pipeline,
        params=task_params,
        # Allow user interruptions of TTS playback (barge-in).
        # In 1.2.1 this is the PipelineTask constructor arg.
        cancel_on_idle_timeout=False,
        # Even though we disabled the cancel, leave the timeout at
        # a long value so any future telemetry / heartbeat that
        # reads ``idle_timeout_secs`` still gets a sane number.
        idle_timeout_secs=24 * 3600,
    )

    return pipeline, context, task, transport


# ─────────────────── interrupt-flag bridge (R34-S29) ───────────────────
async def _interrupt_flag_poller(task: Any) -> None:
    """Poll for an external interrupt request and forward it to Pipecat.

    The dashboard's ``POST /api/interrupt`` writes a flag file at
    ``~/Library/Application Support/jarvis/interrupt.flag``. This
    coroutine, scheduled on the same loop as the pipeline runner,
    notices the file and queues an ``InterruptionFrame`` into the
    active task — which causes Pipecat to cancel in-flight LLM
    streaming + cut off TTS playback cleanly. The flag is removed
    so the next request is a fresh event.

    Polls at 250 ms — fast enough that the user perceives the Stop
    button as instant (≤ 1 frame of TTS audio after click), light
    enough that 4 stat-calls/s is dwarfed by the rest of the loop.
    """
    from pathlib import Path as _P
    base = _P.home() / "Library/Application Support/jarvis"
    flag = base / "interrupt.flag"
    # R34-S30: ALSO poll a transcription-injection flag so the
    # dashboard / tests can drive the LLM+TTS path without going
    # through the microphone (used for end-to-end smoke tests
    # when the audio device is busy or the user is debugging).
    inject_flag = base / "inject_transcription.flag"
    # R34-S45 — control.json carries persistent action verbs from the
    # HUD (Close button writes {"action":"end_session"}). The legacy
    # listener.py path consumes this file; the Pipecat path didn't,
    # so end_session only cancelled the current turn and the daemon
    # would speak again on the next surviving transcript. We now read
    # the same file, set the standby flag, and unlink so re-entry
    # writes are honoured. ``_seen_ctl_ts`` deduplicates against the
    # millisecond-stamp HUD puts in the JSON so we don't loop on
    # the same end_session multiple times.
    ctrl_file = base / "control.json"
    _seen_ctl_ts: float = 0.0
    try:
        # Use late import so a missing pipecat install doesn't break
        # the whole module — _amain only runs when pipecat is loaded.
        from pipecat.frames.frames import (
            InterruptionFrame,
            TranscriptionFrame,
        )
    except Exception:
        debug_log(
            "interrupt poller: InterruptionFrame import failed — "
            "stop button will not work",
            "pipecat",
        )
        return
    while True:
        try:
            # R34-S45 — control.json end_session goes to persistent
            # standby. Check FIRST so that the same poll tick also
            # cancels in-flight TTS via the flag-block below (Close
            # button writes both atomically).
            if ctrl_file.exists():
                try:
                    import json as _json_ctl
                    raw = ctrl_file.read_text(encoding="utf-8")
                    ctl = _json_ctl.loads(raw) if raw.strip() else {}
                    action_ = str(ctl.get("action") or "").lower()
                    ts_ = float(ctl.get("ts") or 0)
                    if ts_ > _seen_ctl_ts and action_ in (
                        "end_session", "stop_session", "session_end",
                        "standby", "close",
                    ):
                        _seen_ctl_ts = ts_
                        _set_standby(True)
                        debug_log(
                            f"control.json: {action_} → STANDBY engaged "
                            "(say the wake word to resume)",
                            "pipecat",
                        )
                        # Kill any in-flight playback right now.
                        try:
                            _stop_current_playback()
                        except Exception:
                            pass
                        try:
                            await task.queue_frame(InterruptionFrame())
                        except Exception:
                            pass
                        # Flip HUD coin to IDLE so the user sees the
                        # daemon is truly silent (not just LISTENING
                        # while waiting). State auto-resets to
                        # LISTENING the next time a transcript flows
                        # through, which only happens after wake-word
                        # clears standby.
                        try:
                            _write_hud_state_atomic("IDLE", level=0.0)
                        except Exception:
                            pass
                except Exception as exc:
                    debug_log(
                        f"control.json poll error: {exc!r}",
                        "pipecat",
                    )
            if flag.exists():
                debug_log("interrupt flag detected → cancelling turn", "pipecat")
                # R34-S34: ALSO kill any in-flight afplay so the user
                # hears immediate silence. Without this, the user
                # presses Stop and waits 3-5 s for the WAV buffer to
                # drain (afplay can't be SIGINT'd cleanly otherwise).
                try:
                    _stop_current_playback()
                except Exception:
                    pass
                try:
                    await task.queue_frame(InterruptionFrame())
                except Exception as exc:
                    debug_log(f"interrupt queue failed: {exc!r}", "pipecat")
                try:
                    flag.unlink()
                except Exception:
                    pass
            if inject_flag.exists():
                try:
                    text = inject_flag.read_text(encoding="utf-8").strip()
                    if text:
                        debug_log(
                            f"inject_transcription: queueing {text[:60]!r}",
                            "pipecat",
                        )
                        from time import time as _t
                        await task.queue_frame(
                            TranscriptionFrame(
                                text=text,
                                user_id="dashboard",
                                timestamp=str(_t()),
                            )
                        )
                except Exception as exc:
                    debug_log(
                        f"inject_transcription error: {exc!r}",
                        "pipecat",
                    )
                try:
                    inject_flag.unlink()
                except Exception:
                    pass
        except asyncio.CancelledError:
            return
        except Exception as exc:  # pragma: no cover — observability
            debug_log(f"poller error: {exc!r}", "pipecat")
        await asyncio.sleep(0.25)


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
            # R34-S39 — startup hygiene: clear orphan TTS files left by
            # prior crashed daemons. Cheap, runs once per boot.
            try:
                n = _cleanup_orphan_tts_wavs()
                if n > 0:
                    debug_log(
                        f"startup: cleared {n} orphan /tmp/jarvis_tts_*.wav files",
                        "pipecat",
                    )
            except Exception as exc:
                debug_log(f"startup wav-sweep failed: {exc!r}", "pipecat")
            # Also register FactStore atexit close so SQLite checkpoints
            # cleanly on SIGTERM (otherwise WAL can grow unboundedly
            # across kill-cycles).
            try:
                import atexit as _atexit
                from ..memory.facts import get_facts_store as _gfs
                def _close_facts():
                    try:
                        _gfs().close()
                    except Exception:
                        pass
                _atexit.register(_close_facts)
            except Exception:
                pass

            _, _, task, _ = _build_pipeline(self.cfg)
            self._task = task
            runner = PipelineRunner(handle_sigint=False)
            self._runner = runner
            debug_log("PipecatLoop starting PipelineRunner", "pipecat")
            # R34-S49 REC 4 — start the cron-style background scheduler.
            # Tasks (events.jsonl rotate, audit.db vacuum, briefs prune,
            # Brain inventory, Ollama keepwarm) run in their own daemon
            # thread, so a slow task can never block the voice pipeline.
            # See loop_workers.py docstring for the registry.
            try:
                from . import loop_workers as _lw
                _lw.start()
            except Exception as exc:
                debug_log(f"loop_workers: start failed: {exc!r}", "pipecat")
            # R34-S29: poll the dashboard's interrupt flag in a side
            # task so the user can stop TTS / abort the current turn
            # without waiting for VAD to detect their voice.
            interrupt_task = asyncio.create_task(
                _interrupt_flag_poller(task)
            )
            try:
                await runner.run(task)
            except asyncio.CancelledError:
                debug_log("PipecatLoop runner cancelled", "pipecat")
            except Exception as exc:  # pragma: no cover — surface to logs
                debug_log(f"PipecatLoop runner crashed: {exc!r}", "pipecat")
                raise
            finally:
                interrupt_task.cancel()
                # R34-S39 — await the interrupt task so its file handle
                # on interrupt.flag and inject_transcription.flag closes
                # cleanly. Without this the .stat()/.unlink() ops can
                # race a fresh daemon's startup.
                try:
                    await asyncio.gather(
                        interrupt_task, return_exceptions=True,
                    )
                except Exception:
                    pass
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

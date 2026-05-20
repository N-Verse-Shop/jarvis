"""R34-S11 — verify legacy config knobs reach Pipecat properly.

The user has been tuning ``vad_aggressiveness``, ``voice_min_energy``,
``endpoint_silence_ms`` and ``whisper_language`` in ``config.json`` for
months. Earlier versions of ``from_settings`` ignored those keys and
left Pipecat on its 0.5 Silero default → in any non-silent room, the
VAD reported ``speaking=True`` continuously → endpoint never fired →
Whisper never ran → wake word never landed.

These tests pin down the bridge so a future refactor can't silently
regress it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.listening.pipecat_loop import (
    PipecatLoopConfig,
    _vad_threshold_from_legacy,
    from_settings,
)


def _make_cfg(**overrides):
    """Build a minimal ``Settings``-shaped object for ``from_settings``."""
    base = {
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_chat_model": "qwen3:8b",
        "whisper_model": "LARGE_V3_TURBO",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ─── threshold mapping ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "aggr, expected",
    [
        # R34-S14 recalibrated mapping (was 0.45/0.55/0.62/0.72 before
        # live mic probe showed real speech peaks at ~0.73, so 0.62
        # caught only 1 % of frames).
        (0, 0.30),
        (1, 0.35),
        (2, 0.42),
        (3, 0.55),
        # Out-of-range / wrong type → safe default (0.42).
        (None, 0.42),
        ("garbage", 0.42),
        (-1, 0.42),
        (99, 0.42),
    ],
)
def test_vad_threshold_from_aggressiveness(aggr, expected):
    assert _vad_threshold_from_legacy(aggr, None) == pytest.approx(expected)


def test_explicit_silero_threshold_overrides_aggressiveness():
    # Explicit config beats the aggressiveness table.
    assert _vad_threshold_from_legacy(2, 0.81) == pytest.approx(0.81)


def test_explicit_silero_threshold_is_clamped():
    # Out-of-range values clamp to [0.10, 0.95] (Silero is undefined
    # outside that range and 1.0 would silence the model entirely).
    assert _vad_threshold_from_legacy(2, 0.0) == pytest.approx(0.10)
    assert _vad_threshold_from_legacy(2, 1.5) == pytest.approx(0.95)


def test_explicit_threshold_bad_value_falls_back():
    # If the explicit knob is unparseable, fall back to the
    # aggressiveness table instead of crashing.
    assert _vad_threshold_from_legacy(2, "not a float") == pytest.approx(0.42)


# ─── full from_settings() bridge ─────────────────────────────────────


def test_from_settings_translates_aggressiveness():
    loop = from_settings(_make_cfg(vad_aggressiveness=3))
    assert loop.vad_threshold == pytest.approx(0.55)


def test_from_settings_respects_explicit_silero():
    loop = from_settings(
        _make_cfg(vad_aggressiveness=2, silero_vad_threshold=0.83)
    )
    assert loop.vad_threshold == pytest.approx(0.83)


def test_from_settings_passes_endpoint_silence_ms():
    loop = from_settings(_make_cfg(endpoint_silence_ms=900))
    assert loop.vad_min_silence_ms == 900


def test_from_settings_clamps_endpoint_silence():
    # Insanely short → floor at 150ms (Silero needs SOME silence to
    # call endpoint).
    loop_short = from_settings(_make_cfg(endpoint_silence_ms=5))
    assert loop_short.vad_min_silence_ms == 150
    # Insanely long → cap at 2s.
    loop_long = from_settings(_make_cfg(endpoint_silence_ms=10_000))
    assert loop_long.vad_min_silence_ms == 2000


def test_from_settings_falls_back_to_whisper_language():
    # No active_language set → use whisper_language (legacy listener key).
    loop = from_settings(_make_cfg(whisper_language="ru"))
    assert loop.stt_language == "ru"
    assert loop.active_language == "ru"


def test_from_settings_prefers_active_language():
    loop = from_settings(
        _make_cfg(active_language="uk", whisper_language="ru")
    )
    assert loop.stt_language == "uk"
    assert loop.active_language == "uk"


def test_from_settings_final_default_is_ru():
    # Neither field present → "ru" (May-16+ project default).
    loop = from_settings(_make_cfg())
    assert loop.stt_language == "ru"


def test_extras_carry_diagnostic_floors():
    # Legacy voice_min_energy + max_utterance_ms surface in extras so
    # downstream diagnostic emitters can correlate ambient noise floor
    # with Silero output.
    loop = from_settings(
        _make_cfg(voice_min_energy=0.0030, max_utterance_ms=8000)
    )
    assert loop.extra["voice_min_energy"] == pytest.approx(0.0030)
    assert loop.extra["max_utterance_ms"] == 8000


def test_extras_use_defaults_on_missing():
    loop = from_settings(_make_cfg())
    assert loop.extra["voice_min_energy"] == pytest.approx(0.0025)
    assert loop.extra["max_utterance_ms"] == 7000


def test_pipecat_loop_config_default_threshold_is_realistic():
    # R34-S14: default must be reachable by real speech on a typical
    # consumer mic. Live probe on MBP M1 showed real-speech peaks at
    # ~0.73 → default must be well under that. 0.42 catches the meaty
    # middle of normal utterances.
    cfg = PipecatLoopConfig()
    assert cfg.vad_threshold >= 0.35
    assert cfg.vad_threshold <= 0.50


# ─── Whisper hallucination filter (R34-S14) ──────────────────────────


def _get_hallucination_fn():
    """Pull the inner ``_is_hallucination`` out of the factory so we
    can test it without standing up Pipecat. The factory exposes it as
    a staticmethod on the produced class for exactly this purpose.
    """
    # We avoid importing pipecat at module top so this test module
    # stays loadable on CI runners without ML extras.
    from jarvis.listening.pipecat_loop import _make_echo_filter_processor

    cls = _make_echo_filter_processor()
    return cls._is_hallucination


@pytest.mark.parametrize(
    "text",
    [
        # English Whisper tail
        "Thank you for watching",
        "thanks for watching",
        "Thank you.",
        "Thanks!",
        "what time is it",
        "Will it rain today?",
        "is a cool name",
        "[Music]",
        "(Applause)",
        "Bye",
        "okay.",
        "Um",
        # Ukrainian
        "Дякую за перегляд",
        "Дякую!",
        "Підпишіться на канал",
        # Russian
        "Спасибо за просмотр",
        "Подписывайтесь на канал!",
        # Punctuation-only
        ".",
        "—",
        " ... ",
        # Repetition artifact
        "ого ого ого ого",
        "yes yes yes",
    ],
)
def test_hallucination_phrases_are_dropped(text):
    is_hallucination = _get_hallucination_fn()
    assert is_hallucination(text) is True, f"should drop {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        # Real wake-word + command — never confuse for hallucination.
        "Джарвіс відкрий браузер",
        "Jarvis what time is it actually",  # contains hallucinated phrase
        "thanks for watching the demo with me",  # contains "thanks for watching"
        "Привіт як справи",
        "ого як круто це працює",  # 4 words but not all identical
        "тест тест",  # 2-word repetition is too short to flag
        "open Safari please",
    ],
)
def test_real_commands_are_preserved(text):
    is_hallucination = _get_hallucination_fn()
    assert is_hallucination(text) is False, f"must NOT drop {text!r}"


def test_empty_text_is_dropped():
    is_hallucination = _get_hallucination_fn()
    assert is_hallucination("") is True
    assert is_hallucination("   ") is True


def test_hallucination_matching_is_case_insensitive():
    is_hallucination = _get_hallucination_fn()
    assert is_hallucination("THANK YOU FOR WATCHING") is True
    assert is_hallucination("Thank You For Watching") is True
    assert is_hallucination("дякую за перегляд") is True
    assert is_hallucination("ДЯКУЮ ЗА ПЕРЕГЛЯД") is True


# ─── input device resolver (R34-S16) ─────────────────────────────────


def test_resolve_input_device_index_passthrough():
    """Explicit integer index passes through unchanged."""
    from jarvis.listening.pipecat_loop import _resolve_input_device_index
    assert _resolve_input_device_index(3, "MacBook") == 3
    assert _resolve_input_device_index(0, None) == 0


def test_resolve_input_device_index_no_input():
    """Both args None / empty → returns None (let PortAudio pick)."""
    from jarvis.listening.pipecat_loop import _resolve_input_device_index
    assert _resolve_input_device_index(None, None) is None
    assert _resolve_input_device_index(None, "") is None
    assert _resolve_input_device_index(None, "   ") is None


def test_resolve_input_device_index_bad_integer_falls_back_to_name():
    """A garbage explicit-index → fall through to name match."""
    from jarvis.listening.pipecat_loop import _resolve_input_device_index
    # "garbage" can't be int-cast; with no name, returns None.
    assert _resolve_input_device_index("garbage", None) is None


def test_resolve_input_device_index_uses_pyaudio_when_named(monkeypatch):
    """When a name substring is given, scan PyAudio devices."""
    import jarvis.listening.pipecat_loop as loop_mod

    class _FakePA:
        def __init__(self):
            self._closed = False

        def get_device_count(self):
            return 3

        def get_default_input_device_info(self):
            return {"index": 1}

        def get_device_info_by_index(self, i):
            return [
                {"name": "iPhone Continuity", "maxInputChannels": 1},
                {"name": "MacBook Pro Microphone", "maxInputChannels": 1},
                {"name": "Speakers (output)", "maxInputChannels": 0},
            ][i]

        def terminate(self):
            self._closed = True

    class _FakeModule:
        PyAudio = _FakePA

    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **kw: _FakeModule() if name == "pyaudio" else __builtins__["__import__"](name, *a, **kw)
        if hasattr(__builtins__, "__import__") else __import__(name, *a, **kw),
    )
    # Substring match → returns the MacBook index (1).
    assert loop_mod._resolve_input_device_index(None, "MacBook") == 1
    # Match against iPhone instead.
    assert loop_mod._resolve_input_device_index(None, "iPhone") == 0


def test_resolve_input_device_index_no_match_returns_none(monkeypatch):
    """Substring with no matching device → None (PortAudio default)."""
    import jarvis.listening.pipecat_loop as loop_mod

    class _FakePA:
        def get_device_count(self): return 1
        def get_default_input_device_info(self): return {"index": 0}
        def get_device_info_by_index(self, i):
            return {"name": "Some Other Mic", "maxInputChannels": 1}
        def terminate(self): pass

    class _FakeModule:
        PyAudio = _FakePA

    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *a, **kw: _FakeModule() if name == "pyaudio" else __import__(name, *a, **kw),
    )
    assert loop_mod._resolve_input_device_index(None, "NonExistent") is None


def test_resolve_input_device_index_swallows_pyaudio_errors(monkeypatch):
    """Probe failure → None, no exception leaks to daemon boot."""
    import jarvis.listening.pipecat_loop as loop_mod

    def _bad_import(name, *a, **kw):
        if name == "pyaudio":
            raise RuntimeError("PortAudio init crashed")
        return __import__(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _bad_import)
    # Must not raise.
    assert loop_mod._resolve_input_device_index(None, "MacBook") is None

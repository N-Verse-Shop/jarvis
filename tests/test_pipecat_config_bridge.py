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
        (0, 0.45),
        (1, 0.55),
        (2, 0.62),
        (3, 0.72),
        # Out-of-range / wrong type → safe default (0.62).
        (None, 0.62),
        ("garbage", 0.62),
        (-1, 0.62),
        (99, 0.62),
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
    assert _vad_threshold_from_legacy(2, "not a float") == pytest.approx(0.62)


# ─── full from_settings() bridge ─────────────────────────────────────


def test_from_settings_translates_aggressiveness():
    loop = from_settings(_make_cfg(vad_aggressiveness=3))
    assert loop.vad_threshold == pytest.approx(0.72)


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


def test_pipecat_loop_config_default_threshold_is_aggressive_enough():
    # Sanity check: PipecatLoopConfig() with NO settings (no config.json
    # at all) still uses 0.62 — not the old Pipecat default of 0.5,
    # which produced the 9-hour silent-pipeline regression.
    cfg = PipecatLoopConfig()
    assert cfg.vad_threshold >= 0.6
    assert cfg.vad_threshold <= 0.7

"""Tests for jarvis.listening.warmup — best-effort warmup paths.

The warmup module is the SAFEST module in the codebase: every
external dependency is allowed to be missing, every call is
allowed to fail, none of it can block. The tests verify exactly
those invariants — not that warmup actually warmed.
"""

from __future__ import annotations

import threading
import time

from jarvis.listening.warmup import (
    _warmup_ollama,
    _warmup_piper,
    _warmup_whisper_mlx,
    warmup_voice_stack,
)


class TestWarmupFailureSafety:
    def test_ollama_unreachable_does_not_raise(self):
        # 1.0.0.0 is reserved and won't accept connections quickly.
        _warmup_ollama(
            base_url="http://1.0.0.0:9999",
            model="phantom",
            system_prompt="ping",
            timeout_s=0.5,
        )

    def test_whisper_missing_repo_no_op(self):
        _warmup_whisper_mlx(None)

    def test_piper_missing_voice_no_op(self, tmp_path):
        _warmup_piper("nonexistent-voice", piper_dir=tmp_path)


class TestWarmupVoiceStack:
    def test_kickoff_returns_immediately(self):
        # Kicking off warmup should NEVER block the caller. We measure
        # the call duration and assert it's well under any plausible
        # network timeout.
        t0 = time.monotonic()
        warmup_voice_stack(
            ollama_base_url="http://1.0.0.0:9999",
            chat_model="phantom",
            system_prompt="x",
            whisper_mlx_repo=None,
            piper_voice_id=None,
        )
        dt = time.monotonic() - t0
        assert dt < 0.2, (
            f"warmup_voice_stack blocked for {dt:.3f}s — should be non-blocking"
        )

    def test_threads_are_daemons(self):
        # Snapshot threads before/after; any new warmup-* thread must
        # be daemon. Some warmups may finish before the snapshot (the
        # ollama one fails fast on an unreachable address), so we
        # patch the worker functions with a small sleep to guarantee
        # they're alive when we check.
        import jarvis.listening.warmup as warmup_mod

        long_held: list[threading.Thread] = []

        def _slow(*a, **kw):
            time.sleep(0.5)

        original = (
            warmup_mod._warmup_ollama,
            warmup_mod._warmup_whisper_mlx,
            warmup_mod._warmup_piper,
        )
        warmup_mod._warmup_ollama = _slow
        warmup_mod._warmup_whisper_mlx = _slow
        warmup_mod._warmup_piper = _slow
        try:
            before = {t.name for t in threading.enumerate()}
            warmup_voice_stack(
                ollama_base_url="http://x",
                chat_model="phantom",
                system_prompt="x",
                whisper_mlx_repo="x",
                piper_voice_id="x",
            )
            time.sleep(0.05)
            warm = [
                t for t in threading.enumerate()
                if t.name not in before and t.name.startswith("warmup-")
            ]
            assert warm, "no warmup threads spawned"
            for t in warm:
                assert t.daemon is True, f"{t.name} is not a daemon thread"
        finally:
            (
                warmup_mod._warmup_ollama,
                warmup_mod._warmup_whisper_mlx,
                warmup_mod._warmup_piper,
            ) = original

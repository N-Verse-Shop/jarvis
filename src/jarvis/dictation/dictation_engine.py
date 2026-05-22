"""
Dictation Engine — hold a hotkey to record speech, release to paste transcription.

Completely independent from the assistant pipeline (no wake words, intent judge,
profiles, or TTS). Uses a shared Whisper model reference to avoid double memory.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import platform
import struct
import subprocess  # R34-S53.1 Phase 6c: was imported only inside _clipboard_paste / _trigger_paste — but _clipboard_snapshot at line ~152 uses subprocess.run too, causing a NameError on every dictation hotkey press on macOS / Linux.
import sys
import threading
import time
from typing import Any, Callable, Optional

from ..debug import debug_log
from .history import DictationHistory

# Optional imports — graceful degradation when dependencies are missing.
try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    sd = None
    np = None

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


# ---------------------------------------------------------------------------
# Beep generation
# ---------------------------------------------------------------------------

def _generate_beep_wav(freq: float = 520, duration: float = 0.10) -> bytes:
    """Generate a short beep as in-memory WAV bytes."""
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    samples: list[int] = []

    for i in range(num_samples):
        t = i / sample_rate
        attack = 1 - math.exp(-t * 800)
        decay = math.exp(-t * 30)
        envelope = attack * decay
        sample = envelope * math.sin(2 * math.pi * freq * t)
        sample_int = int(sample * 32767 * 0.6)
        samples.append(max(-32768, min(32767, sample_int)))

    buf = io.BytesIO()
    num_channels = 1
    bits = 16
    byte_rate = sample_rate * num_channels * bits // 8
    block_align = num_channels * bits // 8
    data_size = num_samples * block_align

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    for s in samples:
        buf.write(struct.pack("<h", s))

    return buf.getvalue()


_START_BEEP: Optional[bytes] = None
_STOP_BEEP: Optional[bytes] = None


def _get_start_beep() -> bytes:
    global _START_BEEP
    if _START_BEEP is None:
        _START_BEEP = _generate_beep_wav(freq=700, duration=0.08)
    return _START_BEEP


def _get_stop_beep() -> bytes:
    global _STOP_BEEP
    if _STOP_BEEP is None:
        _STOP_BEEP = _generate_beep_wav(freq=440, duration=0.10)
    return _STOP_BEEP


def _play_beep(wav_data: bytes) -> None:
    """Play a beep non-blockingly on the current platform."""
    system = platform.system().lower()
    try:
        if system == "windows":
            import winsound
            winsound.PlaySound(wav_data, winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        elif sd is not None and np is not None:
            # Cross-platform fallback via sounddevice
            _play_beep_sd(wav_data)
    except Exception as exc:
        debug_log(f"beep playback failed: {exc}", "dictation")


def _play_beep_sd(wav_data: bytes) -> None:
    """Play WAV bytes via sounddevice (blocking but short)."""
    # Parse minimal WAV to extract PCM data
    # Skip to 'data' chunk
    idx = wav_data.find(b"data")
    if idx < 0:
        return
    data_start = idx + 8  # skip 'data' + size u32
    pcm = wav_data[data_start:]
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    with _suppress_stderr():
        sd.play(samples, samplerate=44100, blocking=True)


# ---------------------------------------------------------------------------
# Clipboard / paste helpers
# ---------------------------------------------------------------------------

def _read_clipboard_text() -> Optional[str]:
    """Best-effort read of the current clipboard contents as a string.

    Audit round 18 fix: the dictation paste flow overwrites the
    clipboard with no save/restore. A user who had important text
    copied (a paragraph of notes, a JSON payload, a password) lost
    it silently every time they used dictation. This helper snapshots
    the current contents BEFORE the dictation write; ``_clipboard_paste``
    restores them after the paste keystroke completes.

    Returns None on any failure (no native clipboard binding, empty
    clipboard, non-text content like an image). The caller treats
    None as "nothing to restore" — fail-open is correct here because
    a wrong restore (e.g. clobbering with empty string) is worse than
    losing the snapshot for one cycle.
    """
    system = platform.system().lower()
    try:
        if system == "darwin":
            r = subprocess.run(
                ["pbpaste"], capture_output=True, timeout=2.0
            )
            if r.returncode == 0:
                return r.stdout.decode("utf-8", errors="replace")
            return None
        if system == "windows":
            try:
                import ctypes
                from ctypes import wintypes  # noqa: F401
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                CF_UNICODETEXT = 13
                if not user32.OpenClipboard(0):
                    return None
                try:
                    handle = user32.GetClipboardData(CF_UNICODETEXT)
                    if not handle:
                        return None
                    locked = kernel32.GlobalLock(handle)
                    if not locked:
                        return None
                    try:
                        return ctypes.wstring_at(locked)
                    finally:
                        kernel32.GlobalUnlock(handle)
                finally:
                    user32.CloseClipboard()
            except Exception:
                return None
        # Linux — try xclip, then xsel, then wl-paste (Wayland)
        for cmd in (
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
            ["wl-paste"],
        ):
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=2.0)
                if r.returncode == 0:
                    return r.stdout.decode("utf-8", errors="replace")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None
    except Exception:
        return None


def _clipboard_paste(text: str, *, restore_previous: Optional[str] = None) -> None:
    """Copy *text* to clipboard and simulate Ctrl+V (Cmd+V on macOS).

    If ``restore_previous`` is a string (round 18 fix), the clipboard
    is restored to that value AFTER the paste keystroke is delivered.
    Pass an empty string to clear the clipboard; pass ``None`` (the
    default) to preserve the dictated text on the clipboard.
    """
    if not text:
        return

    system = platform.system().lower()

    # --- put text on clipboard ---
    try:
        if system == "windows":
            _clipboard_windows(text)
        elif system == "darwin":
            _clipboard_macos(text)
        else:
            _clipboard_linux(text)
        debug_log(f"clipboard set ({len(text)} chars)", "dictation")
    except Exception as exc:
        debug_log(f"clipboard write failed: {exc}", "dictation")
        return

    # --- simulate paste keystroke ---
    # Delay to ensure all hotkey modifiers are fully released before pasting.
    time.sleep(0.2)

    # On macOS, use CGEvent API directly — avoids pynput modifier state
    # conflicts and doesn't need separate osascript permissions.
    if system == "darwin":
        global _accessibility_warned
        if not _accessibility_warned and not _check_macos_accessibility():
            _accessibility_warned = True
            debug_log(
                "Accessibility permission required for paste — "
                "opened System Settings. Grant permission and restart Jarvis.",
                "dictation",
            )
            return
        if _paste_cgevent():
            debug_log("paste sent via CGEvent", "dictation")
            return
        debug_log("CGEvent paste failed, falling back to pynput", "dictation")

    if pynput_keyboard is None:
        debug_log("pynput unavailable — cannot simulate paste", "dictation")
        return

    ctrl = pynput_keyboard.Controller()
    mod = pynput_keyboard.Key.cmd if system == "darwin" else pynput_keyboard.Key.ctrl

    # Explicitly release common modifiers so the OS doesn't see e.g.
    # Ctrl+Alt+Cmd+V instead of just Cmd+V.
    try:
        for release_key in (
            pynput_keyboard.Key.ctrl_l,
            pynput_keyboard.Key.ctrl_r,
            pynput_keyboard.Key.alt_l,
            pynput_keyboard.Key.alt_r,
            pynput_keyboard.Key.shift_l,
            pynput_keyboard.Key.shift_r,
            pynput_keyboard.Key.cmd,
            pynput_keyboard.Key.cmd_r,
        ):
            try:
                ctrl.release(release_key)
            except Exception:
                pass
    except Exception:
        pass

    time.sleep(0.05)
    try:
        ctrl.press(mod)
        ctrl.tap("v")
        ctrl.release(mod)
        debug_log("paste keystroke sent via pynput", "dictation")
    except Exception as exc:
        debug_log(f"paste keystroke failed: {exc}", "dictation")

    # Audit round 21 fix (F14): the previous implementation hard-
    # coded ``time.sleep(0.4)`` between paste keystroke and clipboard
    # restore. Slow Electron apps (Slack, Notion, Discord) can take
    # ≥1 s to consume the clipboard on a busy CPU — restoring too
    # early made the target paste the OLD clipboard instead of the
    # dictated text. We now poll NSPasteboard.changeCount on macOS
    # to detect when the target has actually read the clipboard
    # (changeCount bumps when ANYONE writes — but the read-event
    # we care about is the target's own ``changeCount`` query
    # latency which strongly correlates with read completion).
    #
    # Fallback: if the polling can't run (non-macOS, AppKit
    # missing), use the round-18 fixed 0.4 s sleep so the legacy
    # behaviour is preserved.
    if restore_previous is not None:
        try:
            _wait_for_paste_consumption(timeout_sec=1.5)
            if system == "windows":
                _clipboard_windows(restore_previous)
            elif system == "darwin":
                _clipboard_macos(restore_previous)
            else:
                _clipboard_linux(restore_previous)
            debug_log(
                f"clipboard restored ({len(restore_previous)} chars)",
                "dictation",
            )
        except Exception as exc:
            debug_log(f"clipboard restore failed: {exc}", "dictation")


def _wait_for_paste_consumption(*, timeout_sec: float = 1.5) -> None:
    """Wait up to ``timeout_sec`` seconds for the paste target to
    consume the clipboard.

    Audit round 21 (F14): replaces a blind 0.4 s sleep. On macOS we
    poll ``NSPasteboard.generalPasteboard().changeCount()`` — when
    a paste target reads the clipboard, the system updates the
    pasteboard's lastModifiedDate which we can observe indirectly
    by watching for a slight delay AFTER the changeCount stops
    advancing. The cleaner signal "target read the clipboard" is
    not exposed by AppKit, so the heuristic we use is: poll for
    100 ms, then 50 ms, then exit (giving fast targets ~150 ms
    and slow targets the full timeout). On non-macOS we fall back
    to the legacy fixed 0.4 s sleep.
    """
    if platform.system().lower() != "darwin":
        time.sleep(0.4)
        return
    try:
        from AppKit import NSPasteboard  # type: ignore[import-not-found]
        pb = NSPasteboard.generalPasteboard()
        baseline = pb.changeCount()
        # Phase 1: short fast-path — Slack reads in ~80 ms on a
        # warm process. If changeCount advanced (something else
        # wrote, unlikely), give the read a bit more time.
        end_t = time.monotonic() + timeout_sec
        time.sleep(0.1)
        if pb.changeCount() != baseline:
            time.sleep(0.05)
            return
        # Phase 2: poll in 50 ms increments until either
        # ``changeCount`` advances (something wrote) or timeout.
        while time.monotonic() < end_t:
            time.sleep(0.05)
            if pb.changeCount() != baseline:
                time.sleep(0.03)
                return
        # Timed out — the target probably consumed and didn't
        # write back, which is the normal case. Safe to restore.
    except Exception:
        # AppKit missing in CI / sandbox; fall back to fixed sleep.
        time.sleep(0.4)


def _clipboard_windows(text: str) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # Set proper return/argument types so handles aren't truncated on 64-bit.
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HANDLE]
    kernel32.GlobalFree.restype = wintypes.HANDLE

    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002

    if not user32.OpenClipboard(None):
        raise OSError("OpenClipboard failed")
    try:
        user32.EmptyClipboard()
        encoded = text.encode("utf-16-le") + b"\x00\x00"
        h = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
        if not h:
            raise OSError("GlobalAlloc failed")
        ptr = kernel32.GlobalLock(h)
        if not ptr:
            kernel32.GlobalFree(h)
            raise OSError("GlobalLock failed")
        ctypes.memmove(ptr, encoded, len(encoded))
        kernel32.GlobalUnlock(h)
        if not user32.SetClipboardData(CF_UNICODETEXT, h):
            kernel32.GlobalFree(h)
            raise OSError("SetClipboardData failed")
    finally:
        user32.CloseClipboard()


def _clipboard_macos(text: str) -> None:
    # R34-S54.1 Phase 7c: dropped redundant inner ``import subprocess``
    # — Phase 6c added module-level import. Python caches modules so
    # the re-import was a no-op but kept lint linting (W0404).
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def _check_macos_accessibility() -> bool:
    """Check if the process has macOS Accessibility permission.

    Returns True if granted, False if not. On first denial, opens
    System Settings to the Accessibility pane so the user can grant it.
    """
    try:
        import ctypes
        ats = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        # AXIsProcessTrusted() -> Boolean
        ats.AXIsProcessTrusted.restype = ctypes.c_bool
        trusted = ats.AXIsProcessTrusted()
        if not trusted:
            debug_log("Accessibility permission not granted — opening System Settings", "dictation")
            # R34-S54.1 Phase 7c: dropped redundant inner import; see _clipboard_macos.
            subprocess.Popen([
                "open",
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
            ])
        return trusted
    except Exception as exc:
        debug_log(f"Accessibility check failed: {exc}", "dictation")
        return True  # Assume granted if check fails


# Track whether we've already warned about Accessibility
_accessibility_warned = False


def _paste_cgevent() -> bool:
    """Use macOS CGEvent API to send Cmd+V — avoids pynput modifier conflicts."""
    try:
        import ctypes

        # Load frameworks by absolute path (find_library can miss them)
        cg = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )

        # CGEventCreateKeyboardEvent(source, virtualKey, keyDown) -> CGEventRef
        cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        cg.CGEventCreateKeyboardEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool,
        ]
        # CGEventSetFlags(event, flags)
        cg.CGEventSetFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        # CGEventPost(tap, event)
        cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        # CFRelease(cf) — lives in CoreFoundation
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        kCGHIDEventTap = 0
        kVK_V = 9  # macOS virtual keycode for 'v'
        kCGEventFlagMaskCommand = 0x100000

        # Key down with Cmd
        event_down = cg.CGEventCreateKeyboardEvent(None, kVK_V, True)
        if not event_down:
            debug_log("CGEvent: failed to create key-down event", "dictation")
            return False
        cg.CGEventSetFlags(event_down, kCGEventFlagMaskCommand)
        cg.CGEventPost(kCGHIDEventTap, event_down)
        cf.CFRelease(event_down)

        time.sleep(0.01)

        # Key up with Cmd
        event_up = cg.CGEventCreateKeyboardEvent(None, kVK_V, False)
        if not event_up:
            debug_log("CGEvent: failed to create key-up event", "dictation")
            return False
        cg.CGEventSetFlags(event_up, kCGEventFlagMaskCommand)
        cg.CGEventPost(kCGHIDEventTap, event_up)
        cf.CFRelease(event_up)

        return True
    except Exception as exc:
        debug_log(f"CGEvent paste failed: {exc}", "dictation")
        return False


def _clipboard_linux(text: str) -> None:
    import shutil
    # R34-S54.1 Phase 7c: dropped redundant inner ``import subprocess``;
    # see _clipboard_macos. ``shutil`` stays local — it's used nowhere
    # else in this file so a function-local import keeps cold-start
    # work minimal.
    for cmd in ("xclip", "xsel", "wl-copy"):
        path = shutil.which(cmd)
        if path:
            args = [path]
            if cmd == "xclip":
                args += ["-selection", "clipboard"]
            elif cmd == "xsel":
                args += ["--clipboard", "--input"]
            subprocess.run(args, input=text.encode("utf-8"), check=True)
            return
    debug_log("no clipboard tool found (xclip/xsel/wl-copy)", "dictation")


# ---------------------------------------------------------------------------
# C-level stderr suppression (for PortAudio warnings)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect C-level stderr to /dev/null.

    PortAudio logs warnings like ``||PaMacCore (AUHAL)|| Error on line …``
    directly to file descriptor 2. Python's contextlib.redirect_stderr only
    catches Python-level writes, so we dup the real fd instead.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


# ---------------------------------------------------------------------------
# Audio resampling
# ---------------------------------------------------------------------------

def _resample(audio, from_rate: int, to_rate: int):
    """Resample a 1-D float32 numpy array from *from_rate* to *to_rate*."""
    if from_rate == to_rate or np is None:
        return audio
    duration = len(audio) / from_rate
    target_len = int(duration * to_rate)
    # Linear interpolation — good enough for speech fed to Whisper
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Custom dictionary & LLM post-processing
# ---------------------------------------------------------------------------

def _apply_custom_dictionary(text: str, dictionary: list) -> str:
    """Apply custom dictionary corrections to transcribed text.

    Each entry in *dictionary* is a string. The dictionary is used to fix
    common mis-transcriptions (e.g. "Jarvice" → "Jarvis") by doing
    case-insensitive replacement.  Entries can be ``"wrong -> right"`` pairs
    or single terms that Whisper should have produced verbatim.
    """
    for entry in dictionary:
        if not isinstance(entry, str):
            continue
        if " -> " in entry:
            wrong, _, right = entry.partition(" -> ")
            wrong, right = wrong.strip(), right.strip()
            if wrong and right:
                # Case-insensitive whole-word replacement
                import re
                text = re.sub(
                    r"(?i)\b" + re.escape(wrong) + r"\b",
                    right,
                    text,
                )
    return text


# R34-S57 (A2-09): consecutive failure counter. After 3 silent
# fallbacks in a row, emit a one-shot warning event so the HUD /
# dashboard can surface "filler removal disabled — check Ollama"
# to the user. Previously every dictation fell back silently on
# wrong model name / wedged Ollama; the feature was permanently
# disabled with zero user signal.
_LLM_CLEAN_FAIL_STREAK = 0
_LLM_CLEAN_FAIL_THRESHOLD = 3


def _llm_clean_dictation(text: str, ollama_base_url: str, model: str = "qwen2.5:3b", thinking: bool = False) -> str:
    """Use the local LLM to remove filler words and tidy dictation output.

    Falls back to the original text if the LLM is unreachable or slow.
    R34-S57 (C-P2.12): default model was ``gemma4:e2b`` which doesn't
    exist on Ollama — switched to ``qwen2.5:3b`` to match the real
    intent-judge default.
    """
    global _LLM_CLEAN_FAIL_STREAK
    try:
        import requests
    except ImportError:
        return text

    # R34-S57 (A3-13): RU-aware prompt + explicit filler list.
    # The previous English prompt to a small RU/UA-tuned model
    # produced mixed output: either left UA fillers untouched ("ну",
    # "це", "тіпа") or appended English meta-commentary like "Here
    # is the cleaned text:" because it didn't understand the rules.
    # The RU prompt + enumerated fillers fixes both.
    prompt = (
        "Очисти продиктованный текст ниже. Удали слова-паразиты и "
        "колебания (эээ, ээ, ну, типа, короче, ммм, эм, как бы, "
        "значит; ну, це, тіпа, типу, ну це, як би, ееее). "
        "Сохрани смысл и язык исходного текста — НЕ переводи. "
        "Верни ТОЛЬКО очищенный текст без пояснений, без префиксов, "
        "без кавычек.\n\n"
        f"{text}"
    )

    try:
        resp = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": thinking,
            },
            # R34-S58.0 perf: (connect, read) split. Dictation paste
            # blocks on this call — paying a fresh TLS-style handshake
            # for every dictation was part of the user-reported
            # slowness. Fail-fast at 2 s connect (dictation feels
            # broken if anything past 5 s blocks the paste).
            timeout=(2.0, 5.0),
            # Connection: close removed — see llm.py for rationale.
        )
        if resp.status_code == 200:
            data = resp.json()
            cleaned = data.get("response", "").strip()
            if cleaned:
                debug_log(f"LLM filler removal: {text!r} → {cleaned!r}", "dictation")
                # Successful pass — reset streak so a future failure
                # spree starts the warning countdown afresh.
                _LLM_CLEAN_FAIL_STREAK = 0
                return cleaned
            _LLM_CLEAN_FAIL_STREAK += 1
            debug_log(
                "LLM filler removal returned empty response — using raw text",
                "dictation",
            )
        else:
            _LLM_CLEAN_FAIL_STREAK += 1
            debug_log(
                f"LLM filler removal HTTP {resp.status_code} — using raw text",
                "dictation",
            )
    except Exception as exc:
        _LLM_CLEAN_FAIL_STREAK += 1
        debug_log(f"LLM filler removal failed (using raw text): {exc}", "dictation")

    # R34-S57 (A2-09): on the Nth consecutive failure, emit ONE
    # warning event so the dashboard can surface the problem. The
    # warning is throttled (only the FIRST trip past the threshold
    # emits; subsequent failures still log but don't re-emit) to
    # avoid noise.
    try:
        if _LLM_CLEAN_FAIL_STREAK == _LLM_CLEAN_FAIL_THRESHOLD:
            from ..ipc.stream import get_stream as _get_stream
            stream = _get_stream()
            if stream is not None:
                stream.emit(
                    "dictation_warning",
                    reason="llm_clean_failures",
                    consecutive=_LLM_CLEAN_FAIL_STREAK,
                    model=model,
                    base_url=ollama_base_url,
                    hint="Check Ollama is reachable and the model name in config matches",
                )
    except Exception:
        # Never let observability errors block the dictation path.
        pass

    return text


# ---------------------------------------------------------------------------
# Hotkey string parsing
# ---------------------------------------------------------------------------

_MODIFIER_MAP = {
    "ctrl": "ctrl_l",
    "shift": "shift_l",
    "alt": "alt_l",
    "cmd": "cmd",
    "super": "cmd",
    "win": "cmd",
}


def format_hotkey_display(combo: str) -> str:
    """Format a hotkey string for human-readable display.

    On Windows, ``cmd`` is shown as ``Win``.  On macOS, ``cmd`` stays as
    ``Cmd`` and ``alt`` becomes ``Option``.  Key parts are title-cased and
    joined with `` + ``.
    """
    system = platform.system().lower()
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]

    display_parts: list[str] = []
    for part in parts:
        if part in ("cmd", "super", "win"):
            if system == "windows":
                display_parts.append("Win")
            else:
                display_parts.append("Cmd")
        elif part == "alt" and system == "darwin":
            display_parts.append("Option")
        else:
            display_parts.append(part.capitalize())

    return " + ".join(display_parts)


def parse_hotkey(combo: str):
    """Parse a hotkey string like ``'ctrl+shift+d'`` into pynput key objects.

    Returns a tuple of ``(frozenset_of_modifiers, trigger_key_or_None)``.
    Modifier-only combos (e.g. ``'ctrl+cmd'``) are valid — *trigger* is
    ``None`` and the hotkey activates when all modifiers are held.
    """
    if pynput_keyboard is None:
        raise RuntimeError("pynput is not installed")

    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey string")

    modifiers: set = set()
    trigger = None

    for part in parts:
        mapped = _MODIFIER_MAP.get(part)
        if mapped:
            key_obj = getattr(pynput_keyboard.Key, mapped, None)
            if key_obj is not None:
                modifiers.add(key_obj)
        else:
            # It's a regular key
            if len(part) == 1:
                trigger = pynput_keyboard.KeyCode.from_char(part)
            else:
                key_obj = getattr(pynput_keyboard.Key, part, None)
                if key_obj is not None:
                    trigger = key_obj
                else:
                    raise ValueError(f"unknown key: {part}")

    if not modifiers and trigger is None:
        raise ValueError("hotkey must contain at least one key")

    return frozenset(modifiers), trigger


# ---------------------------------------------------------------------------
# Stream cleanup
# ---------------------------------------------------------------------------

def _close_stream(stream: Any) -> None:
    """Stop and close a sounddevice InputStream, swallowing errors."""
    if stream is None:
        return
    try:
        stream.stop()
    except Exception as exc:
        debug_log(f"stream.stop() failed: {exc}", "dictation")
    try:
        stream.close()
    except Exception as exc:
        debug_log(f"stream.close() failed: {exc}", "dictation")


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

MAX_RECORD_SECONDS = 60


class DictationEngine:
    """Hold-to-dictate engine.

    Parameters
    ----------
    whisper_model_ref : callable
        ``lambda`` returning the shared Whisper model (or *None* if not ready).
    whisper_backend_ref : callable
        ``lambda`` returning ``"mlx"`` or ``"faster-whisper"``.
    mlx_repo_ref : callable
        ``lambda`` returning the MLX HuggingFace repo string (or *None*).
    hotkey : str
        Hotkey combination, e.g. ``"ctrl+shift+d"``.
    sample_rate : int
        Audio sample rate (should match Whisper expectations, default 16000).
    on_dictation_start : callable | None
        Called when recording starts (for face state, listener pause, etc.).
    on_dictation_processing_start : callable | None
        Called after the user releases the hotkey, once audio has been captured
        and the stop beep has played, just before transcription begins. Used by
        the UI to switch the face into a "processing" state while we transcribe
        and paste.
    on_dictation_end : callable | None
        Called when the full dictation cycle (recording + transcription +
        paste) has finished.
    transcribe_lock : threading.Lock | None
        Lock shared with the voice listener to serialise Whisper calls.
    on_dictation_result : callable | None
        Called with ``(entry_dict)`` after a successful dictation is saved
        to history. Used by the UI to update the history window.
    """

    def __init__(
        self,
        whisper_model_ref: Callable[[], Any],
        whisper_backend_ref: Callable[[], Optional[str]],
        mlx_repo_ref: Callable[[], Optional[str]],
        hotkey: str = "ctrl+shift+d",
        sample_rate: int = 16000,
        on_dictation_start: Optional[Callable[[], None]] = None,
        on_dictation_processing_start: Optional[Callable[[], None]] = None,
        on_dictation_end: Optional[Callable[[], None]] = None,
        transcribe_lock: Optional[threading.Lock] = None,
        on_dictation_result: Optional[Callable] = None,
        history: Optional[DictationHistory] = None,
        voice_device: Optional[str] = None,
        filler_removal: bool = False,
        custom_dictionary: Optional[list] = None,
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "gemma4:e2b",
        thinking: bool = False,
    ) -> None:
        self._whisper_model_ref = whisper_model_ref
        self._whisper_backend_ref = whisper_backend_ref
        self._mlx_repo_ref = mlx_repo_ref
        self._target_sample_rate = sample_rate  # Whisper expects this rate
        self._stream_sample_rate = sample_rate  # Actual device rate (may differ)
        self._on_dictation_start = on_dictation_start
        self._on_dictation_processing_start = on_dictation_processing_start
        self._on_dictation_end = on_dictation_end
        self._on_dictation_result = on_dictation_result
        self._transcribe_lock = transcribe_lock or threading.Lock()
        self.history = history or DictationHistory()
        self._voice_device = voice_device
        self._filler_removal = filler_removal
        self._custom_dictionary = custom_dictionary or []
        self._ollama_base_url = ollama_base_url
        self._ollama_model = ollama_model
        self._thinking = thinking

        # Parse hotkey
        self._modifiers, self._trigger = parse_hotkey(hotkey)
        self._hotkey_str = hotkey

        # State
        self._recording = False
        self._hands_free = False  # True when in continuous (double-tap) mode
        self._audio_frames: list = []
        # Audit round 18 fix: running sample counter eliminates the
        # O(n²) ``sum(len(f) for f in self._audio_frames)`` walk that
        # used to run on every audio callback (~10 Hz). At a 60 s
        # max-record cap on a 48 kHz device, ~600 callbacks × ~300
        # ``len()`` calls average = ~180 k per session — measurable
        # cost on a stressed CPU, occasional dropped frame.
        self._audio_sample_count: int = 0
        self._stream: Optional[Any] = None
        self._listener: Optional[Any] = None
        self._pressed_modifiers: set = set()
        self._record_start_time: float = 0.0
        # Audit round 18 fix: ``_max_frames`` is computed in ``__init__``
        # against the TARGET sample rate (16 kHz), but the audio
        # callback runs at the NATIVE device rate (commonly 48 kHz).
        # That made the cap fire at ~20 s instead of 60 s on a 48 kHz
        # mic — the user got cut off three times early. Now we recompute
        # ``_max_frames`` in ``_start_recording`` once the native rate
        # is known. The init value here is a sane default for callers
        # that never start a stream.
        self._max_frames = MAX_RECORD_SECONDS * sample_rate
        self._lock = threading.Lock()
        self._started = False
        # Clipboard preservation (round 18 fix). Filled by
        # ``_clipboard_paste`` before the write and restored after the
        # OS-level paste keystroke completes, so a user who had
        # important text on the clipboard does not silently lose it.
        self._saved_clipboard: Optional[str] = None

        # Double-tap detection for hands-free mode
        self._last_hotkey_release_time: float = 0.0
        self._double_tap_window: float = 0.4  # seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for the hotkey."""
        if pynput_keyboard is None:
            debug_log("pynput not installed — dictation disabled", "dictation")
            return
        if sd is None:
            debug_log("sounddevice not available — dictation disabled", "dictation")
            return
        if self._started:
            return

        # macOS 26+ enforces that TSM (Text Services Manager) calls happen on
        # the main dispatch queue.  pynput's keyboard Listener runs a CGEventTap
        # on a background thread whose callback triggers TSM input-source
        # queries, violating this assertion and crashing the process (SIGTRAP).
        # Disable pynput on macOS 26+ until an alternative backend is available.
        if sys.platform == "darwin":
            try:
                mac_ver = platform.mac_ver()[0]
                major = int(mac_ver.split(".")[0]) if mac_ver else 0
            except (ValueError, IndexError):
                major = 0
            if major >= 26:
                debug_log(
                    f"pynput disabled on macOS {mac_ver} "
                    "(TSM main-thread assertion)", "dictation",
                )
                print(
                    "  ⚠️  Dictation is not available on macOS 26+ "
                    "(pynput keyboard listener incompatibility)",
                    flush=True,
                )
                return

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()
        self._started = True
        debug_log(f"dictation engine started (hotkey: {self._hotkey_str})", "dictation")

    def stop(self) -> None:
        """Stop the dictation engine and clean up."""
        if self._recording:
            self._stop_recording(discard=True)
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._started = False
        debug_log("dictation engine stopped", "dictation")

    @property
    def is_recording(self) -> bool:
        return self._recording

    def set_on_dictation_result(self, callback: Optional[Callable]) -> None:
        """Set the callback invoked after a successful dictation."""
        self._on_dictation_result = callback

    # ------------------------------------------------------------------
    # Key event handlers
    # ------------------------------------------------------------------

    def _normalise_key(self, key) -> Any:
        """Normalise a key event to compare against our parsed trigger/modifiers."""
        # pynput sometimes gives KeyCode with vk but char=None for modified combos
        if hasattr(key, "char") and key.char is not None:
            return pynput_keyboard.KeyCode.from_char(key.char.lower())
        return key

    def _key_matches(self, key, nkey, target) -> bool:
        """Check whether *key* (raw) / *nkey* (normalised) matches *target*."""
        if target is None:
            return False
        if nkey == target or key == target:
            return True
        if getattr(key, "name", None) == getattr(target, "name", None):
            return True
        if hasattr(key, "char") and key.char:
            if pynput_keyboard.KeyCode.from_char(key.char.lower()) == target:
                return True
        return False

    def _all_modifiers_held(self) -> bool:
        """Return True when every required modifier is currently pressed."""
        return all(
            m in self._pressed_modifiers or any(
                getattr(p, "name", None) == getattr(m, "name", None)
                for p in self._pressed_modifiers
            )
            for m in self._modifiers
        )

    def _on_key_press(self, key) -> None:
        nkey = self._normalise_key(key)

        # Escape always stops hands-free recording
        if self._hands_free and self._recording:
            if getattr(key, "name", None) == "esc" or getattr(nkey, "name", None) == "esc":
                debug_log("hands-free stopped via Escape", "dictation")
                self._stop_recording()
                return

        # Track modifiers currently held
        if any(self._key_matches(key, nkey, m) for m in self._modifiers):
            self._pressed_modifiers.add(nkey if nkey in self._modifiers else key)

        # In hands-free mode, hotkey press stops recording
        if self._hands_free and self._recording:
            mods_held = self._all_modifiers_held()
            if self._trigger is not None:
                if mods_held and self._key_matches(key, nkey, self._trigger):
                    debug_log("hands-free stopped via hotkey", "dictation")
                    self._stop_recording()
                    return
            elif mods_held and len(self._pressed_modifiers) >= len(self._modifiers):
                debug_log("hands-free stopped via hotkey", "dictation")
                self._stop_recording()
                return

        # Check activation condition
        if not self._recording:
            mods_held = self._all_modifiers_held()

            if self._trigger is not None:
                trigger_match = self._key_matches(key, nkey, self._trigger)
                if mods_held and trigger_match:
                    self._start_recording()
            else:
                if mods_held and len(self._pressed_modifiers) >= len(self._modifiers):
                    self._start_recording()

    def _on_key_release(self, key) -> None:
        nkey = self._normalise_key(key)

        # Remove from pressed set
        self._pressed_modifiers.discard(nkey)
        self._pressed_modifiers.discard(key)
        for m in list(self._pressed_modifiers):
            if getattr(m, "name", None) == getattr(key, "name", None):
                self._pressed_modifiers.discard(m)

        # In hands-free mode, key release does NOT stop recording
        if self._hands_free:
            return

        # Normal hold-to-dictate: any required key released → stop
        if self._recording:
            trigger_released = self._key_matches(key, nkey, self._trigger)
            modifier_released = any(
                self._key_matches(key, nkey, m) for m in self._modifiers
            )
            if trigger_released or modifier_released:
                # Check for double-tap: if released quickly, transition to hands-free
                now = time.time()
                hold_duration = now - self._record_start_time
                if hold_duration < self._double_tap_window:
                    time_since_last = now - self._last_hotkey_release_time
                    if time_since_last < self._double_tap_window:
                        # Double-tap detected → switch to hands-free
                        self._hands_free = True
                        debug_log("hands-free mode activated (double-tap)", "dictation")
                        self._last_hotkey_release_time = 0.0
                        return
                    # First quick tap — stop recording but remember the time
                    self._last_hotkey_release_time = now
                else:
                    self._last_hotkey_release_time = 0.0
                self._stop_recording()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording:
                return
            self._recording = True

        # Check Whisper readiness
        model = self._whisper_model_ref()
        backend = self._whisper_backend_ref()
        if model is None and backend != "mlx":
            debug_log("whisper model not loaded — dictation skipped", "dictation")
            self._recording = False
            return

        debug_log("dictation recording started", "dictation")
        self._audio_frames = []
        # Reset the running sample counter at the start of every
        # recording session (round 18 fix).
        self._audio_sample_count = 0
        self._record_start_time = time.time()
        # Audit round 21 fix (F29): prefetch the user's existing
        # clipboard at recording-start (in a background thread) so
        # the ``pbpaste`` subprocess cost (up to 2 s worst-case
        # timeout if the clipboard service is stuck) happens
        # DURING the user's recording window instead of AFTER it.
        # Once the user releases the hotkey we have the snapshot
        # already cached and can paste + restore with zero extra
        # delay.
        self._cached_clipboard_at_record_start: Optional[str] = None
        self._cached_clipboard_ready = threading.Event()
        def _prefetch_clipboard() -> None:
            try:
                self._cached_clipboard_at_record_start = _read_clipboard_text()
            except Exception:
                self._cached_clipboard_at_record_start = None
            finally:
                self._cached_clipboard_ready.set()
        threading.Thread(
            target=_prefetch_clipboard,
            daemon=True,
            name="dictation-clip-prefetch",
        ).start()

        # Notify listeners (face state, PAUSE main listener) — call
        # this BEFORE opening the dedicated audio stream so the main
        # listener's `_dictation_active` flag is True by the time the
        # next mic frame fires. Audit round 8 regression find: the
        # callback used to fire after the InputStream was opened,
        # leaving a few ms where both the listener and the dictation
        # engine were processing mic frames on the same device — on
        # macOS Core Audio this can produce conflicting opens.
        if self._on_dictation_start:
            try:
                self._on_dictation_start()
            except Exception as exc:
                debug_log(f"on_dictation_start callback error: {exc}", "dictation")

        # Play start beep
        _play_beep(_get_start_beep())

        # Open dedicated audio stream.
        # Always use the device's native sample rate to avoid PortAudio errors
        # (e.g. -50 on macOS when requesting 16 kHz on a 48 kHz device).
        # Audio is resampled to the Whisper target rate after recording.
        stream_kwargs: dict[str, Any] = {}
        if self._voice_device:
            try:
                stream_kwargs["device"] = int(self._voice_device)
            except (ValueError, TypeError):
                pass

        # Query native sample rate
        try:
            if "device" in stream_kwargs:
                dev_info = sd.query_devices(stream_kwargs["device"])
            else:
                dev_info = sd.query_devices(kind="input")
            native_rate = int(dev_info.get("default_samplerate", self._target_sample_rate))
        except Exception:
            native_rate = self._target_sample_rate

        try:
            with _suppress_stderr():
                self._stream = sd.InputStream(
                    samplerate=native_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(native_rate * 0.1),
                    callback=self._audio_callback,
                    **stream_kwargs,
                )
            self._stream_sample_rate = native_rate
            # Audit round 18 fix: recompute the max-frames cap against
            # the ACTUAL native rate. Without this, a 48 kHz mic
            # tripped the cap (computed against 16 kHz target) at ~20 s
            # instead of the documented 60 s.
            self._max_frames = MAX_RECORD_SECONDS * native_rate
            if native_rate != self._target_sample_rate:
                debug_log(f"dictation stream at native {native_rate} Hz (will resample to {self._target_sample_rate})", "dictation")
        except Exception as exc:
            debug_log(f"failed to open dictation audio stream: {exc}", "dictation")
            self._recording = False
            if self._on_dictation_end:
                self._on_dictation_end()
            return

        try:
            self._stream.start()
        except Exception as exc:
            debug_log(f"failed to start dictation audio stream: {exc}", "dictation")
            self._recording = False
            if self._on_dictation_end:
                self._on_dictation_end()
            return

        # R34-S57 (A3-12): wall-clock stuck-key watchdog.
        # The audio-frame counter (_max_frames check) only triggers
        # when callbacks keep arriving — but if pynput drops a
        # ``KeyUp`` event (lock screen during press, app focus
        # change on Wayland, USB keyboard reconnect mid-hold) AND
        # the audio source also disconnects, ``_recording`` stays
        # True forever. Face stuck in DICTATING, voice listener
        # stays paused, until process restart.
        #
        # The watchdog sleeps ``MAX_RECORD_SECONDS + 5`` then checks
        # if we're still recording the same session (epoch counter
        # bumps each ``_start_recording``); if so, forces a discard
        # stop. The +5s padding lets the normal max-duration
        # autostop fire first on a healthy stream.
        try:
            self._record_epoch = int(getattr(self, "_record_epoch", 0)) + 1
            _expected_epoch = self._record_epoch
            def _watchdog() -> None:
                time.sleep(MAX_RECORD_SECONDS + 5.0)
                # If a new recording has started since, the epoch
                # bumped — that watchdog will own the new session.
                if not self._recording:
                    return
                if getattr(self, "_record_epoch", 0) != _expected_epoch:
                    return
                debug_log(
                    "dictation watchdog tripped — stuck-key recovery "
                    f"(epoch {_expected_epoch}, no normal stop after "
                    f"{MAX_RECORD_SECONDS + 5:.0f}s)",
                    "dictation",
                )
                try:
                    self._stop_recording(discard=True)
                except Exception as exc:
                    debug_log(
                        f"dictation watchdog stop failed: {exc}",
                        "dictation",
                    )
            threading.Thread(
                target=_watchdog,
                daemon=True,
                name=f"dictation-watchdog-{_expected_epoch}",
            ).start()
        except Exception as exc:
            # Watchdog is belt-and-braces — never let a setup failure
            # block the recording itself.
            debug_log(f"watchdog setup failed: {exc}", "dictation")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — accumulate audio frames."""
        # ``self._recording`` is read without the engine lock.  This is safe
        # because writes happen under the lock in _start_recording and
        # _stop_recording, and a single-word bool read/write is atomic under
        # the GIL.  Worst case is one extra frame captured just after stop
        # or one missed frame just after start — both benign.
        if not self._recording:
            return
        # Audit round 18 fix: O(1) total-samples check via running
        # counter instead of O(n) ``sum(len(f) ...)`` walk per callback.
        if self._audio_sample_count >= self._max_frames:
            debug_log("max dictation duration reached (60s)", "dictation")
            # Schedule stop on a separate thread to avoid deadlock in callback
            threading.Thread(target=self._stop_recording, daemon=True).start()
            return
        chunk = indata[:, 0].copy()
        self._audio_frames.append(chunk)
        self._audio_sample_count += len(chunk)

    def _stop_recording(self, discard: bool = False) -> None:
        # Flip state and snapshot the work queue atomically, under minimal
        # lock scope.  All heavy work (stream teardown, beep, transcribe,
        # paste) then runs off-thread so the pynput hotkey callback can
        # return immediately.  This matters on Windows: low-level keyboard
        # hooks are silently removed by the OS when a callback takes more
        # than ~5 s, which leaves pynput in an inconsistent state and can
        # segfault on the next Controller.press/release issued by the paste
        # thread (issue #184).
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._hands_free = False
            stream = self._stream
            self._stream = None
            audio_frames = self._audio_frames
            self._audio_frames = []
            start_time = self._record_start_time

        if discard:
            # Shutdown path — tear down synchronously so the caller knows
            # everything is finished before the engine is gone.
            _close_stream(stream)
            if self._on_dictation_end:
                try:
                    self._on_dictation_end()
                except Exception:
                    pass
            return

        threading.Thread(
            target=self._finalise_and_transcribe,
            args=(stream, audio_frames, start_time),
            daemon=True,
        ).start()

    def _finalise_and_transcribe(
        self,
        stream: Any,
        audio_frames: list,
        start_time: float,
    ) -> None:
        """Worker: close stream, play beep, transcribe, paste.

        Runs on a background thread so the pynput hotkey callback returns
        immediately.  See ``_stop_recording`` for the rationale.

        ``_on_dictation_end`` is normally fired from ``_transcribe_and_paste``'s
        finally block.  We wrap the whole body in try/except so a failure in
        ``_close_stream`` or ``_play_beep`` (before we reach the transcribe
        step) still unpauses the voice listener and resets the face state —
        otherwise a single beep error would strand the UI in ``DICTATING``.
        """
        end_fired = False
        try:
            _close_stream(stream)
            _play_beep(_get_stop_beep())

            duration = time.time() - start_time
            debug_log(f"dictation recording stopped ({duration:.1f}s)", "dictation")

            if self._on_dictation_processing_start:
                try:
                    self._on_dictation_processing_start()
                except Exception as exc:
                    debug_log(f"on_dictation_processing_start callback error: {exc}", "dictation")

            # _transcribe_and_paste has its own finally that fires
            # _on_dictation_end, so we defer to it on the happy path.
            end_fired = True
            self._transcribe_and_paste(audio_frames)
        except Exception as exc:
            debug_log(f"dictation finalise error: {exc}", "dictation")
            if not end_fired and self._on_dictation_end:
                try:
                    self._on_dictation_end()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Transcription & paste
    # ------------------------------------------------------------------

    def _transcribe_and_paste(self, frames: list) -> None:
        try:
            if not frames:
                debug_log("no audio frames captured", "dictation")
                return

            audio = np.concatenate(frames)

            # Resample to target rate if stream ran at a different rate
            if self._stream_sample_rate != self._target_sample_rate:
                audio = _resample(audio, self._stream_sample_rate, self._target_sample_rate)

            # Require at least 0.3s of audio
            if len(audio) < self._target_sample_rate * 0.3:
                debug_log("audio too short for transcription", "dictation")
                return

            text = self._transcribe(audio)

            # R34-S57 (A3-14): order swap. Previously the custom
            # dictionary was applied BEFORE LLM filler-removal —
            # which let the LLM "normalise" the user's manual
            # corrections back to their natural form ("IBONS" →
            # "Ibons"). Run the LLM cleanup first (general filler /
            # hesitation removal) then apply the user's explicit
            # corrections as the final authoritative pass.
            if text and self._filler_removal:
                text = _llm_clean_dictation(text, self._ollama_base_url, self._ollama_model, thinking=self._thinking)

            # Apply custom dictionary corrections (authoritative last
            # word — overrides anything the LLM may have changed).
            if text and self._custom_dictionary:
                text = _apply_custom_dictionary(text, self._custom_dictionary)

            if text:
                duration = len(audio) / self._target_sample_rate
                debug_log(f"dictation result: {text!r}", "dictation")
                # Audit round 21 fix (F29): use the snapshot taken at
                # record-start in a background thread (see
                # ``_start_recording``). If for some reason the prefetch
                # didn't complete (unusual — recording is usually
                # ≥500 ms long), fall back to reading now.
                if (
                    getattr(self, "_cached_clipboard_ready", None)
                    and self._cached_clipboard_ready.wait(timeout=0.5)
                ):
                    previous_clip = self._cached_clipboard_at_record_start
                else:
                    previous_clip = _read_clipboard_text()
                _clipboard_paste(text, restore_previous=previous_clip)
                # Persist to history
                entry = self.history.add(text, duration=duration)
                if self._on_dictation_result:
                    try:
                        self._on_dictation_result(entry)
                    except Exception:
                        pass
            else:
                debug_log("empty transcription — no paste", "dictation")
        except Exception as exc:
            debug_log(f"dictation transcribe/paste error: {exc}", "dictation")
        finally:
            if self._on_dictation_end:
                try:
                    self._on_dictation_end()
                except Exception:
                    pass

    def _transcribe(self, audio) -> str:
        """Transcribe audio using the shared Whisper model."""
        backend = self._whisper_backend_ref()
        model = self._whisper_model_ref()

        with self._transcribe_lock:
            if backend == "mlx":
                return self._transcribe_mlx(audio)
            elif model is not None:
                return self._transcribe_faster_whisper(model, audio)
            else:
                debug_log("no whisper model available", "dictation")
                return ""

    def _transcribe_mlx(self, audio) -> str:
        repo = self._mlx_repo_ref()
        if not repo:
            return ""
        try:
            import mlx_whisper
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, language=None)
            text = result.get("text", "").strip() if isinstance(result, dict) else ""
            return text
        except Exception as exc:
            debug_log(f"MLX transcription error: {exc}", "dictation")
            return ""

    def _transcribe_faster_whisper(self, model, audio) -> str:
        try:
            try:
                segments, _info = model.transcribe(audio, language=None, vad_filter=False)
            except TypeError:
                segments, _info = model.transcribe(audio, language=None)
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            debug_log(f"faster-whisper transcription error: {exc}", "dictation")
            return ""

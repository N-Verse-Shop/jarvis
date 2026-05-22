"""Typed event stream — append-only JSONL channel between daemon and HUD.

Inspired by quiet-node/thuki Tauri Channel API. Instead of every IPC
consumer parsing free-form JSON state, we emit typed events with a
stable schema. Each line is one event, easy to tail with `tail -f`
or `fs.watch()` in Electron.

EVENT SCHEMA (single source of truth)
─────────────────────────────────────

All events share these fields:
  type     — discriminator tag (see TYPES below)
  ts       — unix timestamp in seconds (float, microsecond precision)
  seq      — monotonic sequence number (uint64) — gap-detection helper

Per-type payload (the rest of the JSON object):

  state            { state: "IDLE"|"LISTENING"|"THINKING"|"SPEAKING", level: 0..1 }
  vad              { speaking: bool, level: 0..1 }
  stt_partial      { text: str, lang: str|null }
  stt_final        { text: str, lang: str, confidence: float, duration_ms: int }
  token            { content: str, sentence_idx: int, total_chars: int }
  sentence         { text: str, idx: int }          # complete sentence ready for TTS
  tts_start        { text: str, estimated_ms: int }
  tts_done         { duration_ms: int }
  tool_call        { tool: str, args: dict, status: "starting"|"completed"|"failed",
                     result?: any, error?: str, duration_ms?: int }
  action           { name: str, description: str, async: bool,
                     result?: { ok: bool, message: str } }
  wake_word        { word: str, confidence: float }
  hot_window       { active: bool, expires_in_ms: int }
  language_switch  { from_lang: str, to_lang: str, ack: str }
  memory           { event: "load"|"save"|"compact", count: int }
  llm_request      { model: str, num_messages: int, num_tools: int }
  llm_response     { model: str, total_ms: int, tokens: int, tokens_per_sec: float }
  error            { component: str, message: str, traceback?: str }
  log              { level: "DEBUG"|"INFO"|"WARN"|"ERROR", component: str, message: str }
  daemon_startup   { version: str, channel: str, pid: int }
  daemon_shutdown  { reason: str }

USAGE (daemon side):

    from jarvis.ipc import get_stream
    stream = get_stream()
    stream.emit("stt_final", text=transcript, lang="uk",
                confidence=0.95, duration_ms=1300)

USAGE (Electron HUD side):

    fs.watch(eventsPath, () => {
      readNewLinesIncrementally(eventsPath).forEach(line => {
        const ev = JSON.parse(line);
        switch (ev.type) {
          case "stt_final": showTranscript(ev.text); break;
          case "tool_call": animateToolBadge(ev.tool); break;
          case "token":     streamTokenIntoCaption(ev.content); break;
          ...
        }
      });
    });

ROTATION:

The file caps at MAX_BYTES (default 8MB). When exceeded, we rename
events.jsonl → events.jsonl.1 and start fresh. We keep at most 3
rotations on disk. Most consumers re-open on rotation transparently
since they watch the parent directory, not the inode.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

# Single module-level singleton so daemon + threads all write to the
# same file with a shared write lock.
_INSTANCE: Optional["EventStream"] = None
_INSTANCE_LOCK = threading.Lock()


# Default location — same dir as state.json/control.json. macOS-friendly,
# Linux falls back to $XDG_DATA_HOME or ~/.local/share.
def _default_events_path() -> Path:
    # Audit round 9 fix: `os.name == "darwin"` was dead code (os.name
    # is "posix" on macOS, not "darwin"). The macOS branch only worked
    # because the `or os.uname().sysname == "Darwin"` half caught it.
    # Use sys.platform for the explicit check.
    import sys as _sys
    if _sys.platform == "darwin":
        return Path.home() / "Library/Application Support/jarvis/events.jsonl"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".local/share")
    return base / "jarvis/events.jsonl"


# Max bytes per file before rotation. 8MB ≈ ~80k typical events.
MAX_BYTES = 8 * 1024 * 1024
MAX_ROTATIONS = 3


class EventStream:
    """Thread-safe append-only typed-event writer.

    All writes are line-buffered + flushed so consumers see events
    within ~1ms of emit(). Use get_stream() for the singleton instance.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or _default_events_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._enabled = True
        # Touch file so consumers (Electron fs.watch) don't trip on ENOENT
        if not self.path.exists():
            self.path.touch()
        # Audit round 11 fix: `touch()` uses the process umask (typically
        # 022 → 644 = world-readable). This file contains stt_final
        # transcripts, tool_call args, log payloads — often after redact
        # but never PII-empty. On multi-user macOS another local account
        # could `tail -f events.jsonl`. Lock down to owner-only.
        try:
            self.path.chmod(0o600)
        except OSError:
            # Best effort: failures here are benign (non-POSIX FS).
            pass
        # Audit round 9 fix C5: keep ONE file handle open instead of
        # opening/closing on every emit. Audio callback emits `vad`
        # events ~10/sec; the open/stat/write/close cycle was costing
        # ~150-400µs per emit under the lock — wasted syscall storm.
        # `buffering=1` = line-buffered so consumers still see lines
        # within ~1ms (flush on \n). Atomicity vs other writers is
        # provided by `self._lock` (single-process file).
        try:
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            self._bytes_written = self.path.stat().st_size
        except Exception:
            self._fh = None
            self._bytes_written = 0

    def emit(self, event_type: str, **payload: Any) -> None:
        """Emit one typed event. Drops payload silently if disabled.

        Never raises — IPC must not crash the voice loop. Failures
        get logged via debug_log only if available.
        """
        if not self._enabled:
            return
        # Audit round 15 fix: serialise the payload OUTSIDE the lock.
        # Holding the lock during ``json.dumps`` meant any non-trivial
        # payload (bytes / numpy / custom class whose ``__str__``
        # raises / circular containers) raised mid-serialise. Even
        # though the outer try/except caught it, every other thread
        # waiting on _lock paid the full attempt latency, and the bad
        # event silently disappeared with no breadcrumb. Now we build
        # the line first; on failure we synthesise a minimal sanitised
        # error event so the consumer at least sees that something
        # was dropped (and what type).
        ts = time.time()
        # Audit round 17 fix: previously the placeholder was the literal
        # string ``"seq": 0`` and we used ``str.replace(..., count=1)``
        # to patch it under the lock. Any payload value whose JSON
        # encoding contained the substring ``"seq": 0`` (e.g. a nested
        # ``{"matches": [{"seq": 0, ...}]}`` from a tool result, or a
        # captured string like ``'foo "seq": 0 bar'``) caused replace
        # to patch the WRONG occurrence — leaving the framework's
        # ``seq`` at 0 and breaking gap-detection on the consumer
        # side, while corrupting the payload. Replacing with a
        # high-entropy sentinel (UUID4 hex) eliminates the collision
        # window: a payload would need to coincidentally embed the
        # exact same UUID we just generated in-memory, which is
        # cryptographically negligible.
        import uuid as _uuid
        seq_sentinel = f"__seq_placeholder_{_uuid.uuid4().hex}__"
        try:
            line = json.dumps(
                {
                    "type": event_type,
                    "ts": ts,
                    "seq": seq_sentinel,  # patched to a real int under the lock
                    **payload,
                },
                ensure_ascii=False,
                default=str,
            )
        except Exception as serialise_exc:
            try:
                from ..debug import debug_log
                debug_log(
                    f"EventStream: dropped {event_type} — serialise failed: {serialise_exc}",
                    "ipc",
                )
            except Exception:
                pass
            # Emit a sanitised replacement so the HUD knows an event
            # was lost. Wrapped in its own try so an error here can't
            # cascade.
            try:
                replacement = json.dumps(
                    {
                        "type": "error",
                        "ts": ts,
                        "seq": seq_sentinel,
                        "component": "ipc",
                        "message": f"dropped {event_type}: {type(serialise_exc).__name__}",
                    },
                    ensure_ascii=False,
                )
                line = replacement
            except Exception:
                return
        try:
            with self._lock:
                self._seq += 1
                # Patch the unique sentinel with the real monotonic seq.
                # The sentinel is a freshly-generated UUID4 in hex so a
                # collision against any payload string is cryptographically
                # negligible. Quote-wrapped to swap a JSON string for a
                # JSON int in one shot.
                line = line.replace(
                    f'"{seq_sentinel}"',
                    str(self._seq),
                    1,
                ) + "\n"
                encoded_len = len(line.encode("utf-8"))
                # Rotate BEFORE appending if we'd exceed cap.
                if self._bytes_written + encoded_len > MAX_BYTES:
                    self._rotate_locked()
                # Lazy-reopen if rotation closed us OR init failed.
                if self._fh is None:
                    try:
                        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
                        self._bytes_written = self.path.stat().st_size
                    except Exception:
                        return
                self._fh.write(line)
                self._bytes_written += encoded_len
        except Exception as e:
            # Lazy import so circular-import risk stays low.
            try:
                from ..debug import debug_log
                debug_log(f"EventStream emit failed: {e}", "ipc")
            except Exception:
                pass

    def _rotate_locked(self) -> None:
        """Caller must hold self._lock.

        Shift events.jsonl → events.jsonl.1 → events.jsonl.2 → ...
        Drop oldest beyond MAX_ROTATIONS. Re-opens the persistent
        handle on the fresh file.
        """
        try:
            # Close the current handle BEFORE renaming — on Windows
            # rename on an open file fails; on POSIX it works but
            # the handle would point at the moved file, defeating
            # the rotation. Both fixed by closing first.
            try:
                if self._fh is not None:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None
            # Drop the oldest if it exists
            oldest = self.path.with_suffix(f".jsonl.{MAX_ROTATIONS}")
            if oldest.exists():
                oldest.unlink()
            # Shift backwards
            for i in range(MAX_ROTATIONS - 1, 0, -1):
                src = self.path.with_suffix(f".jsonl.{i}")
                dst = self.path.with_suffix(f".jsonl.{i + 1}")
                if src.exists():
                    src.rename(dst)
            # Move current → .1
            if self.path.exists():
                self.path.rename(self.path.with_suffix(".jsonl.1"))
            # Create fresh empty file + reopen handle
            self.path.touch()
            # Audit round 11 fix: re-apply owner-only perms after rotation
            # — touch() on the fresh file uses process umask (typically 644).
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            self._bytes_written = 0
        except Exception:
            # Rotation failures are non-fatal — worst case we exceed
            # the cap. Better than dropping events. Attempt to reopen
            # the handle so we don't lose ALL subsequent emits.
            try:
                self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            except Exception:
                self._fh = None

    def disable(self) -> None:
        """Stop emitting (used during shutdown so we don't race with
        unlink + atexit handlers).

        Audit round 10 fix R1: take the lock BEFORE closing the
        handle. Without this, a thread mid-emit (past the
        `if not self._enabled: return` guard but holding the lock
        for the write) races with `disable()` closing the handle
        from under it → `ValueError: I/O on closed file`. The outer
        try/except in emit() swallows the error, but events get
        silently dropped during shutdown. Locking serialises us
        behind any in-flight emit.
        """
        self._enabled = False
        with self._lock:
            try:
                if self._fh is not None:
                    # R34-S57 (B2.1): explicit flush + fsync on
                    # shutdown so the final N lines (typically the
                    # ``daemon_shutdown`` event plus any in-flight
                    # tool replies) actually hit disk before the
                    # process exits. Pre-fix, line-buffered ``flush``
                    # only sent bytes to the OS — a SIGKILL during
                    # the shutdown window could lose the last
                    # several events from the kernel page cache.
                    try:
                        self._fh.flush()
                    except Exception:
                        pass
                    try:
                        import os as _os
                        _os.fsync(self._fh.fileno())
                    except (OSError, AttributeError):
                        # fsync isn't supported on every filesystem
                        # (encrypted overlays, some FUSE mounts).
                        # flush() is still better than nothing.
                        pass
                    self._fh.close()
                    self._fh = None
            except Exception:
                pass


def get_stream() -> EventStream:
    """Return the process-wide singleton EventStream."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = EventStream()
    return _INSTANCE

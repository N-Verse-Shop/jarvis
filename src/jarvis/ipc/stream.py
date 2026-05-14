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
    if os.name == "darwin" or os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
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

    def emit(self, event_type: str, **payload: Any) -> None:
        """Emit one typed event. Drops payload silently if disabled.

        Never raises — IPC must not crash the voice loop. Failures
        get logged via debug_log only if available.
        """
        if not self._enabled:
            return
        try:
            with self._lock:
                self._seq += 1
                record = {
                    "type": event_type,
                    "ts": time.time(),
                    "seq": self._seq,
                    **payload,
                }
                line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
                # Rotate BEFORE appending if we'd exceed cap with this
                # event — cheap stat() check vs always-writing-then-truncating.
                try:
                    cur_size = self.path.stat().st_size
                except OSError:
                    cur_size = 0
                if cur_size + len(line.encode("utf-8")) > MAX_BYTES:
                    self._rotate_locked()
                # Append + flush. line-buffered (buffering=1) ensures
                # fsync-light durability without sync()ing every line.
                with open(self.path, "a", encoding="utf-8", buffering=1) as f:
                    f.write(line)
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
        Drop oldest beyond MAX_ROTATIONS.
        """
        try:
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
            # Create fresh empty file
            self.path.touch()
        except Exception:
            # Rotation failures are non-fatal — worst case we exceed
            # the cap. Better than dropping events.
            pass

    def disable(self) -> None:
        """Stop emitting (used during shutdown so we don't race with
        unlink + atexit handlers)."""
        self._enabled = False


def get_stream() -> EventStream:
    """Return the process-wide singleton EventStream."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = EventStream()
    return _INSTANCE

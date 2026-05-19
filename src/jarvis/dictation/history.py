"""
Dictation history — persists transcription results to a local JSON file.

Privacy-first: all data stays on disk, never leaves the machine.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_history_path() -> Path:
    """Return the default path for dictation history storage."""
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "dictation_history.json"


class DictationHistory:
    """Thread-safe, file-backed dictation history.

    Each entry is a dict with keys:
        id       – unique identifier (UUID4 hex)
        text     – transcribed text
        timestamp – epoch seconds (float)
        duration – recording duration in seconds (float)
    """

    def __init__(self, path: Optional[Path] = None, max_entries: int = 500) -> None:
        self._path = path or _default_history_path()
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # Audit round 15 fix: sibling .lock file used to serialise
        # multi-process writes via ``flock``. Touched lazily inside
        # ``_cross_process_lock`` so a read-only consumer doesn't
        # create the file just by existing.
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._entries: List[Dict[str, Any]] = self._load()

    @contextmanager
    def _cross_process_lock(self):
        """Acquire an exclusive POSIX advisory lock on a sibling .lock file.

        Audit round 15 fix: the desktop app and daemon construct
        independent ``DictationHistory`` instances pointed at the same
        JSON file. ``self._lock`` is per-instance and useless across
        processes — interleaved load→mutate→save in two processes
        loses writes. Each writer takes ``flock(LOCK_EX)`` on a
        sibling file before its load+save cycle; the lock is released
        on context-manager exit. Read-only ``_load`` doesn't need this
        — the JSON write is atomic via ``os.replace``.

        Best-effort: on Windows or filesystems without ``fcntl``, the
        ``try/except ImportError`` falls back to the threading lock
        alone (pre-fix behaviour). Logged once at first failure so
        the regression is visible.
        """
        try:
            import fcntl as _fcntl
        except ImportError:
            yield
            return
        # Touch the lock file lazily — its existence is the lock, but
        # we don't want a read-only consumer creating it.
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = open(self._lock_path, "a+")
        except OSError as exc:
            from jarvis.debug import debug_log
            debug_log(f"dictation-history: file-lock open failed: {exc}", "dictation")
            yield
            return
        try:
            _fcntl.flock(fd.fileno(), _fcntl.LOCK_EX)
            try:
                yield
            finally:
                try:
                    _fcntl.flock(fd.fileno(), _fcntl.LOCK_UN)
                except OSError:
                    pass
        except OSError as exc:
            # flock can fail on some network FSes; degrade gracefully.
            from jarvis.debug import debug_log
            debug_log(f"dictation-history: flock failed: {exc}", "dictation")
            yield
        finally:
            try:
                fd.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, duration: float = 0.0) -> Dict[str, Any]:
        """Append a new dictation entry and persist. Returns the new entry."""
        entry: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "text": text,
            "timestamp": time.time(),
            "duration": round(duration, 1),
        }
        # Audit round 15 fix: serialise cross-process writes with a
        # POSIX file-lock. ``self._lock`` is per-instance; the desktop
        # app constructs its OWN DictationHistory pointed at the same
        # file (the default path is shared). Daemon ``add()`` and
        # desktop ``delete()`` could interleave load→mutate→save and
        # the daemon's save would resurrect the just-deleted entry.
        # ``os.replace`` is atomic for the swap but doesn't serialise
        # two concurrent writers. flock on a sibling .lock file (the
        # JSON file itself can't be locked because we rewrite it via
        # rename) bounds the load→mutate→save sequence.
        with self._lock:
            with self._cross_process_lock():
                self._entries = self._load()
                self._entries.append(entry)
                # Trim oldest entries if over limit
                if len(self._entries) > self._max_entries:
                    self._entries = self._entries[-self._max_entries:]
                self._save()
        return entry

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all entries, newest first."""
        with self._lock:
            return list(reversed(self._entries))

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns True if found and removed."""
        # Audit round 15 fix: same cross-process lock as ``add`` so
        # daemon-side ``add`` and desktop-side ``delete`` can't lose
        # updates to each other.
        with self._lock:
            with self._cross_process_lock():
                # Re-read in case another process appended since last load.
                self._entries = self._load()
                before = len(self._entries)
                self._entries = [e for e in self._entries if e["id"] != entry_id]
                if len(self._entries) < before:
                    self._save()
                    return True
                return False

    def clear(self) -> None:
        """Delete all entries."""
        with self._lock:
            with self._cross_process_lock():
                self._entries = []
                self._save()

    def reload_from_disk(self) -> None:
        """Re-read entries from the JSON file (thread-safe).

        Useful for external consumers (e.g. the desktop app) that need to
        pick up changes written by another process.
        """
        with self._lock:
            self._entries = self._load()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict[str, Any]]:
        # Audit round 17 fix: previously a corrupt or truncated
        # ``dictation_history.json`` was silently swallowed, returning
        # ``[]`` and tricking the very next ``_save()`` into
        # overwriting the broken file with a zero-entry list — losing
        # EVERY prior dictation forever, with no warning. Now we
        # rename the broken file to a ``.corrupt-<ts>`` sibling
        # before returning empty, so the user can recover it (and
        # so a future repair sweep can attempt structured recovery).
        if not self._path.exists():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # Top-level shape wrong but file parses — also broken.
            raise ValueError(
                f"dictation history root is {type(data).__name__}, expected list"
            )
        except Exception as exc:
            try:
                from datetime import datetime as _dt
                ts = _dt.now().strftime("%Y%m%dT%H%M%S")
                quarantine = self._path.with_suffix(
                    self._path.suffix + f".corrupt-{ts}"
                )
                # ``os.replace`` is atomic on POSIX/Windows; the
                # quarantine path will not collide because ``ts`` is
                # second-resolution and reads only happen once per
                # writer-lock acquisition.
                import os as _os
                _os.replace(self._path, quarantine)
                from jarvis.debug import debug_log
                debug_log(
                    f"dictation history corrupt ({type(exc).__name__}: "
                    f"{exc}); quarantined to {quarantine} and starting "
                    "fresh",
                    "dictation",
                )
            except Exception as _qe:
                from jarvis.debug import debug_log
                debug_log(
                    f"dictation history corrupt AND quarantine failed "
                    f"({type(_qe).__name__}: {_qe}); starting fresh",
                    "dictation",
                )
            return []

    def _save(self) -> None:
        # Audit round 13 fix: atomic .tmp + os.replace. Previously
        # ``self._path.open("w")`` truncated the file on open — a
        # SIGTERM during the json.dump dropped EVERY dictation entry
        # to disk-zero. Same pattern as config.py / graph.py backups.
        try:
            import os as _os
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
                f.flush()
                try:
                    _os.fsync(f.fileno())
                except OSError:
                    pass
            _os.replace(tmp, self._path)
        except Exception as exc:
            from jarvis.debug import debug_log
            debug_log(f"failed to save dictation history: {exc}", "dictation")
            # Cleanup stale tmp if it survived.
            try:
                tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

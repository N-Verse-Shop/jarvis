"""Debug logging utilities for Jarvis.

Audit round 9 rewrite (C4):
  - Cache check now uses `time.monotonic()`, not `time.time()`. An NTP
    backdate (clock jumps backwards) used to make `now - _last_check`
    negative, which kept the cached value pinned indefinitely.
  - Double-checked locking around `load_settings()` so two threads
    entering simultaneously after expiry don't both reload — defeats
    the cache exactly when contention is highest (audio callback
    emitting many debug_log calls in quick succession).
  - `_print_lock` so multi-thread prints don't interleave mid-line
    (audio thread + intent judge + HUD watcher were producing torn
    lines in jarvis-assistant.err.log).
  - `JARVIS_VOICE_DEBUG=1` env override for trace-on-demand without
    restarting the daemon to flip the config.
  - Explicit `flush=True` — stderr is line-buffered only on tty;
    when daemon is spawned by launchd or the Electron HUD, stderr is
    redirected to a file and gets block-buffered → debug lines were
    invisible for minutes during HUD-piped runs.
"""
import os
import sys
import threading
import time
from typing import Optional
from .config import load_settings

_lock = threading.Lock()
_last_check: float = 0.0
_cached: Optional[bool] = None
_TTL = 2.0

# Audit round 11 fix: `_ENV_FORCE` used to be sampled once at import,
# which contradicted the file-level docstring claim of "trace-on-demand
# without restarting the daemon". `export JARVIS_VOICE_DEBUG=1`
# after the daemon was up did nothing. Now re-read on every call —
# cheap (single dict lookup) and matches the documented contract.
_ENV_VAR = "JARVIS_VOICE_DEBUG"
_ENV_TRUTHY = ("1", "true", "yes", "on")


def _env_force_on() -> bool:
    return os.environ.get(_ENV_VAR, "").lower() in _ENV_TRUTHY


def _is_debug_enabled() -> bool:
    if _env_force_on():
        return True
    now = time.monotonic()
    # Fast path — no lock needed for the common case where cache is fresh.
    cached = _cached
    last = _last_check
    if cached is not None and (now - last) <= _TTL:
        return cached
    # Slow path — acquire lock, double-check, then reload.
    with _lock:
        # Another thread may have refreshed while we were waiting.
        if _cached is not None and (time.monotonic() - _last_check) <= _TTL:
            return _cached
        try:
            new_val = bool(load_settings().voice_debug)
        except Exception:
            new_val = False
        # Globals write inside the lock — safe across threads.
        globals()["_cached"] = new_val
        globals()["_last_check"] = time.monotonic()
        return new_val


# Serialise stderr writes so simultaneous calls from audio + intent
# judge + HUD watcher threads don't interleave their lines.
_print_lock = threading.Lock()


def debug_log(message: str, category: str = "debug") -> None:
    """Unified debug logging function for Jarvis.

    Args:
        message: The debug message to log
        category: The log category (e.g., "debug", "voice", "echo", "tts", etc.)

    Audit round 19 fix (HIGH): every call site across the codebase
    formats arbitrary values (fact text, tool args, transcript
    chunks, exception messages that may echo input) directly into
    the message string. The output goes to stderr, which launchd
    redirects to ``~/Library/Logs/jarvis-assistant.err.log`` — a
    persistent file that can be tailed, attached to a bug report,
    or rotated into off-machine backups. Without scrubbing, a user
    utterance containing ``OPENAI_API_KEY=sk-proj-...`` or a tool
    response embedding a Bearer token landed verbatim in that
    permanent log. We now run the message through ``scrub_secrets``
    (the same structural redactor used for diary entries and IPC
    events) before printing. The scrub is best-effort: any failure
    falls through to the raw message rather than silencing the log
    line, since silent log loss would defeat debugging entirely.
    """
    if not _is_debug_enabled():
        return
    try:
        # Lazy import: ``utils.redact`` imports ``re`` only and has no
        # cycle with this module, but lazy-loading keeps the cold-start
        # path of ``debug.py`` (used by EVERY module) minimal.
        try:
            from .utils.redact import scrub_secrets as _scrub
            safe_message = _scrub(str(message))
        except Exception:
            safe_message = str(message)
        with _print_lock:
            print(f"[{category:^10}] {safe_message}", file=sys.stderr, flush=True)
    except Exception:
        pass

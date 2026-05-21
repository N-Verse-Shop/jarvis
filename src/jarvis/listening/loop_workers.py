"""Background scheduler — periodic maintenance tasks.

R34-S49 REC 4 — the user asked for Jarvis to "audit / optimize tasks
overnight" without blocking voice. This module is a tiny cron-style
scheduler that runs inside the daemon thread; each registered task
declares an interval (or a daily HH:MM clock time) and a callable.

Why pure-Python, no APScheduler / no crontab?
  - Zero new dependencies (the user is on a CCX23 with 15 GB RAM and
    we already audit every wheel that lands on the box).
  - Tasks are bounded (file rotates, DB vacuums) — no need for a
    distributed scheduler.
  - One thread, daemon=True, so it dies with the process; no orphan
    PID files to manage.

Tasks registered by default (see ``_DEFAULT_TASKS`` at bottom):
  * ``rotate_events_jsonl`` — every hour, trims events.jsonl to last
    100 MB / 200 k lines as a safety-net rotation. (The primary 8-MB
    rotation lives in ``ipc/stream.py`` and runs on every emit; this
    one only fires if that one ever fails / lags.)
  * ``vacuum_audit_db``   — every 6 h, SQLite VACUUM on the audit
    trail; keeps query latency stable.
  * ``prune_briefs``      — daily at 03:00, removes upgrade-briefs
    older than 30 days from ``~/.config/jarvis/upgrade-briefs/``.
  * ``brain_inventory``   — daily at 04:00, logs Brain file count
    + total size to events.jsonl so the dashboard can show health.

NOT registered by default:
  * ``keepwarm_ollama``   — covered by the dedicated
    ``_keepwarm_ollama_loop`` thread in ``warmup.py``. The callable
    is kept here so operators can re-enable it via
    ``register_task("keepwarm_ollama", task_keepwarm_ollama,
    interval_sec=600.0)`` if the main thread is disabled.

Design notes for future maintainers:
  - Each task callable MUST be safe to invoke from a non-asyncio
    thread (use ``urllib`` over ``aiohttp`` for HTTP, blocking I/O
    is fine because this is a daemon thread).
  - A task that raises is logged but doesn't kill the scheduler.
  - Tasks run sequentially in the scheduler thread — no two run in
    parallel — so a slow task (e.g. VACUUM on a 500 MB db) won't
    overlap a write to events.jsonl.
  - ``interval_sec`` and ``daily_hhmm`` are mutually exclusive; pass
    one or the other to ``register_task``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from ..debug import debug_log


# ─── Task model ──────────────────────────────────────────────────────────


@dataclass
class _ScheduledTask:
    """Single periodic task."""

    name: str
    fn: Callable[[], Optional[str]]
    interval_sec: Optional[float] = None
    daily_hhmm: Optional[str] = None  # "HH:MM" 24h local time
    last_run_monotonic: float = 0.0
    last_run_date: str = ""  # YYYY-MM-DD, for daily_hhmm
    runs_completed: int = 0
    runs_failed: int = 0
    last_error: str = ""
    last_status: str = "pending"
    last_message: str = ""

    def due(self, now_mono: float, now_dt: datetime) -> bool:
        """Should this task run right now?"""
        if self.interval_sec is not None:
            if self.last_run_monotonic == 0.0:
                return True  # first run on startup
            return (now_mono - self.last_run_monotonic) >= self.interval_sec
        if self.daily_hhmm is not None:
            today = now_dt.strftime("%Y-%m-%d")
            if self.last_run_date == today:
                return False
            try:
                hh, mm = (int(x) for x in self.daily_hhmm.split(":"))
            except Exception:
                return False
            return (now_dt.hour, now_dt.minute) >= (hh, mm)
        return False


_TASKS_LOCK = threading.Lock()
_TASKS: list[_ScheduledTask] = []
_SCHEDULER_THREAD: Optional[threading.Thread] = None
_SCHEDULER_STOP = threading.Event()


def register_task(
    name: str,
    fn: Callable[[], Optional[str]],
    *,
    interval_sec: Optional[float] = None,
    daily_hhmm: Optional[str] = None,
) -> None:
    """Register a periodic task. Safe to call before or after start().

    Either ``interval_sec`` (run every N seconds) or ``daily_hhmm``
    (run once per day at given HH:MM local) — not both.
    """
    if (interval_sec is None) == (daily_hhmm is None):
        raise ValueError(
            "register_task: exactly one of interval_sec / daily_hhmm required"
        )
    with _TASKS_LOCK:
        # If a task with this name is already registered, replace it.
        # Lets tests / hot-reloads override defaults without growing
        # the list unbounded.
        for i, t in enumerate(_TASKS):
            if t.name == name:
                _TASKS[i] = _ScheduledTask(
                    name=name,
                    fn=fn,
                    interval_sec=interval_sec,
                    daily_hhmm=daily_hhmm,
                )
                return
        _TASKS.append(
            _ScheduledTask(
                name=name,
                fn=fn,
                interval_sec=interval_sec,
                daily_hhmm=daily_hhmm,
            )
        )


def get_task_status() -> list[dict]:
    """Snapshot for dashboard / debug. Cheap, lock-protected."""
    with _TASKS_LOCK:
        return [
            {
                "name": t.name,
                "interval_sec": t.interval_sec,
                "daily_hhmm": t.daily_hhmm,
                "runs_completed": t.runs_completed,
                "runs_failed": t.runs_failed,
                "last_status": t.last_status,
                "last_message": t.last_message[:200],
                "last_error": t.last_error[:200],
            }
            for t in _TASKS
        ]


# ─── Scheduler thread ────────────────────────────────────────────────────


def _scheduler_loop() -> None:
    """Single-threaded scheduler — checks tasks every 30 s, runs due ones."""
    debug_log("loop_workers: scheduler started", "voice")
    while not _SCHEDULER_STOP.is_set():
        try:
            now_mono = time.monotonic()
            now_dt = datetime.now()
            due_tasks: list[_ScheduledTask] = []
            with _TASKS_LOCK:
                for t in _TASKS:
                    if t.due(now_mono, now_dt):
                        due_tasks.append(t)
            for t in due_tasks:
                try:
                    t.last_status = "running"
                    msg = t.fn()
                    t.last_status = "ok"
                    t.last_message = (msg or "")[:240]
                    t.runs_completed += 1
                    t.last_error = ""
                    debug_log(
                        f"loop_workers: {t.name} ok — {t.last_message[:120]}",
                        "voice",
                    )
                except Exception as exc:
                    t.last_status = "error"
                    t.last_error = repr(exc)[:240]
                    t.runs_failed += 1
                    debug_log(
                        f"loop_workers: {t.name} FAILED — {exc!r}",
                        "voice",
                    )
                finally:
                    t.last_run_monotonic = time.monotonic()
                    t.last_run_date = now_dt.strftime("%Y-%m-%d")
        except Exception as exc:
            # Scheduler MUST never die — log and continue.
            debug_log(f"loop_workers: scheduler tick crashed: {exc!r}", "voice")
        # Sleep 30 s, but wake on stop event for clean shutdown.
        _SCHEDULER_STOP.wait(timeout=30.0)
    debug_log("loop_workers: scheduler stopped", "voice")


def start() -> None:
    """Start the scheduler thread. Idempotent."""
    global _SCHEDULER_THREAD
    if _SCHEDULER_THREAD is not None and _SCHEDULER_THREAD.is_alive():
        return
    _register_default_tasks()
    _SCHEDULER_STOP.clear()
    _SCHEDULER_THREAD = threading.Thread(
        target=_scheduler_loop,
        name="jarvis-loop-workers",
        daemon=True,
    )
    _SCHEDULER_THREAD.start()


def stop(timeout: float = 2.0) -> None:
    """Signal the scheduler to exit. Called from daemon shutdown."""
    _SCHEDULER_STOP.set()
    t = _SCHEDULER_THREAD
    if t is not None and t.is_alive():
        try:
            t.join(timeout=timeout)
        except Exception:
            pass


# ─── Default tasks ───────────────────────────────────────────────────────


def _events_jsonl_path() -> Path:
    """Same logic as audit/events module — keep in sync if that moves."""
    return Path.home() / "Library/Application Support/jarvis/events.jsonl"


def _audit_db_path() -> Path:
    return Path.home() / "Library/Application Support/jarvis/audit.db"


def _briefs_dir() -> Path:
    return Path.home() / ".config/jarvis/upgrade-briefs"


def _brain_dir() -> Path:
    return Path.home() / "Documents/Nexus-Brain"


def task_rotate_events_jsonl() -> str:
    """Trim events.jsonl if > 100 MB or > 200k lines.

    Strategy: rename to ``events.jsonl.1``, keep only the last 50k
    lines in the new ``events.jsonl``. Cheap, atomic-ish (the old
    file is preserved for forensics).
    """
    p = _events_jsonl_path()
    if not p.exists():
        return "no events.jsonl"
    size = p.stat().st_size
    if size < 100 * 1024 * 1024:
        return f"events.jsonl size={size // 1024} KB, no rotate"
    # Count lines cheaply (read in chunks)
    lines_seen = 0
    with p.open("rb") as f:
        for _ in f:
            lines_seen += 1
    keep = 50_000
    if lines_seen <= keep:
        return f"events.jsonl lines={lines_seen}, no rotate"
    # Copy last `keep` lines into a tmp, rename old to .1, swap in.
    tmp = p.with_suffix(p.suffix + ".rotating")
    with p.open("rb") as src, tmp.open("wb") as dst:
        # Skip the first (lines_seen - keep) lines, write the rest.
        skip = lines_seen - keep
        for i, line in enumerate(src):
            if i < skip:
                continue
            dst.write(line)
    backup = p.with_suffix(p.suffix + ".1")
    if backup.exists():
        try:
            backup.unlink()
        except Exception:
            pass
    p.rename(backup)
    tmp.rename(p)
    return f"rotated: {lines_seen} → {keep} lines, backup={backup.name}"


def task_vacuum_audit_db() -> str:
    """SQLite VACUUM on the audit DB; reclaims fragmented space."""
    import sqlite3
    p = _audit_db_path()
    if not p.exists():
        return "no audit.db"
    size_before = p.stat().st_size
    try:
        # PRAGMA optimize is cheaper than full VACUUM and runs in a
        # tx; VACUUM rewrites the whole file but releases pages.
        # Alternate every other run so we get both benefits.
        with sqlite3.connect(str(p), timeout=10.0) as conn:
            conn.execute("PRAGMA optimize")
            conn.execute("VACUUM")
    except sqlite3.OperationalError as exc:
        return f"vacuum skipped: {exc}"
    size_after = p.stat().st_size
    delta = size_before - size_after
    return f"vacuumed: {size_before // 1024} → {size_after // 1024} KB (-{delta // 1024} KB)"


def task_prune_briefs() -> str:
    """Delete upgrade-briefs older than 30 days."""
    d = _briefs_dir()
    if not d.exists():
        return "no briefs dir"
    cutoff = time.time() - 30 * 24 * 3600
    deleted = 0
    saved = 0
    for f in d.iterdir():
        if not f.is_file():
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
            else:
                saved += 1
        except Exception:
            pass
    return f"pruned briefs: deleted={deleted}, kept={saved}"


def task_brain_inventory() -> str:
    """Walk Brain vault, log file count + total size."""
    d = _brain_dir()
    if not d.exists():
        return "no Brain vault"
    n_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(d):
        # Skip Obsidian / git internals to keep the walk cheap.
        if "/.obsidian" in root or "/.git" in root:
            dirs[:] = []
            continue
        for name in files:
            if name.endswith((".md", ".txt")):
                n_files += 1
                try:
                    total_bytes += (Path(root) / name).stat().st_size
                except Exception:
                    pass
    return f"Brain: {n_files} files, {total_bytes // 1024} KB"


def task_keepwarm_ollama() -> str:
    """Ping the chat model with a 1-token completion to keep KV warm.

    NOTE: there's an existing dedicated thread ``_keepwarm_ollama_loop``
    in ``listening/warmup.py`` that fires every 240 s with the FULL
    system prompt (so the KV-cache prefix stays primed). This task
    runs only as an OPT-IN failsafe — it's NOT registered by default
    anymore (see ``_register_default_tasks``). Kept as a callable so
    operators can re-enable it after a config crisis.

    If you re-register: pick an interval >= 600 s so it doesn't race
    with the dedicated thread (which would double the load on Hetzner
    without any extra benefit).
    """
    # Read URL + model from the same config the daemon uses.
    try:
        from ..config import load_settings  # type: ignore
        cfg = load_settings()
        url = str(getattr(cfg, "ollama_base_url", "")).rstrip("/")
        model = str(getattr(cfg, "ollama_chat_model", ""))
    except Exception:
        # If config import fails (very unlikely — module would not load
        # at all then), skip silently.
        return "config unavailable"
    if not url or not model:
        return "ollama not configured"
    payload = json.dumps({
        "model": model,
        "prompt": " ",  # noop prompt
        "stream": False,
        "keep_alive": "24h",
        "options": {"num_predict": 1, "temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            resp.read()
    except urllib.error.URLError as exc:
        return f"keepwarm skipped: {exc}"
    except Exception as exc:
        return f"keepwarm error: {exc}"
    ms = int((time.monotonic() - t0) * 1000)
    return f"keepwarm {model}: {ms} ms"


# ─── Default registration ────────────────────────────────────────────────


_DEFAULT_TASKS_REGISTERED = False


def _register_default_tasks() -> None:
    """Idempotent — called by ``start()``; tests can call register_task
    BEFORE start() to override / supplement the defaults."""
    global _DEFAULT_TASKS_REGISTERED
    if _DEFAULT_TASKS_REGISTERED:
        return
    _DEFAULT_TASKS_REGISTERED = True
    register_task(
        "rotate_events_jsonl",
        task_rotate_events_jsonl,
        interval_sec=3600.0,
    )
    register_task(
        "vacuum_audit_db",
        task_vacuum_audit_db,
        interval_sec=6 * 3600.0,
    )
    register_task(
        "prune_briefs",
        task_prune_briefs,
        daily_hhmm="03:00",
    )
    register_task(
        "brain_inventory",
        task_brain_inventory,
        daily_hhmm="04:00",
    )
    # NOTE: ``keepwarm_ollama`` intentionally NOT registered by default
    # — the dedicated ``_keepwarm_ollama_loop`` thread in warmup.py
    # already handles this with the full system prompt, which keeps
    # the KV-cache prefix primed. Registering this task too would
    # double the Hetzner Ollama load with no extra benefit.

"""Audit trail — durable SQLite log of voice events + tool calls.

Inspired by KAOS ``kronos/audit.py`` + the dashboard ``audit-trail/``
API. The events.jsonl stream is great for live tailing but
horrible for querying ("show me failed `focus_app` calls in the
last 24h"). SQLite gives us proper filtering, time-range queries,
and aggregations.

Schema:

    audit_events (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      ts          REAL NOT NULL,        -- unix epoch seconds, float
      kind        TEXT NOT NULL,        -- 'tool_call' | 'wake' | 'state' | 'voice_turn' | 'error'
      tool        TEXT,                 -- mac_control op / skill / etc.
      status      TEXT,                 -- 'starting' | 'completed' | 'failed' | 'info'
      args_json   TEXT,                 -- redacted JSON args dict
      result_json TEXT,                 -- truncated JSON result
      error       TEXT,
      duration_ms INTEGER,
      lang        TEXT,
      session_id  TEXT                  -- voice-turn grouping
    )

The writer is append-only + non-blocking — emits are queued and
flushed by a background thread. Even a hung disk doesn't block the
voice loop.

The reader is read-only + indexed on (kind, ts). Sub-millisecond
queries up to ~100k rows.

PII safety:

* ``args_json`` and ``result_json`` are passed through ``_redact``
  before insert. We strip values for keys named like ``token``,
  ``password``, ``secret``, ``apikey``, ``cookie``.
* The DB file is 0o600. Even if ``~/Library/Application Support``
  permissions slip, the file itself is owner-readable only.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger("jarvis.audit")


# ─────────────────────── paths ──────────────────────────


def _default_db_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/jarvis/audit.db"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".local/share")
    return base / "jarvis/audit.db"


# ─────────────────────── PII redaction ──────────────────────────


_SENSITIVE_KEY_PATTERNS = (
    "password",
    "passwd",
    "secret",
    "token",
    "apikey",
    "api_key",
    "cookie",
    "authorization",
    "bearer",
    "credential",
    "private_key",
    "ssh_key",
)


# Value-shape patterns — catch credentials embedded in free-text
# fields (e.g. ``"notes": "the key is sk-proj-..."``). The same set
# as ``memory.facts._REDACT_VALUE_RE`` so behaviour is consistent.
import re as _re  # noqa: E402
_VALUE_REDACT_RE = _re.compile(
    r"(?:sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|"
    r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|"
    r"Bearer\s+[A-Za-z0-9_\-\.]{8,})",
)


def _redact(obj: Any, _depth: int = 0) -> Any:
    """Deep-walk a JSON-able value, redacting sensitive keys.

    Caps depth at 16 levels (defends against cyclic refs in custom
    objects that managed to serialise). Two redaction passes:

      1. Key-driven: any dict key matching :data:`_SENSITIVE_KEY_PATTERNS`
         (substring match, case-insensitive) → value becomes ``"<redacted>"``.
      2. Value-shape: known credential token shapes inside any string
         (API keys, JWTs, bearer headers) collapse to ``"<redacted>"``.
    """
    if _depth > 16:
        return "<redacted: too deep>"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(p in kl for p in _SENSITIVE_KEY_PATTERNS):
                out[str(k)] = "<redacted>"
            else:
                out[str(k)] = _redact(v, _depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact(v, _depth + 1) for v in obj]
    if isinstance(obj, str):
        s = obj
        # Pattern scrub — cheap regex, runs even on short strings.
        if "sk-" in s or "ghp_" in s or "eyJ" in s or "Bearer" in s:
            s = _VALUE_REDACT_RE.sub("<redacted>", s)
        # Cap raw strings at 4 KB so a giant tool result doesn't blow
        # up the DB. Anything longer is informational only.
        if len(s) > 4096:
            return s[:4096] + f"…<+{len(s) - 4096} chars truncated>"
        return s
    return obj


def _to_json(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(_redact(obj), ensure_ascii=False, default=str)
    except Exception as exc:
        log.warning("audit json failed: %s", exc)
        return None


# ─────────────────────── dataclass ──────────────────────────


@dataclass
class AuditEvent:
    id: int
    ts: float
    kind: str
    tool: Optional[str]
    status: Optional[str]
    args: Optional[dict]
    result: Optional[Any]
    error: Optional[str]
    duration_ms: Optional[int]
    lang: Optional[str]
    session_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "ts": self.ts,
            "ts_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.gmtime(self.ts),
            ) + f".{int((self.ts % 1) * 1000):03d}Z",
            "kind": self.kind,
            "tool": self.tool,
            "status": self.status,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "lang": self.lang,
            "session_id": self.session_id,
        }


# ─────────────────────── store ──────────────────────────


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS audit_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  ts           REAL NOT NULL,
  kind         TEXT NOT NULL,
  tool         TEXT,
  status       TEXT,
  args_json    TEXT,
  result_json  TEXT,
  error        TEXT,
  duration_ms  INTEGER,
  lang         TEXT,
  session_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_kind_ts ON audit_events(kind, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_tool_ts ON audit_events(tool, ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_events(status);
"""


class AuditStore:
    """Background-thread-backed audit writer + reader.

    Public API:

    * :meth:`emit` — non-blocking; queues an event for the writer thread.
    * :meth:`query` — synchronous read with filters; returns AuditEvent[].
    * :meth:`stats` — quick aggregates for the dashboard overview.
    * :meth:`close` — flushes the queue + closes the connection.

    Threading model:

    The writer thread owns ``self._wconn``. Every emit pushes a tuple
    into ``self._q``; the writer drains the queue in batches of up to
    100 events at a time, committing once per batch — under ~5ms even
    on a slow disk. Read queries open their own short-lived connection
    so they don't contend with the writer (WAL handles isolation).
    """

    _DRAIN_BATCH = 100
    _DRAIN_IDLE_MS = 250  # max sleep between drains

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Initialise schema on a one-shot connection so subsequent
        # opens are guaranteed to see the tables/indexes.
        with sqlite3.connect(str(self._path), isolation_level=None) as conn:
            conn.executescript(_SCHEMA)
        try:
            self._path.chmod(0o600)
        except OSError:
            pass

        self._q: queue.Queue = queue.Queue(maxsize=10_000)
        self._closed = False
        self._wconn: Optional[sqlite3.Connection] = None
        self._thread = threading.Thread(
            target=self._writer_loop, name="audit-writer", daemon=True
        )
        self._thread.start()

    # ---------------------------------------------------- write side --
    def emit(
        self,
        *,
        kind: str,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        args: Optional[dict] = None,
        result: Any = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        lang: Optional[str] = None,
        session_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Non-blocking append. Drops with a log warning if queue is full."""
        if self._closed:
            return
        record = (
            ts if ts is not None else time.time(),
            kind,
            tool,
            status,
            _to_json(args),
            _to_json(result),
            error,
            duration_ms,
            lang,
            session_id,
        )
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # Worst case: voice loop runs fast enough to overflow the
            # writer. Drop oldest so newest still lands. The DB is
            # paginated history; missing a few events under saturation
            # is acceptable.
            try:
                self._q.get_nowait()
                self._q.put_nowait(record)
                log.warning("audit queue full; dropped oldest")
            except queue.Empty:
                pass

    # Sentinel object pushed on close() so the writer loop unblocks
    # from queue.get() promptly. None is fine because real records
    # are always tuples.
    _CLOSE_SENTINEL = None

    def _writer_loop(self) -> None:
        try:
            self._wconn = sqlite3.connect(
                str(self._path), isolation_level=None, check_same_thread=False
            )
            self._wconn.execute("PRAGMA journal_mode=WAL")
            self._wconn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as exc:
            log.error("audit writer cannot open DB: %s", exc)
            return

        while True:
            batch: list[tuple] = []
            try:
                first = self._q.get(timeout=self._DRAIN_IDLE_MS / 1000)
            except queue.Empty:
                if self._closed:
                    break  # closed + drained
                continue
            if first is self._CLOSE_SENTINEL:
                # Drain anything queued before the sentinel, then exit.
                while True:
                    try:
                        item = self._q.get_nowait()
                        if item is self._CLOSE_SENTINEL:
                            continue
                        batch.append(item)
                    except queue.Empty:
                        break
                if batch:
                    self._flush_batch(batch)
                break
            batch.append(first)
            while len(batch) < self._DRAIN_BATCH:
                try:
                    nxt = self._q.get_nowait()
                    if nxt is self._CLOSE_SENTINEL:
                        # Honour close: flush what we have + exit
                        if batch:
                            self._flush_batch(batch)
                        try:
                            if self._wconn is not None:
                                self._wconn.close()
                        except sqlite3.Error:
                            pass
                        return
                    batch.append(nxt)
                except queue.Empty:
                    break
            self._flush_batch(batch)

        try:
            if self._wconn is not None:
                self._wconn.close()
        except sqlite3.Error:
            pass

    def _flush_batch(self, batch: list[tuple]) -> None:
        try:
            assert self._wconn is not None
            self._wconn.executemany(
                """INSERT INTO audit_events
                   (ts, kind, tool, status, args_json, result_json,
                    error, duration_ms, lang, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch,
            )
        except (sqlite3.Error, AssertionError) as exc:
            log.warning("audit insert failed (batch %d): %s", len(batch), exc)

    def close(self, *, timeout: float = 2.0) -> None:
        """Flush + stop the writer.

        Pushes a sentinel so the writer drains queued events and
        exits promptly. Joins the writer thread with ``timeout``;
        if it doesn't exit, we just bail (DB writes are durable
        per-batch anyway thanks to autocommit + WAL).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._q.put_nowait(self._CLOSE_SENTINEL)
        except queue.Full:
            pass
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass

    # ---------------------------------------------------- read side --
    def _read_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def query(
        self,
        *,
        kind: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since_ts: Optional[float] = None,
        until_ts: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if tool is not None:
            clauses.append("tool = ?")
            params.append(tool)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        if until_ts is not None:
            clauses.append("ts <= ?")
            params.append(float(until_ts))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, ts, kind, tool, status, args_json, result_json, "
            "       error, duration_ms, lang, session_id "
            "  FROM audit_events"
            f"{where} "
            " ORDER BY ts DESC, id DESC "
            " LIMIT ? OFFSET ?"
        )
        params.extend([int(limit), int(offset)])

        with self._read_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        out: list[AuditEvent] = []
        for r in rows:
            try:
                args = json.loads(r["args_json"]) if r["args_json"] else None
            except (json.JSONDecodeError, TypeError):
                args = None
            try:
                result = json.loads(r["result_json"]) if r["result_json"] else None
            except (json.JSONDecodeError, TypeError):
                result = None
            out.append(
                AuditEvent(
                    id=r["id"],
                    ts=r["ts"],
                    kind=r["kind"],
                    tool=r["tool"],
                    status=r["status"],
                    args=args,
                    result=result,
                    error=r["error"],
                    duration_ms=r["duration_ms"],
                    lang=r["lang"],
                    session_id=r["session_id"],
                )
            )
        return out

    def stats(self, *, since_ts: Optional[float] = None) -> dict:
        """Aggregate counts for the dashboard overview."""
        clauses: list[str] = []
        params: list[Any] = []
        if since_ts is not None:
            clauses.append("ts >= ?")
            params.append(float(since_ts))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._read_conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM audit_events{where}", params
            ).fetchone()["n"]
            by_kind = {
                row["kind"]: row["n"]
                for row in conn.execute(
                    f"SELECT kind, COUNT(*) AS n FROM audit_events{where}"
                    " GROUP BY kind",
                    params,
                ).fetchall()
            }
            by_status = {
                row["status"]: row["n"]
                for row in conn.execute(
                    f"SELECT status, COUNT(*) AS n FROM audit_events{where}"
                    " GROUP BY status",
                    params,
                ).fetchall()
            }
            top_tools = [
                {"tool": row["tool"], "n": row["n"]}
                for row in conn.execute(
                    f"SELECT tool, COUNT(*) AS n FROM audit_events"
                    f"{where}{' AND ' if where else ' WHERE '}tool IS NOT NULL"
                    " GROUP BY tool ORDER BY n DESC LIMIT 10",
                    params,
                ).fetchall()
            ]
        return {
            "total": total,
            "by_kind": by_kind,
            "by_status": by_status,
            "top_tools": top_tools,
        }

    @property
    def path(self) -> Path:
        return self._path


# ─────────────────────── singleton ──────────────────────────


_INSTANCE: Optional[AuditStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_audit_store() -> AuditStore:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = AuditStore()
    return _INSTANCE


def reset_audit_store_singleton() -> None:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            try:
                _INSTANCE.close()
            except Exception:
                pass
        _INSTANCE = None

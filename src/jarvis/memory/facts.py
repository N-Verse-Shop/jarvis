"""Atomic facts store — persistent context fragments about the user.

Ported from KAOS ``kronos/memory/facts.py``. Companion to (NOT
replacement for) ``conversation_summaries``:

  conversation_summaries  →  daily diary, "what happened on day X"
  memory_facts            →  long-lived traits, "who the user IS"

A *fact* is an atomic statement that survives across sessions and is
injected into the chat system prompt for personalisation:

  ("user.preference",  "drinks black coffee, no sugar")
  ("user.location",    "Rehburg-Loccum, Germany")
  ("user.work",        "CEO of Nexus Studio, DACH B2B agency")
  ("user.tech.likes",  "Cloudflare, Shopify Hydrogen, React Native")

Each fact carries:

  - ``key``         — dotted namespace, free-form.
  - ``value``       — the actual statement, one sentence.
  - ``source``      — who created it (``"voice"``, ``"manual"``,
                      ``"skill:<name>"``, ``"chat"``).
  - ``confidence``  — 0.0..1.0 from extractor.
  - ``ts_utc``      — when it was learned (REAL epoch seconds).
  - ``last_used``   — when it was last surfaced to the chat layer.
  - ``hits``        — total times it has been recalled.

Retrieval ranking is **time-decayed**::

    score = confidence
          * exp(-age_days / half_life_days)
          * log1p(hits)            # mild recency boost
          * fts_match_bonus(query) # if a query is supplied

Old, unused facts naturally lose to fresh ones. A nightly sweep
prunes anything whose score drops below ``min_score`` so the table
doesn't grow unbounded.

Schema
------

::

    facts (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      key         TEXT    NOT NULL,
      value       TEXT    NOT NULL,
      source      TEXT    NOT NULL,
      confidence  REAL    NOT NULL DEFAULT 1.0,
      ts_utc      REAL    NOT NULL,
      last_used   REAL    NOT NULL,
      hits        INTEGER NOT NULL DEFAULT 0,
      tombstone   INTEGER NOT NULL DEFAULT 0    -- soft delete
    )

    CREATE UNIQUE INDEX facts_key_value ON facts(key, value);
    CREATE INDEX facts_key             ON facts(key);
    CREATE INDEX facts_ts              ON facts(ts_utc DESC);

    -- FTS5 over (key, value) for fuzzy retrieval.
    facts_fts (key, value, content='facts', content_rowid='id')

Privacy
-------

Facts are stored in ``~/.local/share/jarvis/facts.db`` with file mode
``0o600``. Values pass through the same ``_redact`` scrubber as
``audit.py`` before write — any string that looks like a key/token
field collapses to ``"<redacted>"``.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


# Default decay parameters — calibrated for personal-assistant use.
# A fact loses half its weight after 30 days of non-use.
DEFAULT_HALF_LIFE_DAYS = 30.0
# Anything below this is pruned by the nightly sweep.
DEFAULT_MIN_SCORE = 0.05


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS facts (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  key         TEXT    NOT NULL,
  value       TEXT    NOT NULL,
  source      TEXT    NOT NULL,
  confidence  REAL    NOT NULL DEFAULT 1.0,
  ts_utc      REAL    NOT NULL,
  last_used   REAL    NOT NULL,
  hits        INTEGER NOT NULL DEFAULT 0,
  tombstone   INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS facts_key_value ON facts(key, value);
CREATE INDEX IF NOT EXISTS facts_key   ON facts(key);
CREATE INDEX IF NOT EXISTS facts_ts    ON facts(ts_utc DESC);
CREATE INDEX IF NOT EXISTS facts_score ON facts(tombstone, ts_utc);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  key, value,
  content='facts',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES('delete', old.id, old.key, old.value);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, key, value) VALUES('delete', old.id, old.key, old.value);
  INSERT INTO facts_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
"""


# ---- redaction (same pattern as audit.py) ----

_REDACT_KEY_RE = re.compile(
    r"(password|passwd|pwd|secret|token|apikey|api[_\-]?key|bearer|"
    r"cookie|authorization|credential|private[_\-]?key|ssh[_\-]?key)",
    re.IGNORECASE,
)
_REDACT_VALUE_RE = re.compile(
    # API keys: sk-..., sk-proj-..., ghp_..., JWTs, bearer tokens.
    # Allow internal dashes (sk-proj-XXX is common) — match up to the
    # next whitespace boundary.
    r"(?:sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|"
    r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}|"
    r"Bearer\s+[A-Za-z0-9_\-\.]{8,})",
)


def _redact(text: str, key: str = "") -> str:
    """Best-effort secret scrubbing for fact values.

    1. If the *key* mentions a credential field, redact the whole value.
       (The caller is doing ``add_fact("user.api_key", "abc")`` — the
        whole value is sensitive regardless of its shape.)
    2. Otherwise pattern-match known credential shapes inside the value.

    Always returns a string, never raises.
    """
    if not text:
        return text
    # Key-driven hard redact.
    if key and _REDACT_KEY_RE.search(key):
        return "<redacted>"
    # Value-shape based.
    return _REDACT_VALUE_RE.sub("<redacted>", text)


def _default_path() -> Path:
    base = os.environ.get("JARVIS_DATA_DIR")
    if base:
        return Path(base).expanduser() / "facts.db"
    return Path.home() / ".local/share/jarvis/facts.db"


# ---- dataclass ----


@dataclass(frozen=True)
class Fact:
    id: int
    key: str
    value: str
    source: str
    confidence: float
    ts_utc: float
    last_used: float
    hits: int

    def score(
        self,
        *,
        now: Optional[float] = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> float:
        """Decay-weighted score for ranking. Pure function — no side
        effects on the fact itself.

        Components:

          - ``confidence``     — extractor's belief.
          - exponential decay  — ``exp(-age / half_life)`` where
                                 ``age`` is days since ``last_used``.
          - log(hits + 1)      — gentle bump for recurring facts.
        """
        ref_ts = now if now is not None else time.time()
        age_days = max(0.0, (ref_ts - self.last_used) / 86400.0)
        # True half-life: 2^(-age/half_life) — score halves every
        # ``half_life_days``. exp(-age/half_life) would be the
        # exponential time-constant form (drops to ~37% per period).
        decay = math.pow(2.0, -age_days / max(half_life_days, 0.001))
        recurrence = math.log1p(self.hits)  # 0 for fresh, ~3 for 20 hits
        return self.confidence * decay * (1.0 + 0.2 * recurrence)


# ---- store ----


class FactStore:
    """Thread-safe SQLite-backed facts store.

    All public methods acquire ``self._lock`` — sqlite3 connections
    are not safe for concurrent writes from multiple threads even
    with ``check_same_thread=False``.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.path = path or _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._half_life_days = half_life_days
        self._min_score = min_score
        # Mode 0o600 — private user-data file.
        try:
            if not self.path.exists():
                self.path.touch(mode=0o600)
            else:
                os.chmod(self.path, 0o600)
        except OSError:
            pass
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage txns explicitly
        )
        self._conn.row_factory = sqlite3.Row
        # R34-S40 — explicit busy_timeout. Without this, concurrent
        # writes from the voice loop's _store_facts() and the dashboard's
        # CRUD endpoints can collide on SQLITE_BUSY (instant locked error)
        # because WAL doesn't serialise writers — only readers. 5s gives
        # any contending writer time to commit + release the journal.
        try:
            self._conn.execute("PRAGMA busy_timeout=5000")
        except Exception:
            pass
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ---- mutation ----

    def add(
        self,
        key: str,
        value: str,
        *,
        source: str = "manual",
        confidence: float = 1.0,
    ) -> int:
        """Insert OR refresh a fact.

        Idempotency: ``(key, value)`` is UNIQUE — re-adding the same
        statement bumps ``ts_utc`` + ``last_used`` and (slightly)
        raises confidence toward 1.0. This is exactly what we want
        when the user repeats themselves ("I told you, I drink
        BLACK coffee!").
        """
        key = (key or "").strip()
        value = _redact((value or "").strip(), key=key)
        if not key or not value:
            raise ValueError("key and value must be non-empty")
        confidence = max(0.0, min(1.0, float(confidence)))
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, confidence FROM facts WHERE key=? AND value=?",
                (key, value),
            )
            row = cur.fetchone()
            if row is None:
                cur = self._conn.execute(
                    "INSERT INTO facts(key, value, source, confidence, "
                    "                  ts_utc, last_used, hits, tombstone) "
                    "VALUES(?, ?, ?, ?, ?, ?, 0, 0)",
                    (key, value, source, confidence, now, now),
                )
                return int(cur.lastrowid)
            # Bump existing row.
            new_conf = min(1.0, row["confidence"] + (1.0 - row["confidence"]) * 0.5)
            self._conn.execute(
                "UPDATE facts SET ts_utc=?, last_used=?, confidence=?, "
                "                 tombstone=0 WHERE id=?",
                (now, now, new_conf, row["id"]),
            )
            return int(row["id"])

    def delete(self, fact_id: int) -> bool:
        """Soft delete (tombstone). Returns True if a row was hit."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE facts SET tombstone=1 WHERE id=? AND tombstone=0",
                (int(fact_id),),
            )
            return cur.rowcount > 0

    def forget_key(self, key: str) -> int:
        """Tombstone every fact under a key. Returns rows affected."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE facts SET tombstone=1 WHERE key=? AND tombstone=0",
                (key,),
            )
            return cur.rowcount

    # ---- retrieval ----

    def get(self, fact_id: int) -> Optional[Fact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE id=? AND tombstone=0",
                (int(fact_id),),
            ).fetchone()
        return _row_to_fact(row) if row else None

    def all(
        self,
        *,
        include_tombstones: bool = False,
        limit: int = 1000,
    ) -> List[Fact]:
        with self._lock:
            if include_tombstones:
                rows = self._conn.execute(
                    "SELECT * FROM facts ORDER BY ts_utc DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM facts WHERE tombstone=0 "
                    "ORDER BY ts_utc DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def by_key(self, key_prefix: str, limit: int = 100) -> List[Fact]:
        """All facts whose key starts with ``key_prefix``.

        Used by the prompt injector: ``store.by_key("user.")`` pulls
        every persistent user trait.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE tombstone=0 AND key LIKE ? "
                "ORDER BY ts_utc DESC LIMIT ?",
                (f"{key_prefix}%", limit),
            ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        key_prefix: Optional[str] = None,
    ) -> List[Fact]:
        """FTS5 search across key+value with decay-weighted ranking.

        Empty query falls back to ``by_key`` (or ``all`` if no prefix).
        FTS syntax errors are silently caught — we treat the query as
        a plain-text fallback and AND the tokens together.
        """
        q = (query or "").strip()
        if not q:
            return self.by_key(key_prefix or "", limit)
        fts_q = _normalise_fts_query(q)
        with self._lock:
            sql = (
                "SELECT facts.* FROM facts "
                "JOIN facts_fts ON facts.id = facts_fts.rowid "
                "WHERE facts_fts MATCH ? AND facts.tombstone=0"
            )
            params: List = [fts_q]
            if key_prefix:
                sql += " AND facts.key LIKE ?"
                params.append(f"{key_prefix}%")
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit * 4)  # over-fetch, rerank with decay
            try:
                rows = self._conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                # FTS syntax error → plain LIKE fallback.
                tokens = re.findall(r"\w+", q.lower())
                like = "%" + "%".join(tokens) + "%" if tokens else f"%{q}%"
                rows = self._conn.execute(
                    "SELECT * FROM facts WHERE tombstone=0 "
                    "AND (lower(key) LIKE ? OR lower(value) LIKE ?) "
                    "ORDER BY ts_utc DESC LIMIT ?",
                    (like, like, limit * 4),
                ).fetchall()
        facts = [_row_to_fact(r) for r in rows]
        now = time.time()
        facts.sort(
            key=lambda f: f.score(now=now, half_life_days=self._half_life_days),
            reverse=True,
        )
        return facts[:limit]

    def touch(self, fact_id: int) -> None:
        """Mark a fact as recently used — updates ``last_used`` + hits.

        Call this when a fact is surfaced to the chat layer so the
        decay clock resets.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET last_used=?, hits=hits+1 "
                "WHERE id=? AND tombstone=0",
                (time.time(), int(fact_id)),
            )

    def touch_many(self, fact_ids: Iterable[int]) -> None:
        ids = [int(i) for i in fact_ids]
        if not ids:
            return
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE facts SET last_used=?, hits=hits+1 "
                "WHERE id=? AND tombstone=0",
                [(now, i) for i in ids],
            )

    # ---- housekeeping ----

    def prune(
        self,
        *,
        min_score: Optional[float] = None,
        hard_age_days: float = 365.0,
    ) -> int:
        """Tombstone facts whose decay-weighted score has decayed below
        ``min_score`` OR whose age exceeds ``hard_age_days``.

        Returns the count of newly-tombstoned rows. Safe to call from
        a nightly cron — idempotent on already-tombstoned rows.
        """
        cutoff = min_score if min_score is not None else self._min_score
        now = time.time()
        hard_cutoff = now - hard_age_days * 86400.0
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, confidence, last_used, hits FROM facts WHERE tombstone=0"
            ).fetchall()
            to_prune: List[int] = []
            for row in rows:
                if row["last_used"] < hard_cutoff:
                    to_prune.append(int(row["id"]))
                    continue
                age_days = (now - row["last_used"]) / 86400.0
                score = (
                    row["confidence"]
                    * math.pow(2.0, -age_days / max(self._half_life_days, 0.001))
                    * (1.0 + 0.2 * math.log1p(row["hits"]))
                )
                if score < cutoff:
                    to_prune.append(int(row["id"]))
            if to_prune:
                self._conn.executemany(
                    "UPDATE facts SET tombstone=1 WHERE id=?",
                    [(i,) for i in to_prune],
                )
        return len(to_prune)

    def vacuum(self) -> None:
        """Hard-delete tombstoned rows + reclaim space. Call sparingly —
        VACUUM holds an exclusive lock.
        """
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE tombstone=1")
            self._conn.execute("VACUUM")

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) AS c FROM facts WHERE tombstone=0"
            ).fetchone()["c"]
            tomb = self._conn.execute(
                "SELECT COUNT(*) AS c FROM facts WHERE tombstone=1"
            ).fetchone()["c"]
            keys = self._conn.execute(
                "SELECT key, COUNT(*) AS c FROM facts WHERE tombstone=0 "
                "GROUP BY key ORDER BY c DESC LIMIT 10"
            ).fetchall()
        return {
            "active": total,
            "tombstoned": tomb,
            "top_keys": [{"key": r["key"], "count": r["c"]} for r in keys],
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ---- helpers ----


def _row_to_fact(row: sqlite3.Row) -> Fact:
    return Fact(
        id=int(row["id"]),
        key=str(row["key"]),
        value=str(row["value"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
        ts_utc=float(row["ts_utc"]),
        last_used=float(row["last_used"]),
        hits=int(row["hits"]),
    )


def _normalise_fts_query(raw: str) -> str:
    """Convert free-text into an FTS5-safe query.

    Strategy: tokenise on word chars, drop empties, join with space
    (implicit AND in FTS5). Each token gets a trailing ``*`` so prefix
    matches work — important for languages with rich morphology
    (Ukrainian, German).
    """
    tokens = re.findall(r"\w+", raw, re.UNICODE)
    if not tokens:
        return raw
    return " ".join(f"{t}*" for t in tokens if len(t) >= 2)


# ---- singleton accessor ----

_GLOBAL_STORE: Optional[FactStore] = None
_GLOBAL_LOCK = threading.Lock()


def get_facts_store() -> FactStore:
    """Process-wide singleton — opens the default DB on first call."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        return _GLOBAL_STORE
    with _GLOBAL_LOCK:
        if _GLOBAL_STORE is None:
            _GLOBAL_STORE = FactStore()
    return _GLOBAL_STORE


def render_facts_for_prompt(
    store: Optional[FactStore] = None,
    *,
    key_prefix: str = "user.",
    limit: int = 12,
    max_chars: int = 600,
) -> str:
    """Pack the top facts into a compact prompt block.

    Returns an empty string if there are no facts. Caller is expected
    to splice this into the system prompt between the persona and
    the skill catalogue. Format::

        Known about you:
        - drinks black coffee, no sugar
        - lives in Rehburg-Loccum, Germany
        - prefers Ukrainian for voice chat

    Decay-aware: facts are picked top-K by ``Fact.score`` so stale
    items naturally drop out. Touches the IDs we returned so they
    won't decay further before the next prompt build.
    """
    s = store or get_facts_store()
    facts = s.by_key(key_prefix, limit=limit * 3)
    if not facts:
        return ""
    now = time.time()
    facts.sort(
        key=lambda f: f.score(now=now, half_life_days=s._half_life_days),
        reverse=True,
    )
    picked: List[Fact] = []
    total = 0
    for f in facts:
        line_len = len(f.value) + 4
        if total + line_len > max_chars:
            break
        picked.append(f)
        total += line_len
        if len(picked) >= limit:
            break
    if not picked:
        return ""
    # R34-S52 H: was English "Known about you:" — injected as the
    # header on every voice turn's system prompt, biasing the model
    # toward English. RU-only persona means the header is RU too.
    lines = ["Что я знаю о тебе:"]
    # R34-S52 H: filter out facts that would re-inject a non-Russian
    # language preference into the prompt. Even after the
    # language_lock_block always returns RU, a stored
    # ``user.preference.voice_lang: uk`` fact rendered here gives the
    # model conflicting signals ("user wants UA"). We hide any
    # language-preference fact whose value is not "ru".
    def _is_ua_or_en_lang_pref(fact: Fact) -> bool:
        key_lower = (fact.key or "").lower()
        val_lower = (fact.value or "").lower()
        if "voice_lang" in key_lower or "language" in key_lower:
            if "ru" not in val_lower:
                return True
        return False
    # R34-S53.1 (Phase 6b — N-2): apply the filter BEFORE deciding which
    # facts to mark as touched. Previously ``touch_many`` ran on the
    # full ``picked`` list (line 613 in S52), then we filtered for the
    # output — which boosted the recency score of a UA voice-lang fact
    # the next render-cycle WOULDN'T even show. Net effect: filtered-out
    # rows kept rising in the FTS ranking, eventually evicting genuinely
    # useful facts. Touch only what we render.
    rendered: List[Fact] = []
    for f in picked:
        if _is_ua_or_en_lang_pref(f):
            continue
        rendered.append(f)
        lines.append(f"- {f.value}")
    if rendered:
        s.touch_many([f.id for f in rendered])
    if len(lines) == 1:
        return ""
    return "\n".join(lines)

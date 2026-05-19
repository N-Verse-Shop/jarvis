from __future__ import annotations
import sqlite3
import re
from typing import Sequence, Optional
from pathlib import Path
import threading
from datetime import datetime, timezone

from ..debug import debug_log

# Audit round 16 fix: ``date_utc`` is a UNIQUE-key component on the
# conversation_summaries table. The schema accepts any TEXT, so a caller
# that hands us " 2026-05-17", "2026-5-17", or even "today" creates a
# brand-new row instead of UPDATE-ing the existing one — silently breaking
# daily-flush idempotency and producing duplicate diary entries the user
# never sees. Validate once at the API boundary so the FK-cascade UPDATE
# path in ``upsert_conversation_summary`` actually fires.
_DATE_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date_utc(date_utc: str) -> str:
    """Validate a ``YYYY-MM-DD`` date string and return the canonical form.

    Raises ``ValueError`` if the input is not a real calendar date in that
    exact shape. Returns the input unchanged on success so callers can
    inline the check (``cur.execute(..., (_validate_date_utc(d), ...))``).
    """
    if not isinstance(date_utc, str) or not _DATE_UTC_RE.match(date_utc):
        raise ValueError(
            f"date_utc must be 'YYYY-MM-DD' (got {date_utc!r})"
        )
    # Reject 2026-13-40, 2026-02-30, etc.
    try:
        datetime.strptime(date_utc, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date_utc is not a valid calendar date: {date_utc!r}") from exc
    return date_utc


_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Structured meals log (optional feature)
CREATE TABLE IF NOT EXISTS meals (
  id            INTEGER PRIMARY KEY,
  ts_utc        TEXT NOT NULL,
  source_app    TEXT NOT NULL,
  description   TEXT NOT NULL,
  calories_kcal REAL,
  protein_g     REAL,
  carbs_g       REAL,
  fat_g         REAL,
  fiber_g       REAL,
  sugar_g       REAL,
  sodium_mg     REAL,
  potassium_mg  REAL,
  micros_json   TEXT,
  confidence    REAL
);

-- Conversation summaries for diary/memory system
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id         INTEGER PRIMARY KEY,
  date_utc   TEXT NOT NULL,  -- YYYY-MM-DD format
  ts_utc     TEXT NOT NULL,  -- When summary was created
  summary    TEXT NOT NULL,  -- Concise summary of the day's conversations
  topics     TEXT,           -- Comma-separated list of main topics discussed
  source_app TEXT NOT NULL,  -- Source app that generated the conversation
  UNIQUE(date_utc, source_app)
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
  summary,
  topics,
  content='conversation_summaries',
  content_rowid='id',
  tokenize='porter'
);

-- Triggers for conversation summaries FTS
CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(rowid, summary, topics) VALUES (new.id, new.summary, new.topics);
END;
CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(summaries_fts, rowid, summary, topics) VALUES('delete', old.id, old.summary, old.topics);
END;
CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(summaries_fts, rowid, summary, topics) VALUES('delete', old.id, old.summary, old.topics);
  INSERT INTO summaries_fts(rowid, summary, topics) VALUES (new.id, new.summary, new.topics);
END;
"""

_VSS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vss0(
  id INTEGER PRIMARY KEY,
  vec FLOAT[768]
);

CREATE TABLE IF NOT EXISTS summary_vec (
  summary_id INTEGER PRIMARY KEY REFERENCES conversation_summaries(id) ON DELETE CASCADE,
  emb_id     INTEGER NOT NULL REFERENCES embeddings(id)
);
"""


def _normalize_fts_query(raw: str) -> str:
    # Use improved fuzzy search query generation
    try:
        from .fuzzy_search import generate_flexible_fts_query
        flexible_query = generate_flexible_fts_query(raw)
        if flexible_query:
            return flexible_query
    except ImportError:
        pass
    
    # Fallback: Extract alphanumeric tokens and join them with spaces (logical AND)
    tokens = re.findall(r"[A-Za-z0-9_]+", raw)
    return " ".join(tokens)


class Database:
    def __init__(self, db_path: str, sqlite_vss_path: Optional[str] = None) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.is_vss_enabled = False
        self._python_vector_store = None
        
        if sqlite_vss_path:
            try:
                self.conn.enable_load_extension(True)
                self.conn.load_extension(sqlite_vss_path)
                self.is_vss_enabled = True
            except Exception:
                self.is_vss_enabled = False
        
        # If sqlite-vss is not available, use best available vector store (FAISS or Python fallback)
        if not self.is_vss_enabled:
            from ..utils.vector_store import get_best_vector_store
            self._python_vector_store = get_best_vector_store(db_path, dimension=768)
            
            # Log which vector store implementation is being used
            import sys
            store_type = type(self._python_vector_store).__name__
            if store_type == "FAISSVectorStore":
                debug_log("Using FAISS vector store for fast search", "jarvis")
            else:
                debug_log("Using Python fallback vector store", "jarvis")
        
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            # Audit round 16 fix: set busy_timeout BEFORE schema init.
            # The diary flush worker, foreground summary writes, the
            # meals/nutrition tool, and the FAISS sidecar all operate
            # against this connection's database file. Without
            # busy_timeout SQLite returns SQLITE_BUSY instantly on any
            # writer contention — the writer silently drops the write
            # and the user loses (e.g.) a logged meal. 5s matches the
            # graph store's round-14 setting and is well above any
            # realistic single-statement duration.
            cur.execute("PRAGMA busy_timeout = 5000")
            cur.executescript(_SCHEMA_SQL)
            if self.is_vss_enabled:
                cur.executescript(_VSS_SCHEMA_SQL)
            self.conn.commit()

    

    def search_hybrid(self, fts_query: str, query_vec_json: Optional[str], top_k: int = 8) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.cursor()
            safe_q = _normalize_fts_query(fts_query)

            # Use Python vector store if sqlite-vss is not available
            if not self.is_vss_enabled and self._python_vector_store and query_vec_json is not None and safe_q:
                # Parse query vector
                import json as _json
                query_vec = _json.loads(query_vec_json)
                
                # Get vector search results (use max of top_k*3 and 50 for good hybrid scoring)
                vector_search_limit = max(top_k * 3, 50)
                vector_results = self._python_vector_store.search(query_vec, top_k=vector_search_limit)
                
                # Get FTS results (use max of top_k*3 and 50 for good hybrid scoring).
                # Audit round 20 P2 — bind the LIMIT via parameter,
                # not f-string. The value is computed locally so this
                # is not an injection today, but the shape is a latent
                # footgun: any future refactor that lets a caller
                # supply ``top_k`` from a tool argument would
                # immediately produce a real SQL injection here.
                # SQLite supports parameterised LIMIT (``?``) since
                # 3.8.0 — no functional cost.
                fts_search_limit = max(top_k * 3, 50)
                fts_sql = """
                SELECT s.id, bm25(summaries_fts) AS bm
                FROM summaries_fts
                JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY bm
                LIMIT ?
                """
                fts_rows = cur.execute(fts_sql, (safe_q, int(fts_search_limit))).fetchall()
                fts_scores = {row['id']: row['bm'] for row in fts_rows}
                
                # Combine scores
                combined_scores = {}
                
                # Add vector scores (60% weight)
                for summary_id, distance in vector_results:
                    combined_scores[summary_id] = (1.0 / (1.0 + distance)) * 0.6
                
                # Add FTS scores (40% weight)
                for summary_id, bm_score in fts_scores.items():
                    if summary_id in combined_scores:
                        combined_scores[summary_id] += (1.0 / (1.0 + bm_score)) * 0.4
                    else:
                        combined_scores[summary_id] = (1.0 / (1.0 + bm_score)) * 0.4
                
                # Sort by combined score and fetch summaries
                sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                if sorted_ids:
                    # Fetch summaries for top results
                    placeholders = ','.join('?' * len(sorted_ids))
                    summary_sql = f"""
                    SELECT s.id, 
                           '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                           'summary' AS result_type
                    FROM conversation_summaries s
                    WHERE s.id IN ({placeholders})
                    """
                    rows = cur.execute(summary_sql, [sid for sid, _ in sorted_ids]).fetchall()
                    
                    # Create result rows with scores
                    results = []
                    id_to_score = {sid: score for sid, score in sorted_ids}
                    for row in rows:
                        # Create a new row dict with score
                        result = dict(row)
                        result['score'] = id_to_score.get(row['id'], 0.0)
                        results.append(result)
                    
                    # Sort by score again (in case DB returned in different order)
                    results.sort(key=lambda x: x['score'], reverse=True)
                    return results
                else:
                    return []
                    
            elif self.is_vss_enabled and query_vec_json is not None and safe_q:
                # Hybrid search: 60% vector similarity (semantic) + 40% FTS (exact terms)
                # This balances finding semantically related content with keyword matches
                # Use dynamic limits for efficiency on large datasets.
                # Audit round 20 P2 — bind every LIMIT as a parameter
                # (was f-string). Today ``top_k`` and ``search_limit``
                # are ints by code-path; a later refactor that lets a
                # caller pass a string would otherwise produce a
                # latent SQL-injection.
                search_limit = int(max(top_k * 3, 50))
                summary_sql = """
                WITH fts_sum AS (
                  SELECT s.id, bm25(summaries_fts) AS bm
                  FROM summaries_fts
                  JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                  WHERE summaries_fts MATCH ?
                  ORDER BY bm LIMIT ?
                ),
                v_sum AS (
                  SELECT sv.summary_id AS id, distance
                  FROM vss_search(embeddings, 'vec', ?)
                  JOIN summary_vec sv ON sv.emb_id = rowid
                  LIMIT ?
                )
                SELECT s.id, (
                    (1.0/(1.0+COALESCE(v_sum.distance, 1))) * 0.6 +
                    (1.0/(1.0+COALESCE(fts_sum.bm, 10))) * 0.4
                  ) AS score,
                  '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                  'summary' AS result_type
                FROM conversation_summaries s
                LEFT JOIN v_sum     ON v_sum.id = s.id
                LEFT JOIN fts_sum   ON fts_sum.id = s.id
                WHERE v_sum.id IS NOT NULL OR fts_sum.id IS NOT NULL
                ORDER BY score DESC
                LIMIT ?;
                """
                rows = cur.execute(
                    summary_sql,
                    (safe_q, search_limit, query_vec_json, search_limit, int(top_k)),
                ).fetchall()

            elif safe_q:
                # FTS-only search over conversation summaries.
                # Audit round 20 P2 — parameterised LIMIT (see above).
                summary_sql = """
                SELECT s.id, bm25(summaries_fts) AS score,
                       '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                       'summary' AS result_type
                FROM summaries_fts
                JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY score
                LIMIT ?;
                """
                rows = cur.execute(summary_sql, (safe_q, int(top_k))).fetchall()

            else:
                # Fallback: latest conversation summaries.
                # Audit round 20 P2 — parameterised LIMIT.
                summary_sql = """
                SELECT id, 0.0 AS score,
                       '[' || date_utc || '] ' || summary || ' (Topics: ' || COALESCE(topics, '') || ')' AS text,
                       'summary' AS result_type
                FROM conversation_summaries
                ORDER BY date_utc DESC
                LIMIT ?;
                """
                rows = cur.execute(summary_sql, (int(top_k),)).fetchall()

            return rows

    @staticmethod
    def _pack_vector(vec: Sequence[float]) -> bytes:
        # SQLite-vss expects a float array; packing via array('f') ensures binary blob layout.
        import array
        arr = array.array('f', [float(x) for x in vec])
        return arr.tobytes()

    # --- Meals API ---
    def insert_meal(
        self,
        ts_utc: str,
        source_app: str,
        description: str,
        calories_kcal: Optional[float] = None,
        protein_g: Optional[float] = None,
        carbs_g: Optional[float] = None,
        fat_g: Optional[float] = None,
        fiber_g: Optional[float] = None,
        sugar_g: Optional[float] = None,
        sodium_mg: Optional[float] = None,
        potassium_mg: Optional[float] = None,
        micros_json: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO meals(ts_utc, source_app, description, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, potassium_mg, micros_json, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    source_app,
                    description,
                    calories_kcal,
                    protein_g,
                    carbs_g,
                    fat_g,
                    fiber_g,
                    sugar_g,
                    sodium_mg,
                    potassium_mg,
                    micros_json,
                    confidence,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_meals_between(self, ts_utc_min: str, ts_utc_max: str) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM meals
                WHERE ts_utc >= ? AND ts_utc <= ?
                ORDER BY ts_utc ASC
                """,
                (ts_utc_min, ts_utc_max),
            ).fetchall()
            return rows

    def delete_meal(self, meal_id: int) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM meals WHERE id = ?", (meal_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # --- Conversation Summaries API ---
    def upsert_conversation_summary(
        self,
        date_utc: str,  # YYYY-MM-DD format
        summary: str,
        topics: Optional[str] = None,
        source_app: str = "jarvis",
        ts_utc: Optional[str] = None,
    ) -> int:
        """Insert or update a conversation summary for a given date.

        ``ts_utc`` defaults to "now". Maintenance ops that rewrite an
        existing row's content without changing what it represents (e.g.
        the deflection scrub bulk sweep) should pass through the row's
        original ``ts_utc`` so the audit trail is preserved.
        """
        # Audit round 16 fix: reject malformed ``date_utc`` BEFORE we
        # take the write lock or hit the DB. Whitespace, two-digit
        # months, or non-dates like ``"today"`` would each create a
        # duplicate UNIQUE-key row that breaks daily-flush idempotency.
        date_utc = _validate_date_utc(date_utc)
        if ts_utc is None:
            ts_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            # Audit round 10 fix C1: `INSERT OR REPLACE` deletes the
            # existing row (firing FTS delete trigger AND cascading
            # `summary_vec.summary_id` → ON DELETE removes embedding),
            # then inserts a fresh autoincrement id. Result: every
            # daily diary re-flush dropped + recreated the embedding
            # row → forced the FAISS index to rebuild from scratch
            # → continuous O(N) churn for a quiet feature.
            # `ON CONFLICT DO UPDATE` preserves the rowid, keeping
            # the FK + FTS rows valid, so the lazy-rebuild path in
            # `fast_vector_store.add_vector` no longer fires on dupes.
            cur.execute(
                """
                INSERT INTO conversation_summaries(date_utc, ts_utc, summary, topics, source_app)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(date_utc, source_app) DO UPDATE SET
                    ts_utc = excluded.ts_utc,
                    summary = excluded.summary,
                    topics = excluded.topics
                """,
                (date_utc, ts_utc, summary, topics, source_app),
            )
            self.conn.commit()
            # `lastrowid` returns 0 on the UPDATE path of ON CONFLICT;
            # resolve the actual id via lookup. This also matters for
            # the caller that hands the id straight to
            # `upsert_summary_embedding(summary_id, vec)`.
            row = cur.execute(
                "SELECT id FROM conversation_summaries WHERE date_utc=? AND source_app=?",
                (date_utc, source_app),
            ).fetchone()
            if row is None:
                # Insert genuinely failed — fall back so caller doesn't
                # crash on `None.lastrowid`.
                return int(cur.lastrowid) if cur.lastrowid else 0
            return int(row["id"])

    def get_conversation_summary(self, date_utc: str, source_app: str = "jarvis") -> Optional[sqlite3.Row]:
        """Get conversation summary for a specific date."""
        # Audit round 16 fix: same validation as upsert — keep the
        # read side in lock-step with the write side so a typo in a
        # caller surfaces immediately rather than as a silent miss.
        date_utc = _validate_date_utc(date_utc)
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute(
                """
                SELECT * FROM conversation_summaries
                WHERE date_utc = ? AND source_app = ?
                """,
                (date_utc, source_app),
            ).fetchone()
            return row

    def get_recent_conversation_summaries(self, days: int = 7) -> list[sqlite3.Row]:
        """Get conversation summaries from the last N days."""
        from datetime import datetime, timedelta, timezone
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_summaries
                WHERE date_utc >= ?
                ORDER BY date_utc DESC
                """,
                (cutoff_date,),
            ).fetchall()
            return rows

    def get_all_conversation_summaries(self) -> list[sqlite3.Row]:
        """Get all conversation summaries, ordered by date ascending (oldest first).

        Used for bulk import into graph memory — processes diary entries
        chronologically so the graph builds up naturally.
        """
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_summaries
                ORDER BY date_utc ASC
                """,
            ).fetchall()
            return rows

    def upsert_summary_embedding(self, summary_id: int, vec: Sequence[float]) -> Optional[int]:
        """Store or update embedding for a conversation summary.

        Audit round 10 fix C3: the previous version inserted a brand-new
        vss row on every call (vss has no UPDATE path) and then rebound
        ``summary_vec.emb_id`` to point at it. The OLD ``embeddings``
        row was never deleted — vss has no FK back, no cascading
        cleanup, no GC. Over time every diary re-flush left a dead
        vector forever; ANN search slowed down on stale rows that no
        summary even referenced. Now we look up the previous ``emb_id``
        BEFORE insert and delete the orphan AFTER the rebind commits.
        """
        if self.is_vss_enabled:
            # Use sqlite-vss
            #
            # Audit round 16 fix: wrap the three statements (INSERT
            # embeddings, INSERT-OR-REPLACE summary_vec, DELETE
            # orphan) in a SINGLE explicit transaction via ``with
            # self.conn``. Previously each statement ran in its own
            # implicit autocommit; a crash between the INSERT and
            # the REPLACE left a dangling vss row (round-10 comment
            # acknowledged it), and a concurrent writer firing the
            # FK ON DELETE CASCADE could nuke the rebind in flight.
            # The ``with self.conn`` block commits on clean exit and
            # rolls back on exception — atomic.
            with self._lock:
                cur = self.conn.cursor()
                try:
                    with self.conn:
                        # Look up the previous binding so we can clean it up
                        # after the new vector is in and the FK has moved over.
                        old_row = cur.execute(
                            "SELECT emb_id FROM summary_vec WHERE summary_id=?",
                            (summary_id,),
                        ).fetchone()
                        old_emb_id = int(old_row["emb_id"]) if old_row else None

                        cur.execute("INSERT INTO embeddings(vec) VALUES (?)", (sqlite3.Binary(self._pack_vector(vec)),))
                        emb_id = cur.lastrowid
                        cur.execute(
                            "INSERT OR REPLACE INTO summary_vec(summary_id, emb_id) VALUES (?, ?)",
                            (summary_id, emb_id),
                        )
                        # GC the orphan AFTER the rebind so a crash between the
                        # two statements leaves the new row dangling (harmless)
                        # rather than losing the embedding entirely. Now that
                        # all three statements are in one transaction, a
                        # rollback drops the partial insert too.
                        if old_emb_id is not None and old_emb_id != emb_id:
                            try:
                                cur.execute("DELETE FROM embeddings WHERE rowid=?", (old_emb_id,))
                            except Exception as e:
                                # Don't fail the whole upsert if cleanup hits a
                                # vss quirk — just log so the leak is visible.
                                debug_log(f"upsert_summary_embedding: orphan GC failed for emb_id={old_emb_id}: {e}", "jarvis")
                except sqlite3.Error as exc:
                    debug_log(f"upsert_summary_embedding: rolled back — {exc}", "jarvis")
                    return None
                return int(emb_id)
        elif self._python_vector_store:
            # Use Python vector store — already an upsert-by-id under the
            # hood (the FAISS/python fallback indexes by summary_id), so
            # no orphan-vector accumulation there.
            self._python_vector_store.add_vector(summary_id, list(vec))
            return summary_id  # Return summary_id as a placeholder for emb_id
        else:
            return None

    def delete_conversation_summary(self, summary_id: int) -> bool:
        """Delete a conversation summary AND its vector representation.

        Audit round 21 fix (F01+F08): the previous "delete a memory"
        path (``memory_viewer.delete_memory``) issued
        ``DELETE FROM conversation_summaries WHERE id = ?`` in
        isolation. That works on the sqlite-vss path because
        ``embeddings`` has an FK cascade via the ``summary_vec``
        bridge table. But on the FAISS-fallback path
        (``_python_vector_store``) the in-memory FAISS index +
        sidecar table (``faiss_vector_store``) were NEVER touched.
        Result: subsequent hybrid search returned the deleted
        summary_id with a phantom distance, and the JOIN with the
        now-deleted summary row produced silent corruption (NULL
        text rows or missing entries in the score blend).

        This method handles both backends atomically:
          1. Delete the summary row in the same transaction as the
             sqlite-vss bridge cascade (if vss is on).
          2. If FAISS is the backend, call ``delete_vector`` AFTER
             the SQL delete commits — that ordering ensures a busy
             SQLite that rolls back doesn't leave the FAISS index
             inconsistent with the on-disk row.
        Returns True if a row was deleted.
        """
        with self._lock:
            try:
                with self.conn:
                    cur = self.conn.execute(
                        "DELETE FROM conversation_summaries WHERE id = ?",
                        (int(summary_id),),
                    )
                    deleted = cur.rowcount > 0
                    # On the sqlite-vss path, FK cascade handles
                    # ``summary_vec`` → ``embeddings``. Belt-and-
                    # braces: explicitly clean ``summary_vec`` in
                    # case FKs were ever disabled.
                    if self.is_vss_enabled:
                        try:
                            self.conn.execute(
                                "DELETE FROM summary_vec WHERE summary_id = ?",
                                (int(summary_id),),
                            )
                        except sqlite3.Error:
                            pass
            except sqlite3.Error as exc:
                debug_log(
                    f"delete_conversation_summary: SQL failed for id={summary_id}: {exc}",
                    "jarvis",
                )
                return False
        # FAISS path — outside the SQLite lock so a slow FAISS
        # remove_ids doesn't hold the connection.
        if not self.is_vss_enabled and self._python_vector_store is not None:
            try:
                self._python_vector_store.delete_vector(int(summary_id))
            except Exception as exc:
                debug_log(
                    f"delete_conversation_summary: vector store delete failed for "
                    f"id={summary_id}: {exc}",
                    "jarvis",
                )
        return bool(deleted)

    def close(self) -> None:
        try:
            with self._lock:
                self.conn.close()
        except Exception:
            pass

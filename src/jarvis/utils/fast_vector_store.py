"""
High-performance vector store implementation using FAISS for fast vector search.
This replaces the slow pure Python vector store with a much faster C++ implementation.

Audit round 11 rewrite:
  * **C1 fixed** — switched the inner index from ``IndexFlatIP`` to
    ``IndexIDMap2(IndexFlatIP)``. Updates of an existing summary_id are
    now ``remove_ids + add_with_ids`` (O(log N)) instead of dropping the
    lazy-rebuild flag that caused the next ``search()`` to re-read the
    entire ``faiss_vector_store`` table from SQLite and rebuild from
    scratch. The diary re-flush hot path now never touches SQLite for
    the index, only for durability.
  * **C2 fixed** — module-level lock around singleton creation so the
    intent-judge thread and reply thread don't construct two parallel
    stores at startup (writes to one would be invisible to the other).
  * **H1 fixed** — singleton keyed by ``(db_path, dimension)`` rather
    than a bare global; tests and reconfigured installs no longer get a
    silently-mismatched instance.
  * **H3 fixed** — held one persistent ``sqlite3.Connection`` per
    instance (lock-guarded). Diary flush no longer pays open + journal
    + commit + close per vector.
  * **L1 fixed** — drop the double-normalize; only ``faiss.normalize_L2``
    is applied (one consistent code path).
  * **L2 fixed** — distinguish ``DatabaseError`` (genuine corruption,
    surface to operator via warning + raise) from transient open
    failures.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sqlite3
import threading
import logging

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class FAISSVectorStore:
    """High-performance vector store using FAISS for fast similarity search."""

    def __init__(self, db_path: str, dimension: int = 768):
        """Initialize the FAISS vector store with a database path."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        self.db_path = db_path
        self.dimension = dimension
        self.index = None  # type: ignore[assignment]
        self._lock = threading.RLock()

        # Persistent connection — opened once, reused for every add/delete.
        # `check_same_thread=False` is safe because every call goes through
        # ``self._lock`` (RLock — safe for re-entrant operations).
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            # Non-fatal — defaults still work, just slower under contention.
            pass
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faiss_vector_store (
                summary_id INTEGER PRIMARY KEY,
                vector_blob BLOB NOT NULL
            )
        """)
        self._conn.commit()

        self._load_vectors()

    def _build_empty_index(self) -> None:
        """Build an empty FAISS index wrapped in IndexIDMap2 so the FAISS-side
        IDs match our SQLite summary_id and we can ``add_with_ids`` /
        ``remove_ids`` without rebuild."""
        with self._lock:
            base = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap2(base)

    def _load_vectors(self) -> None:
        """Load vectors from SQLite database and build FAISS index.

        Audit round 11 fix L2: previously caught every Exception silently
        and silently built an empty index — so a corrupt SQLite file
        looked identical to a fresh install. Now ``DatabaseError`` is
        re-raised so the daemon log shows the corruption; transient
        open failures still fall through to an empty index.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT summary_id, vector_blob FROM faiss_vector_store"
                ).fetchall()
            except sqlite3.DatabaseError:
                # Likely a genuine corruption — let it surface.
                logging.exception("FAISSVectorStore: SQLite reports DB corruption — refusing to mask")
                self._build_empty_index()
                raise
            except Exception as e:
                logging.warning(f"FAISSVectorStore: transient load failure ({e}); starting empty")
                self._build_empty_index()
                return

            self._build_empty_index()
            if not rows:
                return

            vectors: List[np.ndarray] = []
            summary_ids: List[int] = []
            for summary_id, vector_blob in rows:
                vec = np.frombuffer(vector_blob, dtype=np.float32)
                if len(vec) != self.dimension:
                    # Skip rows from an incompatible dimension — happens
                    # after model swap. Embeddings.py refuses mismatched
                    # writes now (audit round 10), so this only matters
                    # for legacy rows.
                    continue
                vectors.append(vec)
                summary_ids.append(int(summary_id))

            if not vectors:
                return

            arr = np.vstack(vectors).astype(np.float32, copy=False)
            faiss.normalize_L2(arr)
            ids = np.array(summary_ids, dtype=np.int64)
            self.index.add_with_ids(arr, ids)

    def _save_vector(self, summary_id: int, vector_blob: bytes) -> None:
        """Persist a single vector to SQLite (durable record).

        Always called under ``self._lock``.
        """
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO faiss_vector_store (summary_id, vector_blob) VALUES (?, ?)",
                (summary_id, vector_blob),
            )
            self._conn.commit()
        except Exception as e:
            logging.warning(f"Failed to save vector to database: {e}")

    def add_vector(self, summary_id: int, vector: List[float]) -> None:
        """Add or update a vector for a summary.

        Audit round 11 rewrite: the round 9 lazy-rebuild fix avoided
        O(N) work on every add but moved it to the next ``search`` —
        the SAME O(N) cost just shifted in time. Now we use the FAISS
        IndexIDMap2 API: an update is a ``remove_ids + add_with_ids``
        pair, both O(log N). No more SQLite re-read, no more rebuild
        flag. The diary re-flush hot path stays cheap.
        """
        with self._lock:
            if self.index is None:
                self._build_empty_index()

            vec_array = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            # Single normalize via FAISS — dropping the previous numpy
            # `vec/norm` round (L1 from audit). FAISS normalize is what
            # the search query also uses, so we stay self-consistent.
            faiss.normalize_L2(vec_array)

            # Durable record first — even if the in-memory update below
            # races on shutdown, the next load picks up the new vector.
            self._save_vector(summary_id, vec_array.tobytes())

            ids = np.array([summary_id], dtype=np.int64)
            # remove_ids returns 0 when the id is absent — fine to call
            # unconditionally. Wrapping in a try because some FAISS
            # builds throw if the underlying index doesn't expose remove.
            try:
                self.index.remove_ids(ids)
            except RuntimeError:
                pass
            self.index.add_with_ids(vec_array, ids)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors using FAISS.
        Returns list of (summary_id, distance) tuples sorted by similarity.
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            query_array = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query_array)

            k = min(top_k, self.index.ntotal)
            similarities, ids = self.index.search(query_array, k)

            results: List[Tuple[int, float]] = []
            for i in range(len(ids[0])):
                summary_id = int(ids[0][i])
                if summary_id < 0:  # FAISS marker for "no result in this slot"
                    continue
                similarity = float(similarities[0][i])
                # IP on unit-normalized vectors == cosine similarity.
                # Convert to a 0..2 distance for callers expecting "lower = better".
                distance = 1.0 - similarity
                results.append((summary_id, distance))

            return results

    def delete_vector(self, summary_id: int) -> None:
        """Remove a vector from the store (both index and DB).

        Audit round 21 fix (F24): reordered operations — SQL DELETE
        FIRST, then ``remove_ids`` on the FAISS index. Previously the
        FAISS removal ran first; if the SQL DELETE then failed
        (SQLITE_BUSY exhausted busy_timeout), the in-memory FAISS
        index had lost the entry but the disk row remained — and on
        the next process restart, ``_load_vectors`` re-added it from
        the persisted row. User saw "deleted" memories silently
        reappear after restart. With SQL-first ordering, a SQL
        failure means we never touch FAISS, so the state is at
        worst "delete failed entirely" instead of "split brain".
        """
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM faiss_vector_store WHERE summary_id = ?",
                    (summary_id,),
                )
                self._conn.commit()
            except Exception as e:
                logging.warning(f"Failed to delete vector from database: {e}")
                # Don't touch the FAISS index on SQL failure — keeps
                # the in-memory and on-disk state in sync ("not
                # deleted") rather than split.
                return
            if self.index is not None:
                try:
                    self.index.remove_ids(np.array([summary_id], dtype=np.int64))
                except RuntimeError:
                    # FAISS raises when the id isn't in the index;
                    # benign because we already removed the disk row.
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        with self._lock:
            ntotal = int(self.index.ntotal) if self.index is not None else 0
            return {
                "total_vectors": ntotal,
                "dimension": self.dimension,
                "index_type": type(self.index).__name__ if self.index is not None else None,
                "faiss_available": FAISS_AVAILABLE,
            }

    def close(self) -> None:
        """Release the persistent SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ── Singleton registry ───────────────────────────────────────────────────────
# Audit round 11 fixes C2 + H1: previously the singleton was a bare global
# with no lock and no key — the intent-judge thread and reply thread could
# race past `if _store is None` and end up with TWO stores backed by the
# same DB; writes to one were invisible to the other until process restart.
# Keying by (db_path, dimension) also stops a test override from silently
# inheriting a stale instance from earlier in the process lifetime.
_singleton_lock = threading.Lock()
_singletons: Dict[Tuple[str, int], FAISSVectorStore] = {}


def get_faiss_vector_store(db_path: str, dimension: int = 768) -> Optional[FAISSVectorStore]:
    """Get or create the FAISS vector store for ``(db_path, dimension)``."""
    if not FAISS_AVAILABLE:
        return None

    key = (str(db_path), int(dimension))
    # Fast path — already constructed, no lock needed for read.
    existing = _singletons.get(key)
    if existing is not None:
        return existing

    with _singleton_lock:
        # Double-check under lock.
        existing = _singletons.get(key)
        if existing is not None:
            return existing
        try:
            store = FAISSVectorStore(db_path, dimension)
        except Exception as e:
            logging.warning(f"Failed to create FAISS vector store: {e}")
            return None
        _singletons[key] = store
        return store

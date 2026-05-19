"""
Pure Python vector store implementation for out-of-the-box vector search.
Falls back to this when sqlite-vss is not available.

Audit round 11 fixes:
  * **H2** — ``_load_vectors`` is now lock-guarded (same Lock as
    ``add_vector``). The previous version mutated ``self.vectors`` from
    ``__init__`` without the lock; a concurrent ``add_vector`` from a
    second thread that got the partly-built singleton (see H1 + C2)
    would race the dict and at minimum drop a vector — at worst raise
    ``RuntimeError: dictionary changed size during iteration``.
  * **H3** — held one persistent SQLite connection per instance instead
    of opening + committing + closing on every add/delete. Diary flush
    used to pay a full journal cycle per vector.
  * **C2/H1** — module-level lock around singleton creation; singleton
    keyed by ``db_path`` so tests / reconfigured installs don't inherit
    a stale instance from earlier in the process.
"""

import json
import numpy as np
from typing import List, Tuple, Optional, Dict
import sqlite3
import threading


class PythonVectorStore:
    """Simple in-memory vector store with SQLite persistence."""

    def __init__(self, db_path: str):
        """Initialize the vector store with a database path."""
        self.db_path = db_path
        self.vectors: Dict[int, np.ndarray] = {}  # summary_id -> vector
        self._lock = threading.RLock()

        # Persistent connection (lock-guarded) — eliminates the open +
        # journal + commit + close storm on diary flushes.
        # ``check_same_thread=False`` because all use sites go through
        # ``self._lock``.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            pass
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS python_vector_store (
                summary_id INTEGER PRIMARY KEY,
                vector_json TEXT NOT NULL
            )
        """)
        self._conn.commit()

        self._load_vectors()

    def _load_vectors(self) -> None:
        """Load vectors from SQLite database. Lock-guarded so a racing
        ``add_vector`` (from a thread that got an in-flight singleton)
        can't mutate ``self.vectors`` mid-iteration."""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT summary_id, vector_json FROM python_vector_store"
                ).fetchall()
                for summary_id, vector_json in rows:
                    self.vectors[int(summary_id)] = np.asarray(
                        json.loads(vector_json), dtype=np.float32
                    )
            except sqlite3.DatabaseError:
                # Surface corruption to operator — same policy as the
                # FAISS store. Better than silently starting empty.
                import logging
                logging.exception("PythonVectorStore: SQLite DB corruption")
                raise
            except Exception:
                # Transient open failure — start empty, log to debug.
                pass

    def _save_vector(self, summary_id: int, vector: np.ndarray) -> None:
        """Persist a single vector to SQLite. Always called under self._lock."""
        try:
            vector_json = json.dumps(vector.tolist())
            self._conn.execute(
                "INSERT OR REPLACE INTO python_vector_store (summary_id, vector_json) VALUES (?, ?)",
                (summary_id, vector_json),
            )
            self._conn.commit()
        except Exception:
            # Fail silently - in-memory still works
            pass

    def add_vector(self, summary_id: int, vector: List[float]) -> None:
        """Add or update a vector for a summary."""
        with self._lock:
            vec_array = np.asarray(vector, dtype=np.float32)
            # Normalize vector for cosine similarity
            norm = np.linalg.norm(vec_array)
            if norm > 0:
                vec_array = vec_array / norm
            self.vectors[summary_id] = vec_array
            self._save_vector(summary_id, vec_array)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar vectors using cosine similarity.
        Returns list of (summary_id, distance) tuples sorted by similarity.
        """
        with self._lock:
            if not self.vectors:
                return []

            # Normalize query vector
            query_array = np.asarray(query_vector, dtype=np.float32)
            query_norm = np.linalg.norm(query_array)
            if query_norm > 0:
                query_array = query_array / query_norm

            # Calculate cosine similarities. Snapshot items() under the
            # lock so a concurrent add doesn't mutate mid-loop.
            similarities = []
            for summary_id, vector in self.vectors.items():
                similarity = float(np.dot(query_array, vector))
                distance = 1.0 - similarity
                similarities.append((summary_id, distance))

            similarities.sort(key=lambda x: x[1])
            return similarities[:top_k]

    def delete_vector(self, summary_id: int) -> None:
        """Remove a vector from the store."""
        with self._lock:
            if summary_id in self.vectors:
                del self.vectors[summary_id]
                try:
                    self._conn.execute(
                        "DELETE FROM python_vector_store WHERE summary_id = ?",
                        (summary_id,),
                    )
                    self._conn.commit()
                except Exception:
                    pass

    def close(self) -> None:
        """Release the persistent SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ── Singleton registry (lock-guarded, keyed by db_path) ──────────────────────
_singleton_lock = threading.Lock()
_singletons: Dict[str, PythonVectorStore] = {}


def get_python_vector_store(db_path: str) -> PythonVectorStore:
    """Get or create a Python vector store for ``db_path``."""
    key = str(db_path)
    existing = _singletons.get(key)
    if existing is not None:
        return existing
    with _singleton_lock:
        existing = _singletons.get(key)
        if existing is not None:
            return existing
        store = PythonVectorStore(db_path)
        _singletons[key] = store
        return store


def get_best_vector_store(db_path: str, dimension: int = 768):
    """Get the best available vector store (FAISS if available, otherwise Python fallback)."""
    # Try FAISS first (much faster)
    try:
        from .fast_vector_store import get_faiss_vector_store
        faiss_store = get_faiss_vector_store(db_path, dimension)
        if faiss_store is not None:
            return faiss_store
    except ImportError:
        pass

    # Fallback to Python implementation
    return get_python_vector_store(db_path)

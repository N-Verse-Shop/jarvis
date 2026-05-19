"""
🧠 Knowledge Graph

A self-organising node graph that stores the assistant's accumulated world
knowledge — anything learned during conversations that it wouldn't already know.
Three fast-access entry points (recent nodes, top nodes, root node) ensure the
most relevant knowledge is always reachable without exhaustive search.

See graph.spec.md for the full specification.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from ..debug import debug_log


# ── Mutation listeners ─────────────────────────────────────────────────────
#
# Lightweight observer registry so consumers (e.g. DialogueMemory's warm
# profile cache) can invalidate derived state when a node is created,
# updated, or deleted. The listener receives the action name, node id, and
# the FIXED_BRANCH ancestor (e.g. ``"user"``, ``"directives"``, ``"world"``)
# so it can scope its reaction. Failures in listeners are logged and
# swallowed so they cannot break a write.

_MUTATION_LISTENERS: "list[Callable[..., None]]" = []


def register_graph_mutation_listener(cb: Callable[..., None]) -> None:
    """Register a callback invoked after every node mutation.

    The callback is invoked with keyword arguments ``action``, ``node_id``,
    and ``branch`` where ``branch`` is the id of the FIXED_BRANCH ancestor
    (or the node id itself when the node is a fixed branch), or ``None``
    when the branch cannot be resolved (e.g. root mutations).
    """
    if cb not in _MUTATION_LISTENERS:
        _MUTATION_LISTENERS.append(cb)


def unregister_graph_mutation_listener(cb: Callable[..., None]) -> None:
    """Remove a previously registered mutation listener (idempotent)."""
    try:
        _MUTATION_LISTENERS.remove(cb)
    except ValueError:
        pass


def _notify_graph_mutation(action: str, node_id: str, branch: Optional[str]) -> None:
    for cb in list(_MUTATION_LISTENERS):
        try:
            cb(action=action, node_id=node_id, branch=branch)
        except Exception as exc:
            debug_log(f"graph mutation listener failed (non-fatal): {exc}", "memory")


# ── Fact normalisation ─────────────────────────────────────────────────────
#
# Used for dedupe comparisons. Locale-safe — the user base includes
# non-Latin scripts (e.g. Turkish, where ``"İ".lower()`` returns ``"i"``
# but Turkish lowercase is ``"ı"``), so we use ``unicodedata.NFKC`` plus
# ``str.casefold`` rather than ``str.lower``. ``casefold`` also folds
# German ß to ss, and NFKC collapses visually identical code points.

_WS_RE = re.compile(r"\s+")


def normalise_fact(text: str) -> str:
    """Lowercase (Unicode-aware) + collapse all whitespace, including
    newlines, into single spaces for fuzzy equality. ``_WS_RE`` matches
    ``\\s+``, so any newline embedded in an extracted fact collapses to
    a space on the candidate side, keeping the dedupe key well-formed
    even if the extractor accidentally emits a multi-line statement."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WS_RE.sub(" ", folded.strip())


# ── Configuration defaults ──────────────────────────────────────────────────

SPLIT_THRESHOLD = 1500       # tokens — when to split a node into children
MERGE_THRESHOLD = 200        # tokens — when to collapse sparse children back
RECENT_NODES_COUNT = 10      # number of recently-accessed nodes to track
TOP_NODES_COUNT = 15         # most-accessed nodes to surface
TOP_NODES_WINDOW_DAYS = 30   # time window for top-nodes ranking (legacy, kept for compat)
MAX_TRAVERSAL_DEPTH = 8      # safety limit on graph traversal
SUMMARY_MAX_LENGTH = 300     # max characters for a node description
DECAY_HALF_LIFE_DAYS = 14    # days until a node's access score halves


# ── Fixed top-level branches ────────────────────────────────────────────────
#
# The root is seeded with three fixed children on first run. The graph
# is still self-organising below these — auto-split/merge runs within
# each branch — but the top level is purpose-shaped, not content-shaped,
# so the extractor can route each new fact into the right semantic slot.
#
# - USER: everything about the person the assistant serves (identity,
#   tastes, preferences, plans, opinions). Warm-loaded into the system
#   prompt on every turn.
# - DIRECTIVES: imperatives the user issued at the assistant about its
#   own behaviour ("be concise", "use British English", "stop apologising").
#   Verbatim rules, never summarised. Warm-loaded on every turn.
# - WORLD: external facts with attribution (current graph content —
#   films, businesses, recipes, techniques). Unbounded. Not warm-loaded;
#   retrieved on demand via searchMemory.
#
# The IDs are stable strings so re-opening an existing graph is
# idempotent — no duplicate branches get seeded if the store already
# has them.

BRANCH_USER = "user"
BRANCH_DIRECTIVES = "directives"
BRANCH_WORLD = "world"

FIXED_BRANCHES: tuple[tuple[str, str, str], ...] = (
    (
        BRANCH_USER,
        "User",
        "Everything about the user: identity, location, relationships, "
        "tastes, preferences, history, plans, opinions. Always injected "
        "into the system prompt.",
    ),
    (
        BRANCH_DIRECTIVES,
        "Directives",
        "Imperatives the user issued at the assistant about its own "
        "behaviour — tone, verbosity, language, style rules. Verbatim, "
        "never summarised. Always injected into the system prompt.",
    ),
    (
        BRANCH_WORLD,
        "World",
        "External facts the assistant has learned and wants to carry "
        "forward: films, businesses, recipes, techniques, events. "
        "Retrieved on demand via searchMemory.",
    ),
)

FIXED_BRANCH_IDS: frozenset[str] = frozenset(bid for bid, _, _ in FIXED_BRANCHES)


# ── SQL helpers ────────────────────────────────────────────────────────────

def _decay_score_sql(half_life_days: int = DECAY_HALF_LIFE_DAYS) -> str:
    """Return a SQL expression that computes a time-decayed access score.

    Uses hyperbolic decay: access_count / (1 + age_days / half_life).
    A node accessed 100 times 14 days ago scores the same as one
    accessed 50 times today (with default half-life of 14 days).

    The raw access_count is never modified — decay is computed at query time
    so no data is lost and the half-life can be changed freely.
    """
    return (
        f"(access_count * 1.0 / "
        f"(1.0 + MAX(0, julianday('now') - julianday(last_accessed)) / {half_life_days}.0))"
    )


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class MemoryNode:
    """A single node in the memory graph."""
    id: str
    name: str
    description: str
    data: str = ""
    parent_id: Optional[str] = None
    access_count: int = 0
    last_accessed: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data_token_count: int = 0

    def to_dict(self) -> dict:
        """Serialise to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "data": self.data,
            "parent_id": self.parent_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "data_token_count": self.data_token_count,
        }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — ~4 chars per token for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Schema ──────────────────────────────────────────────────────────────────

_GRAPH_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memory_nodes (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT NOT NULL,
    data             TEXT NOT NULL DEFAULT '',
    parent_id        TEXT REFERENCES memory_nodes(id) ON DELETE SET NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed    TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    data_token_count INTEGER NOT NULL DEFAULT 0,
    CHECK(parent_id IS NULL OR parent_id != id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_parent ON memory_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON memory_nodes(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_access_count ON memory_nodes(access_count DESC);
"""


# ── Graph Memory Store ──────────────────────────────────────────────────────

class GraphMemoryStore:
    """
    Self-organising node graph for persistent memory.

    Backed by SQLite with thread-safe access. Provides three entry points
    for fast retrieval: recent nodes, top nodes, and the root node.
    """

    def __init__(self, db_path: str) -> None:
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()
        self._ensure_root()
        # Audit round 14 fix: restrict the on-disk file to the owner.
        # The graph holds the assistant's persistent memory (facts about
        # the user, directives, world knowledge) — group/other read is
        # never appropriate. Best-effort: ignore failures on FSes that
        # don't support POSIX modes.
        try:
            import os as _os
            _os.chmod(db_path, 0o600)
        except Exception:
            pass

    # ── Schema & bootstrap ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Audit round 14 fix: without a busy_timeout, two writers
            # racing on the same SQLite file get SQLITE_BUSY (the diary
            # flush worker and the foreground graph write happen at the
            # same time during a voice turn). Set 5 s — well above any
            # realistic single-write duration but short enough that a
            # truly hung writer surfaces instead of stalling forever.
            self.conn.execute("PRAGMA busy_timeout = 5000")
            # Audit round 20 fix (P1): switch to WAL journal mode so
            # readers never block on a writer. Without WAL, every
            # mutation (touch_node, append_to_node, bulk_split_node)
            # held a global lock that even foreground SELECTs
            # (search_nodes, find_node_by_name, get_recent_nodes —
            # used in the hot voice path) had to wait on. Under
            # bursty mutation (warm-profile rebuild during a diary
            # flush) read latency could spike to 5 s (the busy_timeout
            # above), manifesting as visible voice latency. Sibling
            # stores (memory/db.py, utils/vector_store.py) already use
            # WAL; this brings the graph store into parity.
            #
            # WAL is durable across crashes and persists across opens,
            # so issuing the PRAGMA on every open is idempotent.
            # ``synchronous=NORMAL`` is the recommended companion
            # (WAL + NORMAL = full crash safety for committed txns,
            # without the per-COMMIT fsync of FULL).
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.executescript(_GRAPH_SCHEMA_SQL)
            self.conn.commit()

    def _ensure_root(self) -> None:
        """Create the root node and the three fixed top-level branches
        (User / Directives / World) if they don't exist.

        Idempotent: each branch has a stable string id, so re-opening an
        existing graph never duplicates them. Branches are also created
        on first boot for existing graphs that predate the taxonomy —
        this is the migration path.
        """
        with self._lock:
            self._seed_root_locked()

    def _seed_root_locked(self) -> None:
        """Caller-locked variant of ``_ensure_root``.

        Used by ``migrate_legacy_shape`` to keep the DELETE + re-seed
        inside ONE transaction — if a crash interrupts the wipe the
        next boot must not see an empty ``memory_nodes`` table. Caller
        MUST hold ``self._lock`` and (ideally) be inside a
        ``with self.conn:`` transaction.
        """
        now = datetime.now(timezone.utc).isoformat()
        row = self.conn.execute(
            "SELECT id FROM memory_nodes WHERE parent_id IS NULL LIMIT 1"
        ).fetchone()
        if row is None:
            self.conn.execute(
                """INSERT INTO memory_nodes
                   (id, name, description, data, parent_id,
                    access_count, last_accessed, created_at, updated_at,
                    data_token_count)
                   VALUES (?, ?, ?, ?, NULL, 0, ?, ?, ?, 0)""",
                ("root", "Root", "Top-level memory node — contains all knowledge domains.", "", now, now, now),
            )
            debug_log("Created root memory node", "memory")

        # Seed fixed top-level branches under root. Each row is
        # inserted with INSERT OR IGNORE keyed on the stable id so
        # repeated boots are no-ops.
        for branch_id, name, description in FIXED_BRANCHES:
            self.conn.execute(
                """INSERT OR IGNORE INTO memory_nodes
                   (id, name, description, data, parent_id,
                    access_count, last_accessed, created_at, updated_at,
                    data_token_count)
                   VALUES (?, ?, ?, '', 'root', 0, ?, ?, ?, 0)""",
                (branch_id, name, description, now, now, now),
            )

    def migrate_legacy_shape(self) -> bool:
        """Wipe the graph if it has a non-conforming (pre-taxonomy) shape.

        The purpose-driven taxonomy (root → User / Directives / World)
        is a hard reorganisation: pre-existing nodes under root that
        don't match this shape would sit invisible to the warm profile
        forever.
        Rather than carrying them as dead weight, we wipe on daemon
        start-up and let the diary re-import repopulate with correctly
        classified facts.

        Called ONLY from the daemon start-up path — the memory viewer
        instantiates ``GraphMemoryStore`` read-mostly and must not
        trigger a wipe mid-session.

        Non-conforming shape is defined as:
          - root has a direct child whose id is not in ``FIXED_BRANCHES``
          - OR root's own ``data`` column is non-empty (cold-start writes
            that landed on root before the taxonomy existed).

        Returns True if a wipe happened, False if the graph was already
        in the expected shape.
        """
        expected_ids = FIXED_BRANCH_IDS
        with self._lock:
            root_row = self.conn.execute(
                "SELECT data FROM memory_nodes WHERE id = 'root'"
            ).fetchone()
            root_has_data = bool(root_row and (root_row["data"] or "").strip())

            rogue_child = self.conn.execute(
                "SELECT id FROM memory_nodes "
                "WHERE parent_id = 'root' AND id NOT IN ({}) LIMIT 1".format(
                    ",".join("?" * len(expected_ids))
                ),
                tuple(expected_ids),
            ).fetchone()

            if not root_has_data and rogue_child is None:
                return False

            reason = (
                "root holds pre-taxonomy data"
                if root_has_data
                else f"found non-conforming root child: {rogue_child['id']!r}"
            )
            debug_log(
                f"wiping knowledge graph ({reason}); will re-seed fixed branches",
                "memory",
            )
            # Audit round 10 fix: BACKUP before destructive DELETE.
            # Previously a misclassified rogue child wiped the entire
            # User branch with NO recovery path — months of accreted
            # diary nodes gone in one startup. Now: dump current rows
            # to a timestamped JSON file under ~/.config/jarvis/backups/
            # so the user can recover via `jq` + reinsert script. The
            # backup is best-effort — failure does NOT abort the wipe
            # (we want migration progress even if disk is full), but
            # the path is logged for forensics.
            try:
                from datetime import datetime as _dt
                import json as _json
                import os as _os
                from pathlib import Path as _P
                backup_dir = _P.home() / ".config" / "jarvis" / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                # Audit round 14 fix: tighten perms on the backup dir
                # — pre-wipe backups contain the full memory dump and
                # default umask leaves them group/other readable.
                try:
                    _os.chmod(backup_dir, 0o700)
                except Exception:
                    pass
                ts = _dt.now().strftime("%Y%m%dT%H%M%S")
                backup_path = backup_dir / f"graph-pre-wipe-{ts}.json"
                rows = self.conn.execute(
                    "SELECT * FROM memory_nodes"
                ).fetchall()
                # sqlite3.Row → dict
                dumped = [dict(r) for r in rows]
                _tmp = backup_path.with_suffix(".json.tmp")
                _tmp.write_text(
                    _json.dumps(dumped, ensure_ascii=False, default=str, indent=2),
                    encoding="utf-8",
                )
                _os.replace(_tmp, backup_path)
                # Audit round 14 fix: chmod 0o600 + bounded retention.
                # Previously every wipe wrote a new file with no
                # rotation — a chatty diary could fill the disk over
                # months. Keep the 5 most-recent backups.
                try:
                    _os.chmod(backup_path, 0o600)
                except Exception:
                    pass
                try:
                    backups = sorted(
                        backup_dir.glob("graph-pre-wipe-*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for old in backups[5:]:
                        try:
                            old.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
                debug_log(
                    f"graph pre-wipe backup written: {backup_path} ({len(dumped)} rows)",
                    "memory",
                )
            except Exception as _be:
                debug_log(f"graph backup failed (proceeding with wipe): {_be}", "memory")
            # Audit round 16 fix: wrap DELETE + re-seed in ONE
            # transaction. The previous flow committed the DELETE
            # FIRST and then released the lock to call ``_ensure_root``
            # (which re-acquired the lock). A crash, OOM, or even a
            # foreign-key cascade error between those two commits left
            # ``memory_nodes`` empty — the next read crashed in
            # ``get_warm_profile`` because ``root`` was missing.
            # ``with self.conn:`` wraps the inner statements in a
            # transaction that COMMITS on clean exit and ROLLS BACK on
            # any exception, so the table either has the new root +
            # branches or stays in its pre-wipe state. The legacy rows
            # being un-DELETE-able on rollback is the *better* failure
            # mode here — they were already in a non-conforming shape
            # and the next migration attempt will retry the wipe.
            try:
                with self.conn:
                    self.conn.execute("DELETE FROM memory_nodes")
                    self._seed_root_locked()
            except Exception as e:
                debug_log(
                    f"graph migration rolled back (DB unchanged): {e}",
                    "memory",
                )
                # Re-raise so the caller (daemon start-up) surfaces
                # the failure instead of silently continuing with a
                # graph that still has the pre-taxonomy shape.
                raise
        return True

    # ── CRUD ────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Fetch a single node by ID."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memory_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_node(row)

    def get_children(self, node_id: str) -> list[MemoryNode]:
        """Get all direct children of a node, ordered by decayed access score."""
        score = _decay_score_sql()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memory_nodes WHERE parent_id = ? ORDER BY {score} DESC",
                (node_id,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_root(self) -> MemoryNode:
        """Return the root node."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memory_nodes WHERE parent_id IS NULL LIMIT 1"
            ).fetchone()
            return self._row_to_node(row)

    def _resolve_branch(self, node_id: Optional[str]) -> Optional[str]:
        """Walk parents from ``node_id`` up to find the FIXED_BRANCH id it
        belongs to (or itself, if the node IS a fixed branch). Returns
        ``None`` for the root or when the node cannot be located.

        Capped at ``MAX_TRAVERSAL_DEPTH`` so a corrupt parent cycle cannot
        spin the loop. SQLite reads only — safe to call from write paths.
        """
        if not node_id or node_id == "root":
            return None
        if node_id in FIXED_BRANCH_IDS:
            return node_id
        current = node_id
        for _ in range(MAX_TRAVERSAL_DEPTH):
            row = self.conn.execute(
                "SELECT parent_id FROM memory_nodes WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                return None
            parent = row["parent_id"]
            if parent is None or parent == "root":
                return None
            if parent in FIXED_BRANCH_IDS:
                return parent
            current = parent
        return None

    def create_node(
        self,
        name: str,
        description: str,
        data: str = "",
        parent_id: Optional[str] = None,
    ) -> MemoryNode:
        """Create a new node and return it.

        Raises ValueError if parent_id references a non-existent node.
        """
        if parent_id is not None:
            parent = self.get_node(parent_id)
            if parent is None:
                raise ValueError(f"Parent node '{parent_id}' does not exist")

        # Enforce description length limit from spec
        if len(description) > SUMMARY_MAX_LENGTH:
            description = description[:SUMMARY_MAX_LENGTH]

        node_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        token_count = _estimate_tokens(data)

        with self._lock:
            self.conn.execute(
                """INSERT INTO memory_nodes
                   (id, name, description, data, parent_id,
                    access_count, last_accessed, created_at, updated_at,
                    data_token_count)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                (node_id, name, description, data, parent_id, now, now, now, token_count),
            )
            self.conn.commit()

        debug_log(f"Created memory node '{name}' ({node_id[:8]})", "memory")
        _notify_graph_mutation("create", node_id, self._resolve_branch(parent_id))
        return MemoryNode(
            id=node_id,
            name=name,
            description=description,
            data=data,
            parent_id=parent_id,
            access_count=0,
            last_accessed=now,
            created_at=now,
            updated_at=now,
            data_token_count=token_count,
        )

    def update_node(
        self,
        node_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        data: Optional[str] = None,
        parent_id: Optional[str] = ...,  # type: ignore[assignment]
    ) -> Optional[MemoryNode]:
        """Update fields on an existing node. Returns the updated node."""
        node = self.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        if name is not None:
            node.name = name
        if description is not None:
            if len(description) > SUMMARY_MAX_LENGTH:
                description = description[:SUMMARY_MAX_LENGTH]
            node.description = description
        if data is not None:
            node.data = data
            node.data_token_count = _estimate_tokens(data)
        if parent_id is not ...:
            # Audit round 14 fix: reparenting a node to one of its own
            # descendants creates a cycle (A → B → A), which makes
            # ``_resolve_branch``'s capped traversal hit the depth
            # limit and return None — i.e. the node permanently falls
            # out of the warm-profile branch index. Walk the proposed
            # new parent's ancestor chain and refuse the update if
            # ``node_id`` appears anywhere in it.
            if parent_id == node_id:
                raise ValueError(f"update_node: refusing to set parent_id == node_id ({node_id})")
            if parent_id:
                cursor = parent_id
                with self._lock:
                    for _ in range(MAX_TRAVERSAL_DEPTH * 2):
                        if cursor == node_id:
                            raise ValueError(
                                f"update_node: refusing to create cycle — node {node_id} "
                                f"is already an ancestor of proposed parent {parent_id}"
                            )
                        if cursor is None or cursor == "root":
                            break
                        row = self.conn.execute(
                            "SELECT parent_id FROM memory_nodes WHERE id = ?",
                            (cursor,),
                        ).fetchone()
                        if row is None:
                            break
                        cursor = row["parent_id"]
            node.parent_id = parent_id
        node.updated_at = now

        with self._lock:
            self.conn.execute(
                """UPDATE memory_nodes
                   SET name = ?, description = ?, data = ?, parent_id = ?,
                       updated_at = ?, data_token_count = ?
                   WHERE id = ?""",
                (node.name, node.description, node.data, node.parent_id,
                 node.updated_at, node.data_token_count, node_id),
            )
            self.conn.commit()

        _notify_graph_mutation("update", node_id, self._resolve_branch(node_id))
        return node

    def bulk_split_node(
        self,
        parent_id: str,
        children: "list[dict]",
        parent_summary: str,
    ) -> "list[str]":
        """Atomically split a node into N children + clear parent data
        + update parent description.

        Audit round 14 fix: ``auto_split_node`` in ``graph_ops.py`` used
        to issue N+1 separate commits (one per ``create_node`` call,
        then one for the ``update_node`` clear). A crash or process
        kill between commits left the graph half-split — the children
        existed AND the parent still held the original data → next
        warm-profile rebuild saw every fact twice. Wrap everything in
        a single transaction so a failure rolls back to the pre-split
        state.

        Each entry in ``children`` is a dict with keys ``name``,
        ``description``, ``data`` (already joined into a single
        string).
        Returns the list of new child node ids in insertion order.
        """
        if not children:
            return []
        parent = self.get_node(parent_id)
        if parent is None:
            raise ValueError(f"bulk_split_node: parent '{parent_id}' does not exist")

        now = datetime.now(timezone.utc).isoformat()
        new_ids: list[str] = []
        rows_to_insert: list[tuple] = []
        for cat in children:
            name = str(cat.get("name", ""))[:SUMMARY_MAX_LENGTH] or "Unnamed"
            desc = str(cat.get("description", f"Memories about: {name}"))[:SUMMARY_MAX_LENGTH]
            data = str(cat.get("data", ""))
            cid = str(uuid.uuid4())
            new_ids.append(cid)
            rows_to_insert.append(
                (cid, name, desc, data, parent_id, now, now, now, _estimate_tokens(data))
            )

        if len(parent_summary) > SUMMARY_MAX_LENGTH:
            parent_summary = parent_summary[:SUMMARY_MAX_LENGTH]

        # Audit round 19 fix: re-check the split precondition INSIDE the
        # lock. ``auto_split_node`` reads ``node.data_token_count`` and
        # then calls back into here, but there is a window between the
        # read and this transaction during which another caller (a
        # second audio-callback thread, the cumulative diary flusher,
        # the warm-profile rebuilder) can complete its own split. Without
        # this guard the second caller wipes the parent again AND
        # inserts a second set of children — producing duplicate
        # sibling sets with identical content. The condition we re-test
        # is "the parent still holds enough tokens to warrant splitting".
        # If a peer already split it, ``data_token_count`` has been reset
        # to 0 (see the UPDATE below) and we abort cleanly.
        with self._lock:
            current = self.conn.execute(
                "SELECT data_token_count FROM memory_nodes WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if current is None:
                # Parent was deleted out from under us between the
                # outer get_node and now. Nothing to split.
                return []
            current_tokens = int(current["data_token_count"] or 0)
            if current_tokens <= SPLIT_THRESHOLD:
                # Either we lost the race or the parent was emptied by
                # an unrelated op. Either way, splitting again would
                # produce duplicate children — refuse.
                return []
            try:
                self.conn.execute("BEGIN")
                self.conn.executemany(
                    """INSERT INTO memory_nodes
                       (id, name, description, data, parent_id,
                        access_count, last_accessed, created_at, updated_at,
                        data_token_count)
                       VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                    rows_to_insert,
                )
                self.conn.execute(
                    """UPDATE memory_nodes
                       SET data = '', description = ?, updated_at = ?,
                           data_token_count = 0
                       WHERE id = ?""",
                    (parent_summary, now, parent_id),
                )
                self.conn.commit()
            except Exception:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                raise

        # Listeners fire OUTSIDE the transaction so a slow listener
        # can't hold the SQLite write lock. Order: children created
        # first, then parent updated — matches the pre-bulk semantics.
        branch = self._resolve_branch(parent_id)
        for cid in new_ids:
            _notify_graph_mutation("create", cid, branch)
        _notify_graph_mutation("update", parent_id, branch)
        return new_ids

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and reparent its children to its grandparent.

        Audit round 10 fix: the previous implementation relied on
        `ON DELETE SET NULL` for the parent_id FK, which left children
        with `parent_id=NULL` — i.e. they became disconnected roots.
        That broke `_resolve_branch` for them (warm-profile lookups
        rooted at FIXED_BRANCHES no longer reached the orphans), they
        stopped showing in tree traversals, and the next `find_root()`
        call would either skip them or — on a freshly emptied tree —
        promote one of them to root with no obvious owner.

        Now we explicitly **reparent** children to the deleted node's
        own parent before the delete. The subtree shape is preserved
        as best we can: a → b → {c, d} becomes a → {c, d} when b is
        deleted, rather than a + (orphan c) + (orphan d).

        Root and the seeded fixed branches (see ``FIXED_BRANCHES``) are
        non-deletable — the warm profile and extractor routing rely on
        their stable presence (graph.spec.md §"Fixed Top-Level Branches").
        """
        if node_id == "root" or node_id in FIXED_BRANCH_IDS:
            return False
        # Resolve branch BEFORE the delete so listeners get a meaningful
        # branch attribution even though the row is about to vanish.
        branch = self._resolve_branch(node_id)
        with self._lock:
            # Look up the doomed node's own parent so we can reparent
            # its children to it. If the doomed node was a top-level
            # branch root (parent_id == 'root' or NULL), children get
            # reparented to 'root' so they remain reachable from the
            # canonical entry point. Falling back to NULL would put
            # them back into the orphan state we're trying to avoid.
            parent_row = self.conn.execute(
                "SELECT parent_id FROM memory_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if parent_row is None:
                # Node didn't exist — nothing to do.
                return False
            new_parent = parent_row["parent_id"] or "root"

            # Reparent in one statement BEFORE the delete; ordering is
            # important because the FK cascade would fire SET NULL the
            # instant we delete the parent row.
            self.conn.execute(
                "UPDATE memory_nodes SET parent_id = ? WHERE parent_id = ?",
                (new_parent, node_id),
            )
            cur = self.conn.execute(
                "DELETE FROM memory_nodes WHERE id = ?", (node_id,)
            )
            self.conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            _notify_graph_mutation("delete", node_id, branch)
        return deleted

    def node_contains_fact(self, node_id: str, fact: str) -> bool:
        """True if ``fact`` matches any line of the node's data after
        ``normalise_fact`` folding. Used to dedupe graph appends when the
        cumulative daily summary re-seeds the same facts across diary flushes.
        """
        node = self.get_node(node_id)
        if node is None or not node.data:
            return False
        target = normalise_fact(fact)
        if not target:
            return False
        for line in node.data.split("\n"):
            if normalise_fact(line) == target:
                return True
        return False

    def append_to_node(self, node_id: str, text: str) -> bool:
        """Append text to a node's data field.

        Returns True if the node's data_token_count now exceeds SPLIT_THRESHOLD.

        Audit round 19 fix: previously this method performed a
        ``get_node`` → ``update_node`` round-trip, each acquiring
        ``self._lock`` separately. Two concurrent diary flushes that
        both landed on the same node (User branch is dominant) could
        each read ``data=X``, then both write ``X + fact_A`` and
        ``X + fact_B`` respectively — one fact was silently lost. The
        new implementation holds ``self._lock`` across the read, the
        UPDATE statement, and the post-write token-count read, so the
        sequence is serialised against any other writer.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT data FROM memory_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                return False
            current = row["data"] or ""
            separator = "\n" if current else ""
            new_data = current + separator + text
            new_token_count = _estimate_tokens(new_data)
            now = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """UPDATE memory_nodes
                   SET data = ?, data_token_count = ?, updated_at = ?
                   WHERE id = ?""",
                (new_data, new_token_count, now, node_id),
            )
            self.conn.commit()
            crossed_threshold = new_token_count > SPLIT_THRESHOLD

        # Fire the graph-mutation listener OUTSIDE the lock so a slow
        # listener cannot stall the next append. ``_notify_graph_mutation``
        # is the same notification path ``update_node`` uses.
        _notify_graph_mutation("update", node_id, self._resolve_branch(node_id))
        return crossed_threshold

    def touch_node(self, node_id: str) -> None:
        """Increment access_count and update last_accessed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """UPDATE memory_nodes
                   SET access_count = access_count + 1, last_accessed = ?
                   WHERE id = ?""",
                (now, node_id),
            )
            self.conn.commit()

    # ── Entry points ────────────────────────────────────────────────────

    def get_recent_nodes(self, limit: int = RECENT_NODES_COUNT) -> list[MemoryNode]:
        """Get the most recently accessed nodes."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT * FROM memory_nodes
                   WHERE id != 'root'
                   ORDER BY last_accessed DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_top_nodes(
        self,
        limit: int = TOP_NODES_COUNT,
        window_days: int = TOP_NODES_WINDOW_DAYS,
    ) -> list[MemoryNode]:
        """Get nodes with the highest time-decayed access score.

        Uses hyperbolic decay so frequently accessed nodes that haven't
        been touched in a while naturally fall off without needing a hard
        window cutoff. The ``window_days`` parameter is kept for backward
        compatibility but is no longer used for filtering.
        """
        score = _decay_score_sql()
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT * FROM memory_nodes
                   WHERE id != 'root'
                   ORDER BY {score} DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    # ── Tree queries ────────────────────────────────────────────────────

    def get_subtree(self, node_id: str, max_depth: int = 3) -> dict:
        """
        Return a nested dict representing the subtree rooted at node_id.

        Each dict has keys: node (MemoryNode.to_dict()) and children (list of subtrees).
        Useful for the tree sidebar in the UI.
        """
        node = self.get_node(node_id)
        if node is None:
            return {}

        def _build(nid: str, depth: int) -> dict:
            n = self.get_node(nid)
            if n is None:
                return {}
            children = []
            if depth < max_depth:
                for child in self.get_children(nid):
                    children.append(_build(child.id, depth + 1))
            return {"node": n.to_dict(), "children": children}

        return _build(node_id, 0)

    def get_ancestors(self, node_id: str) -> list[MemoryNode]:
        """Return the path from root to this node (inclusive), root first."""
        ancestors: list[MemoryNode] = []
        visited: set[str] = set()
        current = self.get_node(node_id)
        while current is not None:
            if current.id in visited or len(ancestors) > MAX_TRAVERSAL_DEPTH:
                debug_log(f"Cycle or depth limit hit in get_ancestors for {node_id}", "memory")
                break
            visited.add(current.id)
            ancestors.append(current)
            if current.parent_id is None:
                break
            current = self.get_node(current.parent_id)
        ancestors.reverse()
        return ancestors

    def get_all_nodes(self) -> list[MemoryNode]:
        """Return all nodes — use with care on large graphs."""
        score = _decay_score_sql()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memory_nodes ORDER BY {score} DESC"
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_node_count(self) -> int:
        """Return total number of nodes in the graph."""
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM memory_nodes").fetchone()
            return row["cnt"]

    def get_total_tokens(self) -> int:
        """Return total data tokens across all nodes. Zero means no knowledge stored."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(data_token_count), 0) as total FROM memory_nodes"
            ).fetchone()
            return int(row["total"])

    # ── Search ─────────────────────────────────────────────────────────

    def search_nodes(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search nodes by keyword match across name, description, and data.

        Uses case-insensitive LIKE matching on each keyword (split by whitespace).
        Scoring weights: name/description matches are worth 3× data matches, so
        specific nodes about a topic rank above broad category nodes that merely
        contain the keyword somewhere in their data blob.
        Excludes the root node from results and touches matched nodes.
        """
        keywords = [k.strip() for k in query.split() if k.strip()]
        if not keywords:
            return []

        # Build a score expression: name/description matches worth 3, data worth 1
        score_parts: list[str] = []
        params: list[str] = []
        for kw in keywords:
            # Escape LIKE wildcards so literal %, _, \ are matched exactly
            escaped = kw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            score_parts.append(
                "(CASE WHEN name LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END"
                " + CASE WHEN description LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END"
                " + CASE WHEN data LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"
            )
            params.extend([pattern, pattern, pattern])

        score_expr = " + ".join(score_parts)
        # Use a subquery to avoid duplicating the score expression (and its bindings)
        sql = f"""
            SELECT * FROM (
                SELECT *, ({score_expr}) AS relevance
                FROM memory_nodes
                WHERE id != 'root'
            ) WHERE relevance > 0
            ORDER BY relevance DESC, {_decay_score_sql()} DESC
            LIMIT ?
        """
        params.append(str(limit))

        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
            nodes = [self._row_to_node(r) for r in rows]

        # Touch matched nodes (updates access tracking)
        for node in nodes:
            self.touch_node(node.id)

        debug_log(f"Graph search for '{query}' found {len(nodes)} nodes", "memory")
        return nodes

    def find_node_by_name(self, name: str, parent_id: Optional[str] = None) -> Optional[MemoryNode]:
        """Find a node by exact name match (case-insensitive), optionally under a specific parent.

        Audit round 10 fix: SQLite `LOWER()` is ASCII-only by default
        — it does NOT lowercase Cyrillic / Turkish / German umlauts.
        Result: a graph node stored as "Ярвіс" would NOT be found
        when searching for "ярвіс". The rest of the module uses
        Python's `casefold()` + NFKC normalisation via
        `normalise_fact`. Align this lookup by fetching candidates
        via a permissive WHERE clause and filtering with Python's
        casefold in the application. Small N (per-parent or
        global-no-root) so the post-filter is cheap.
        """
        try:
            needle = name.strip().casefold()
        except Exception:
            needle = (name or "").strip().lower()
        if not needle:
            return None
        with self._lock:
            if parent_id is not None:
                rows = self.conn.execute(
                    "SELECT * FROM memory_nodes WHERE parent_id = ?",
                    (parent_id,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM memory_nodes WHERE id != 'root'",
                ).fetchall()
            for row in rows:
                try:
                    if (row["name"] or "").strip().casefold() == needle:
                        return self._row_to_node(row)
                except Exception:
                    continue
            return None

    # ── Graph edges for visualisation ───────────────────────────────────

    def get_graph_data(self, root_id: str = "root", max_depth: int = 4) -> dict:
        """
        Return nodes and edges suitable for graph visualisation.

        Returns:
            {"nodes": [...], "edges": [...]}
            Each node: {id, name, description, data_token_count, access_count,
                        last_accessed, parent_id, has_children, depth}
            Each edge: {source, target}
        """
        nodes_out: list[dict] = []
        edges_out: list[dict] = []
        visited: set[str] = set()

        def _walk(nid: str, depth: int) -> None:
            if nid in visited or depth > max_depth:
                return
            visited.add(nid)

            node = self.get_node(nid)
            if node is None:
                return

            children = self.get_children(nid)
            nodes_out.append({
                "id": node.id,
                "name": node.name,
                "description": node.description,
                "data_token_count": node.data_token_count,
                "access_count": node.access_count,
                "last_accessed": node.last_accessed,
                "parent_id": node.parent_id,
                "has_children": len(children) > 0,
                "depth": depth,
            })

            for child in children:
                edges_out.append({"source": nid, "target": child.id})
                _walk(child.id, depth + 1)

        _walk(root_id, 0)
        return {"nodes": nodes_out, "edges": edges_out}

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            data=row["data"],
            parent_id=row["parent_id"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            data_token_count=row["data_token_count"],
        )

    def close(self) -> None:
        """Close the database connection."""
        try:
            with self._lock:
                self.conn.close()
        except Exception:
            pass

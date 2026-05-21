"""Jarvis memory subsystem.

Three persistent stores back the assistant's long-term memory:

- :mod:`jarvis.memory.db` — SQLite "diary" of conversations (FTS5-indexed
  message history + embeddings + summary table). The primary store.
- :mod:`jarvis.memory.graph` — knowledge graph of nodes/edges with
  hierarchical FIXED_BRANCH_IDS. Used for warm-profile rendering.
- :mod:`jarvis.memory.facts` — small key/value store with decay scoring
  for "facts the user has told us" (preferences, addresses, etc.).

Plus four helpers:

- :mod:`jarvis.memory.embeddings` — Ollama-backed text embedder with
  cache + dim guard (fails closed on mismatched dimensions).
- :mod:`jarvis.memory.conversation` — diary-flush, summarisation,
  search-by-keyword over conversations.
- :mod:`jarvis.memory.self_learning` — heuristic fact extractor from
  conversation turns + language-lock prompt builders.
- :mod:`jarvis.memory.recall_gate` — gates whether a turn warrants a
  full memory recall (cheap pre-check).
- :mod:`jarvis.memory.graph_ops` — LLM-driven graph mutations.

R34-S52 C-7: this file used to be empty. Submodule imports worked
because the directory was a valid Python package, but the package had
no formal public surface — every consumer reached past it via
``from jarvis.memory.<submodule> import …``. That works today but
silently breaks if anything is renamed. The ``__all__`` below declares
the canonical entry points; new submodules should be added here so
the package's intent stays grep-able.
"""

from __future__ import annotations

# Submodules are imported lazily by consumers via the full dotted path.
# We don't eagerly import them here because (a) the diary store opens
# SQLite at import time and (b) test fixtures want to monkey-patch
# individual submodules without paying the cross-import cost.

__all__ = [
    # core stores
    "db",
    "graph",
    "facts",
    # helpers
    "embeddings",
    "conversation",
    "self_learning",
    "recall_gate",
    "graph_ops",
]

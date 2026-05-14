"""Inter-process communication primitives for jarvis-isair.

Currently exposes the typed event stream — append-only JSONL channel
between the voice daemon and any number of consumers (HUD overlay,
Telegram bridge, debug viewer, log shipper).

Inspired by quiet-node/thuki's Tauri Channel API typed StreamChunk
pattern. Trades a little I/O overhead for a clean contract: instead
of every consumer parsing free-form JSON state, they pattern-match on
a typed event tag.
"""

from .stream import EventStream, get_stream

__all__ = ["EventStream", "get_stream"]

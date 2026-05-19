"""
Opt-in per-turn prompt dump for the reply engine.

Motivation: PR #232's harness evals cannot reproduce the live confab where
`gemma4:e2b` answers "Tell me about the movie Possessor" with "The movie is
Under the Skin" despite a successful webSearch fetch. To bridge the
harness-vs-field gap, this module writes the exact `messages` array, the
selected tool schema, and the raw LLM response to disk for each turn, so a
user-side reproduction can be replayed verbatim in an eval.

Gated on the env var `JARVIS_DUMP_PROMPTS=1` — off by default because the
dumps contain the full system prompt, memory digest and tool output (likely
PII). Users opt in only when hunting a bug.

Files are written to `~/.local/share/jarvis/prompts/` as per-turn JSON so
each dump is self-contained and easy to `cat` or paste into a test.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..debug import debug_log
from ..utils.redact import redact


_ENV_VAR = "JARVIS_DUMP_PROMPTS"

# Audit round 10: cap on dumped-files retention. Without this the
# directory grows unbounded — each turn writes one JSON file (~30-200KB)
# and a single 30-min voice session can produce hundreds. If the user
# flips JARVIS_DUMP_PROMPTS=1 and forgets, the disk fills silently.
_MAX_DUMP_FILES = 200
_MAX_DUMP_AGE_SECS = 7 * 24 * 3600  # 7 days


def _prune_dumps(dump_dir: Path) -> None:
    """Drop oldest dumps + anything older than 7 days.

    Best-effort: failures are silent. Runs O(N) once per dump call —
    cheap at N≤200.
    """
    try:
        files = sorted(
            dump_dir.glob("turn-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        now = time.time()
        # Drop by age first (cheap), then by count.
        for f in files:
            try:
                if (now - f.stat().st_mtime) > _MAX_DUMP_AGE_SECS:
                    f.unlink()
            except Exception:
                pass
        # Re-list and trim by count.
        files = sorted(
            dump_dir.glob("turn-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files[_MAX_DUMP_FILES:]:
            try:
                f.unlink()
            except Exception:
                pass
    except Exception:
        pass


def is_enabled() -> bool:
    """Return True when the user has opted in via the env var."""
    return os.environ.get(_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def new_session_id() -> str:
    """A short per-reply identifier so a session's turns sort together on disk."""
    return uuid.uuid4().hex[:8]


def _dump_dir() -> Path:
    base = Path.home() / ".local" / "share" / "jarvis" / "prompts"
    base.mkdir(parents=True, exist_ok=True)
    return base


def dump_reply_turn(
    *,
    session_id: str,
    turn: int,
    query: str,
    model: str,
    messages: list,
    tools_schema: Optional[list],
    use_text_tools: bool,
    response: Any = None,
    error: Optional[str] = None,
) -> Optional[Path]:
    """Write one turn's full LLM input/output to disk.

    Returns the path written, or None when dumping is disabled or failed.
    Failure is swallowed — diagnostics must never break the reply loop.
    """
    if not is_enabled():
        return None
    try:
        dump_dir = _dump_dir()
        # Audit round 10: redact + retention.
        # `redact` strips PII patterns (tokens, secrets, hex hashes,
        # JWTs) from each message before writing. The dump still
        # contains the structural info needed for an eval reproduce
        # (model, tool schema, message order, query intent) but not
        # the verbatim secrets that may have flowed through a tool.
        # If the user opts in and forgets, at least the on-disk
        # exposure is bounded — and rotation keeps the directory
        # from filling the disk.
        _prune_dumps(dump_dir)
        ts = time.strftime("%Y%m%dT%H%M%S")
        path = dump_dir / f"turn-{ts}-{session_id}-t{turn:02d}.json"
        redacted_messages = []
        for m in (messages or []):
            try:
                if isinstance(m, dict):
                    rm = dict(m)  # shallow copy
                    if isinstance(rm.get("content"), str):
                        rm["content"] = redact(rm["content"])
                    redacted_messages.append(rm)
                else:
                    redacted_messages.append(m)
            except Exception:
                redacted_messages.append(m)
        redacted_response = response
        if isinstance(response, str):
            try:
                redacted_response = redact(response)
            except Exception:
                pass
        payload = {
            "timestamp": time.time(),
            "session_id": session_id,
            "turn": turn,
            "query": redact(query) if query else query,
            "model": model,
            "use_text_tools": use_text_tools,
            "tools_schema": tools_schema,
            "messages": redacted_messages,
            "response": redacted_response,
            "error": error,
            "_redacted": True,  # marker so debugging knows dump is sanitised
        }
        # default=str keeps us safe if something non-serialisable slips in
        # (e.g. a bytes field from an upstream response body).
        # Audit round 13 fix: atomic write via .tmp + os.replace so a
        # crash mid-write doesn't leave a corrupt JSON file behind
        # (the dumps are diagnostic, but a partially-written .json
        # file confuses post-mortem readers).
        import os as _os
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        _os.replace(tmp_path, path)
        print(f"  📝 Prompt dump (redacted): {path}", flush=True)
        debug_log(f"Wrote prompt dump to {path}", "planning")
        return path
    except Exception as exc:  # pragma: no cover — diagnostics must not crash the reply loop
        debug_log(f"prompt dump failed: {exc}", "planning")
        return None

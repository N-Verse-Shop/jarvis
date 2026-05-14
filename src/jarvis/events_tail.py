"""CLI viewer for the typed event stream.

Like `tail -f` but with colour-coded types and pretty-printed payloads.
Useful for debugging: spot what the daemon is actually doing without
digging through journalctl.

Run:
    python -m jarvis.events_tail              # tail current events
    python -m jarvis.events_tail --filter tool_call,error
    python -m jarvis.events_tail --since 1m   # show last minute
    python -m jarvis.events_tail --all        # include rotated files
    python -m jarvis.events_tail --json       # raw JSONL passthrough

Exits cleanly on Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reuse path resolution from the writer side.
from .ipc.stream import _default_events_path

# Per-type colour codes — ANSI 256.
COLORS = {
    "daemon_startup":  "\033[1;36m",  # bold cyan
    "daemon_shutdown": "\033[1;36m",
    "state":           "\033[35m",    # magenta
    "stt_final":       "\033[32m",    # green
    "stt_partial":     "\033[2;32m",  # dim green
    "sentence":        "\033[36m",    # cyan
    "token":           "\033[2;36m",  # dim cyan
    "tool_call":       "\033[33m",    # yellow
    "action":          "\033[33m",
    "language_switch": "\033[1;34m",  # bold blue
    "error":           "\033[1;31m",  # bold red
    "log":             "\033[2;37m",  # dim grey
    "vad":             "\033[2;36m",
    "wake_word":       "\033[1;33m",  # bold yellow
    "hot_window":      "\033[34m",
    "llm_request":     "\033[35m",
    "llm_response":    "\033[35m",
    "memory":          "\033[33m",
}
RESET = "\033[0m"
DIM = "\033[2m"


def _format_ts(ts: float) -> str:
    """Format seconds-since-epoch as HH:MM:SS.mmm."""
    lt = time.localtime(ts)
    ms = int((ts - int(ts)) * 1000)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}.{ms:03d}"


def _format_event(ev: dict, no_color: bool = False) -> str:
    """Render a single event for terminal display."""
    typ = ev.get("type", "?")
    ts = _format_ts(ev.get("ts", time.time()))
    seq = ev.get("seq", 0)

    color = "" if no_color else COLORS.get(typ, "")
    reset = "" if no_color else RESET
    dim = "" if no_color else DIM

    # Build a compact payload string — strip control fields.
    payload = {k: v for k, v in ev.items() if k not in ("type", "ts", "seq")}
    if payload:
        # Special-case the most common types for readability.
        if typ == "stt_final":
            text = payload.get("text", "")
            lang = payload.get("lang", "")
            extra = f"[{lang}] «{text}»"
        elif typ == "sentence":
            text = payload.get("text", "")
            idx = payload.get("idx", "?")
            extra = f"#{idx} «{text}»"
        elif typ == "tool_call":
            tool = payload.get("tool", "?")
            status = payload.get("status", "?")
            dur = payload.get("duration_ms")
            dur_s = f" ({dur}ms)" if dur is not None else ""
            res = payload.get("result", {})
            msg = res.get("message", "") if isinstance(res, dict) else ""
            extra = f"{tool} {status}{dur_s} {msg}".strip()
        elif typ == "state":
            extra = f"{payload.get('previous', '?')} → {payload.get('state', '?')}"
        elif typ == "daemon_startup":
            extra = (f"v{payload.get('version', '?')} pid={payload.get('pid', '?')} "
                     f"chat={payload.get('chat_model', '?')}")
        elif typ == "daemon_shutdown":
            extra = payload.get("reason", "")
        elif typ == "language_switch":
            extra = f"{payload.get('from_lang', '?')} → {payload.get('to_lang', '?')}"
        elif typ == "error":
            extra = f"[{payload.get('component', '?')}] {payload.get('message', '')}"
        else:
            extra = json.dumps(payload, ensure_ascii=False, default=str)[:200]
    else:
        extra = ""

    return f"{dim}{ts} #{seq:>5}{reset} {color}{typ:<16}{reset} {extra}"


def _read_lines_incrementally(path: Path, offset: int) -> tuple[list[str], int]:
    """Read appended lines starting at byte offset. Return (lines, new_offset)."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return [], offset
    # Rotation: file shrank → restart
    if stat.st_size < offset:
        offset = 0
    if stat.st_size == offset:
        return [], offset
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(stat.st_size - offset)
        text = chunk.decode("utf-8", errors="replace")
        new_offset = stat.st_size
        # Keep incomplete trailing line in offset (don't advance past it)
        if not text.endswith("\n"):
            last_nl = text.rfind("\n")
            if last_nl == -1:
                return [], offset  # no complete line yet
            new_offset = offset + len(text[:last_nl + 1].encode("utf-8"))
            text = text[:last_nl + 1]
        return [ln for ln in text.split("\n") if ln.strip()], new_offset
    except Exception:
        return [], offset


def _parse_since(s: str) -> float:
    """Parse '30s', '5m', '1h' into seconds-from-now (negative offset).
    Returns absolute timestamp."""
    if not s:
        return 0.0
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    s = s.strip().lower()
    if s[-1] in units:
        try:
            return time.time() - float(s[:-1]) * units[s[-1]]
        except ValueError:
            return 0.0
    try:
        return time.time() - float(s)
    except ValueError:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jarvis.events_tail",
                                description="Tail the daemon's typed event stream.")
    p.add_argument("--filter", "-f", default="",
                   help="Comma-separated event types to show (default: all).")
    p.add_argument("--exclude", "-e", default="",
                   help="Comma-separated event types to hide.")
    p.add_argument("--since", "-s", default="",
                   help="Show events newer than this (e.g. '30s', '5m', '1h').")
    p.add_argument("--all", "-a", action="store_true",
                   help="Include rotated files (.jsonl.1/.2/.3) before tailing.")
    p.add_argument("--json", "-j", action="store_true",
                   help="Output raw JSONL (machine-readable).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colors.")
    p.add_argument("--path", default="",
                   help="Override events file path.")
    p.add_argument("--no-follow", "-n", action="store_true",
                   help="Print existing events and exit (don't tail).")
    args = p.parse_args(argv)

    path = Path(args.path) if args.path else _default_events_path()
    no_color = args.no_color or not sys.stdout.isatty()

    include = set(filter(None, args.filter.split(",")))
    exclude = set(filter(None, args.exclude.split(",")))
    since_ts = _parse_since(args.since) if args.since else 0.0

    def _emit(ev: dict) -> None:
        typ = ev.get("type", "")
        if include and typ not in include:
            return
        if typ in exclude:
            return
        if since_ts and ev.get("ts", 0) < since_ts:
            return
        if args.json:
            print(json.dumps(ev, ensure_ascii=False), flush=True)
        else:
            print(_format_event(ev, no_color=no_color), flush=True)

    # Optionally replay rotated files first
    if args.all:
        for i in range(3, 0, -1):  # .3 oldest → .1 newest archived
            rotated = path.with_suffix(f".jsonl.{i}")
            if rotated.exists():
                try:
                    with open(rotated, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                _emit(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                except OSError:
                    continue

    # Read the current file from start
    offset = 0
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        _emit(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            offset = path.stat().st_size
        except OSError:
            offset = 0
    else:
        # Print friendly hint and wait for daemon to create the file
        if not args.no_follow:
            print(f"# {path} doesn't exist yet — waiting for daemon to start...",
                  file=sys.stderr)

    if args.no_follow:
        return 0

    # Tail loop
    try:
        while True:
            lines, offset = _read_lines_incrementally(path, offset)
            for line in lines:
                try:
                    _emit(json.loads(line))
                except json.JSONDecodeError:
                    continue
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

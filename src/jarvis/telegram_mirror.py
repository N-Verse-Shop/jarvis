"""Telegram event mirror — pipes daemon's typed events into a Telegram chat.

Runs as a separate long-lived process on the Mac (alongside the voice
daemon). Tails events.jsonl, batches sentences with a short debounce
window, and posts to Telegram via Bot API. The voice daemon itself
stays unaware — pure read-only consumer of the event stream.

Why a separate process:
  • Voice loop must never block on outbound HTTP.
  • Easy to restart independently of the daemon.
  • Future: replace with a different sink (Slack, Discord, log shipper)
    without touching the daemon.

Setup:
  export JARVIS_TG_BOT_TOKEN="123:abc"      # @BotFather token
  export JARVIS_TG_CHAT_ID="-1001234567890"  # group/channel/dm id
  python -m jarvis.telegram_mirror &

Optional:
  JARVIS_TG_TYPES   — comma list of event types to forward
                      (default: stt_final,sentence,tool_call,error,
                                language_switch,daemon_startup,
                                daemon_shutdown)
  JARVIS_TG_DEBOUNCE_MS — how long to wait collecting more 'sentence'
                          events before flushing one telegram msg
                          (default: 1500)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from .ipc.stream import _default_events_path

# Defaults — overridable via env.
DEFAULT_TYPES = {
    "stt_final", "sentence", "tool_call", "error",
    "language_switch", "daemon_startup", "daemon_shutdown",
}
DEFAULT_DEBOUNCE_MS = 1500


def _send_telegram(token: str, chat_id: str, text: str,
                   parse_mode: str = "HTML") -> bool:
    """Synchronous Telegram sendMessage. Returns True on 200."""
    if not text.strip():
        return True
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text[:4090],  # Telegram limit 4096; leave buffer for parse_mode
        "parse_mode": parse_mode,
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception as e:
        print(f"# telegram send failed: {e}", file=sys.stderr, flush=True)
        return False


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _format_event(ev: dict) -> Optional[str]:
    """Return HTML-formatted Telegram message for an event, or None to skip."""
    typ = ev.get("type", "")
    payload = {k: v for k, v in ev.items() if k not in ("type", "ts", "seq")}

    if typ == "stt_final":
        text = _html_escape(payload.get("text", ""))
        lang = payload.get("lang", "?")
        return f"🎙 <i>«{text}»</i> <code>[{lang}]</code>"

    if typ == "sentence":
        # Sentences are batched separately — not formatted here.
        return None

    if typ == "tool_call":
        status = payload.get("status", "?")
        tool = payload.get("tool", "?")
        if status == "starting":
            return f"⚙️ <code>{_html_escape(tool)}</code> starting…"
        dur = payload.get("duration_ms")
        dur_s = f" ({dur}ms)" if dur is not None else ""
        result = payload.get("result", {})
        msg = result.get("message", "") if isinstance(result, dict) else ""
        if status == "completed":
            return f"✅ <code>{_html_escape(tool)}</code>{dur_s} {_html_escape(msg)}"
        err = payload.get("error", "")
        return f"❌ <code>{_html_escape(tool)}</code>{dur_s} {_html_escape(err or msg)}"

    if typ == "language_switch":
        return (f"🌐 {payload.get('from_lang', '?')} → "
                f"{payload.get('to_lang', '?')}")

    if typ == "error":
        return (f"⚠️ <b>{_html_escape(payload.get('component', '?'))}</b>: "
                f"{_html_escape(payload.get('message', ''))}")

    if typ == "daemon_startup":
        return (f"🟢 <b>Daemon up</b> v{payload.get('version', '?')} "
                f"pid={payload.get('pid', '?')} "
                f"<code>{payload.get('chat_model', '?')}</code>")

    if typ == "daemon_shutdown":
        return f"🔴 <b>Daemon down</b>: {payload.get('reason', 'unknown')}"

    return None


def _read_lines_incrementally(path: Path, offset: int) -> tuple[list[str], int]:
    """Read appended lines from offset. Returns (lines, new_offset)."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return [], offset
    if stat.st_size < offset:
        offset = 0  # rotation
    if stat.st_size == offset:
        return [], offset
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(stat.st_size - offset)
        text = chunk.decode("utf-8", errors="replace")
        new_offset = stat.st_size
        if not text.endswith("\n"):
            last_nl = text.rfind("\n")
            if last_nl == -1:
                return [], offset
            new_offset = offset + len(text[:last_nl + 1].encode("utf-8"))
            text = text[:last_nl + 1]
        return [ln for ln in text.split("\n") if ln.strip()], new_offset
    except Exception:
        return [], offset


def main() -> int:
    token = os.environ.get("JARVIS_TG_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("JARVIS_TG_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("# missing JARVIS_TG_BOT_TOKEN or JARVIS_TG_CHAT_ID env",
              file=sys.stderr)
        return 1

    types = set(os.environ.get("JARVIS_TG_TYPES", "").split(",")) - {""}
    if not types:
        types = DEFAULT_TYPES.copy()

    debounce_ms = int(os.environ.get("JARVIS_TG_DEBOUNCE_MS",
                                      str(DEFAULT_DEBOUNCE_MS)))

    path = _default_events_path()
    print(f"# tailing {path}", file=sys.stderr, flush=True)
    print(f"# filtering types: {sorted(types)}", file=sys.stderr, flush=True)
    print(f"# debounce: {debounce_ms}ms", file=sys.stderr, flush=True)

    # Seek to end on startup — don't replay historical events.
    offset = path.stat().st_size if path.exists() else 0

    # Sentence accumulator + flush state
    sentence_buf: list[str] = []
    sentence_buf_started: float = 0.0

    def flush_sentences() -> None:
        nonlocal sentence_buf, sentence_buf_started
        if not sentence_buf:
            return
        text = " ".join(s.strip() for s in sentence_buf if s.strip())
        if text:
            _send_telegram(token, chat_id, f"🗣 {_html_escape(text)}")
        sentence_buf = []
        sentence_buf_started = 0.0

    while True:
        lines, offset = _read_lines_incrementally(path, offset)
        now_ms = time.time() * 1000

        for line in lines:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = ev.get("type", "")
            if typ not in types:
                continue

            if typ == "sentence":
                txt = ev.get("text", "")
                if txt:
                    sentence_buf.append(txt)
                    if not sentence_buf_started:
                        sentence_buf_started = now_ms
                continue

            # Non-sentence event — flush any pending sentences first to
            # preserve chronological order in the chat.
            if sentence_buf:
                flush_sentences()

            msg = _format_event(ev)
            if msg:
                _send_telegram(token, chat_id, msg)

        # Debounce flush: if sentences have been accumulating > N ms
        # without any new event triggering flush, send them now.
        if (sentence_buf and sentence_buf_started
                and (now_ms - sentence_buf_started) >= debounce_ms):
            flush_sentences()

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)

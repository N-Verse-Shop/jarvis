"""Self-upgrade — Jarvis evolves its own code via Claude Code CLI.

The user says: "Джарвіс, я хочу щоб ти краще обробляв перебивання".
Voice daemon:

  1. parse_upgrade_request() catches the intent from the LLM reply.
  2. write_upgrade_brief() captures: user's request (verbatim), current
     daemon git SHA, recent error log tail, file map. This is the brief
     Claude Code will work from.
  3. spawn_claude() runs the `claude` CLI under Danylo's Max
     subscription, headless, with permissions auto-accepted.
  4. wait_and_restart() polls git status; when worktree changes,
     reload the daemon.

The Claude CLI must be installed locally (`npm i -g @anthropic-ai/
claude-code`) and authenticated as Danylo. Token-cost is on his Max
subscription, not API.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from ..debug import debug_log


REPO = Path(os.environ.get(
    "JARVIS_REPO", str(Path.home() / "Projects" / "jarvis-isair"),
))
BRIEF_DIR = Path.home() / ".config" / "jarvis" / "upgrade-briefs"
LOG_TAIL_FILE = Path.home() / "Library" / "Logs" / "jarvis-assistant.err.log"


def write_upgrade_brief(user_request: str) -> Path:
    """Persist a brief for Claude Code with all the context it needs."""
    BRIEF_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    brief_path = BRIEF_DIR / f"brief-{ts}.md"

    # Current git SHA so Claude knows the baseline.
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO), text=True,
        ).strip()
    except Exception:
        git_sha = "(unknown)"

    # Last 80 lines of voice error log.
    log_tail = ""
    try:
        if LOG_TAIL_FILE.exists():
            with LOG_TAIL_FILE.open() as f:
                lines = f.readlines()
            log_tail = "".join(lines[-80:])
    except Exception as e:
        log_tail = f"(log read failed: {e})"

    # Key files Claude should know about.
    file_map = (
        "Key entry points:\n"
        "- src/jarvis/listening/listener.py — main voice loop\n"
        "- src/jarvis/listening/state_manager.py — hot window / states\n"
        "- src/jarvis/listening/action_dispatcher.py — PC-control actions\n"
        "- src/jarvis/listening/wake_detection.py — wake word matching\n"
        "- src/jarvis/output/tts.py — Piper + macOS say\n"
        "- ~/.config/jarvis/config.json — runtime config (DO NOT commit secrets)\n"
    )

    content = (
        f"# Jarvis self-upgrade brief — {ts}\n\n"
        f"## User request (verbatim)\n\n"
        f"> {user_request}\n\n"
        f"## Current state\n\n"
        f"- Git SHA: `{git_sha}`\n"
        f"- Repo: `{REPO}`\n"
        f"- Daemon log: `{LOG_TAIL_FILE}`\n\n"
        f"## File map\n\n"
        f"```\n{file_map}```\n\n"
        f"## Recent daemon errors (tail of err log)\n\n"
        f"```\n{log_tail[-4000:]}\n```\n\n"
        f"## Your task\n\n"
        f"1. Read the user request carefully.\n"
        f"2. Find the relevant code in the file map.\n"
        f"3. Make minimal, focused changes.\n"
        f"4. If you're adding config knobs, also update the JSON schema in `config.py`.\n"
        f"5. Run `python3 -c \"import ast; ast.parse(open('src/jarvis/listening/listener.py').read())\"` after edits to catch syntax errors.\n"
        f"6. Stage and commit ONLY the files you changed.\n"
        f"7. Print a short summary at the end so the daemon TTS knows what to report.\n"
    )
    brief_path.write_text(content, encoding="utf-8")
    debug_log(f"upgrade brief written: {brief_path}", "voice")
    return brief_path


def spawn_claude(brief_path: Path) -> subprocess.Popen | None:
    """Spawn `claude` CLI on the brief. Returns the running process.

    Requires `claude` to be in PATH (npm i -g @anthropic-ai/claude-code).
    Auth uses Danylo's Max subscription — already configured if he's
    used Claude Code locally before.
    """
    # Find claude binary.
    claude_bin = None
    for candidate in ("claude", "/usr/local/bin/claude", "/opt/homebrew/bin/claude"):
        try:
            r = subprocess.run(["which", candidate], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                claude_bin = r.stdout.strip()
                break
        except Exception:
            continue
    if not claude_bin:
        debug_log("self-upgrade: `claude` CLI not found in PATH", "voice")
        return None

    # Build the command. Use --print mode so we get output without TTY.
    cmd = [
        claude_bin,
        "--dangerously-skip-permissions",  # auto-accept all tool use
        "--max-turns", "60",
        "--print",  # non-interactive
        f"Read {brief_path} and execute the task described inside. "
        f"Repository root: {REPO}. Work directly on the main branch.",
    ]
    debug_log(f"spawning: {' '.join(cmd[:3])} ...", "voice")
    log_path = BRIEF_DIR / f"{brief_path.stem}.log"
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO),
        stdout=log_f, stderr=subprocess.STDOUT,
        # Important: claude reads ~/.claude credentials, which only
        # exist in Danylo's HOME. We pass HOME explicitly so a
        # launchd-spawned daemon also finds them.
        env={**os.environ, "HOME": str(Path.home())},
    )
    return proc


def wait_for_completion_and_restart(proc: subprocess.Popen, max_wait_sec: int = 1800) -> tuple[bool, str]:
    """Wait up to `max_wait_sec` for Claude to finish. Then restart daemon.

    Returns (success, summary_message_for_tts).
    """
    start = time.monotonic()
    while proc.poll() is None:
        if time.monotonic() - start > max_wait_sec:
            try:
                proc.terminate()
            except Exception:
                pass
            return False, "Самооновлення тривало надто довго, перервав."
        time.sleep(2.0)

    if proc.returncode != 0:
        return False, f"Самооновлення завершилось з помилкою (код {proc.returncode}). Перевір лог."

    # Check if anything actually changed.
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO), text=True,
        ).strip()
        commits = subprocess.check_output(
            ["git", "log", "--oneline", "-5"], cwd=str(REPO), text=True,
        ).strip()
    except Exception:
        status, commits = "", ""

    if not status and "self-upgrade" not in commits.lower():
        return True, "Самооновлення завершено без змін у коді."

    # Syntax check the main file.
    try:
        import ast
        with (REPO / "src/jarvis/listening/listener.py").open() as f:
            ast.parse(f.read())
    except SyntaxError as e:
        return False, f"Самооновлення зробило syntax error: {e}. Не перезапускаю."

    # All good — restart the daemon.
    try:
        subprocess.run(
            ["launchctl", "kickstart", "-k",
             f"gui/{os.getuid()}/com.jarvis.assistant"],
            timeout=10, check=False,
        )
    except Exception as e:
        return True, f"Код оновлено, але не зміг перезапуститись: {e}. Перезапусти вручну."

    return True, "Самооновлення завершено. Перезапускаюсь."


def is_upgrade_request(query: str) -> bool:
    """Detect an upgrade-request intent in the user's utterance."""
    if not query:
        return False
    t = query.lower()
    triggers = [
        "оновись", "оновись сам", "сам себе оновіть",
        "покращ себе", "покращ свою роботу", "виправ собі",
        "самооновлення", "self upgrade", "self-upgrade",
        "я хочу щоб ти краще", "я хочу щоб ти покращ",
    ]
    return any(t.find(w) >= 0 for w in triggers)

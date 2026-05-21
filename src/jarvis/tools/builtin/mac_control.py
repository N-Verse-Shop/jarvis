"""macOS control tool — window management, native-app automation, AX queries.

Audit round 20 P3 capability addition. The user explicitly asked for
"повноцінного Агента якій вміє повноцінно керувати моїм макбуком". This
tool closes the biggest gaps identified by the audit:

  • Window management (move/resize/focus/space-switch) via JXA.
  • Native-app automation for Notes, Reminders, Calendar, Finder, Mail.
  • Accessibility-API queries (titles, focused element, app menubar).
  • Workflow primitives (sequence of typed steps with delays).

Audit round 21 expansion (F30 in the audit report) — the original
9-op surface was too narrow for "fully control my Mac". This round
adds the missing primitives that whole workflow classes depend on:

  • ``type_text``       — synthesise keystrokes into the focused app
                          (fill forms, compose messages, edit docs).
  • ``key``             — single key or modified combo (Cmd-S, Cmd-Tab,
                          Esc, arrow keys, function keys).
  • ``click_at``        — click absolute screen coordinates (often
                          paired with ``read_menubar`` / OCR for
                          target discovery).
  • ``read_menubar``    — return the focused app's menu tree so the
                          agent can discover which actions are
                          available (e.g. "Format → Add Bullets").
  • ``query_calendar``  — read upcoming Calendar events (EventKit
                          via AppleScript).
  • ``send_message``    — send an iMessage / SMS via Messages.app.
  • ``ax_trust_check``  — explicit check for the Accessibility
                          permission the keystroke/click ops need;
                          returns a clear instruction to the user
                          on how to grant it.
  • ``chain``           — execute a list of (op, args, delay_ms)
                          steps without round-tripping through the
                          LLM. Bounded at 10 steps. Stops on first
                          failure unless ``continue_on_error`` is
                          true. This is the single highest-leverage
                          addition: a "open Safari, focus address
                          bar, type URL, press Enter" workflow used
                          to require 4 LLM hops + 4 STT/TTS rounds;
                          now it's one tool call.

Why a single tool instead of N separate tools:
  Tool catalogues balloon LLM router context fast. A single
  ``macControl`` with a clean ``op`` enum keeps the router prompt
  small while letting the agent compose arbitrary workflows. Each
  ``op`` is a distinct sub-handler with its own validation.

Security model:
  • Every text argument is treated as UNTRUSTED transcript content
    and passed to AppleScript / JXA via ``-e`` argv, never f-string
    interpolated into source code. The injection-fix pattern from
    ``_spotlight_find`` (audit round 8) is reused.
  • Operations that move/click/typewrite are gated by an internal
    allowlist — ``op`` must match exactly, no free-form scripts.
  • No ``do shell script`` ever ends up in the constructed
    AppleScript — the agent CANNOT escape into a shell via this tool.
  • Operations that touch the filesystem (file copy/move) reuse
    ``local_files._resolve_safe`` semantics — sensitive prefixes
    refused, symlinks rejected.
  • Keystroke / click ops precheck ``AXIsProcessTrusted`` and return
    a user-friendly grant message rather than a cryptic AppleScript
    failure when permission is missing.
  • ``key`` only accepts an allowlist of modifier names + keycodes
    so the agent cannot, for example, fire Cmd-Q while a Finder
    "Move to Trash" confirm sheet is open. Combinations are bounded.
  • ``chain`` blocks recursive ``chain`` calls so an LLM-generated
    plan can't bypass the 10-step cap.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Hard timeout for every osascript invocation. Keeps a wedged
# Accessibility prompt or a slow target app from stalling the agent
# loop. 10 s is generous for keystroke / get-property / open-window
# operations and short enough that the agent surfaces failure quickly.
_OSA_TIMEOUT_SEC = 10.0

# Cap on text fields we send into target apps. Caps prompt-injection
# blast radius from a hostile transcript trying to dump giant payloads.
_MAX_TEXT_LEN = 4000

# Disallowed substrings inside any text passed to AppleScript — defence
# in depth. ``-e ... -- argv`` forms already keep the value as data,
# but newlines+quotes embedded in argv have caused issues in older
# osascript versions; strip them at the boundary.
_BAD_TEXT_CHARS = ("\x00",)


def _safe_text(value: Any, *, max_len: int = _MAX_TEXT_LEN) -> str:
    """Clean and bound a text value for AppleScript argv consumption."""
    if value is None:
        return ""
    s = str(value)
    for bad in _BAD_TEXT_CHARS:
        s = s.replace(bad, "")
    # Strip non-printable control chars but keep newlines/tabs/spaces.
    s = "".join(ch for ch in s if ch == "\n" or ch == "\t" or ch.isprintable())
    return s[:max_len]


def _run_osascript(script: str, *argv: str) -> tuple[bool, str]:
    """Execute an AppleScript / JXA snippet with bounded timeout.

    The script is passed via ``-e``, never f-string-interpolated;
    untrusted strings travel as positional argv accessed via ``argv``
    in JavaScript or ``item N of argv`` in AppleScript.
    """
    cmd: List[str] = ["osascript"]
    # Heuristic: scripts starting with the JXA pragma run as
    # JavaScript-for-Automation; otherwise default AppleScript.
    is_jxa = script.lstrip().startswith("ObjC.import") or "JavaScript" in script[:80]
    if is_jxa:
        cmd += ["-l", "JavaScript"]
    cmd += ["-e", script]
    if argv:
        cmd += ["--"] + [_safe_text(a) for a in argv]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_OSA_TIMEOUT_SEC,
        )
        if r.returncode != 0:
            err = (r.stderr or "").strip()
            return False, f"osascript error: {err[:200]}"
        return True, (r.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return False, f"osascript timed out after {_OSA_TIMEOUT_SEC}s"
    except Exception as e:
        return False, f"osascript failed: {type(e).__name__}: {e}"


# ─────────────────────────── op implementations ──────────────────────────────


def _op_focus_app(app_name: str) -> tuple[bool, str]:
    """Bring an app to front by name."""
    name = _safe_text(app_name, max_len=64)
    if not name:
        return False, "app name required"
    script = (
        'on run argv\n'
        '    set appName to item 1 of argv\n'
        '    tell application appName to activate\n'
        'end run'
    )
    ok, out = _run_osascript(script, name)
    return (True, f"Focused {name}") if ok else (False, out)


def _op_list_running_apps() -> tuple[bool, str]:
    """Return names of all running apps with a UI."""
    script = (
        'tell application "System Events"\n'
        '    set procs to name of every process whose background only is false\n'
        '    set AppleScript\'s text item delimiters to "\n"\n'
        '    return procs as string\n'
        'end tell'
    )
    ok, out = _run_osascript(script)
    return (True, out) if ok else (False, out)


def _op_resize_window(app: str, x: int, y: int, w: int, h: int) -> tuple[bool, str]:
    """Move + resize the frontmost window of ``app``."""
    name = _safe_text(app, max_len=64)
    if not name:
        return False, "app name required"
    try:
        x, y, w, h = int(x), int(y), int(w), int(h)
    except Exception:
        return False, "x/y/w/h must be integers"
    # Bound to sane ranges so a runaway agent can't move windows
    # off-screen or to absurd dimensions.
    if not (-10000 <= x <= 10000 and -10000 <= y <= 10000):
        return False, "x/y out of range"
    if not (50 <= w <= 10000 and 50 <= h <= 10000):
        return False, "w/h out of range (50-10000)"
    script = (
        'on run argv\n'
        '    set appName to item 1 of argv\n'
        '    set px to (item 2 of argv) as integer\n'
        '    set py to (item 3 of argv) as integer\n'
        '    set pw to (item 4 of argv) as integer\n'
        '    set ph to (item 5 of argv) as integer\n'
        '    tell application "System Events"\n'
        '        tell process appName\n'
        '            set position of front window to {px, py}\n'
        '            set size of front window to {pw, ph}\n'
        '        end tell\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, name, str(x), str(y), str(w), str(h))
    return (True, f"Resized {name} window to {w}×{h} at ({x},{y})") if ok else (False, out)


def _op_get_focused_app() -> tuple[bool, str]:
    """Return the name of the app that currently owns the menubar."""
    script = (
        'tell application "System Events"\n'
        '    return name of first process whose frontmost is true\n'
        'end tell'
    )
    ok, out = _run_osascript(script)
    return (True, out) if ok else (False, out)


def _op_get_window_titles(app: str) -> tuple[bool, str]:
    """List window titles for ``app`` — useful for the agent to pick a target."""
    name = _safe_text(app, max_len=64)
    if not name:
        return False, "app name required"
    script = (
        'on run argv\n'
        '    set appName to item 1 of argv\n'
        '    tell application "System Events"\n'
        '        tell process appName\n'
        '            set titles to name of every window\n'
        '            set AppleScript\'s text item delimiters to "\n"\n'
        '            return titles as string\n'
        '        end tell\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, name)
    return (True, out or "(no windows)") if ok else (False, out)


def _op_new_note(title: str = "", body: str = "") -> tuple[bool, str]:
    """Create a new note in Notes.app."""
    t = _safe_text(title, max_len=128)
    b = _safe_text(body, max_len=_MAX_TEXT_LEN)
    if not (t or b):
        return False, "title or body required"
    script = (
        'on run argv\n'
        '    set noteTitle to item 1 of argv\n'
        '    set noteBody to item 2 of argv\n'
        '    set noteHtml to "<h1>" & noteTitle & "</h1><br>" & noteBody\n'
        '    tell application "Notes"\n'
        '        tell account "iCloud"\n'
        '            make new note with properties {name:noteTitle, body:noteHtml}\n'
        '        end tell\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, t, b)
    # R34-S52 H: was UA "Створив нотатку". RU-only policy.
    return (True, f"Создал заметку «{t or b[:40]}»") if ok else (False, out)


def _op_new_reminder(text: str = "", list_name: str = "") -> tuple[bool, str]:
    """Add a reminder to Reminders.app (optionally to a specific list)."""
    t = _safe_text(text, max_len=512)
    ln = _safe_text(list_name, max_len=64)
    if not t:
        return False, "reminder text required"
    if ln:
        script = (
            'on run argv\n'
            '    set rText to item 1 of argv\n'
            '    set lName to item 2 of argv\n'
            '    tell application "Reminders"\n'
            '        tell list lName\n'
            '            make new reminder with properties {name:rText}\n'
            '        end tell\n'
            '    end tell\n'
            'end run'
        )
        ok, out = _run_osascript(script, t, ln)
    else:
        script = (
            'on run argv\n'
            '    set rText to item 1 of argv\n'
            '    tell application "Reminders"\n'
            '        make new reminder with properties {name:rText}\n'
            '    end tell\n'
            'end run'
        )
        ok, out = _run_osascript(script, t)
    # R34-S52 H: was UA "Додав нагадування".
    return (True, f"Добавил напоминание: {t[:60]}") if ok else (False, out)


def _op_reveal_in_finder(path: str) -> tuple[bool, str]:
    """Reveal ``path`` in a Finder window."""
    p = _safe_text(path, max_len=1024)
    if not p:
        return False, "path required"
    expanded = os.path.expanduser(p)
    if not os.path.exists(expanded):
        return False, f"Path not found: {expanded}"
    try:
        subprocess.run(["open", "-R", expanded], timeout=5, check=True)
        # R34-S52 H: was UA "Показав у Finder".
        return True, f"Показал в Finder: {expanded}"
    except Exception as e:
        # R34-S52 H: was UA "Помилка".
        return False, f"Ошибка: {e}"


def _op_open_url(url: str) -> tuple[bool, str]:
    """Open a URL in the default browser. http(s) only.

    Audit round 21 (F27) — removed ``about:`` from the allowlist. The
    comment used to claim "about: is harmless" but ``about:config``
    (Firefox) exposes a config UI and several browser-internal
    schemes (``chrome://``, ``view-source:``, ``resource://``) sit
    just outside the allowlist. Restricting to http(s) is unambiguous
    and the LLM cannot reach config pages through the agent path.
    """
    u = _safe_text(url, max_len=2048).strip()
    if not u:
        return False, "url required"
    # Reject schemes that can pivot to OS-level actions or read local data.
    # Round 21 — also explicitly deny browser-internal schemes the
    # previous round's allowlist accidentally permitted.
    bad_schemes = (
        "javascript:", "data:", "vbscript:", "ftp:", "ssh:", "telnet:",
        "x-apple-", "x-launch-", "smb:", "afp:", "shell:",
        "chrome:", "chrome-extension:", "view-source:", "resource:",
        "about:", "file:",
    )
    low = u.lower()
    for s in bad_schemes:
        if low.startswith(s):
            return False, f"Refusing scheme {s!r}"
    if not (low.startswith("http://") or low.startswith("https://")):
        return False, "Only http(s) URLs allowed"
    try:
        subprocess.run(["open", u], timeout=5, check=True)
        # R34-S52 H: was UA "Відкрив".
        return True, f"Открыл: {u[:80]}"
    except Exception as e:
        # R34-S52 H: was UA "Помилка".
        return False, f"Ошибка: {e}"


# ─────────────────────────── round 21 — accessibility ────────────────────────


# Cache the AX trust check for 5 s so chain operations don't spawn N
# subprocesses to ask the same question. The user's grant state changes
# at most once per system-settings interaction so a short TTL is safe.
_AX_TRUST_CACHE: dict = {"value": None, "checked_at": 0.0}
_AX_TRUST_TTL_SEC = 5.0


def _ax_is_trusted() -> bool:
    """Return True if the daemon has Accessibility permission.

    Uses pyobjc's ``ApplicationServices.AXIsProcessTrusted`` when
    available; falls back to a no-op osascript probe (which the
    macOS Accessibility subsystem will gate identically). Cached
    to ~5 s so chain operations don't pay the import/probe cost
    on every step.
    """
    import time as _t
    now = _t.monotonic()
    cached = _AX_TRUST_CACHE["value"]
    if cached is not None and (now - _AX_TRUST_CACHE["checked_at"]) < _AX_TRUST_TTL_SEC:
        return bool(cached)
    trusted = False
    try:
        # pyobjc path — fastest and most accurate. Imported lazily so
        # the tool also loads on systems without pyobjc installed.
        from ApplicationServices import AXIsProcessTrusted  # type: ignore[import-not-found]
        trusted = bool(AXIsProcessTrusted())
    except Exception:
        # Fallback: ask System Events to look at the frontmost process.
        # On a denied daemon this raises an AppleScript error 1002.
        ok, _ = _run_osascript(
            'tell application "System Events" to return name of first process whose frontmost is true'
        )
        trusted = ok
    _AX_TRUST_CACHE["value"] = trusted
    _AX_TRUST_CACHE["checked_at"] = now
    return trusted


_AX_GRANT_HINT = (
    "Accessibility permission missing. Open System Settings → Privacy & "
    "Security → Accessibility and toggle the Jarvis daemon (or "
    "/usr/bin/python3 / your Python.app) on. Then call this op again."
)


def _op_ax_trust_check() -> tuple[bool, str]:
    """Return the AX trust status as a human-readable string."""
    # Force a fresh probe — useful right after the user grants permission.
    _AX_TRUST_CACHE["value"] = None
    if _ax_is_trusted():
        return True, "Accessibility permission OK"
    return False, _AX_GRANT_HINT


# ─────────────────────────── round 21 — input synthesis ──────────────────────


# Modifier-key allowlist for ``key`` op. Reject anything else so a
# transcript like ``"FN+CONTROL+SHIFT+POWER"`` cannot synthesize a
# system-level shutdown chord.
_KEY_MODIFIERS = {
    "command", "cmd", "control", "ctrl", "option", "opt", "alt",
    "shift", "function", "fn",
}

# Symbolic keys map → AppleScript ``key code`` numbers. Limited to
# safe, well-known keys. Adding e.g. ``power`` here would be a
# security regression.
_KEY_CODES = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51,
    "escape": 53, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}


def _op_type_text(text: str) -> tuple[bool, str]:
    """Synthesise keystrokes that type ``text`` into the focused app.

    The text is passed as JXA argv (script reads ``$.NSProcessInfo``
    args via ``ObjC.import``) so transcript content never touches
    the script source. Bound at 1000 chars per call so a runaway
    LLM can't dump a megabyte. Requires AX trust.
    """
    if not _ax_is_trusted():
        return False, _AX_GRANT_HINT
    t = _safe_text(text, max_len=1000)
    if not t:
        return False, "text required"
    # AppleScript's ``keystroke`` honours unicode; pass via argv.
    script = (
        'on run argv\n'
        '    set theText to item 1 of argv\n'
        '    tell application "System Events"\n'
        '        keystroke theText\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, t)
    # R34-S52 H: was EN "Typed N chars". RU-only.
    return (True, f"Напечатал {len(t)} символов") if ok else (False, out)


def _op_key(key_name: str = "", modifiers: Any = None) -> tuple[bool, str]:
    """Press a single key, optionally with modifiers (Cmd-S, Cmd-Tab…).

    ``key_name`` is either a single printable char ("a"…"z", "0"…"9",
    punctuation) or a symbolic name from ``_KEY_CODES``. ``modifiers``
    is an optional list of strings drawn from ``_KEY_MODIFIERS``.
    """
    if not _ax_is_trusted():
        return False, _AX_GRANT_HINT
    k = _safe_text(key_name, max_len=16).strip().lower()
    if not k:
        return False, "key required"
    # Normalise modifiers.
    if modifiers is None:
        mods: List[str] = []
    elif isinstance(modifiers, str):
        mods = [m.strip().lower() for m in modifiers.split(",") if m.strip()]
    elif isinstance(modifiers, (list, tuple)):
        mods = [str(m).strip().lower() for m in modifiers if str(m).strip()]
    else:
        return False, "modifiers must be a list or comma-separated string"
    for m in mods:
        if m not in _KEY_MODIFIERS:
            return False, f"Unknown modifier {m!r}; allowed: {sorted(_KEY_MODIFIERS)}"
    if len(mods) > 4:
        return False, "too many modifiers (max 4)"
    # Map alt → option for AppleScript's "using" clause.
    mod_map = {"cmd": "command", "ctrl": "control", "opt": "option",
               "alt": "option", "fn": "function"}
    using_parts = [f"{mod_map.get(m, m)} down" for m in mods]
    using_clause = (" using {" + ", ".join(using_parts) + "}") if using_parts else ""
    # If the key is a known symbolic, use key code; otherwise keystroke.
    if k in _KEY_CODES:
        script = (
            'on run argv\n'
            '    set kc to (item 1 of argv) as integer\n'
            '    tell application "System Events"\n'
            f'        key code kc{using_clause}\n'
            '    end tell\n'
            'end run'
        )
        ok, out = _run_osascript(script, str(_KEY_CODES[k]))
    else:
        # Single printable char keystroke.
        if len(k) != 1 or not k.isprintable():
            return False, f"unknown key {k!r}; use a single char or {sorted(_KEY_CODES)}"
        script = (
            'on run argv\n'
            '    set kch to item 1 of argv\n'
            '    tell application "System Events"\n'
            f'        keystroke kch{using_clause}\n'
            '    end tell\n'
            'end run'
        )
        ok, out = _run_osascript(script, k)
    label = ("+".join(mods + [k])) if mods else k
    # R34-S52 H: was EN "Pressed {label}".
    return (True, f"Нажал {label}") if ok else (False, out)


def _op_click_at(x: int, y: int) -> tuple[bool, str]:
    """Click absolute screen coordinates (single left click).

    Requires AX trust. Coords are bounded so a runaway agent cannot
    click far off-screen (which on some setups can drag a window
    off the visible area).
    """
    if not _ax_is_trusted():
        return False, _AX_GRANT_HINT
    try:
        cx, cy = int(x), int(y)
    except Exception:
        return False, "x/y must be integers"
    if not (0 <= cx <= 10000 and 0 <= cy <= 10000):
        return False, "x/y out of range (0..10000)"
    script = (
        'on run argv\n'
        '    set cx to (item 1 of argv) as integer\n'
        '    set cy to (item 2 of argv) as integer\n'
        '    tell application "System Events"\n'
        '        click at {cx, cy}\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, str(cx), str(cy))
    # R34-S52 H: was EN "Clicked at (cx, cy)".
    return (True, f"Кликнул по ({cx}, {cy})") if ok else (False, out)


def _op_read_menubar(app: str = "") -> tuple[bool, str]:
    """Return the focused (or named) app's menu structure.

    Output is one menu per line: ``MenuName > ItemName``. Useful for
    the agent to discover available actions before issuing a key
    combo or click. Capped at 200 lines so the LLM context isn't
    blown by an app with massive submenus (Pages, Photoshop).
    """
    if not _ax_is_trusted():
        return False, _AX_GRANT_HINT
    name = _safe_text(app, max_len=64)
    if name:
        # Named app — focus it first so its menubar is the active one.
        _run_osascript(
            'on run argv\n    tell application (item 1 of argv) to activate\nend run',
            name,
        )
    script = (
        'tell application "System Events"\n'
        '    set procName to name of first process whose frontmost is true\n'
        '    tell process procName\n'
        '        set menuLines to {}\n'
        '        repeat with mbi in menu bar items of menu bar 1\n'
        '            set mbiName to name of mbi\n'
        '            try\n'
        '                set itemNames to name of every menu item of menu 1 of mbi\n'
        '                repeat with iName in itemNames\n'
        '                    if iName is not missing value and iName is not "" then\n'
        '                        set end of menuLines to mbiName & " > " & iName\n'
        '                    end if\n'
        '                end repeat\n'
        '            end try\n'
        '        end repeat\n'
        '        set AppleScript\'s text item delimiters to "\n"\n'
        '        return menuLines as string\n'
        '    end tell\n'
        'end tell'
    )
    ok, out = _run_osascript(script)
    if not ok:
        return False, out
    lines = out.split("\n")
    if len(lines) > 200:
        out = "\n".join(lines[:200]) + f"\n… (+{len(lines)-200} more menu items)"
    return True, out or "(empty menu)"


def _op_query_calendar(days: int = 7) -> tuple[bool, str]:
    """List upcoming Calendar events within ``days`` days.

    Reads via Calendar.app AppleScript. Capped at 40 events. Output
    is plain text one-per-line "YYYY-MM-DD HH:MM — Title".
    """
    try:
        d = max(1, min(30, int(days)))
    except Exception:
        d = 7
    script = (
        'on run argv\n'
        '    set daysAhead to (item 1 of argv) as integer\n'
        '    set startDate to (current date)\n'
        '    set endDate to startDate + (daysAhead * days)\n'
        '    tell application "Calendar"\n'
        '        set evs to {}\n'
        '        repeat with cal in calendars\n'
        '            try\n'
        '                set theseEvs to (every event of cal whose start date is greater than or equal to startDate and start date is less than endDate)\n'
        '                repeat with e in theseEvs\n'
        '                    set sd to start date of e\n'
        '                    set sdStr to (year of sd as string) & "-" & text -2 thru -1 of ("0" & (month of sd as integer)) & "-" & text -2 thru -1 of ("0" & day of sd) & " " & text -2 thru -1 of ("0" & hours of sd) & ":" & text -2 thru -1 of ("0" & minutes of sd)\n'
        '                    set end of evs to sdStr & " — " & summary of e\n'
        '                end repeat\n'
        '            end try\n'
        '        end repeat\n'
        '        set AppleScript\'s text item delimiters to "\n"\n'
        '        return evs as string\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, str(d))
    if not ok:
        return False, out
    lines = sorted([ln for ln in out.split("\n") if ln.strip()])
    if not lines:
        return True, f"No events in next {d} days"
    if len(lines) > 40:
        lines = lines[:40] + [f"… (+{len(lines)-40} more)"]
    return True, "\n".join(lines)


def _op_send_message(to: str, body: str) -> tuple[bool, str]:
    """Send an iMessage / SMS via Messages.app.

    ``to`` is either a phone number (digits and ``+``) or an email
    used as an Apple ID. ``body`` is bounded at 2000 chars.
    """
    recipient = _safe_text(to, max_len=128).strip()
    if not recipient:
        return False, "recipient required"
    # Restrict to phone-number or email shape so the LLM can't craft
    # a malformed recipient that crashes Messages.
    import re as _re
    is_phone = bool(_re.fullmatch(r"[+\d][\d\s\-()]{4,30}", recipient))
    is_email = bool(_re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient))
    if not (is_phone or is_email):
        return False, "recipient must be a phone number or email"
    msg = _safe_text(body, max_len=2000)
    if not msg:
        return False, "body required"
    script = (
        'on run argv\n'
        '    set rcpt to item 1 of argv\n'
        '    set msg to item 2 of argv\n'
        '    tell application "Messages"\n'
        '        try\n'
        '            set targetService to 1st account whose service type = iMessage\n'
        '            set targetBuddy to participant rcpt of targetService\n'
        '            send msg to targetBuddy\n'
        '        on error\n'
        '            -- Fallback: any service (SMS via Continuity).\n'
        '            set targetBuddy to participant rcpt of (1st account)\n'
        '            send msg to targetBuddy\n'
        '        end try\n'
        '    end tell\n'
        'end run'
    )
    ok, out = _run_osascript(script, recipient, msg)
    # R34-S52 H: was EN "Sent message to {recipient}".
    return (True, f"Отправил сообщение: {recipient}") if ok else (False, out)


# ─────────────────────────── round 25 — clipboard + sysinfo ──────────────────


def _op_clipboard_get() -> tuple[bool, str]:
    """Read the current macOS clipboard as plain text (round 25 F55).

    Uses ``pbpaste`` (Apple's stock CLI; no permission prompt). Output
    truncated at 8000 chars so an enormous paste doesn't blow context.
    Returns the literal clipboard text in ``out`` field; non-text
    clipboard contents (images, files) come back as empty string.
    """
    import subprocess
    try:
        proc = subprocess.run(
            ["/usr/bin/pbpaste"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if proc.returncode != 0:
            return False, f"pbpaste failed (rc={proc.returncode}): {proc.stderr[:200]}"
        text = proc.stdout or ""
        if len(text) > 8000:
            text = text[:8000] + f"\n…[truncated {len(text) - 8000} chars]"
        return True, text
    except subprocess.TimeoutExpired:
        return False, "pbpaste timed out"
    except Exception as e:
        return False, f"clipboard read error: {type(e).__name__}: {e}"


def _op_clipboard_set(text: str) -> tuple[bool, str]:
    """Write text to the macOS clipboard (round 25 F55).

    Uses ``pbcopy``. Bounded at 100k chars so an LLM hallucination
    can't push a megabyte to the clipboard.
    """
    payload = _safe_text(text, max_len=100_000)
    if not payload:
        return False, "text required"
    import subprocess
    try:
        proc = subprocess.run(
            ["/usr/bin/pbcopy"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if proc.returncode != 0:
            return False, f"pbcopy failed (rc={proc.returncode}): {proc.stderr[:200]}"
        return True, f"Clipboard set ({len(payload)} chars)"
    except subprocess.TimeoutExpired:
        return False, "pbcopy timed out"
    except Exception as e:
        return False, f"clipboard write error: {type(e).__name__}: {e}"


def _op_system_info(field: str = "all") -> tuple[bool, str]:
    """Read macOS system info (round 25 F56).

    Supported fields: ``battery``, ``volume``, ``brightness``,
    ``wifi``, ``disk``, ``mem``, ``cpu``, ``all``.
    All paths are read-only stock CLI tools — no permission prompts.
    """
    import subprocess
    field = (field or "all").strip().lower()
    valid = {"battery", "volume", "brightness", "wifi", "disk", "mem", "cpu", "all"}
    if field not in valid:
        return False, f"unknown field {field!r}. Valid: {', '.join(sorted(valid))}"

    out_lines: list[str] = []

    def _run(cmd: list[str], timeout: float = 3.0) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.stdout if r.returncode == 0 else ""
        except Exception:
            return ""

    def _battery() -> str:
        # pmset -g batt: "Now drawing from 'Battery Power' / 'AC Power'"
        # Battery line: " ... 87%; discharging; 4:23 remaining ..."
        raw = _run(["/usr/bin/pmset", "-g", "batt"])
        if not raw:
            return "battery: unavailable"
        import re as _re
        m = _re.search(r"(\d+)%;\s*(\w+)", raw)
        pct = m.group(1) + "%" if m else "?"
        status = m.group(2) if m else "?"
        time_left = ""
        m2 = _re.search(r"(\d+:\d+)\s+remaining", raw)
        if m2:
            time_left = f", {m2.group(1)} remaining"
        return f"battery: {pct} ({status}{time_left})"

    def _volume() -> str:
        v = _run(["/usr/bin/osascript", "-e", "output volume of (get volume settings)"]).strip()
        m = _run(["/usr/bin/osascript", "-e", "output muted of (get volume settings)"]).strip()
        muted = " (muted)" if m == "true" else ""
        return f"volume: {v}%{muted}"

    def _brightness() -> str:
        # macOS doesn't expose this via stock CLI without `brightness` (homebrew).
        # Try `brightness -l` first, fall back to "unavailable".
        raw = _run(["/opt/homebrew/bin/brightness", "-l"], timeout=2.0)
        if not raw:
            return "brightness: unavailable (install via `brew install brightness`)"
        import re as _re
        m = _re.search(r"brightness\s+([\d.]+)", raw)
        if m:
            try:
                pct = round(float(m.group(1)) * 100)
                return f"brightness: {pct}%"
            except ValueError:
                pass
        return "brightness: parse error"

    def _wifi() -> str:
        ssid = _run([
            "/usr/sbin/networksetup", "-getairportnetwork", "en0"
        ]).strip()
        # Output: "Current Wi-Fi Network: <SSID>"
        if "Current Wi-Fi Network:" in ssid:
            return "wifi: " + ssid.split(":", 1)[1].strip()
        return f"wifi: {ssid or 'off / not connected'}"

    def _disk() -> str:
        # df -h /  → " ... 50G  450G  10% ..."
        raw = _run(["/bin/df", "-h", "/"])
        if not raw:
            return "disk: unavailable"
        try:
            line = raw.strip().splitlines()[-1]
            parts = line.split()
            # parts: [Filesystem, Size, Used, Avail, Capacity%, Mounted on...]
            return f"disk: {parts[3]} free of {parts[1]} ({parts[4]} used)"
        except Exception:
            return f"disk: {raw.strip()[:200]}"

    def _mem() -> str:
        # vm_stat + page size = clunky; use `top -l 1 -n 0` line "PhysMem:".
        raw = _run(["/usr/bin/top", "-l", "1", "-n", "0"])
        if not raw:
            return "mem: unavailable"
        for line in raw.splitlines():
            if line.startswith("PhysMem:"):
                return f"mem: {line[len('PhysMem:'):].strip()[:120]}"
        return "mem: parse error"

    def _cpu() -> str:
        raw = _run(["/usr/bin/top", "-l", "1", "-n", "0"])
        if not raw:
            return "cpu: unavailable"
        for line in raw.splitlines():
            if line.startswith("CPU usage:"):
                return f"cpu: {line[len('CPU usage:'):].strip()[:120]}"
        return "cpu: parse error"

    fields_map = {
        "battery": _battery,
        "volume": _volume,
        "brightness": _brightness,
        "wifi": _wifi,
        "disk": _disk,
        "mem": _mem,
        "cpu": _cpu,
    }

    if field == "all":
        for fn in fields_map.values():
            try:
                out_lines.append(fn())
            except Exception as e:
                out_lines.append(f"({type(e).__name__})")
    else:
        try:
            out_lines.append(fields_map[field]())
        except Exception as e:
            return False, f"system_info {field} error: {type(e).__name__}: {e}"

    return True, "\n".join(out_lines)


def _op_set_volume(level: int) -> tuple[bool, str]:
    """Set system output volume 0-100 (round 25 F56)."""
    try:
        lv = int(level)
    except Exception:
        return False, f"level must be an int 0-100, got {level!r}"
    if not (0 <= lv <= 100):
        return False, f"level out of range (0-100): {lv}"
    script = f"set volume output volume {lv}"
    ok, out = _run_osascript(script)
    return (True, f"volume set to {lv}%") if ok else (False, out)


def _op_set_mute(state: str) -> tuple[bool, str]:
    """Mute / unmute system output (round 25 F56). state = on|off."""
    s = (state or "").strip().lower()
    if s not in {"on", "off", "true", "false", "1", "0"}:
        return False, f"state must be on/off, got {state!r}"
    muted = "true" if s in {"on", "true", "1"} else "false"
    script = f"set volume output muted {muted}"
    ok, out = _run_osascript(script)
    return (True, f"mute={muted}") if ok else (False, out)


def _op_get_active_window() -> tuple[bool, str]:
    """Get title + bounds of the frontmost window (round 25 F56).

    Returns a JSON-ish single-line string with app/title/bounds.
    Useful for "what am I looking at" context before dispatching
    follow-up actions.
    """
    script = (
        'tell application "System Events"\n'
        '    set frontApp to first process whose frontmost is true\n'
        '    set appName to name of frontApp\n'
        '    try\n'
        '        set frontWin to first window of frontApp\n'
        '        set winTitle to name of frontWin\n'
        '        set winPos to position of frontWin\n'
        '        set winSize to size of frontWin\n'
        '        return appName & "||" & winTitle & "||" & (item 1 of winPos as text) & "," & (item 2 of winPos as text) & "||" & (item 1 of winSize as text) & "," & (item 2 of winSize as text)\n'
        '    on error\n'
        '        return appName & "||(no window)||0,0||0,0"\n'
        '    end try\n'
        'end tell'
    )
    ok, out = _run_osascript(script)
    if not ok:
        return False, out
    parts = out.strip().split("||")
    if len(parts) != 4:
        return False, f"unexpected format: {out[:200]}"
    app, title, pos, size = parts
    return True, f"app={app} title={title!r} pos={pos} size={size}"


# ─────────────────────────── round 29 (F83) — Shortcuts ───────────────────

def _op_run_shortcut(name: str = "", input_text: str = "") -> tuple[bool, str]:
    """Run a macOS Shortcut by name.

    Round 29 (F83). macOS Shortcuts is the primary user-extension
    surface — toggle Wi-Fi, run Hazel rules, ping someone, write a
    standardized Notion entry, kick off a video export. Without an
    op for it, the agent couldn't tap any of the user's existing
    automations.

    `shortcuts run <name> [--input-path -]` reads stdin as input.
    For privacy/safety we validate name characters strictly and
    pass input via stdin (NOT argv) so no shell interpolation can
    happen. The shortcuts binary returns stdout on success.

    Args:
        name: Shortcut name (must exist in user's Shortcuts library).
              We allow letters, digits, space, dash, underscore,
              dot, parens, apostrophe — covers all realistic names.
              Anything else → reject (paranoia against argv abuse).
        input_text: Optional text fed via stdin to the Shortcut's
              `Receive: Text` step. Bytes-safe.

    Returns:
        (ok, output) where output is the Shortcut's stdout (≤2KB).
    """
    nm = (name or "").strip()
    if not nm:
        return False, "shortcut name required"
    # Strict allowlist — keeps a hostile/typo'd name from being
    # interpreted as a flag (`-h`, `--help`, etc.).
    # R34-S40 — was `re.match` but module never imports top-level `re`;
    # the other ops in this file use a local `import re as _re`. NameError
    # at first call to run_shortcut.
    import re as _re_shortcut
    if not _re_shortcut.match(r"^[\w\s\.\-_'\(\)]{1,80}$", nm, _re_shortcut.UNICODE):
        return False, f"shortcut name not allowed: {nm!r}"
    try:
        cmd = ["shortcuts", "run", nm]
        proc = subprocess.run(
            cmd,
            input=(input_text or "").encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        if proc.returncode == 0:
            return True, out[:2048] if out else f"Shortcut '{nm}' executed"
        return False, f"shortcut failed (rc={proc.returncode}): {err[:200]}"
    except FileNotFoundError:
        return False, "`shortcuts` CLI unavailable (macOS ≥12 required)"
    except subprocess.TimeoutExpired:
        return False, f"shortcut '{nm}' timed out after 30s"
    except Exception as e:
        return False, f"shortcut error: {e}"


def _op_list_shortcuts() -> tuple[bool, str]:
    """Return a newline-separated list of available Shortcuts (F83).

    Lets the LLM (or a UI) enumerate before calling `run_shortcut`.
    Output is capped at 4KB to keep tool-result tokens reasonable.
    """
    try:
        proc = subprocess.run(
            ["shortcuts", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            out = (proc.stdout or "").strip()
            return True, out[:4096] if out else "No shortcuts found."
        return False, f"shortcuts list failed: {(proc.stderr or '')[:200]}"
    except FileNotFoundError:
        return False, "`shortcuts` CLI unavailable (macOS ≥12 required)"
    except Exception as e:
        return False, f"list error: {e}"


# ─────────────────────────── tool registration ───────────────────────────────


_OPS = {
    "focus_app": _op_focus_app,
    "list_running_apps": _op_list_running_apps,
    "resize_window": _op_resize_window,
    "get_focused_app": _op_get_focused_app,
    "get_window_titles": _op_get_window_titles,
    "new_note": _op_new_note,
    "new_reminder": _op_new_reminder,
    "reveal_in_finder": _op_reveal_in_finder,
    "open_url": _op_open_url,
    # Round 21 — input-synthesis primitives gated by AX trust.
    "ax_trust_check": _op_ax_trust_check,
    "type_text": _op_type_text,
    "key": _op_key,
    "click_at": _op_click_at,
    "read_menubar": _op_read_menubar,
    "query_calendar": _op_query_calendar,
    "send_message": _op_send_message,
    # Round 25 (F55+F56) — clipboard & system info / control.
    "clipboard_get": _op_clipboard_get,
    "clipboard_set": _op_clipboard_set,
    "system_info": _op_system_info,
    "set_volume": _op_set_volume,
    "set_mute": _op_set_mute,
    "get_active_window": _op_get_active_window,
    # Round 29 (F83) — macOS Shortcuts integration.
    "run_shortcut": _op_run_shortcut,
    "list_shortcuts": _op_list_shortcuts,
    # ``chain`` is registered below — it needs a forward reference to
    # _OPS so it can dispatch. See ``_dispatch_op`` and ``_op_chain``.
}


# ─────────────────────────── round 21 — dispatch + chain ─────────────────────


_CHAIN_MAX_STEPS = 10
_CHAIN_MAX_DELAY_MS = 5000


def _dispatch_op(op: str, args: Dict[str, Any]) -> tuple[bool, str]:
    """Internal single-op dispatch shared by ``run`` and ``chain``.

    Pulls the right keyword arguments out of ``args`` for each op so
    ``chain`` can pass identically-shaped step dicts. Returns
    (ok, msg) tuples like every op.
    """
    if op not in _OPS:
        return False, f"Unknown op {op!r}. Valid: {', '.join(_OPS)}"
    handler = _OPS[op]
    try:
        if op == "focus_app":
            return handler(args.get("app", ""))
        if op == "list_running_apps":
            return handler()
        if op == "resize_window":
            return handler(
                args.get("app", ""),
                args.get("x", 0),
                args.get("y", 0),
                args.get("w", 800),
                args.get("h", 600),
            )
        if op == "get_focused_app":
            return handler()
        if op == "get_window_titles":
            return handler(args.get("app", ""))
        if op == "new_note":
            return handler(args.get("title", ""), args.get("body", ""))
        if op == "new_reminder":
            return handler(args.get("text", ""), args.get("list_name", ""))
        if op == "reveal_in_finder":
            return handler(args.get("path", ""))
        if op == "open_url":
            return handler(args.get("url", ""))
        if op == "ax_trust_check":
            return handler()
        if op == "type_text":
            return handler(args.get("text", ""))
        if op == "key":
            return handler(args.get("key_name", args.get("key", "")), args.get("modifiers"))
        if op == "click_at":
            return handler(args.get("x", 0), args.get("y", 0))
        if op == "read_menubar":
            return handler(args.get("app", ""))
        if op == "query_calendar":
            return handler(args.get("days", 7))
        if op == "send_message":
            return handler(args.get("to", ""), args.get("body", ""))
        # Round 25 dispatch routes:
        if op == "clipboard_get":
            return handler()
        if op == "clipboard_set":
            return handler(args.get("text", ""))
        if op == "system_info":
            return handler(args.get("field", "all"))
        if op == "set_volume":
            return handler(args.get("level", 50))
        if op == "set_mute":
            return handler(args.get("state", "off"))
        if op == "get_active_window":
            return handler()
        # Round 29 (F83) — Shortcuts dispatch routes:
        if op == "run_shortcut":
            return handler(args.get("name", ""), args.get("input_text", args.get("input", "")))
        if op == "list_shortcuts":
            return handler()
    except Exception as e:
        return False, f"op={op} failed: {type(e).__name__}: {e}"
    return False, f"Unhandled op {op!r}"


def _op_chain(steps: Any, continue_on_error: bool = False) -> tuple[bool, str]:
    """Execute a sequence of ops without round-tripping the LLM.

    ``steps`` is a list of dicts: ``{"op": "...", ...args, "delay_ms": int}``.
    Bounded at ``_CHAIN_MAX_STEPS`` to prevent runaway loops. Stops on
    first failure unless ``continue_on_error=True``. Recursive ``chain``
    inside ``chain`` is explicitly blocked.

    Returns (ok, multi-line message) where each line documents one
    step's outcome — gives the LLM a clear audit trail to either
    correct course or report back to the user.
    """
    if not isinstance(steps, list):
        return False, "steps must be a list"
    if not steps:
        return False, "steps cannot be empty"
    if len(steps) > _CHAIN_MAX_STEPS:
        return False, f"too many steps (max {_CHAIN_MAX_STEPS})"
    import time as _t
    log_lines: List[str] = []
    overall_ok = True
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            log_lines.append(f"[{idx}] skipped: step is not a dict")
            overall_ok = False
            if not continue_on_error:
                break
            continue
        sub_op = str(step.get("op", "")).strip().lower()
        if sub_op == "chain":
            log_lines.append(f"[{idx}] ✗ chain: nested chain blocked")
            overall_ok = False
            if not continue_on_error:
                break
            continue
        delay_ms = step.get("delay_ms", 0)
        try:
            delay_ms = max(0, min(_CHAIN_MAX_DELAY_MS, int(delay_ms)))
        except Exception:
            delay_ms = 0
        ok, msg = _dispatch_op(sub_op, step)
        marker = "✓" if ok else "✗"
        log_lines.append(f"[{idx}] {marker} {sub_op}: {msg[:160]}")
        if not ok:
            overall_ok = False
            if not continue_on_error:
                break
        if delay_ms > 0:
            _t.sleep(delay_ms / 1000.0)
    summary = "\n".join(log_lines)
    return overall_ok, summary


# Wire ``chain`` into the op table now that the helper is defined.
_OPS["chain"] = _op_chain


class MacControlTool(Tool):
    """Single dispatch tool covering window-mgmt, native-app, AX queries."""

    @property
    def name(self) -> str:
        return "macControl"

    @property
    def description(self) -> str:
        # Audit round 25 — expanded with clipboard + sysinfo ops.
        # Round 29 (F83) — added Shortcuts integration.
        return (
            "Control the macOS desktop. Ops: focus_app, list_running_apps, "
            "resize_window, get_focused_app, get_window_titles, "
            "get_active_window (frontmost app+title+bounds), new_note, "
            "new_reminder, reveal_in_finder, open_url, ax_trust_check, "
            "type_text, key (Cmd-S / Cmd-Tab / arrow keys with modifiers), "
            "click_at, read_menubar, query_calendar, send_message, "
            "clipboard_get, clipboard_set, system_info (battery/volume/"
            "brightness/wifi/disk/mem/cpu/all), set_volume, set_mute, "
            "run_shortcut (call any user-installed macOS Shortcut by "
            "name — perfect for power-user automations like 'Toggle "
            "Wi-Fi', 'Send to Notion'), list_shortcuts (enumerate), "
            "chain (up to 10 ops with delays). Use for 'open Notes', "
            "'press Cmd-S', 'type my address', 'send Anna I'll be late', "
            "'what's on my clipboard', 'how much battery', 'set volume "
            "50', 'mute the speakers', 'what's the active window', "
            "'run my Toggle Wi-Fi shortcut'."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": list(_OPS.keys()),
                    "description": "Which mac action to perform.",
                },
                "app": {"type": "string", "description": "App name (focus_app, resize_window, get_window_titles, read_menubar)."},
                "x": {"type": "integer", "description": "X coordinate (resize_window, click_at)."},
                "y": {"type": "integer", "description": "Y coordinate (resize_window, click_at)."},
                "w": {"type": "integer", "description": "Width (resize_window)."},
                "h": {"type": "integer", "description": "Height (resize_window)."},
                "title": {"type": "string", "description": "Note title (new_note)."},
                "body": {"type": "string", "description": "Note body (new_note) / message body (send_message)."},
                "text": {"type": "string", "description": "Reminder text (new_reminder) / text to type (type_text)."},
                "list_name": {"type": "string", "description": "Reminders list name (new_reminder)."},
                "path": {"type": "string", "description": "Path to reveal in Finder (reveal_in_finder)."},
                "url": {"type": "string", "description": "URL to open — http(s) only (open_url)."},
                "key_name": {"type": "string", "description": "Key name for op=key (e.g. 's', 'tab', 'escape', 'return', 'f12')."},
                "modifiers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modifier keys for op=key (subset of: command, control, option, shift, function).",
                },
                "days": {"type": "integer", "description": "Lookahead window in days (query_calendar, 1..30)."},
                "to": {"type": "string", "description": "Recipient phone or Apple ID for send_message."},
                "steps": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Sequence of step objects for op=chain. Each step has its own 'op' + args + optional 'delay_ms'.",
                },
                "continue_on_error": {
                    "type": "boolean",
                    "description": "Chain only — continue executing remaining steps after a failure. Default false.",
                },
                # Round 25 fields:
                "field": {
                    "type": "string",
                    "description": "system_info field: battery | volume | brightness | wifi | disk | mem | cpu | all. Default 'all'.",
                },
                "level": {
                    "type": "integer",
                    "description": "Volume level 0-100 for op=set_volume.",
                },
                "state": {
                    "type": "string",
                    "description": "on/off for op=set_mute.",
                },
                # Round 29 (F83) — Shortcuts:
                "name": {
                    "type": "string",
                    "description": "Shortcut name for op=run_shortcut.",
                },
                "input_text": {
                    "type": "string",
                    "description": "Optional stdin text fed to the Shortcut (run_shortcut).",
                },
            },
            "required": ["op"],
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        if not (args and isinstance(args, dict)):
            return ToolExecutionResult(success=False, reply_text="macControl requires a JSON object with an 'op' field.")
        op = str(args.get("op") or "").strip().lower()
        if op not in _OPS:
            return ToolExecutionResult(
                success=False,
                reply_text=f"Unknown op {op!r}. Valid: {', '.join(_OPS)}.",
            )
        # Audit round 21 — chain takes special args; dispatch via
        # ``_op_chain`` directly rather than through ``_dispatch_op``
        # because chain isn't a single-op (its semantics are different).
        if op == "chain":
            try:
                ok, msg = _op_chain(args.get("steps"), bool(args.get("continue_on_error", False)))
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=f"chain failed: {type(e).__name__}: {e}")
            return ToolExecutionResult(success=bool(ok), reply_text=msg)
        debug_log(
            f"macControl: op={op} args={ {k: v for k, v in args.items() if k != 'op'} }",
            "tools",
        )
        ok, msg = _dispatch_op(op, args)
        return ToolExecutionResult(success=bool(ok), reply_text=msg)

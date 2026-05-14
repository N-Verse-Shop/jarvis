"""Action dispatcher — turns LLM replies into PC-control intents.

Also handles language switching (UA default, RU/EN/DE on explicit
request with confirmation).


Flow:
  1. LLM reply is scanned for action markers (e.g. "збираюся відкрити X").
  2. If a recognised action is found AND the user hasn't yet said
     "виконуй" (or any synonym), the dispatcher returns
     PENDING and stores the planned action. The LLM reply is spoken as
     usual ("Зараз я зроблю Y. Підтверди?").
  3. When the user's next utterance contains "виконуй"/"давай"/"так",
     the dispatcher executes the pending action and TTS reports the
     result.
  4. If user says "відміни"/"стоп"/"ні", action is dropped.

Each action is a Python callable defined in `ACTIONS` below. Adding a
new capability = add an entry. The keyword router uses simple regex
matching against the LLM reply text — keeping the contract simple and
inspectable rather than relying on a JSON schema the small model can't
reliably emit.

Examples of triggering reply text:
  "Зараз відкрию Safari" → action=open_app, args={"name": "Safari"}
  "Запишу нотатку в inbox: купити каву" → action=write_note, args=...
  "Виконаю pull main у репо" → action=git_pull, args=...
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Callable, Optional

from ..debug import debug_log


# ─── async worker pool ───────────────────────────────────────────────────
#
# Subprocess-based actions (open -a Safari, pmset, osascript) can take
# 5-10s on cold system caches. Running them inline blocks the voice
# loop — TTS doesn't speak the "Відкриваю Safari" confirmation until
# AFTER Safari is fully launched. From user POV: voice freezes.
#
# Solution: ThreadPoolExecutor runs the subprocess work in background.
# Voice loop speaks the ack immediately, action completes async, the
# result message is logged only (TTS already done). For actions that
# MUST report a result (battery level, clipboard read), we still run
# sync — those return data the user needs to hear.
#
# Worker pool is module-global, max 4 concurrent — enough for chained
# voice commands ("відкрий Safari та зменши гучність") without burst.

_ACTION_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jarvis-action")


def _run_async(fn: Callable[[], tuple[bool, str]], action_name: str) -> Future:
    """Submit an action to the worker pool, logging result when done."""
    def _wrap():
        try:
            ok, msg = fn()
            debug_log(f"async action {action_name}: ok={ok} msg={msg!r}", "voice")
            return ok, msg
        except Exception as e:
            debug_log(f"async action {action_name} crashed: {e}", "voice")
            return False, f"Помилка: {e}"
    return _ACTION_POOL.submit(_wrap)


# Actions that SHOULD run sync because they return data the user needs:
SYNC_ACTIONS = {
    "battery",      # returns "Батарея 80%, заряджається"
    "say_time",     # returns "Зараз 14:30, четвер"
    "clipboard",    # returns "В буфері: ..."
    "read_clipboard",
}


# ─── action definitions ──────────────────────────────────────────────────

@dataclass
class Action:
    """A single planned PC-control action waiting for confirmation."""
    name: str
    description: str  # what to say before executing ("Зараз відкрию Safari")
    fn: Callable[[], tuple[bool, str]]  # returns (success, message)
    created_ts: float


# App name aliases — Whisper often transcribes English app names in
# Ukrainian transliteration ("Сафарі" instead of "Safari"). macOS
# `open -a` needs the canonical English name, so we normalise first.
# Maps both Ukrainian and Russian transliterations to canonical names.
APP_ALIASES = {
    # Browsers
    "сафарі": "Safari", "сафари": "Safari", "safari": "Safari",
    "хром": "Google Chrome", "chrome": "Google Chrome", "гугл хром": "Google Chrome",
    "фаерфокс": "Firefox", "файрфокс": "Firefox", "firefox": "Firefox",
    "арк": "Arc", "arc": "Arc",
    # Messaging
    "телеграм": "Telegram", "телеграмм": "Telegram", "telegram": "Telegram",
    "вотсап": "WhatsApp", "ватсап": "WhatsApp", "whatsapp": "WhatsApp",
    "slack": "Slack", "слак": "Slack", "слек": "Slack",
    "diskord": "Discord", "дискорд": "Discord", "discord": "Discord",
    "сігнал": "Signal", "сигнал": "Signal", "signal": "Signal",
    "imessage": "Messages", "айместрітч": "Messages", "повідомлення": "Messages",
    # Productivity
    "notion": "Notion", "ноушн": "Notion", "ноушен": "Notion",
    "obsidian": "Obsidian", "обсідіан": "Obsidian", "обсидіан": "Obsidian",
    "todoist": "Todoist", "тудуіст": "Todoist",
    "ноутс": "Notes", "нотатки": "Notes", "notes": "Notes",
    "календар": "Calendar", "calendar": "Calendar",
    "пошта": "Mail", "mail": "Mail",
    "контакти": "Contacts", "contacts": "Contacts",
    "файндер": "Finder", "finder": "Finder",
    "налаштування": "System Settings", "settings": "System Settings",
    "system settings": "System Settings",
    # Dev
    "вс код": "Visual Studio Code", "вс-код": "Visual Studio Code",
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "termінал": "Terminal", "термінал": "Terminal", "terminal": "Terminal",
    "iterm": "iTerm", "айтерм": "iTerm", "iterm2": "iTerm",
    "warp": "Warp", "ворп": "Warp",
    "ghostty": "Ghostty", "ґостті": "Ghostty",
    "хорхе": "Cursor", "курсор": "Cursor", "cursor": "Cursor",
    "ксcode": "Xcode", "ікскод": "Xcode", "xcode": "Xcode",
    "docker": "Docker", "докер": "Docker",
    "github desktop": "GitHub Desktop", "гітхаб": "GitHub Desktop",
    # Media
    "спотіфай": "Spotify", "спотифай": "Spotify", "spotify": "Spotify",
    "музика": "Music", "music": "Music", "apple music": "Music",
    "youtube": "YouTube", "ютуб": "YouTube",
    "netflix": "Netflix", "нетфлікс": "Netflix",
    # Meeting
    "zoom": "zoom.us", "зум": "zoom.us",
    "google meet": "Google Meet", "гугл міт": "Google Meet",
    "teams": "Microsoft Teams", "тімс": "Microsoft Teams",
    # AI
    "клод": "Claude", "claude": "Claude",
    "чат гпт": "ChatGPT", "chatgpt": "ChatGPT", "гпт": "ChatGPT",
    "perplexity": "Perplexity", "перплексіті": "Perplexity",
    # System
    "calculator": "Calculator", "калькулятор": "Calculator",
    "preview": "Preview", "preview app": "Preview", "перегляд": "Preview",
    "spotlight": "Spotlight",
}


def _resolve_app_name(raw: str) -> str:
    """Normalize transliterated app name to canonical macOS name.

    Whisper STT transcribes "Safari" as "Сафарі" in Ukrainian context.
    `open -a Сафарі` fails because Mac only knows "Safari". This
    function does case-insensitive lookup in APP_ALIASES, falling
    back to the raw name if no alias matches (might still work for
    apps named with cyrillic, e.g. "Telegram" is fine either way).
    """
    if not raw:
        return raw
    key = raw.strip().lower()
    # Direct match
    if key in APP_ALIASES:
        return APP_ALIASES[key]
    # Try without trailing punctuation
    key2 = re.sub(r"[\.,!?;:]+$", "", key)
    if key2 in APP_ALIASES:
        return APP_ALIASES[key2]
    return raw.strip()


def _open_app(app_name: str) -> tuple[bool, str]:
    """Open a macOS app via `open -a`.

    Tries canonical name first (via alias map), then falls back to
    -b bundle-id lookup if direct -a fails. Returns (ok, message).
    """
    canonical = _resolve_app_name(app_name)
    try:
        r = subprocess.run(
            ["open", "-a", canonical],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return True, f"Відкрив {canonical}"
        # Try once more with the raw (untransformed) name in case
        # the user said the canonical name and we mis-aliased.
        if canonical != app_name.strip():
            r2 = subprocess.run(
                ["open", "-a", app_name.strip()],
                capture_output=True, text=True, timeout=10,
            )
            if r2.returncode == 0:
                return True, f"Відкрив {app_name.strip()}"
        err = (r.stderr or "").strip()[:120]
        return False, f"Не зміг відкрити {canonical}. {err}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _write_note(text: str) -> tuple[bool, str]:
    """Append a quick note to the local vault inbox."""
    import os
    from datetime import datetime
    inbox = os.path.expanduser("~/Documents/Nexus-Brain/00-INBOX/voice-notes.md")
    try:
        os.makedirs(os.path.dirname(inbox), exist_ok=True)
        with open(inbox, "a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M}\n\n{text}\n")
        return True, "Записав нотатку"
    except Exception as e:
        return False, f"Не зміг записати: {e}"


def _say_time() -> tuple[bool, str]:
    """Return current local time."""
    from datetime import datetime
    now = datetime.now()
    weekday = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"][now.weekday()]
    return True, f"Зараз {now.strftime('%H:%M')}, {weekday}, {now.day} число."


def _show_screen() -> tuple[bool, str]:
    """Take a screenshot, save to /tmp/jarvis-screen.png."""
    try:
        subprocess.run(["screencapture", "-x", "/tmp/jarvis-screen.png"], timeout=5, check=True)
        return True, "Зробив скріншот, збережено в /tmp/jarvis-screen.png"
    except Exception as e:
        return False, f"Помилка скріншоту: {e}"


def _open_url(url: str) -> tuple[bool, str]:
    """Open a URL in the default browser."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        r = subprocess.run(
            ["open", url],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return True, f"Відкрив {url}"
        return False, f"Не зміг відкрити {url}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _web_search(query: str) -> tuple[bool, str]:
    """Open browser with a Google search."""
    from urllib.parse import quote_plus
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    try:
        subprocess.run(["open", url], timeout=10, check=True)
        return True, f"Шукаю в Google: {query[:60]}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _set_volume(level: int) -> tuple[bool, str]:
    """Set system volume (0-100)."""
    level = max(0, min(100, int(level)))
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            timeout=5, check=True,
        )
        return True, f"Гучність {level} відсотків"
    except Exception as e:
        return False, f"Не зміг змінити гучність: {e}"


def _mute_system() -> tuple[bool, str]:
    """Mute system audio."""
    try:
        subprocess.run(
            ["osascript", "-e", "set volume with output muted"],
            timeout=5, check=True,
        )
        return True, "Звук вимкнено"
    except Exception as e:
        return False, f"Помилка: {e}"


def _unmute_system() -> tuple[bool, str]:
    """Unmute system audio."""
    try:
        subprocess.run(
            ["osascript", "-e", "set volume without output muted"],
            timeout=5, check=True,
        )
        return True, "Звук увімкнено"
    except Exception as e:
        return False, f"Помилка: {e}"


def _lock_screen() -> tuple[bool, str]:
    """Lock the screen."""
    try:
        subprocess.run(
            ["pmset", "displaysleepnow"],
            timeout=5, check=True,
        )
        return True, "Екран заблоковано"
    except Exception as e:
        return False, f"Помилка: {e}"


def _play_pause_music() -> tuple[bool, str]:
    """Toggle play/pause on Music or Spotify (whichever is running)."""
    # Try Spotify first, then Music
    for app in ("Spotify", "Music"):
        try:
            r = subprocess.run(
                ["osascript", "-e", f'tell application "{app}" to playpause'],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return True, f"Перемикнув відтворення у {app}"
        except Exception:
            continue
    return False, "Не знайшов запущений Spotify або Music"


def _next_track() -> tuple[bool, str]:
    """Skip to next track."""
    for app in ("Spotify", "Music"):
        try:
            r = subprocess.run(
                ["osascript", "-e", f'tell application "{app}" to next track'],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return True, f"Наступний трек у {app}"
        except Exception:
            continue
    return False, "Не знайшов музичний плеєр"


def _read_clipboard() -> tuple[bool, str]:
    """Return current clipboard contents."""
    try:
        r = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
        text = (r.stdout or "").strip()
        if not text:
            return True, "Буфер обміну порожній"
        snippet = text[:200].replace("\n", " ")
        return True, f"В буфері: {snippet}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _write_clipboard(text: str) -> tuple[bool, str]:
    """Put text into clipboard."""
    try:
        subprocess.run(
            ["pbcopy"], input=text, text=True, timeout=5, check=True,
        )
        return True, f"Скопіював у буфер: {text[:60]}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _spotlight_find(query: str) -> tuple[bool, str]:
    """Open Spotlight search with a pre-filled query."""
    try:
        # Activate Spotlight via AppleScript and type query
        script = f'''
        tell application "System Events"
            keystroke space using {{command down}}
            delay 0.3
            keystroke "{query}"
        end tell
        '''
        subprocess.run(["osascript", "-e", script], timeout=10, check=True)
        return True, f"Шукаю '{query}' у Spotlight"
    except Exception as e:
        return False, f"Помилка: {e}"


def _new_email(to: str = "") -> tuple[bool, str]:
    """Open Mail with a new compose window."""
    url = f"mailto:{to}" if to else "mailto:"
    try:
        subprocess.run(["open", url], timeout=5, check=True)
        return True, f"Відкрив нового листа{' до ' + to if to else ''}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _say_battery() -> tuple[bool, str]:
    """Report current battery level."""
    try:
        r = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
        out = r.stdout
        m = re.search(r"(\d+)%", out)
        if m:
            pct = int(m.group(1))
            charging = "AC Power" in out or "charged" in out.lower()
            state = "заряджається" if charging else "розряджається"
            return True, f"Батарея {pct} відсотків, {state}."
        return False, "Не зміг визначити рівень батареї"
    except Exception as e:
        return False, f"Помилка: {e}"


# Regex-based parser. Each tuple is (pattern, action_factory).
# The action_factory takes the regex match and returns an Action.
#
# Pattern design rule: each regex must match phrases Jarvis ACTUALLY says
# in its replies, not the user's commands. The flow is:
#   user: "відкрий Safari"
#   LLM:  "Зараз відкрию Safari." ← we match THIS, then ask "виконуй?"
#   user: "виконуй"
#   us:   _open_app("Safari")
# So patterns target LLM phrasing: "Зараз/Я відкрию/відкриваю/...".
ACTION_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Action]]] = [
    # ── URL / web search ── (must come BEFORE open_app — URLs contain dots)
    (
        re.compile(r"(?:зараз\s+)?(?:від|за)крию\s+(?:сторінку\s+|сайт\s+)?(https?://\S+|[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?)", re.IGNORECASE),
        lambda m: Action(
            name="open_url",
            description=f"Зараз відкрию {m.group(1).strip()}",
            fn=lambda: _open_url(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── apps ── (no dot in name — those go to open_url above)
    (
        re.compile(r"(?:зараз\s+)?(?:від|за)крию[\s,\.]?\s+(?:додаток\s+)?([A-Za-z][A-Za-z0-9\s]{1,30})", re.IGNORECASE),
        lambda m: Action(
            name="open_app",
            description=f"Зараз відкрию {m.group(1).strip()}",
            fn=lambda: _open_app(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"(?:зараз\s+)?(?:по)?шукаю\s+(?:в\s+google|в\s+гуглі|в\s+браузері)[:\s]+(.{2,80})", re.IGNORECASE),
        lambda m: Action(
            name="web_search",
            description=f"Шукаю в Google: {m.group(1).strip()[:50]}",
            fn=lambda: _web_search(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── notes ──
    (
        re.compile(r"(?:зараз\s+)?(?:за|пере)пишу(?:[\s,]+нотатку)?(?:[\s,]+в\s+inbox)?[\s,:]+(.{3,200})", re.IGNORECASE),
        lambda m: Action(
            name="write_note",
            description=f"Запишу нотатку: {m.group(1).strip()[:60]}",
            fn=lambda: _write_note(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── screenshot ──
    (
        re.compile(r"(?:зараз\s+)?(?:зроблю|роблю)\s+скрін(?:шот)?", re.IGNORECASE),
        lambda m: Action(
            name="screenshot",
            description="Зроблю скріншот екрана",
            fn=_show_screen,
            created_ts=time.time(),
        ),
    ),
    # ── volume ──
    (
        re.compile(r"(?:зараз\s+)?(?:зменшу|підвищу|вставлю|поставлю|зроблю)\s+гучн[іо]ст[ьі]?\s+(?:на\s+|до\s+)?(\d{1,3})", re.IGNORECASE),
        lambda m: Action(
            name="set_volume",
            description=f"Поставлю гучність {m.group(1)}",
            fn=lambda: _set_volume(int(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"(?:зараз\s+)?вимкну\s+звук\b", re.IGNORECASE),
        lambda m: Action(
            name="mute",
            description="Вимкну звук",
            fn=_mute_system,
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"(?:зараз\s+)?увімкну\s+звук\b", re.IGNORECASE),
        lambda m: Action(
            name="unmute",
            description="Увімкну звук",
            fn=_unmute_system,
            created_ts=time.time(),
        ),
    ),
    # ── screen lock ──
    (
        re.compile(r"(?:зараз\s+)?(?:за)?блокую\s+(?:екран|комп|маку?|ноут)", re.IGNORECASE),
        lambda m: Action(
            name="lock_screen",
            description="Заблокую екран",
            fn=_lock_screen,
            created_ts=time.time(),
        ),
    ),
    # ── music ──
    (
        re.compile(r"(?:зараз\s+)?(?:зап(?:устю|ущу)|пауза|спаузу|пау?з[ао]|включу|вимкну)\s+(?:музику|плеєр|трек)", re.IGNORECASE),
        lambda m: Action(
            name="play_pause_music",
            description="Перемкну відтворення",
            fn=_play_pause_music,
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"(?:зараз\s+)?(?:наступ|перемкну|наступ.+трек)", re.IGNORECASE),
        lambda m: Action(
            name="next_track",
            description="Наступний трек",
            fn=_next_track,
            created_ts=time.time(),
        ),
    ),
    # ── clipboard ──
    (
        re.compile(r"(?:зараз\s+)?(?:прочитаю|зачитаю|подивлюся\s+у?)\s+(?:в\s+)?буфер(?:і)?(?:\s+обміну)?", re.IGNORECASE),
        lambda m: Action(
            name="read_clipboard",
            description="Дивлюся в буфер обміну",
            fn=_read_clipboard,
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"(?:зараз\s+)?(?:с?копіюю|занесу|збережу)\s+(?:у|в|до)?\s*буфер(?:\s+обміну)?[:\s]+(.{2,500})", re.IGNORECASE),
        lambda m: Action(
            name="write_clipboard",
            description=f"Скопіюю в буфер: {m.group(1).strip()[:50]}",
            fn=lambda: _write_clipboard(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── spotlight / file search ──
    (
        re.compile(r"(?:зараз\s+)?(?:знайду|пошукаю)(?:\s+файл|\s+у\s+spotlight)?[\s:]+([^\.\n]{2,80})", re.IGNORECASE),
        lambda m: Action(
            name="spotlight",
            description=f"Шукаю '{m.group(1).strip()[:40]}' у Spotlight",
            fn=lambda: _spotlight_find(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── mail ──
    (
        re.compile(r"(?:зараз\s+)?(?:напишу|створю)\s+(?:листа|email|імейл|новий\s+лист)(?:\s+(?:до|на)\s+(\S+@\S+\.\S+))?", re.IGNORECASE),
        lambda m: Action(
            name="new_email",
            description=f"Відкрию нового листа{' до ' + m.group(1) if m.group(1) else ''}",
            fn=lambda: _new_email(m.group(1) or ""),
            created_ts=time.time(),
        ),
    ),
    # ── time ──
    (
        re.compile(r"(?:зараз\s+)?(?:скажу|повідомлю|подивлюся)\s+(?:котра|який)\s+(?:година|час)", re.IGNORECASE),
        lambda m: Action(
            name="say_time",
            description="Подивлюся котра година",
            fn=_say_time,
            created_ts=time.time(),
        ),
    ),
    # ── battery ──
    (
        re.compile(r"(?:зараз\s+)?(?:перевірю|подивлюся)\s+(?:рівень\s+)?батарею?", re.IGNORECASE),
        lambda m: Action(
            name="battery",
            description="Перевіряю рівень батареї",
            fn=_say_battery,
            created_ts=time.time(),
        ),
    ),
]


# Words that mean "go ahead, execute".
CONFIRM_WORDS = {
    "виконуй", "виконай", "давай", "так", "ага", "ок", "окей",
    "підтверджую", "роби", "зроби", "поїхали", "так-так",
    "yes", "do it", "go", "execute", "ok",
}

# Words that mean "cancel".
DENY_WORDS = {
    "відміни", "відмінити", "ні", "не треба", "забудь", "стоп",
    "no", "cancel", "skip",
}


def parse_action(llm_reply: str) -> Optional[Action]:
    """Try to extract a planned action from an LLM reply.

    Returns the first match (we don't queue multiple actions per
    utterance — keeps the confirmation UX simple).
    """
    if not llm_reply:
        return None
    for pattern, factory in ACTION_PATTERNS:
        m = pattern.search(llm_reply)
        if m:
            try:
                action = factory(m)
                debug_log(f"action parsed: {action.name} — {action.description}", "voice")
                return action
            except Exception as e:
                debug_log(f"action factory failed: {e}", "voice")
    return None


def is_confirmation(user_text: str) -> bool:
    """Check if the user's reply is a confirmation."""
    if not user_text:
        return False
    t = user_text.lower().strip()
    # Whole-word match — "виконуй" alone or anywhere.
    for w in CONFIRM_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False


def is_denial(user_text: str) -> bool:
    """Check if the user's reply is a denial."""
    if not user_text:
        return False
    t = user_text.lower().strip()
    for w in DENY_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False


# ─── language switching ──────────────────────────────────────────────────

# Phrases that request a switch to a non-default language. Default is
# always Ukrainian — these triggers require explicit user confirmation.
LANG_SWITCH_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Russian
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди|перейти)\s+(?:на\s+)?(?:російськ|русск|русс)\w*\b", re.IGNORECASE),
     "ru", "Переходжу на російську"),
    (re.compile(r"\bна\s+(?:російськ|русск|русс)(?:ой|у|ій|ом)\b", re.IGNORECASE),
     "ru", "Переходжу на російську"),
    (re.compile(r"\bswitch\s+to\s+russian\b", re.IGNORECASE),
     "ru", "Переходжу на російську"),
    # English
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди)\s+(?:на\s+)?англійськ\w*\b", re.IGNORECASE),
     "en", "Switching to English"),
    (re.compile(r"\bна\s+англійськ(?:ій|у|ою|ом)\b", re.IGNORECASE),
     "en", "Switching to English"),
    (re.compile(r"\bswitch\s+to\s+english\b", re.IGNORECASE),
     "en", "Switching to English"),
    # German
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди)\s+(?:на\s+)?німецьк\w*\b", re.IGNORECASE),
     "de", "Wechsle zu Deutsch"),
    (re.compile(r"\bна\s+німецьк(?:ій|у|ою|ом)\b", re.IGNORECASE),
     "de", "Wechsle zu Deutsch"),
    (re.compile(r"\b(?:switch\s+to|auf)\s+(?:german|deutsch)\b", re.IGNORECASE),
     "de", "Wechsle zu Deutsch"),
    # Back to Ukrainian
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди|повернись|обратно)\s+(?:на\s+)?українськ\w*\b", re.IGNORECASE),
     "uk", "Повертаюсь на українську"),
    (re.compile(r"\bback\s+to\s+ukrainian\b", re.IGNORECASE),
     "uk", "Повертаюсь на українську"),
]


def detect_language_switch(user_text: str) -> tuple[str, str] | None:
    """Detect a language-switch request.

    Returns (lang_code, ack_phrase) or None.
    """
    if not user_text:
        return None
    for pat, lang, ack in LANG_SWITCH_PATTERNS:
        if pat.search(user_text):
            return lang, ack
    return None


# ─── DIRECT user-command parser ──────────────────────────────────────────
#
# Bypass the LLM entirely for unambiguous PC-control phrases. The user
# complained: "він всерівно неможе навіть відкрити сафарі" — qwen2.5:7b
# on CPU was either inventing JSON, mis-quoting, or just answering with
# text that didn't match ACTION_PATTERNS. So for the most common
# imperative voice commands we skip the LLM round-trip and execute
# directly. Latency: ~50ms vs 15-25s.
#
# Patterns target what the USER says, not what the LLM says:
#   "відкрий Safari" → _open_app("Safari")
#   "запусти Telegram" → _open_app("Telegram")
#   "гучність 30" → _set_volume(30)
#   "пауза" / "наступний трек" → music
#   "котра година" → say_time
#   "вимкни звук" → mute
#   "заблокуй екран" → lock
#
# These run WITHOUT confirmation — user already said the imperative.
# If the parse is ambiguous, we fall through to the LLM as before.

def _make_open_app_action(raw: str, verb_present: str) -> Action:
    """Factory for open_app action — pre-resolves alias for nicer TTS."""
    canonical = _resolve_app_name(raw)
    return Action(
        name="open_app",
        description=f"{verb_present} {canonical}",
        fn=lambda: _open_app(raw),  # _open_app re-resolves internally
        created_ts=time.time(),
    )


USER_COMMAND_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Action]]] = [
    # ── URL open ── (MUST come before open_app — URLs contain dots)
    (
        re.compile(r"^\s*(?:від|за)крий(?:те)?\s+(?:сайт\s+|сторінку\s+)?((?:https?://)?[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="open_url",
            description=f"Відкриваю {m.group(1).strip()}",
            fn=lambda: _open_url(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── apps ── (must match clean imperatives, no dots)
    (
        re.compile(r"^\s*(?:а\s+)?(?:будь\s+ласка\s+)?(?:від|за)крий(?:те)?\s+(?:додаток\s+|програму\s+)?([A-Za-zА-Яа-яЇїІіЄєҐґ][A-Za-zА-Яа-яЇїІіЄєҐґ0-9\s-]{1,30})\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: _make_open_app_action(m.group(1).strip(), "Відкриваю"),
    ),
    (
        re.compile(r"^\s*(?:будь\s+ласка\s+)?запусти(?:те)?\s+(?:додаток\s+|програму\s+)?([A-Za-zА-Яа-яЇїІіЄєҐґ][A-Za-zА-Яа-яЇїІіЄєҐґ0-9\s-]{1,30})\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: _make_open_app_action(m.group(1).strip(), "Запускаю"),
    ),
    # ── volume ──
    (
        re.compile(r"^\s*(?:встав|постав|зроби|постав\s+на)\s+гучн[іо]ст[ьі]?\s+(?:на\s+|до\s+)?(\d{1,3})\s*(?:відсотк|процент)?\w*\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="set_volume",
            description=f"Гучність {m.group(1)}",
            fn=lambda: _set_volume(int(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"^\s*гучн[іо]ст[ьі]?\s+(\d{1,3})\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="set_volume",
            description=f"Гучність {m.group(1)}",
            fn=lambda: _set_volume(int(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"^\s*вимкни\s+звук\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="mute", description="Вимикаю звук", fn=_mute_system, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*увімкни\s+звук\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="unmute", description="Увімкаю звук", fn=_unmute_system, created_ts=time.time()),
    ),
    # ── screen ──
    (
        re.compile(r"^\s*(?:за)?блокуй(?:те)?\s+(?:екран|комп[`']?ютер|маку?|ноут(?:бук)?)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="lock", description="Блокую екран", fn=_lock_screen, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:зроби|зробити)\s+скрін(?:шот)?\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="screenshot", description="Роблю скріншот", fn=_show_screen, created_ts=time.time()),
    ),
    # ── music ──
    (
        re.compile(r"^\s*(?:пауза|спаузу|постав\s+на\s+паузу|зупини\s+музику)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="music_pause", description="Пауза", fn=_play_pause_music, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:увімкни|включи|запусти)\s+музику\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="music_play", description="Вмикаю музику", fn=_play_pause_music, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:наступн\w+\s+трек|далі|перемкни\s+трек)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="next_track", description="Наступний трек", fn=_next_track, created_ts=time.time()),
    ),
    # ── info ──
    (
        re.compile(r"^\s*(?:котра\s+(?:зараз\s+)?година|скільки\s+(?:зараз\s+)?часу|який\s+зараз\s+час)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="say_time", description="Дивлюся час", fn=_say_time, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:рівень\s+)?батаре[яюїіея]\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="battery", description="Перевіряю батарею", fn=_say_battery, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:скільки|який)\s+(?:заряд|рівень)\s+(?:батаре[яюїіея])\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="battery", description="Перевіряю батарею", fn=_say_battery, created_ts=time.time()),
    ),
    # ── clipboard ──
    (
        re.compile(r"^\s*(?:що|що\s+там)\s+(?:у|в)\s+буфер[іе](?:\s+обміну)?\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="clipboard", description="Дивлюся буфер", fn=_read_clipboard, created_ts=time.time()),
    ),
    # ── web search ──
    (
        re.compile(r"^\s*(?:знайди|шукай|пошукай)\s+(?:в\s+google|в\s+гугл[іе]|у?\s*браузер[іе])\s+(.{2,80})\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="web_search",
            description=f"Шукаю: {m.group(1).strip()[:50]}",
            fn=lambda: _web_search(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── URL open ──
    (
        re.compile(r"^\s*(?:від|за)крий(?:те)?\s+(?:сайт\s+|сторінку\s+)?((?:https?://)?[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="open_url",
            description=f"Відкриваю {m.group(1).strip()}",
            fn=lambda: _open_url(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── notes ──
    (
        re.compile(r"^\s*(?:запиши|занотуй|занеси|створи)\s+нотатку[:\s]+(.{3,200})\s*$", re.IGNORECASE),
        lambda m: Action(
            name="write_note",
            description=f"Записую: {m.group(1).strip()[:60]}",
            fn=lambda: _write_note(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
]


def parse_user_command(user_text: str) -> Optional[Action]:
    """Try to extract a DIRECT user command (skip LLM).

    Returns the first match. The action runs IMMEDIATELY without
    confirmation — the user already issued an imperative.
    Returns None if no clear command pattern matches.
    """
    if not user_text:
        return None
    text = user_text.strip()
    # Strip trailing/leading punctuation for cleaner matching
    text = re.sub(r"^[\s,\.!?]+|[\s,\.!?]+$", "", text)
    if not text:
        return None
    for pattern, factory in USER_COMMAND_PATTERNS:
        m = pattern.match(text) or pattern.search(text)
        if m:
            try:
                action = factory(m)
                debug_log(
                    f"DIRECT user command: {action.name} — {action.description}",
                    "voice",
                )
                return action
            except Exception as e:
                debug_log(f"direct cmd factory failed: {e}", "voice")
    return None

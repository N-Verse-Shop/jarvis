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

import json
import os
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

# Audit round 16 fix: dropped max_workers from 4 → 1.
#
# Rapid-fire voice actions ("заблокуй екран. виконуй... відкрий Safari.
# виконуй") used to race in the pool — ``_lock_screen`` could fire mid-
# ``_open_app`` causing Safari to launch behind the lock screen, mute-
# all-tabs ran AFTER the new app had already mutated the system. None of
# the action handlers are designed for parallelism (each touches global
# macOS state). Serialising at the pool level is the cheapest correct
# fix; the latency cost is invisible (each action completes in ms).
#
# Future: if a truly parallelisable action lands (e.g. background HTTP
# fetch), submit it via a separate Executor — don't bump this back up.
_ACTION_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-action")

# Audit round 17 fix: hard wall-clock timeout per action. ``max_workers=1``
# serialises actions correctly, but if any single action hangs (a stuck
# AppleScript dialog awaiting user confirmation, a flaky network call
# in ``_send_message_to_linear``/``_send_to_github`` that blocks the
# kernel socket past the urlopen timeout, an `osascript` subprocess
# wedged on a permissions prompt) every subsequent voice action stalls
# forever with no feedback to the user. ``_ACTION_TIMEOUT_SEC`` is
# generous enough for legitimate slow paths (osascript spin-up, AS
# bridge cold start) and short enough that the user notices.
_ACTION_TIMEOUT_SEC = 30.0


def shutdown_action_pool(wait: bool = False) -> None:
    """Shutdown the module-level ThreadPoolExecutor.

    Audit round 11 fix H6: previously the pool had no shutdown hook.
    Daemon shutdown left 4 idle worker threads alive until process
    exit (atexit), and any in-process daemon restart (test runner,
    self-upgrade) leaked another 4. `wait=False` matches the prior
    fail-fast shutdown semantics — the listener.stop() path is already
    bounded; we don't want to block on a stuck action.
    """
    try:
        _ACTION_POOL.shutdown(wait=wait, cancel_futures=True)
    except Exception:
        pass


def _run_async(fn: Callable[[], tuple[bool, str]], action_name: str) -> Future:
    """Submit an action to the worker pool, logging result when done.

    Emits typed tool_call events at start and finish so HUD / debug
    viewer / external observers can show progress without polling.
    """
    # Lazy import to avoid circular dep on ..ipc → ..debug → ...
    try:
        from ..ipc import get_stream
        _stream = get_stream()
    except Exception:
        _stream = None

    start_ts = time.time()
    if _stream is not None:
        try:
            _stream.emit("tool_call", tool=action_name, status="starting")
        except Exception:
            pass

    def _wrap():
        # Audit round 17 fix: run the action in a daemon sub-thread and
        # ``join(_ACTION_TIMEOUT_SEC)``. If join returns without the
        # thread having finished, the action is officially "abandoned"
        # — we log it, emit a failure event, and let the worker pool
        # accept the next item. The orphan thread keeps running but
        # is daemonised, so it does NOT keep the process alive at
        # shutdown. We do NOT try to force-kill (Python has no safe
        # primitive for that); the abandonment is the contract.
        import threading as _th
        result_box: list = []  # mutable cell to receive (ok, msg)
        exc_box: list = []

        def _runner():
            try:
                result_box.append(fn())
            except Exception as _e:
                exc_box.append(_e)

        sub = _th.Thread(
            target=_runner,
            name=f"jarvis-action-runner:{action_name}",
            daemon=True,
        )
        sub.start()
        sub.join(_ACTION_TIMEOUT_SEC)
        if sub.is_alive():
            elapsed_ms = int((time.time() - start_ts) * 1000)
            debug_log(
                f"async action {action_name} TIMED OUT after "
                f"{_ACTION_TIMEOUT_SEC}s — abandoning runner thread "
                f"(it will continue in the background but the pool "
                f"is now free)",
                "voice",
            )
            if _stream is not None:
                try:
                    _stream.emit(
                        "tool_call",
                        tool=action_name,
                        status="failed",
                        error=f"timeout after {_ACTION_TIMEOUT_SEC}s",
                        duration_ms=elapsed_ms,
                    )
                except Exception:
                    pass
            return False, (
                f"Дія '{action_name}' не завершилась за "
                f"{int(_ACTION_TIMEOUT_SEC)} секунд — пропускаю."
            )
        if exc_box:
            e = exc_box[0]
            debug_log(f"async action {action_name} crashed: {e}", "voice")
            if _stream is not None:
                try:
                    _stream.emit(
                        "tool_call",
                        tool=action_name,
                        status="failed",
                        error=str(e),
                        duration_ms=int((time.time() - start_ts) * 1000),
                    )
                except Exception:
                    pass
            return False, f"Помилка: {e}"
        ok, msg = result_box[0] if result_box else (False, "no result")
        debug_log(f"async action {action_name}: ok={ok} msg={msg!r}", "voice")
        if _stream is not None:
            try:
                _stream.emit(
                    "tool_call",
                    tool=action_name,
                    status="completed" if ok else "failed",
                    result={"ok": ok, "message": msg},
                    duration_ms=int((time.time() - start_ts) * 1000),
                )
            except Exception:
                pass
        return ok, msg
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
    # Round 29 (F79): expanded macOS-native app coverage. User complaints
    # like "відкрий карти" / "открой фото" / "запусти нагадування" used to
    # fall through to `open -a карти` which fails. Map both UA and RU
    # spoken forms to canonical English names.
    "карти": "Maps", "карты": "Maps", "maps": "Maps", "карта": "Maps",
    "фото": "Photos", "фотки": "Photos", "photos": "Photos", "фотографии": "Photos", "фотографії": "Photos",
    "нагадування": "Reminders", "напоминания": "Reminders", "напоминалки": "Reminders",
    "reminders": "Reminders", "нагадувалки": "Reminders",
    "книги": "Books", "books": "Books", "apple books": "Books", "айбукс": "Books", "ibooks": "Books",
    "shortcuts": "Shortcuts", "шорткатс": "Shortcuts", "команди": "Shortcuts",
    "команды": "Shortcuts", "ярлики": "Shortcuts", "ярлыки": "Shortcuts",
    "app store": "App Store", "епп стор": "App Store", "ап стор": "App Store",
    "магазин": "App Store",
    "погода": "Weather", "weather": "Weather", "погоди": "Weather",
    "годинник": "Clock", "часы": "Clock", "clock": "Clock", "часи": "Clock",
    "будильник": "Clock", "будильники": "Clock", "таймер": "Clock",
    "voice memos": "Voice Memos", "диктофон": "Voice Memos",
    "голосові записи": "Voice Memos", "голосовые записи": "Voice Memos",
    "find my": "Find My", "знайти": "Find My", "найти": "Find My",
    "знайти мене": "Find My", "найти меня": "Find My",
    "freeform": "Freeform", "фріформ": "Freeform", "фриформ": "Freeform",
    "stickies": "Stickies", "стікі": "Stickies", "стикеры": "Stickies",
    "наклейки": "Stickies", "стікери": "Stickies",
    "automator": "Automator", "автоматор": "Automator",
    "script editor": "Script Editor", "скрипт едитор": "Script Editor",
    "activity monitor": "Activity Monitor", "монітор": "Activity Monitor",
    "монітор активності": "Activity Monitor", "монитор": "Activity Monitor",
    "монитор активности": "Activity Monitor",
    "disk utility": "Disk Utility", "дискова утиліта": "Disk Utility",
    "дисковая утилита": "Disk Utility", "дисковая": "Disk Utility",
    "system information": "System Information", "інформація": "System Information",
    "система": "System Information",
    "console": "Console", "консоль": "Console",
    "screenshot": "Screenshot", "скріншот": "Screenshot", "скрин": "Screenshot",
    "system preferences": "System Settings",
    "системні налаштування": "System Settings",
    "системные настройки": "System Settings", "настройки": "System Settings",
    "налаштування системи": "System Settings",
    "textedit": "TextEdit", "текстедіт": "TextEdit", "текстедит": "TextEdit",
    "keynote": "Keynote", "кейноут": "Keynote",
    "pages": "Pages", "пейджес": "Pages", "сторінки": "Pages",
    "numbers": "Numbers", "намберс": "Numbers", "числа": "Numbers",
    "garageband": "GarageBand", "гараж": "GarageBand", "гараж бенд": "GarageBand",
    "imovie": "iMovie", "аймуві": "iMovie", "аймуви": "iMovie",
    "image capture": "Image Capture", "захоплення зображення": "Image Capture",
    "захват изображения": "Image Capture",
    "tv": "TV", "apple tv": "TV", "епл тв": "TV",
    "podcasts": "Podcasts", "подкасти": "Podcasts", "подкасты": "Podcasts",
    "news": "News", "новини": "News", "новости": "News",
    "stocks": "Stocks", "акції": "Stocks", "акции": "Stocks", "біржа": "Stocks",
    "translate": "Translate", "перекладач": "Translate", "переводчик": "Translate",
    "переклад": "Translate", "перевод": "Translate",
    "dictionary": "Dictionary", "словник": "Dictionary", "словарь": "Dictionary",
    "wallet": "Wallet", "гаманець": "Wallet", "кошелек": "Wallet",
    "health": "Health", "здоров'я": "Health", "здоровье": "Health",
    "home": "Home", "дім": "Home", "дом": "Home", "оселя": "Home",
    "facetime": "FaceTime", "фейстайм": "FaceTime",
    "messages": "Messages", "iMessage": "Messages", "айместрідж": "Messages",
    "ipod": "Music",
    "music app": "Music", "епл музик": "Music", "эппл мьюзик": "Music",
    "music studio": "Music",
    "siri": "Siri", "сірі": "Siri", "сири": "Siri",
    # 3rd-party productivity expansions
    "1password": "1Password 7", "1пароль": "1Password 7", "ванпароль": "1Password 7",
    "bitwarden": "Bitwarden", "бітворден": "Bitwarden",
    "raycast": "Raycast", "рейкаст": "Raycast",
    "alfred": "Alfred", "альфред": "Alfred", "альфред 5": "Alfred",
    "rectangle": "Rectangle", "ректангл": "Rectangle",
    "magnet": "Magnet", "магніт": "Magnet", "магнит": "Magnet",
    "cleanmymac": "CleanMyMac X", "клінмаймак": "CleanMyMac X",
    "linear": "Linear", "лінеар": "Linear", "линеар": "Linear",
    "figma": "Figma", "фігма": "Figma", "фигма": "Figma",
    "sketch": "Sketch", "скетч": "Sketch",
    "photoshop": "Adobe Photoshop 2025", "фотошоп": "Adobe Photoshop 2025",
    "illustrator": "Adobe Illustrator 2025", "ілюстратор": "Adobe Illustrator 2025",
    "иллюстратор": "Adobe Illustrator 2025",
    "premiere": "Adobe Premiere Pro 2025", "премʼєр": "Adobe Premiere Pro 2025",
    "after effects": "Adobe After Effects 2025", "афтер ефектс": "Adobe After Effects 2025",
    "afterр effects": "Adobe After Effects 2025",
    "lightroom": "Adobe Lightroom", "лайтрум": "Adobe Lightroom",
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
    # Round 28 (F70 defense-in-depth): reject obvious fillers/prepositions
    # that could have leaked through regex. Examples seen in production
    # logs: _open_app("в"), _open_app("на"), _open_app("і"). Without
    # this guard, AppleScript launches an open-dialog for a nonexistent
    # app and fails silently with cryptic stderr.
    name = (app_name or "").strip()
    if not name or len(name) < 2:
        return False, "Не вказано назву додатка."
    # F80: expanded RU/UA preposition + conjunction blacklist. Covers
    # leakage like "сейчас открою во весь экран", "открой у себя" and
    # the conjunction case "открыть и закрыть".
    _FILLER_APP_NAMES = {
        # RU prepositions
        "в", "во", "у", "на", "до", "от", "об", "обо", "для", "при",
        "по", "с", "со", "ко", "из", "из-за", "из-под", "над", "под",
        "перед", "после", "около", "возле",
        # UA prepositions
        "у", "в", "до", "від", "для", "при", "по", "з", "зі", "із",
        "над", "під", "перед", "після",
        # RU/UA conjunctions
        "и", "а", "но", "или", "ни", "то", "же", "ли",
        "і", "та", "але", "чи", "або", "ні", "не",
        # English prepositions
        "the", "a", "an", "to", "in", "on", "at", "of", "for", "by",
        "with", "from", "into", "onto", "via", "is", "and", "or",
        # other fillers
        "э", "эй", "ох", "ах",
    }
    if name.lower() in _FILLER_APP_NAMES:
        return False, f"'{name}' — не схоже на назву додатка."
    canonical = _resolve_app_name(name)
    app_name = name  # use sanitized name downstream
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
        # Round 29 (F81): bundle-id fallback via Spotlight. The
        # docstring promised it but the code never tried it. When the
        # user says an app name that isn't in APP_ALIASES (rare/3rd-
        # party app, fresh install, regional spelling), `open -a` fails
        # but `mdfind` against /Applications usually finds it. We pick
        # the FIRST .app whose CFBundleDisplayName or filename starts
        # with the user's text (case-insensitive), then re-attempt
        # `open` via -b on the bundle id resolved from Info.plist.
        try:
            from pathlib import Path as _Path
            # mdfind kMDItemKind == 'Application' AND name matches.
            q = (
                f"kMDItemKind == 'Application' && "
                f"(kMDItemDisplayName == '*{canonical}*'cd || "
                f"kMDItemDisplayName == '*{app_name}*'cd)"
            )
            mdr = subprocess.run(
                ["mdfind", q],
                capture_output=True, text=True, timeout=5,
            )
            if mdr.returncode == 0:
                paths = [
                    p.strip()
                    for p in (mdr.stdout or "").splitlines()
                    if p.strip().endswith(".app")
                ]
                # Prefer /Applications over user-installed elsewhere so
                # we don't accidentally launch a sandboxed copy from
                # /System or a stale ~/Downloads installer.
                paths.sort(key=lambda p: (
                    0 if p.startswith("/Applications/") else
                    1 if p.startswith("/System/Applications/") else 2,
                    len(_Path(p).name),
                ))
                for app_path in paths[:3]:
                    r3 = subprocess.run(
                        ["open", app_path],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r3.returncode == 0:
                        return True, f"Відкрив {_Path(app_path).stem}"
        except Exception:
            pass
        err = (r.stderr or "").strip()[:120]
        return False, f"Не зміг відкрити {canonical}. {err}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _write_note(text: str) -> tuple[bool, str]:
    """Append a quick note to the local vault inbox.

    Audit round 16 fix: the note text flows from raw user transcript
    through a permissive ``(.{3,200})`` regex straight into a markdown
    file that Obsidian renders. Without sanitisation, a captured
    utterance like ``inbox: <!-- include `~/.ssh/id_rsa` --> #ignore``
    or ``inbox: ![x](javascript:alert(1))`` would be persisted verbatim
    — Obsidian / mark-down preview would execute the HTML comment
    include directive or render the JS link with hover tooltip.
    Defences:
      * ``scrub_secrets`` strips API tokens / credentials that may have
        been spoken (paranoid layer — listener already redacts upstream).
      * Strip ``<!--`` / ``-->`` HTML-comment delimiters so Obsidian's
        include-via-comment plugin can't be tricked.
      * Strip ``javascript:`` / ``data:`` URL schemes (markdown link
        bodies).
      * Wrap the body in an HTML-safe fence so backticks and pipes
        can't break out of the note's section.
    """
    import os
    from datetime import datetime
    from ..utils.redact import scrub_secrets as _scrub

    safe_text = text or ""
    try:
        safe_text = _scrub(safe_text)
    except Exception:
        # Redact must not block notes — fall back to raw text.
        pass
    # Strip HTML-comment delimiters and dangerous URL schemes.
    safe_text = (
        safe_text
        .replace("<!--", "")
        .replace("-->", "")
        .replace("javascript:", "")
        .replace("data:", "")
    )
    # Bound length defensively — the upstream regex caps at 200 chars
    # but a future loosening shouldn't accidentally let a 2 MB
    # transcript land in the inbox.
    if len(safe_text) > 2000:
        safe_text = safe_text[:2000] + "…"

    inbox = os.path.expanduser("~/Documents/Nexus-Brain/00-INBOX/voice-notes.md")
    try:
        os.makedirs(os.path.dirname(inbox), exist_ok=True)
        with open(inbox, "a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M}\n\n{safe_text}\n")
        return True, "Записав нотатку"
    except Exception as e:
        return False, f"Не зміг записати: {e}"


def _nexus_brain_search(query: str) -> tuple[bool, str]:
    """Search the Nexus-Brain Obsidian vault for ``query`` and read top hits.

    R34-S47 — the user (Danylo Molyanko, CEO Nexus Studio) keeps the
    company's full operational knowledge in
    ``~/Documents/Nexus-Brain/`` (clients, partners, finance,
    knowledge base, daily notes). Without read access Jarvis was a
    Wikipedia-with-extra-steps; with it Jarvis can answer "What's
    Ibons' Hydrogen stack?" or "Who's our Bitrix24 contact?"
    confidently from on-disk truth.

    Strategy:
      1. ``ripgrep`` the vault for the query (case-insensitive,
         markdown + text files only, max 20 hits).
      2. Read the matched files (cap at 3 best hits, 800 chars each)
         so the spoken reply has actual content to summarise.
      3. Return a short paragraph: "Знайшов у Brain: <file1>: <snippet>
         …" — voice-friendly and bounded.

    Failure modes — no rg installed / vault missing / no hits — all
    degrade to ``ok=False`` with an explanatory message so the LLM
    can fall back to its parametric knowledge.
    """
    import os
    import shutil
    import subprocess

    vault = os.path.expanduser("~/Documents/Nexus-Brain")
    if not os.path.isdir(vault):
        return False, "Nexus-Brain не знайдено локально"

    q = (query or "").strip()
    if not q:
        return False, "Порожній запит до Nexus-Brain"
    if len(q) > 200:
        q = q[:200]

    rg = shutil.which("rg") or "/opt/homebrew/bin/rg"
    if not os.path.exists(rg):
        # No ripgrep → fall back to a Python-only walk that grep's
        # the first 4 KB of each .md file. Slower but always works.
        hits: list[tuple[str, str]] = []
        try:
            ql = q.lower()
            for root, _dirs, files in os.walk(vault):
                # Skip Obsidian / git / DS_Store dot-folders.
                if "/.obsidian" in root or "/.git" in root:
                    continue
                for name in files:
                    if not (name.endswith(".md") or name.endswith(".txt")):
                        continue
                    path = os.path.join(root, name)
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            data = f.read(4096)
                    except Exception:
                        continue
                    if ql in data.lower():
                        snip = data.strip().replace("\n", " ")[:400]
                        rel = os.path.relpath(path, vault)
                        hits.append((rel, snip))
                        if len(hits) >= 3:
                            break
                if len(hits) >= 3:
                    break
        except Exception as exc:
            return False, f"Не зміг прочитати vault: {exc}"
        if not hits:
            return False, "В Nexus-Brain ничего не нашёл"
        # R34-S48 — voice-friendly: speak file-name list only.
        names = []
        for rel, _snip in hits[:5]:
            stem = os.path.splitext(os.path.basename(rel))[0]
            parent = os.path.basename(os.path.dirname(rel))
            if parent and parent not in (".", "Nexus-Brain"):
                names.append(f"{parent}/{stem}")
            else:
                names.append(stem)
        n = len(hits)
        listing = ", ".join(names[:5])
        return True, f"Нашёл {n} {'файл' if n == 1 else 'файлов'}: {listing}"

    # ripgrep path — much faster on a 1000-file vault.
    try:
        out = subprocess.run(
            [
                rg, "-i", "-l", "--type", "md", "--type", "txt",
                "--max-count", "1", "--no-messages",
                q, vault,
            ],
            capture_output=True, text=True, timeout=8,
        )
    except subprocess.TimeoutExpired:
        return False, "Поиск в Brain застрял"
    except Exception as exc:
        return False, f"rg failed: {exc}"
    files = [ln for ln in out.stdout.splitlines() if ln.strip()][:5]
    if not files:
        return False, "В Nexus-Brain ничего не нашёл"
    # R34-S48 — voice-friendly summary. The user hears a short
    # phrase ("Нашёл 3 файла: ibons, bauplanung, nexus-studio") and
    # gets the full content in the dashboard tool_call event. We
    # deliberately do NOT read the file contents aloud because:
    #   1. Vault content is in Ukrainian — the RU Piper voice
    #      mangles it (R34-S48 motivation).
    #   2. A 600-char file dump played at 200 wpm = ~30 s of
    #      narration, which is interrupting, not helpful.
    # The dashboard already shows the snippets; voice just routes.
    names = []
    for path in files:
        rel = os.path.relpath(path, vault)
        # Use just the basename (no extension) for the spoken list.
        stem = os.path.splitext(os.path.basename(rel))[0]
        # If the file is inside a meaningful subfolder, prepend the
        # subfolder for context ("ibons/MASTER" > bare "MASTER").
        parent = os.path.basename(os.path.dirname(rel))
        if parent and parent not in (".", "Nexus-Brain"):
            names.append(f"{parent}/{stem}")
        else:
            names.append(stem)
    n = len(files)
    listing = ", ".join(names[:5])
    return True, f"Нашёл {n} {'файл' if n == 1 else 'файлов'}: {listing}"


def _nexus_brain_append(section: str, body: str) -> tuple[bool, str]:
    """Append ``body`` under a date-stamped heading in a Brain file.

    Used for "запиши в мозок що …" — automatically pipes new facts
    about Nexus Studio (or clients / partners) into the vault so
    the next session has them. ``section`` selects the destination:

      ``nexus-studio`` → 04-KNOWLEDGE/nexus-studio.md  (default)
      ``ideas``        → 00-INBOX/ideas/jarvis.md
      ``daily``        → 06-DAILY/<YYYY-MM-DD>.md

    Sanitisation is the same conservative set ``_write_note`` uses
    (HTML-comment stripping, scheme bleach, length cap).
    """
    import os
    from datetime import datetime
    from ..utils.redact import scrub_secrets as _scrub

    safe = (body or "").strip()
    if not safe:
        return False, "Порожній зміст для запису"
    try:
        safe = _scrub(safe)
    except Exception:
        pass
    safe = (
        safe.replace("<!--", "")
            .replace("-->", "")
            .replace("javascript:", "")
            .replace("data:", "")
    )
    if len(safe) > 2000:
        safe = safe[:2000] + "…"

    vault = os.path.expanduser("~/Documents/Nexus-Brain")
    today = datetime.now().strftime("%Y-%m-%d")
    section_l = (section or "").strip().lower()
    if section_l in ("ideas", "ідеї", "идеи"):
        target = os.path.join(vault, "00-INBOX/ideas/jarvis.md")
    elif section_l in ("daily", "щоденник", "дневник"):
        target = os.path.join(vault, "06-DAILY", f"{today}.md")
    else:
        target = os.path.join(vault, "04-KNOWLEDGE/nexus-studio.md")
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(
                f"\n## {datetime.now():%Y-%m-%d %H:%M} — Jarvis\n\n{safe}\n"
            )
        rel = os.path.relpath(target, vault)
        return True, f"Записав у Brain: {rel}"
    except Exception as exc:
        return False, f"Не зміг записати в Brain: {exc}"


def _say_time() -> tuple[bool, str]:
    """Return current local time."""
    from datetime import datetime
    now = datetime.now()
    weekday = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"][now.weekday()]
    return True, f"Зараз {now.strftime('%H:%M')}, {weekday}, {now.day} число."


def _say_date() -> tuple[bool, str]:
    """R34-S40 — fast path for date queries."""
    from datetime import datetime
    now = datetime.now()
    weekday = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"][now.weekday()]
    months = [
        "січня", "лютого", "березня", "квітня", "травня", "червня",
        "липня", "серпня", "вересня", "жовтня", "листопада", "грудня",
    ]
    return True, f"Сьогодні {weekday}, {now.day} {months[now.month-1]} {now.year}."


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
    """Lock the screen.

    Audit round 20 P2 fix: previously called ``pmset displaysleepnow``
    which only puts the display to sleep — it does NOT require a
    password to unlock unless "Require password immediately after sleep"
    is enabled in System Settings → Lock Screen. The comment claimed
    "Екран заблоковано" but the user could move the mouse and resume
    without auth. That is a privacy/security misrepresentation.

    Now uses the canonical macOS lock command:
    ``/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession -suspend``
    which immediately locks the session (login window appears, password
    required). Falls back to the keyboard shortcut Cmd-Ctrl-Q via
    AppleScript if CGSession is missing (heavily restricted in newer
    OS versions). Final fallback is the old ``pmset`` so the action
    still does *something* on systems where the canonical path fails.
    """
    cgsession = (
        "/System/Library/CoreServices/Menu Extras/User.menu/"
        "Contents/Resources/CGSession"
    )
    try:
        if os.path.isfile(cgsession):
            r = subprocess.run(
                [cgsession, "-suspend"],
                timeout=5, capture_output=True,
            )
            if r.returncode == 0:
                return True, "Екран заблоковано"
    except Exception:
        pass
    # Fallback 1: Cmd-Ctrl-Q sends the real lock shortcut.
    try:
        r = subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "q" using {command down, control down}',
            ],
            timeout=5, capture_output=True,
        )
        if r.returncode == 0:
            return True, "Екран заблоковано"
    except Exception:
        pass
    # Fallback 2: display sleep — does NOT actually lock if "Require
    # password" is off, but at least blanks the screen. Returned
    # message is honest about the degraded mode.
    try:
        subprocess.run(
            ["pmset", "displaysleepnow"],
            timeout=5, check=True,
        )
        return True, "Дисплей заснув (без блокування — увімкни 'Require password immediately' у налаштуваннях)"
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


def _previous_track() -> tuple[bool, str]:
    """Skip to previous track. R34-S31."""
    for app in ("Spotify", "Music"):
        try:
            r = subprocess.run(
                ["osascript", "-e", f'tell application "{app}" to previous track'],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return True, f"Попередній трек у {app}"
        except Exception:
            continue
    return False, "Не знайшов музичний плеєр"


def _youtube_search(query: str) -> tuple[bool, str]:
    """Open YouTube search results for the given query. R34-S31.

    The user said "знайди X на YouTube" — we want them landed on the
    actual YouTube SERP, not Google's web search showing YouTube as one
    of many results. We URL-encode and open
    ``https://www.youtube.com/results?search_query=...`` in the default
    browser.
    """
    from urllib.parse import quote_plus
    q = (query or "").strip()
    if not q:
        return False, "Не вказано що шукати"
    url = f"https://www.youtube.com/results?search_query={quote_plus(q)}"
    try:
        subprocess.run(["open", url], timeout=10, check=True)
        return True, f"Шукаю на YouTube: {q[:60]}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _open_in_browser(url: str, browser_app: str) -> tuple[bool, str]:
    """Open URL in a specific browser (Safari/Chrome/Firefox/Arc). R34-S31.

    Uses ``open -a <browser> <url>`` which forces the URL into the
    named browser regardless of system default. Falls back to default
    browser if the specified one isn't installed.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    canonical = _resolve_app_name(browser_app)
    try:
        r = subprocess.run(
            ["open", "-a", canonical, url],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return True, f"Відкрив {url} у {canonical}"
        # Fallback: default browser via plain `open`.
        r2 = subprocess.run(["open", url], timeout=10, capture_output=True)
        if r2.returncode == 0:
            return True, f"Відкрив {url} у браузері за замовчуванням"
        return False, f"Не зміг відкрити {url} у {canonical}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _browser_new_tab(browser_hint: str = "") -> tuple[bool, str]:
    """Open a new tab in the specified (or frontmost) browser. R34-S31.

    If ``browser_hint`` is empty we figure out which browser is
    frontmost via ``osascript`` and send Cmd+T to that one. Falls back
    to Safari if no known browser is active.
    """
    browser = _resolve_app_name(browser_hint) if browser_hint else ""
    if not browser or browser not in (
        "Safari", "Google Chrome", "Firefox", "Arc", "Microsoft Edge", "Brave Browser",
    ):
        # Detect frontmost
        try:
            r = subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to '
                    'get name of first application process whose frontmost is true',
                ],
                capture_output=True, text=True, timeout=5,
            )
            front = (r.stdout or "").strip()
            if front in ("Safari", "Google Chrome", "Firefox", "Arc", "Microsoft Edge", "Brave Browser"):
                browser = front
            else:
                browser = "Safari"
        except Exception:
            browser = "Safari"
    try:
        # Make sure the browser is frontmost first
        subprocess.run(
            ["osascript", "-e", f'tell application "{browser}" to activate'],
            timeout=5, capture_output=True,
        )
        # Cmd+T for new tab — works in all common browsers
        subprocess.run(
            [
                "osascript", "-e",
                f'tell application "System Events" to '
                f'tell process "{browser}" to keystroke "t" using {{command down}}',
            ],
            timeout=5, capture_output=True, check=True,
        )
        return True, f"Нова вкладка у {browser}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _browser_close_tab(browser_hint: str = "") -> tuple[bool, str]:
    """Close current browser tab via Cmd+W. R34-S31."""
    browser = _resolve_app_name(browser_hint) if browser_hint else ""
    if not browser:
        try:
            r = subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to '
                    'get name of first application process whose frontmost is true',
                ],
                capture_output=True, text=True, timeout=5,
            )
            browser = (r.stdout or "").strip() or "Safari"
        except Exception:
            browser = "Safari"
    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'tell application "System Events" to '
                f'tell process "{browser}" to keystroke "w" using {{command down}}',
            ],
            timeout=5, capture_output=True, check=True,
        )
        return True, f"Закрив вкладку у {browser}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _hide_app(app_name: str) -> tuple[bool, str]:
    """Hide / minimize a Mac app to the Dock. R34-S31."""
    canonical = _resolve_app_name(app_name) if app_name else ""
    if not canonical:
        # Hide frontmost
        try:
            subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to '
                    'set visible of (first process whose frontmost is true) to false',
                ],
                timeout=5, capture_output=True, check=True,
            )
            return True, "Сховав активний додаток"
        except Exception as e:
            return False, f"Помилка: {e}"
    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'tell application "System Events" to set visible of process "{canonical}" to false',
            ],
            timeout=5, capture_output=True, check=True,
        )
        return True, f"Сховав {canonical}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _quit_app(app_name: str) -> tuple[bool, str]:
    """Quit a Mac app gracefully. R34-S31.

    We deliberately use AppleScript ``quit`` (sends Cmd+Q to the
    app's main dispatcher) rather than ``killall`` because:
      1. ``quit`` lets the app save state, close documents cleanly,
         and prompt the user about unsaved changes.
      2. ``killall`` would also blow away background helpers that
         share the bundle id (e.g. browser plugin processes).
    """
    canonical = _resolve_app_name(app_name) if app_name else ""
    if not canonical or len(canonical) < 2:
        return False, "Не вказано назву додатка"
    try:
        r = subprocess.run(
            ["osascript", "-e", f'tell application "{canonical}" to quit'],
            timeout=10, capture_output=True, text=True,
        )
        if r.returncode == 0:
            return True, f"Закрив {canonical}"
        return False, f"Не зміг закрити {canonical}"
    except Exception as e:
        return False, f"Помилка: {e}"


def _switch_window() -> tuple[bool, str]:
    """Cycle to next window via Cmd+Tab. R34-S31."""
    try:
        subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to keystroke tab using {command down}',
            ],
            timeout=5, capture_output=True, check=True,
        )
        return True, "Перемикнув вікно"
    except Exception as e:
        return False, f"Помилка: {e}"


def _show_desktop() -> tuple[bool, str]:
    """Show desktop via Mission Control's "Show Desktop" gesture (F11). R34-S31."""
    try:
        subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to key code 103',  # F11
            ],
            timeout=5, capture_output=True, check=True,
        )
        return True, "Показую робочий стіл"
    except Exception as e:
        return False, f"Помилка: {e}"


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
    """Open Spotlight search with a pre-filled query.

    Audit round 8 SECURITY fix: previously interpolated `query` (raw
    user voice text!) directly into AppleScript source via f-string.
    A Whisper transcript like  `тест" \\n do shell script "rm -rf ~`
    would inject arbitrary AppleScript including `do shell script`,
    giving voice → shell RCE. Now passes the query as an argv string
    via `on run argv ... item 1 of argv` so AppleScript treats it as
    pure data, never as code.
    """
    # Defense in depth: also strip non-printable + control chars that
    # could confuse AppleScript even via argv (NULs, escapes, etc.).
    safe_query = "".join(ch for ch in query if ch.isprintable())[:200]
    if not safe_query.strip():
        return False, "Порожній запит."
    try:
        script = (
            'on run argv\n'
            '    set q to item 1 of argv\n'
            '    tell application "System Events"\n'
            '        keystroke space using {command down}\n'
            '        delay 0.3\n'
            '        keystroke q\n'
            '    end tell\n'
            'end run'
        )
        subprocess.run(
            ["osascript", "-e", script, "--", safe_query],
            timeout=10, check=True,
        )
        return True, f"Шукаю '{safe_query}' у Spotlight"
    except Exception as e:
        return False, f"Помилка: {e}"


def _new_email(to: str = "") -> tuple[bool, str]:
    """Open Mail with a new compose window.

    Audit round 20 P2 fix: the regex that fed ``to`` accepted
    ``\\S+@\\S+\\.\\S+`` — which allows ``bob@evil.com?bcc=attacker@x.com``
    or ``victim@x.com&body=secret``. Building ``mailto:{to}`` then
    handed those query-string params straight to ``open`` and the
    OS Mail.app honoured them — producing a compose window with a
    hidden BCC + pre-filled body the user might miss. Strict-validate
    the address via ``email.utils.parseaddr`` and reject anything that
    contains URL query metacharacters.
    """
    safe_to = ""
    if to:
        candidate = str(to).strip()
        # Reject any URL-injection metacharacters BEFORE parseaddr; the
        # parser is lenient and will happily round-trip ``a@b.c?bcc=x``.
        if any(ch in candidate for ch in ("?", "&", "#", "\n", "\r", " ")):
            return False, "Невалідний email — містить заборонені символи"
        try:
            from email.utils import parseaddr
            name, addr = parseaddr(candidate)
            # parseaddr returns ('', '') on garbage. We don't honour the
            # display-name part (would let an attacker inject "Bob"
            # <attacker@x.com>); only the raw addr.
            if not addr or "@" not in addr or "." not in addr.split("@", 1)[1]:
                return False, "Невалідний email"
            safe_to = addr
        except Exception:
            return False, "Невалідний email"
    # URL-encode the validated address so e.g. a ``+`` survives intact
    # without ever being interpretable as a query-param boundary.
    from urllib.parse import quote
    url = f"mailto:{quote(safe_to, safe='@.+-_')}" if safe_to else "mailto:"
    try:
        subprocess.run(["open", url], timeout=5, check=True)
        return True, f"Відкрив нового листа{' до ' + safe_to if safe_to else ''}"
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


# ─── External integration tools (stub level, ready for API keys) ─────────
#
# Each stub:
#   1. Reads required API key from env (JARVIS_<SERVICE>_TOKEN)
#   2. If missing — returns (False, "Підключи <Service> токен у JARVIS_X_TOKEN")
#   3. If present — calls real API (TODO: implement once user provides token)
#   4. Always emits typed tool_call event via _run_async wrapper
#
# This means: actions are wired NOW, hooked into voice loop NOW, HUD shows
# badge NOW. The only missing piece is API credentials. When user adds them,
# the functions Just Start Working — no plumbing changes needed.

def _linear_create_issue(title: str, description: str = "") -> tuple[bool, str]:
    """Create a Linear issue. Requires JARVIS_LINEAR_TOKEN env var."""
    token = os.environ.get("JARVIS_LINEAR_TOKEN", "").strip()
    if not token:
        return False, "Підключи Linear токен у JARVIS_LINEAR_TOKEN (Settings → API → Personal API keys)"
    try:
        import urllib.request as _ur
        import urllib.error as _ue
        team_id = os.environ.get("JARVIS_LINEAR_TEAM_ID", "").strip()
        if not team_id:
            return False, "Постав JARVIS_LINEAR_TEAM_ID (один раз — з URL твоєї команди)"
        body = {
            "query": (
                "mutation IssueCreate($title:String!,$teamId:String!,$desc:String){"
                " issueCreate(input:{title:$title,teamId:$teamId,description:$desc})"
                " { success issue { identifier title url } } }"
            ),
            "variables": {"title": title, "teamId": team_id, "desc": description or ""},
        }
        req = _ur.Request(
            "https://api.linear.app/graphql",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": token, "Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        issue = (((data or {}).get("data") or {}).get("issueCreate") or {}).get("issue") or {}
        if issue:
            return True, f"Створив {issue.get('identifier')}: {issue.get('title')}"
        err = data.get("errors", [{}])[0].get("message", "невідома помилка")
        return False, f"Linear: {err}"
    except Exception as e:
        return False, f"Linear error: {e}"


def _github_create_issue(repo: str, title: str, body: str = "") -> tuple[bool, str]:
    """Create a GitHub issue. Requires JARVIS_GITHUB_TOKEN env var.

    repo format: 'owner/repo' (e.g. 'isair/jarvis')
    """
    token = os.environ.get("JARVIS_GITHUB_TOKEN", "").strip()
    if not token:
        return False, "Підключи GitHub токен у JARVIS_GITHUB_TOKEN (gh auth token)"
    try:
        import urllib.request as _ur
        payload = {"title": title, "body": body or ""}
        req = _ur.Request(
            f"https://api.github.com/repos/{repo}/issues",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        with _ur.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if data.get("number"):
            return True, f"GitHub #{data['number']}: {data.get('title', title)}"
        return False, f"GitHub error: {data}"
    except Exception as e:
        return False, f"GitHub error: {e}"


def _gitlab_create_issue(
    project: str, title: str, body: str = "", *, base_url: str = "https://gitlab.com"
) -> tuple[bool, str]:
    """Create a GitLab issue. Requires ``JARVIS_GITLAB_TOKEN`` env var.

    Audit round 19 fix: the user reported migrating from GitHub to
    GitLab; the existing ``_github_create_issue`` is preserved for
    users still on GitHub, and this sibling action provides parity for
    the GitLab path. Both can be wired up at once (the registry +
    voice patterns choose between them based on the LLM's phrasing).

    Args:
        project: Either a numeric project ID (e.g. ``"12345678"``) OR
            a URL-encoded namespace path (e.g. ``"mygroup/myproject"`` —
            the encoder runs here so the caller can pass the readable
            form). Personal projects use ``"username/projectname"``.
        title: Issue title.
        body: Optional issue description (markdown supported).
        base_url: GitLab instance URL. Defaults to gitlab.com; pass
            ``"https://gitlab.example.com"`` for self-hosted.

    Token format: GitLab personal access tokens start with ``glpat-``.
    See https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html
    """
    token = os.environ.get("JARVIS_GITLAB_TOKEN", "").strip()
    if not token:
        return False, (
            "Підключи GitLab токен у JARVIS_GITLAB_TOKEN "
            "(Personal Access Token, scope: api)"
        )
    # GitLab project identifiers must be URL-encoded when passed as a
    # namespace path. Numeric IDs pass through unchanged.
    try:
        import urllib.parse as _uparse
        project_clean = project.strip().strip("/")
        if project_clean.isdigit():
            project_part = project_clean
        else:
            project_part = _uparse.quote(project_clean, safe="")
    except Exception:
        return False, f"GitLab: bad project identifier {project!r}"
    try:
        import urllib.request as _ur
        # Send title + description as form-encoded — GitLab's REST API
        # is happiest with this shape and it avoids the JSON-body
        # ``Content-Type`` dance that some self-hosted instances reject.
        # We still URL-encode the body to keep multi-line markdown intact.
        import urllib.parse as _uparse
        form = _uparse.urlencode({"title": title, "description": body or ""}).encode("utf-8")
        api_url = f"{base_url.rstrip('/')}/api/v4/projects/{project_part}/issues"
        req = _ur.Request(
            api_url,
            data=form,
            headers={
                # GitLab uses ``PRIVATE-TOKEN`` rather than the
                # ``Authorization: Bearer`` header GitHub takes. Both
                # the cloud (gitlab.com) and self-hosted instances
                # accept this consistently.
                "PRIVATE-TOKEN": token,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            method="POST",
        )
        with _ur.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if isinstance(data, dict) and data.get("iid"):
            # ``iid`` is the project-local issue number ("#42"); ``id``
            # is the GitLab-wide numeric id. Users care about iid.
            return True, f"GitLab #{data['iid']}: {data.get('title', title)}"
        # Self-hosted instances occasionally return a partial body —
        # surface the message field if present.
        msg = (
            data.get("message")
            if isinstance(data, dict)
            else None
        ) or str(data)[:200]
        return False, f"GitLab error: {msg}"
    except Exception as e:
        return False, f"GitLab error: {e}"


def _notion_append_to_page(page_id: str, text: str) -> tuple[bool, str]:
    """Append a paragraph to a Notion page. Requires JARVIS_NOTION_TOKEN."""
    token = os.environ.get("JARVIS_NOTION_TOKEN", "").strip()
    if not token:
        return False, "Підключи Notion токен у JARVIS_NOTION_TOKEN (Internal Integration)"
    try:
        import urllib.request as _ur
        payload = {
            "children": [{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                }
            }]
        }
        req = _ur.Request(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            method="PATCH",
        )
        with _ur.urlopen(req, timeout=15) as r:
            r.read()
        return True, f"Додав до Notion: {text[:60]}"
    except Exception as e:
        return False, f"Notion error: {e}"


# Need json + os imports at top — they are already in the file.


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
    # Round 26 fix (F62): added RU variants. The system migrated UA→RU
    # in May, but action patterns still expected UA "відкрию" phrasing.
    # LLM was correctly outputting "Сейчас открою Safari" (RU) but no
    # pattern matched → no pending_action → no execution path. Live
    # evidence (events.jsonl 1779136981): "Сейчас открою ." → no
    # action fired → Safari/YouTube never opened.
    (
        re.compile(r"(?:сейчас\s+)?(?:от)?крою\s+(?:страницу\s+|сайт\s+)?(https?://\S+|[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?)", re.IGNORECASE),
        lambda m: Action(
            name="open_url",
            description=f"Сейчас открою {m.group(1).strip()}",
            fn=lambda: _open_url(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
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
    # Round 26 fix (F62): RU variants. Common LLM phrasings:
    #   "Сейчас открою Safari"
    #   "Сейчас открою YouTube в Safari"  → matches Safari, then we
    #                                      web-search YouTube inside
    #   "Сейчас запущу Telegram"
    # Round 28 (F70): require ≥3 chars and reject lone prepositions.
    # Previous regex matched group(1)="в" (Cyrillic preposition) when
    # Latin app names got stripped by sanitize → _open_app("в") fail.
    # Negative lookahead blocks the captured word from being a known
    # 1-3 char filler. Length floor moved from {1,30}? to {2,30}? to
    # require ≥3 chars total (1 lead char + 2+ tail chars).
    # F70 char class extended to cover Ukrainian І/Ї/Є/Ґ (U+0406, U+0407,
    # U+0404, U+0490) which are OUTSIDE А-Яа-я (U+0410-U+044F). The Piper
    # transliteration in F69 emits "Сафарі"/"Файрфокс" with these chars.
    # F70 char class extended + word-boundary anchor.
    # Word boundary `\b(?:от)?крою\b` ensures "закрою" inside a
    # different verb does not trigger an open_app match
    # ("сейчас открою и закрою всё" was mis-parsing "закрою всё").
    # Round 30 (F91): pipe captured text through _trim_app_capture
    # to drop trailing "и перейду на Ютуб." sentence-tails that the
    # lazy regex over-extended into when the LLM produced compound
    # actions in one sentence.
    (
        re.compile(
            r"\b(?:сейчас\s+)?(?:от)?крою\b[\s,\.]?\s+"
            r"(?:приложение\s+|app\s+)?"
            r"(?!(?:в|во|у|на|до|от|об|обо|для|при|по|с|со|ко|и|а|но|или|о)[\s\.,!?]|$)"
            r"([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9\s]{2,30}?)"
            r"(?:\s+в\s+[A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+)?(?:[.!?,]|$)",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="open_app",
            description=f"Сейчас открою {_trim_app_capture(m.group(1))}",
            fn=lambda: _open_app(_trim_app_capture(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(
            r"\b(?:сейчас\s+)?запущу\b\s+(?:приложение\s+|app\s+)?"
            r"(?!(?:в|во|у|на|до|от|об|обо|для|при|по|с|со|ко|и|а|но|или|о)[\s\.,!?]|$)"
            r"([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9\s]{2,30}?)"
            r"(?:[.!?,]|$)",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="open_app",
            description=f"Сейчас запущу {_trim_app_capture(m.group(1))}",
            fn=lambda: _open_app(_trim_app_capture(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(
            r"\b(?:зараз\s+)?(?:від|за)крию\b[\s,\.]?\s+(?:додаток\s+)?"
            r"(?!(?:в|у|на|до|від|для|при|по|з|зі|і|та|але|чи|або)[\s\.,!?]|$)"
            r"([A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9\s]{2,30})",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="open_app",
            description=f"Зараз відкрию {_trim_app_capture(m.group(1))}",
            fn=lambda: _open_app(_trim_app_capture(m.group(1))),
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
# RU added after May 16 uk→ru migration. User report: "не може виконувати
# деякі дії на ПК" — LLM was correctly generating "Сейчас открою Safari"
# pending-action pattern, but user's natural RU "выполняй" / "сделай" was
# not in the trigger set, so the action never executed.
CONFIRM_WORDS = {
    # RU (primary)
    "выполняй", "выполни", "выполнить", "давай", "да", "ага", "ок", "окей",
    "подтверждаю", "делай", "сделай", "поехали", "запускай", "запусти",
    "да-да", "конечно",
    # UA (legacy)
    "виконуй", "виконай", "так", "підтверджую", "роби", "зроби",
    "поїхали", "так-так",
    # EN
    "yes", "do it", "go", "execute", "ok",
}

# Words that mean "cancel".
DENY_WORDS = {
    # RU (primary)
    "отмени", "отменить", "нет", "не надо", "забудь", "стоп",
    "отставить", "не нужно",
    # UA (legacy)
    "відміни", "відмінити", "ні", "не треба",
    # EN
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


def _normalised_for_match(user_text: str) -> str:
    """Lowercase + strip surrounding punctuation / whitespace."""
    return user_text.lower().strip().rstrip(".!?,;:").strip()


def is_confirmation(user_text: str) -> bool:
    """Check if the user's reply is a confirmation.

    Audit round 7 fix C5: previously matched ANY confirm word ANYWHERE
    in the utterance via `\\b{w}\\b`. Family/TV speech routinely contains
    "да"/"ок"/"так"/"yes" inside longer sentences ("да я уже сделал",
    "ок поехали в магазин", "yes I think so"). With a pending-action
    awaiting confirmation, those ambient phrases silently executed the
    action (open Safari, lock screen, restart daemon).

    New policy:
      - SHORT (≤3 words): treat the WHOLE utterance (minus punctuation)
        as the candidate. Must match a confirm word exactly OR be a
        multi-word confirm phrase ("да-да", "так-так", "do it", "go ahead").
      - LONG (≥4 words): require an imperative-mood confirm verb
        ("выполняй", "виконуй", "запускай", "делай", "сделай",
        "execute", "do it") — these are very rarely heard in ambient
        speech and almost always signal intent.

    Bare "да"/"ок"/"так"/"yes" in a 4+ word utterance NEVER confirms.
    """
    if not user_text:
        return False
    t = _normalised_for_match(user_text)
    if not t:
        return False
    words = t.split()
    # SHORT path — whole-utterance exact match against confirm set.
    if len(words) <= 3:
        if t in CONFIRM_WORDS:
            return True
        # Allow leading "yes,"/"да,"/"так," followed by a confirm verb
        # ("так виконуй") — common natural pattern after a prompt.
        for w in CONFIRM_WORDS:
            if " " in w and w in t:
                return True
        # Single short confirm at the start ("ок виконуй ці дії") —
        # require at least one IMPERATIVE_CONFIRM verb to also be present.
        for w in _IMPERATIVE_CONFIRM:
            if re.search(rf"\b{re.escape(w)}\b", t):
                return True
        return False
    # LONG path — require an imperative confirm verb.
    for w in _IMPERATIVE_CONFIRM:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False


def is_denial(user_text: str) -> bool:
    """Check if the user's reply is a denial.

    Same conservatism as `is_confirmation` — bare "нет"/"ні"/"no" in
    a multi-word utterance does NOT count. Ambient "не знаю де ключі"
    used to accidentally deny pending actions.
    """
    if not user_text:
        return False
    t = _normalised_for_match(user_text)
    if not t:
        return False
    words = t.split()
    if len(words) <= 3:
        if t in DENY_WORDS:
            return True
        for w in DENY_WORDS:
            if " " in w and w in t:
                return True
        return False
    # LONG path — require imperative deny verb.
    for w in _IMPERATIVE_DENY:
        if re.search(rf"\b{re.escape(w)}\b", t):
            return True
    return False


# Imperative confirm verbs — VERY RARELY appear in ambient speech.
# Conservative subset of CONFIRM_WORDS that we'll match anywhere in
# a long utterance. Single-syllable acks ("да", "ок", "так", "yes")
# are deliberately EXCLUDED — they false-fire on family/TV speech.
#
# Audit round 8 regression find: previously included "поехали", "поїхали",
# "делай", "сделай", "роби", "зроби" — but these are EXTREMELY common
# in family speech ("ок поехали в магазин", "сделай уроки", "роби
# домашку"). With a pending action, ambient "ок поехали в магазин"
# silently executed the action. Removed all 6 — they're still in
# CONFIRM_WORDS so the short-utterance exact-match path still
# confirms bare "сделай" / "поехали" alone (which is fine: a 1-word
# command IS unambiguous).
_IMPERATIVE_CONFIRM = {
    "выполняй", "выполни", "выполнить",
    "виконуй", "виконай",
    "запускай", "запусти",
    "execute", "do it",
    # Round 27 fix (F64): "подтверждаю" / "підтверджую" / "confirm"
    # were in CONFIRM_WORDS (short-utterance match) but NOT in the
    # IMPERATIVE set used for long-utterance match. After round 26
    # F61 LLM now asks "Подтверди моё действие, пожалуйста" — user
    # naturally responds with a longer sentence like "Подтверждаю.
    # Открывай YouTube в Safari" (≥4 words → LONG path). Without
    # "подтверждаю" here, the LONG path required "выполняй" and
    # rejected the user's natural confirmation. Live evidence:
    # events.jsonl seq=336+ had partials containing "Подтверждаю"
    # but no action_run / tool_call followed.
    "подтверждаю", "подтверди",
    "підтверджую", "підтверди",
    "confirm",
}

_IMPERATIVE_DENY = {
    "отмени", "отменить", "відміни", "відмінити",
    "отставить", "забудь",
    "cancel", "skip",
}


# ─── language switching ──────────────────────────────────────────────────

# Phrases that request a switch to a non-default language. Default is
# always Ukrainian — these triggers require explicit user confirmation.
# Audit round 7 fix C6: the bare "на X" fallback patterns ("на російській",
# "на англійській") matched ambient prose like "слышал на русском канале",
# "учу на англійській", "видео на немецком" — any of which fired a
# pending language switch.
#
# Audit round 8 regression: removing the bare patterns also broke
# legitimate wake-prefixed phrases like "Джарвіс, на українську" /
# "Джарвіс на російську". Solution: ALLOW the bare "на X" pattern
# but ONLY when prefixed by the wake word — that proves the user
# is addressing the assistant, not narrating ambient.
LANG_SWITCH_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Russian — imperative verb form
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди|перейти)\s+(?:на\s+)?(?:російськ|русск|русс)\w*\b", re.IGNORECASE),
     "ru", "Переходжу на російську"),
    # Russian — wake-prefixed "Джарвіс на російську"
    (re.compile(r"\b(?:джарвіс|джарвис|jarvis)[\s,]+(?:на\s+)?(?:російськ|русск|русс)\w*\b", re.IGNORECASE),
     "ru", "Переходжу на російську"),
    (re.compile(r"\bswitch\s+to\s+russian\b", re.IGNORECASE),
     "ru", "Переходжу на російську"),
    # English — imperative
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди)\s+(?:на\s+)?англійськ\w*\b", re.IGNORECASE),
     "en", "Switching to English"),
    # English — wake-prefixed
    (re.compile(r"\b(?:джарвіс|джарвис|jarvis)[\s,]+(?:на\s+)?англійськ\w*\b", re.IGNORECASE),
     "en", "Switching to English"),
    (re.compile(r"\bswitch\s+to\s+english\b", re.IGNORECASE),
     "en", "Switching to English"),
    # German — imperative
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди)\s+(?:на\s+)?німецьк\w*\b", re.IGNORECASE),
     "de", "Wechsle zu Deutsch"),
    # German — wake-prefixed
    (re.compile(r"\b(?:джарвіс|джарвис|jarvis)[\s,]+(?:на\s+)?німецьк\w*\b", re.IGNORECASE),
     "de", "Wechsle zu Deutsch"),
    (re.compile(r"\b(?:switch\s+to|auf)\s+(?:german|deutsch)\b", re.IGNORECASE),
     "de", "Wechsle zu Deutsch"),
    # Back to Ukrainian — imperative
    (re.compile(r"\b(?:говори|скажи|відповідай|перейди|повернись)\s+(?:на\s+|обратно\s+)?українськ\w*\b", re.IGNORECASE),
     "uk", "Повертаюсь на українську"),
    # Back to Ukrainian — wake-prefixed
    (re.compile(r"\b(?:джарвіс|джарвис|jarvis)[\s,]+(?:на\s+)?українськ\w*\b", re.IGNORECASE),
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

def _trim_app_capture(raw: str) -> str:
    """Round 30 (F91): trim a captured app name at the first sentence
    conjunction or trailing preposition.

    The LLM frequently produces a multi-clause sentence like
        "Сейчас открою Сафари и перейду на Ютуб."
    The ACTION_PATTERNS regex captures lazily but the trailing
    "(?:[.!?,]|$)" anchor lets the lazy quantifier expand all the
    way to the terminal dot, picking up
        "Сафари и перейду на Ютуб"
    as the "app name". `_open_app` then fails on that phrase. We
    post-process by chopping at the FIRST occurrence of a
    space-bounded conjunction (и, і, та, или, але, чи, або, а, но)
    or a "preposition + something" tail (на, в, у, до, з, с …) so
    only the actual proper noun remains.
    """
    if not raw:
        return raw
    s = raw.strip()
    # Cut at the first conjunction. Word-boundary-ish split using
    # a deliberately small set so common app names like "Find My"
    # (which contains "My" — but no Cyrillic conjunctions) survive.
    s = re.split(
        r"\s+(?:и|і|та|или|але|чи|або|а|но|ні|не)\s+",
        s,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    # Cut a trailing preposition + object ("на новой странице",
    # "в фоне"). We only trim if at least 2 leading word-chars
    # survive, so single-token names aren't damaged.
    s2 = re.split(
        r"\s+(?:на|в|у|до|з|с|со|от|для|при|по|обо?|із|зі)\s+\S",
        s,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    if len(s2) >= 2:
        s = s2
    return s


def _make_open_app_action(raw: str, verb_present: str) -> Action:
    """Factory for open_app action — pre-resolves alias for nicer TTS."""
    raw = _trim_app_capture(raw)
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
    # Round 27 fix (F66): RU URL open. After May UA→RU migration the
    # user speaks RU but USER_COMMAND_PATTERNS only matched UA. Live
    # evidence (events.jsonl): "открой YouTube в Safari" never hit
    # this fast path — fell through to LLM → 15-25s latency vs ~50ms.
    # Round 28 fix (F71): added infinitive forms (открыть/запустить).
    # User said "открыть YouTube" but the imperative-only regex
    # `(?:от)?крой(?:те)?` missed it → fell to LLM → "Сейчас открою
    # YouTube в Safari" → Latin stripped → app="в" fail.
    (
        re.compile(
            r"^\s*(?:пожалуйста[\s,]+)?"
            r"(?:(?:от)?крой(?:те)?|(?:от)?крыть|откры(?:вать|вай(?:те)?))\s+"
            r"(?:сайт\s+|страницу\s+)?"
            r"((?:https?://)?[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?)\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="open_url",
            description=f"Открываю {m.group(1).strip()}",
            fn=lambda: _open_url(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: CLOSE + NEW-TAB patterns MUST come before open_app —
    # the existing open_app verb alternation (?:від|за)крий matches
    # both "відкрий" AND "закрий", and the app-name capture is greedy
    # enough to grab "нову вкладку" too. First-match-wins, so we
    # place close_tab + quit_app + new_tab FIRST.
    (
        re.compile(
            r"^\s*(?:закрий|закрити|закрой|закрыть)\s+(?:вкладку|таб|tab)\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="browser_close_tab",
            description="Закриваю вкладку",
            fn=lambda: _browser_close_tab(""),
            created_ts=time.time(),
        ),
    ),
    (
        # quit_app — "закрий <app>" / "закрой <app>". Must NOT match
        # "вкладку" / "таб" / "tab" (those go to close_tab above).
        re.compile(
            r"^\s*(?:закрий|закрити|закрой|закрыть|вийди\s+з|выйди\s+из)\s+"
            r"(?:додатка?|додатку|програму|приложения?|приложение|app)?\s*"
            r"(?!вкладку\b|таб\b|tab\b)"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґЁё][A-Za-zА-Яа-яЇїІіЄєҐґЁё0-9\s-]{1,30})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="quit_app",
            description=f"Закриваю {_resolve_app_name(m.group(1).strip())}",
            fn=lambda: _quit_app(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # browser_new_tab — handles all variants:
    #   "новий таб", "нова вкладка", "новая вкладка", "new tab",
    #   "відкрий нову вкладку", "відкрий нову вкладку у Safari",
    #   "open new tab", "створи новий таб у Chrome"
    # Verb prefix optional; "нов\w*" matches all gendered/number
    # forms of UA/RU "новий|нова|нову|новую|новой|...".
    (
        re.compile(
            r"^\s*(?:(?:відкрий|відкрити|відкривай|открой|открыть|открывай|"
            r"створи|створити|create|open)\s+)?"
            r"(?:нов\w*\s+|new\s+)?(?:вкладку|вкладка|вкладки|таб|tab)"
            r"(?:\s+(?:у|в|in)\s+"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґЁё][A-Za-zА-Яа-яЇїІіЄєҐґЁё0-9\s-]{1,20}))?"
            r"\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="browser_new_tab",
            description=(
                f"Нова вкладка у {_resolve_app_name(m.group(1).strip())}"
                if m.group(1)
                else "Нова вкладка"
            ),
            fn=lambda: _browser_new_tab((m.group(1) or "").strip()),
            created_ts=time.time(),
        ),
    ),
    # English: "open X" → open_app (compat with existing UA/RU
    # patterns; previously this fell through to the LLM).
    (
        re.compile(
            r"^\s*(?:please\s+)?(?:open|launch|start)\s+"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґЁё][A-Za-zА-Яа-яЇїІіЄєҐґЁё0-9\s-]{1,30})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: _make_open_app_action(m.group(1).strip(), "Opening"),
    ),
    # ── apps ── (must match clean imperatives, no dots)
    # Round 28 (F71): UA infinitive forms (відкрити/запустити) added —
    # same rationale as RU patterns below: intent_judge re-fires often
    # rewrite imperatives as infinitives.
    # R34-S31: tightened verb alternation to ONLY match OPEN verbs
    # (від-, but NOT за-). Previously "(?:від|за)крий" caused "закрий
    # Telegram" to match and OPEN Telegram instead of quitting it.
    # The close patterns above now handle "за-" exclusively.
    (
        re.compile(
            r"^\s*(?:а\s+)?(?:будь\s+ласка\s+)?"
            r"(?:відкрий(?:те)?|відкрити|відкривай(?:те)?)\s+"
            r"(?:додаток\s+|програму\s+)?"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґ][A-Za-zА-Яа-яЇїІіЄєҐґ0-9\s-]{1,30})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: _make_open_app_action(m.group(1).strip(), "Відкриваю"),
    ),
    (
        re.compile(
            r"^\s*(?:будь\s+ласка\s+)?"
            r"(?:запусти(?:те)?|запустити|запускай(?:те)?)\s+"
            r"(?:додаток\s+|програму\s+)?"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґ][A-Za-zА-Яа-яЇїІіЄєҐґ0-9\s-]{1,30})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: _make_open_app_action(m.group(1).strip(), "Запускаю"),
    ),
    # Round 27 (F66): RU apps. Supports "открой Telegram",
    # "открой YouTube в Safari" (we open Safari as the host, group(1)
    # is Safari due to the optional "в X" suffix being non-capturing).
    # Round 28 (F71): added infinitive forms — "открыть Safari",
    # "запустить Telegram". intent_judge frequently rewrites a user's
    # imperative as infinitive when re-firing the hot window, so the
    # fast-path must handle BOTH grammatical forms.
    # Round 30 (F92): add common Whisper misheards to the verb
    # alternation. Live evidence (events.jsonl): user said "відкрий
    # Safari і знайди ютуб" → Whisper transcribed "вкрой сафари и
    # найди в нём youtube" — note the malformed "вкрой" (lost the
    # leading "от"/"від"). qwen2.5:3b Russian fluent enough to
    # rescue the intent at LLM level, but that costs 30-60s on cold
    # cache. With the fast-path now accepting "вкрой"/"кры"/"скрой"
    # we route the obvious case in ~50ms instead.
    (
        re.compile(
            r"^\s*(?:пожалуйста[\s,]+)?"
            r"(?:(?:от|во|в|с|о)?крой(?:те)?|"
            r"(?:от|во|в|с|о)?крыть|"
            r"откры(?:вать|вай(?:те)?))\s+"
            r"(?:приложение\s+|программу\s+|app\s+)?"
            r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\s-]{1,30}?)"
            r"(?:\s+в\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\s-]{1,20}))?"
            r"\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        # For "X в Y" form (m.group(2) populated), prefer opening the
        # HOST (Y, typically Safari) and trust the LLM/user to type
        # X inside. Simpler than launching X directly.
        lambda m: _make_open_app_action(
            (m.group(2) or m.group(1)).strip(),
            "Открываю",
        ),
    ),
    (
        re.compile(
            r"^\s*(?:пожалуйста[\s,]+)?"
            r"(?:запусти(?:те)?|запустить|запускать|запускай(?:те)?)\s+"
            r"(?:приложение\s+|программу\s+|app\s+)?"
            r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9\s-]{1,30})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
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
    # Round 27 (F66): RU volume / mute equivalents.
    (
        re.compile(r"^\s*(?:постав|сделай)\s+громкост[ьи]?\s+(?:на\s+|до\s+)?(\d{1,3})\s*(?:процент)?\w*\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="set_volume",
            description=f"Громкость {m.group(1)}",
            fn=lambda: _set_volume(int(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"^\s*громкост[ьи]?\s+(\d{1,3})\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(
            name="set_volume",
            description=f"Громкость {m.group(1)}",
            fn=lambda: _set_volume(int(m.group(1))),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(r"^\s*выключи\s+звук\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="mute", description="Выключаю звук", fn=_mute_system, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*включи\s+звук\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="unmute", description="Включаю звук", fn=_unmute_system, created_ts=time.time()),
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
    # Round 27 (F66): RU screen lock + screenshot.
    (
        re.compile(r"^\s*(?:за)?блокируй(?:те)?\s+(?:экран|компьютер|мак|ноут(?:бук)?)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="lock", description="Блокирую экран", fn=_lock_screen, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:сделай|сделать)\s+(?:скрин(?:шот)?|снимок\s+экрана)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="screenshot", description="Делаю скриншот", fn=_show_screen, created_ts=time.time()),
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
    # R34-S40 — Russian + English variants for time. Without these,
    # "сколько сейчас времени" / "what time is it" hit the slow LLM
    # path (qwen3:8b cold-eval = 8-12s) instead of the instant
    # ``datetime.now()`` regex.
    (
        re.compile(r"^\s*(?:сколько\s+(?:сейчас\s+)?времени|который\s+(?:сейчас\s+)?час|какое\s+(?:сейчас\s+)?время|время)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="say_time", description="Смотрю время", fn=_say_time, created_ts=time.time()),
    ),
    (
        re.compile(r"^\s*(?:what(?:'s|\s+is)?\s+the\s+time|what\s+time\s+is\s+it)\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="say_time", description="Checking time", fn=_say_time, created_ts=time.time()),
    ),
    # ── date ──
    (
        re.compile(r"^\s*(?:який\s+(?:сьогодні\s+)?(?:день|число|дата)|сьогодні\s+(?:який|яке)\s+(?:день|число|дата)|какое\s+(?:сегодня\s+)?(?:число|день|дата)|сегодня\s+(?:какое|какой)\s+(?:число|день|дата))\s*[!\.\?]?\s*$", re.IGNORECASE),
        lambda m: Action(name="say_date", description="Дивлюсь дату", fn=_say_date, created_ts=time.time()),
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
    # Round 29 (F82): duplicate URL pattern removed. Previously this
    # identical UA "відкрий <URL>" regex appeared twice (here and at
    # the top of USER_COMMAND_PATTERNS line ~1543). The second copy
    # was unreachable shadow but wasted match attempts on every voice
    # turn — small but persistent overhead.
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
    # ── Linear: create issue ──
    (
        re.compile(r"^\s*(?:створи|додай)\s+(?:задачу|тікет|issue|таску)\s+(?:у\s+|в\s+)?linear[:\s]+(.{3,200})\s*$", re.IGNORECASE),
        lambda m: Action(
            name="linear_create_issue",
            description=f"Створюю Linear-задачу: {m.group(1).strip()[:60]}",
            fn=lambda: _linear_create_issue(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── GitHub: create issue ──
    (
        re.compile(r"^\s*(?:створи|додай)\s+(?:issue|тікет|баг)\s+(?:у\s+|в\s+)?github(?:\s+([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+))?[:\s]+(.{3,200})\s*$", re.IGNORECASE),
        lambda m: Action(
            name="github_create_issue",
            description=f"Створюю GitHub issue: {m.group(2).strip()[:60]}",
            fn=lambda: _github_create_issue(
                m.group(1) or os.environ.get("JARVIS_GITHUB_DEFAULT_REPO", ""),
                m.group(2).strip(),
            ),
            created_ts=time.time(),
        ),
    ),
    # ── GitLab: create issue (round 19 — user works with GitLab now) ──
    #
    # Matches:
    #   "створи issue у gitlab: купити каву"
    #   "додай тікет в gitlab mygroup/myproj: тестовий баг"
    #   "create gitlab issue 12345: ..." (numeric project id)
    # The project arg may be a numeric ID OR a namespace path
    # (``mygroup/myproject``). When omitted, falls back to the
    # ``JARVIS_GITLAB_DEFAULT_PROJECT`` env var so the user can voice-
    # create issues without re-stating the project every time.
    (
        re.compile(
            r"^\s*(?:створи|додай|create|add)\s+"
            r"(?:issue|тікет|баг|ticket|bug)\s+"
            r"(?:у\s+|в\s+|in\s+)?gitlab"
            r"(?:\s+([a-zA-Z0-9_./-]+))?"
            r"[:\s]+(.{3,200})\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="gitlab_create_issue",
            description=f"Створюю GitLab issue: {m.group(2).strip()[:60]}",
            fn=lambda: _gitlab_create_issue(
                m.group(1) or os.environ.get("JARVIS_GITLAB_DEFAULT_PROJECT", ""),
                m.group(2).strip(),
            ),
            created_ts=time.time(),
        ),
    ),
    # ── Notion: append note to default page ──
    (
        re.compile(r"^\s*(?:додай|запиши|занотуй)\s+(?:у\s+|в\s+)?notion[:\s]+(.{3,400})\s*$", re.IGNORECASE),
        lambda m: Action(
            name="notion_append",
            description=f"Додаю до Notion: {m.group(1).strip()[:60]}",
            fn=lambda: _notion_append_to_page(
                os.environ.get("JARVIS_NOTION_DEFAULT_PAGE_ID", ""),
                m.group(1).strip(),
            ),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: YouTube search ──
    # "знайди на ютубі котики" / "найди на youtube cats"
    (
        re.compile(
            r"^\s*(?:знайди|пошукай|шукай|найди|поищи|ищи)\s+"
            r"(?:на\s+)?(?:youtube|ютубі|ютубе|ютуб)\s+"
            r"(.{2,80})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="youtube_search",
            description=f"Шукаю на YouTube: {m.group(1).strip()[:50]}",
            fn=lambda: _youtube_search(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(
            r"^\s*(?:відкрий|открой|открыть)\s+(?:на\s+)?(?:youtube|ютубі|ютубе|ютуб)\s+"
            r"(.{2,80})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="youtube_search",
            description=f"Шукаю на YouTube: {m.group(1).strip()[:50]}",
            fn=lambda: _youtube_search(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: open URL in specific browser ──
    # "відкрий google.com у Chrome" / "открой github.com в Safari"
    (
        re.compile(
            r"^\s*(?:відкрий|открой|откры(?:ть|вай(?:те)?)|(?:від|за)крий(?:те)?)\s+"
            r"((?:https?://)?[a-z0-9-]+\.[a-z]{2,}(?:/\S*)?)\s+"
            r"(?:у|в|in)\s+"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґЁё][A-Za-zА-Яа-яЇїІіЄєҐґЁё0-9\s-]{1,20})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="open_in_browser",
            description=f"Відкриваю {m.group(1).strip()} у {_resolve_app_name(m.group(2).strip())}",
            fn=lambda: _open_in_browser(m.group(1).strip(), m.group(2).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: previous track ──
    (
        re.compile(
            r"^\s*(?:попередн\w+\s+трек|назад|перемкни\s+назад|повернись)\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="previous_track",
            description="Попередній трек",
            fn=_previous_track,
            created_ts=time.time(),
        ),
    ),
    (
        re.compile(
            r"^\s*(?:предыдущ\w+\s+трек|обратно|перемени\s+назад)\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="previous_track",
            description="Предыдущий трек",
            fn=_previous_track,
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: hide / minimize app ──
    (
        re.compile(
            r"^\s*(?:сховай|сховати|приховай|сверни|сверни[те]?|спрячь|сховати)\s+"
            r"([A-Za-zА-Яа-яЇїІіЄєҐґЁё][A-Za-zА-Яа-яЇїІіЄєҐґЁё0-9\s-]{1,30})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="hide_app",
            description=f"Ховаю {_resolve_app_name(m.group(1).strip())}",
            fn=lambda: _hide_app(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: switch window ──
    (
        re.compile(
            r"^\s*(?:перемкни|переключи|switch)\s+(?:вікно|окно|window)\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="switch_window",
            description="Перемикаю вікно",
            fn=_switch_window,
            created_ts=time.time(),
        ),
    ),
    # ── R34-S31: show desktop ──
    (
        re.compile(
            r"^\s*(?:покажи|показати|покажу|show)\s+(?:робочий\s+стіл|desktop|рабочий\s+стол)\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="show_desktop",
            description="Показую робочий стіл",
            fn=_show_desktop,
            created_ts=time.time(),
        ),
    ),
    # ── R34-S47: Nexus-Brain search ──
    # "Подивись/знайди/перевір у мозку <запит>" / "поищи в мозге …"
    # / "загугли по brain …" — read-only, never destructive, so this
    # one is in _SAFE_ACTIONS_NO_CONFIRM at the fast-path site.
    (
        re.compile(
            r"^\s*(?:подивись|перевір|перевірь|знайди|пошукай|пошук|search|find|"
            r"посмотри|проверь|найди|поищи|поиск)\s+"
            r"(?:у\s+|в\s+|по\s+|in\s+)?(?:мозку|мозок|мозге|мозг|brain|вашему\s+мозгу|"
            r"моєму\s+мозку|нексус[- ]?брейн|nexus[- ]?brain)\s+"
            r"(.{2,200})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="nexus_brain_search",
            description=f"Шукаю у Brain: {m.group(1).strip()}",
            fn=lambda: _nexus_brain_search(m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S47: Nexus-Brain write — Nexus Studio knowledge ──
    # "Запам'ятай / запиши в мозок: <факт>" / "запомни в мозге …"
    # Targets 04-KNOWLEDGE/nexus-studio.md by default. Destructive
    # (writes to disk) so it goes through the confirmation gate.
    (
        re.compile(
            r"^\s*(?:запам['']ятай|запам['']ятай|запиши|запиши[те]?|"
            r"запомни|сохрани|remember|save)\s+"
            r"(?:у\s+|в\s+|to\s+)?(?:мозок|мозку|мозг|мозге|brain|нексус[- ]?брейн|"
            r"nexus[- ]?brain)\s*[:,]?\s*"
            r"(.{3,500})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="nexus_brain_write",
            description=f"Записую у Brain (Nexus Studio): {m.group(1).strip()[:60]}",
            fn=lambda: _nexus_brain_append("nexus-studio", m.group(1).strip()),
            created_ts=time.time(),
        ),
    ),
    # ── R34-S47: Nexus-Brain write — ideas inbox ──
    (
        re.compile(
            r"^\s*(?:додай|записати|запиши|запомни|save)\s+"
            r"(?:в\s+|до\s+|to\s+)?(?:ідеї|идею|ideas?|inbox)\s*[:,]?\s*"
            r"(.{3,500})\s*[!\.\?]?\s*$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="nexus_brain_write",
            description=f"Записую ідею: {m.group(1).strip()[:60]}",
            fn=lambda: _nexus_brain_append("ideas", m.group(1).strip()),
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
        # Audit round 11 fix M14: `.match() or .search()` defeated every
        # `^\s*` anchor in USER_COMMAND_PATTERNS — a polite "ну я вже
        # відкрив Safari вручну, дякую" matched the open-Safari pattern
        # via .search() and re-fired the action behind the user's back.
        # Anchors exist precisely so the command must be at the start;
        # honour them.
        m = pattern.match(text)
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

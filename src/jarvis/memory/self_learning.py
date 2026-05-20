"""Self-learning + language consistency layer (R34-S37).

Two responsibilities, both tightly coupled to the voice loop:

1. ``detect_user_language(text)`` and ``language_lock_block(lang)``
   produce a hard guardrail string that gets injected into the system
   prompt every turn. Empirically the qwen3 / llama3 / mistral
   families code-switch into RU↔UK↔EN unless told NOT to — even when
   the base prompt is unilingual. We re-state the rule per turn so it
   never gets pushed out of the attention window.

2. ``record_turn(user_text, bot_text, lang)`` extracts facts from the
   user message via regex heuristics (fast, deterministic) and writes
   them to the existing FTS5 fact store. We also schedule an
   async-LLM extraction pass for richer facts (e.g. preferences,
   intentions) — that pass runs AFTER the user has heard the reply,
   so it never blocks the speaking turn.

Both pieces are best-effort: any exception is swallowed and logged.
The voice loop must keep working even if the memory DB is locked or
the LLM is offline.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple


# Module-level state for the deferred LLM extractor (R34-S39).
# - _PENDING_TASKS: hard refs so PEP-3156 task GC doesn't reap them
#   mid-flight (would silently drop the extraction).
# - _PENDING_EXTRACT_TEXT/TASK: most-recent-user-message snapshot so
#   stale extractor calls can detect they've been superseded and exit
#   without burning an Ollama call.
_PENDING_TASKS: Set[asyncio.Task] = set()
_PENDING_EXTRACT_TEXT: str = ""
_PENDING_EXTRACT_TASK: Optional[asyncio.Task] = None


# ---------- language detection ----------------------------------------

# Reuse the tested detector in tts.py — it knows the difference between
# RU, UK, DE, EN and falls back sanely. Import locally inside the
# wrapper so the memory module doesn't drag the TTS stack on cold
# import paths (CLI, tests).
def detect_user_language(text: str) -> str:
    """Return ISO-639-1 lang code: 'uk' | 'ru' | 'de' | 'en'.

    Defaults to 'ru' on empty input — matches the Pipecat config
    ``active_language`` default. Caller can override if needed.
    """
    text = (text or "").strip()
    if not text:
        return "ru"
    try:
        from ..output.tts import _detect_language as _td
        return _td(text)
    except Exception:
        # Conservative fallback: any UA-only Cyrillic glyph wins UK.
        for ch in text:
            if ch in "іїєґІЇЄҐ":
                return "uk"
        for ch in text:
            if "А" <= ch <= "я" or ch in "ёЁ":
                return "ru"
        return "en"


_LANG_NAME = {
    "uk": "українською",
    "ru": "по-русски",
    "de": "auf Deutsch",
    "en": "in English",
}
_LANG_NAME_EN = {"uk": "Ukrainian", "ru": "Russian", "de": "German", "en": "English"}


def language_lock_block(lang: str) -> str:
    """Strong, language-aware instruction to lock the reply language.

    Re-emitted on every turn. The phrasing intentionally uses the
    target language itself — models are more compliant when the
    instruction is in the language they're being asked to produce.
    """
    lang = (lang or "ru").lower()
    if lang == "uk":
        return (
            "МОВНЕ ПРАВИЛО (СУВОРО): відповідай ВИКЛЮЧНО українською. "
            "Жодних російських, англійських чи німецьких слів. Якщо "
            "стикаєшся з іншомовними іменами або термінами — переклади "
            "чи транслітеруй українською. Перевір кожне слово відповіді "
            "перед тим, як її дати."
        )
    if lang == "de":
        return (
            "SPRACHREGEL (STRENG): Antworte AUSSCHLIESSLICH auf Deutsch. "
            "Keine russischen, ukrainischen oder englischen Wörter."
        )
    if lang == "en":
        return (
            "LANGUAGE RULE (STRICT): Reply ONLY in English. "
            "No Ukrainian, Russian, or German words. Transliterate "
            "any foreign names into English."
        )
    # Default — Russian.
    return (
        "ЯЗЫКОВОЕ ПРАВИЛО (СТРОГО): отвечай ИСКЛЮЧИТЕЛЬНО на русском. "
        "Никаких украинских, английских или немецких слов. Иностранные "
        "имена и термины — транслитерируй на русский. Проверь каждое "
        "слово ответа перед тем, как его произнести."
    )


# ---------- fact extraction (regex heuristics) ------------------------

@dataclass(frozen=True)
class FactCandidate:
    """One extracted fact in canonical store format."""
    key: str           # 'user.name', 'user.location', 'user.preference.coffee', ...
    value: str         # the displayable phrase ("живёт в Регбург-Локкум")
    confidence: float  # 0..1 — how sure the heuristic is

    def as_args(self) -> Tuple[str, str, str, float]:
        return (self.key, self.value, "voice_self_learn", self.confidence)


# Each entry: (compiled regex, key, value-template).
# Value-template uses {0} for the first capture, lowercased+stripped.
# Patterns target Ukrainian / Russian / English first-person statements
# typical for the IT-agency owner using the assistant.
_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # Name (uk / ru / en)
    (re.compile(r"\bмене звуть\s+([А-ЯҐЄІЇA-Z][\wʼ'\-]+)", re.IGNORECASE), "user.name",      "ім'я: {0}"),
    (re.compile(r"\bменя зовут\s+([А-ЯA-Z][\w\-]+)",        re.IGNORECASE), "user.name",      "имя: {0}"),
    (re.compile(r"\bmy name is\s+([A-Z][\w\-]+)",           re.IGNORECASE), "user.name",      "name: {0}"),
    (re.compile(r"\bя\s+([А-ЯҐЄІЇA-Z][\wʼ'\-]+),\s+(?:CEO|керівник|основатель|засновник)\b", re.IGNORECASE),
                                                                            "user.role",      "{0} — CEO"),

    # Location
    (re.compile(r"\bя живу\s+(?:в|у)\s+([А-ЯҐЄІЇA-Z][\wʼ'\- ]{1,40})", re.IGNORECASE), "user.location", "живе в {0}"),
    (re.compile(r"\bя живу\s+в\s+([А-ЯA-Z][\w\- ]{1,40})",             re.IGNORECASE), "user.location", "живёт в {0}"),
    (re.compile(r"\bi live in\s+([A-Z][\w\- ,]{1,40})",                re.IGNORECASE), "user.location", "lives in {0}"),

    # Company
    (re.compile(r"\b(?:моя компанія|моя компания|my company)\s+[—–-]\s+([А-ЯҐЄІЇA-Z][\w \-]{1,40})", re.IGNORECASE),
                                                                            "user.company",   "компанія: {0}"),
    (re.compile(r"\b(?:я працюю|я работаю|i work)\s+(?:в|у|at)\s+([А-ЯҐЄІЇA-Z][\w \-]{1,40})", re.IGNORECASE),
                                                                            "user.company",   "роботодавець: {0}"),

    # Preferences — coffee/tea/etc.
    (re.compile(r"\bя п(?:ʼ|')ю\s+([\w\s\-]{1,40}?)\s+кав[уи]\b",       re.IGNORECASE), "user.preference.coffee", "кава: {0}"),
    (re.compile(r"\bпью\s+([\w\s\-]{1,40}?)\s+кофе\b",                  re.IGNORECASE), "user.preference.coffee", "кофе: {0}"),
    (re.compile(r"\bi drink\s+([\w\s\-]{1,40}?)\s+coffee\b",            re.IGNORECASE), "user.preference.coffee", "coffee: {0}"),

    # Language preference for voice
    (re.compile(r"\b(?:говори|відповідай|відповідайте)\s+(?:зі мною\s+)?українською\b",  re.IGNORECASE),
                                                                            "user.preference.voice_lang", "voice language: uk"),
    (re.compile(r"\b(?:говори|отвечай)\s+(?:со мной\s+)?на\s+русском\b",   re.IGNORECASE),
                                                                            "user.preference.voice_lang", "voice language: ru"),
    (re.compile(r"\bspeak\s+(?:to me\s+)?in english\b",                    re.IGNORECASE),
                                                                            "user.preference.voice_lang", "voice language: en"),

    # Birthday / age (light touch — confidence 0.6 because of false positives)
    (re.compile(r"\bмені\s+(\d{1,2})\s+рок[іиу]в?", re.IGNORECASE), "user.age", "вік: {0}"),
    (re.compile(r"\bмне\s+(\d{1,2})\s+(?:год|лет)", re.IGNORECASE), "user.age", "возраст: {0}"),
]


def extract_facts_heuristic(text: str) -> List[FactCandidate]:
    """Pull facts from the user's last message via regex patterns.

    Returns ``[]`` on no-match. Each candidate has confidence ~0.85
    (regex hit is high-signal). The LLM extractor (see below) can
    overwrite with a fresher confidence if it disagrees.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: List[FactCandidate] = []
    for pat, key, tmpl in _PATTERNS:
        for m in pat.finditer(text):
            try:
                raw = (m.group(1) or "").strip().rstrip(".,!?;:")
            except IndexError:
                raw = ""
            if not raw:
                continue
            if len(raw) < 2 or len(raw) > 60:
                continue
            value = tmpl.format(raw)
            # Lower confidence for ambiguous patterns (age, etc.)
            conf = 0.6 if key == "user.age" else 0.85
            out.append(FactCandidate(key=key, value=value, confidence=conf))
    return _dedupe(out)


def _dedupe(cands: Iterable[FactCandidate]) -> List[FactCandidate]:
    seen: dict[Tuple[str, str], FactCandidate] = {}
    for c in cands:
        prev = seen.get((c.key, c.value))
        if prev is None or c.confidence > prev.confidence:
            seen[(c.key, c.value)] = c
    return list(seen.values())


# ---------- LLM-based fact extraction (async, post-turn) --------------

_FACT_EXTRACT_PROMPT = (
    "Extract durable facts about the SPEAKER from the message. Return "
    "JSON ARRAY ONLY, no prose. Schema: "
    '[{"key":"user.<category>","value":"<short phrase>","confidence":0.0-1.0}]. '
    "Categories: user.name, user.role, user.company, user.location, "
    "user.preference.<topic>, user.goal, user.skill, user.tool, "
    "user.relationship.<who>. Return [] if nothing durable is stated. "
    "DO NOT extract: feelings, ephemeral state, weather, current task. "
    "MAX 4 facts. Each value MUST be <= 60 chars."
)


async def extract_facts_llm(
    user_text: str,
    ollama_url: str,
    model: str,
    *,
    timeout_s: float = 8.0,
) -> List[FactCandidate]:
    """Call Ollama to extract richer facts. Best-effort.

    Cheap-and-cheerful: one Ollama call, JSON-mode disabled (Ollama's
    enforcement is unreliable), parse the first ``[...]`` block we
    find. On any error we return ``[]`` so the caller can fall back
    to the heuristic facts.
    """
    user_text = (user_text or "").strip()
    if not user_text or len(user_text) < 6:
        return []
    import json as _json

    # Lazy import — aiohttp is on the voice loop already.
    try:
        import aiohttp as _ah
    except Exception:
        return []
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _FACT_EXTRACT_PROMPT},
            {"role": "user",   "content": user_text[:600]},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": 220, "num_ctx": 1024, "temperature": 0.0},
    }
    try:
        timeout = _ah.ClientTimeout(total=timeout_s, connect=2)
        async with _ah.ClientSession(timeout=timeout) as session:
            async with session.post(
                ollama_url.rstrip("/") + "/api/chat",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                content = (data.get("message", {}) or {}).get("content", "") or ""
    except Exception:
        return []

    # Extract JSON array. Models sometimes wrap in ```json fences;
    # strip those and look for the first bracketed block.
    blob = content.strip()
    if "```" in blob:
        blob = re.sub(r"```(?:json)?", "", blob).strip("` \n")
    m = re.search(r"\[.*\]", blob, re.DOTALL)
    if not m:
        return []
    try:
        parsed = _json.loads(m.group(0))
    except Exception:
        return []
    out: List[FactCandidate] = []
    if isinstance(parsed, list):
        for item in parsed[:6]:
            if not isinstance(item, dict):
                continue
            k = str(item.get("key", "")).strip()
            v = str(item.get("value", "")).strip()
            try:
                c = float(item.get("confidence", 0.6))
            except Exception:
                c = 0.6
            if not k or not v or not k.startswith("user."):
                continue
            if len(v) > 80:
                v = v[:77] + "…"
            out.append(FactCandidate(key=k, value=v, confidence=max(0.0, min(1.0, c))))
    return _dedupe(out)


# ---------- public entry point ---------------------------------------

def record_turn(
    user_text: str,
    bot_text: str = "",
    *,
    lang: Optional[str] = None,
    ollama_url: Optional[str] = None,
    chat_model: Optional[str] = None,
    enable_llm_extract: bool = True,
) -> None:
    """Persist facts + conversation pair after a voice turn.

    Synchronous: stores heuristic facts immediately (cheap regex).
    If ``enable_llm_extract`` is True and an event loop is running,
    schedules a background coroutine that fires the LLM extractor
    and writes any additional facts it returns. The background task
    is fully fire-and-forget — exceptions are swallowed.

    Safe to call from sync or async contexts (auto-detects the
    running loop).
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return

    # 1. Heuristic facts — run inline, cheap.
    facts = extract_facts_heuristic(user_text)
    _store_facts(facts)

    # 2. Conversation row (uses existing conversation memory if available).
    try:
        from .conversation import get_conversation_store  # type: ignore[attr-defined]
        store = get_conversation_store()
        if hasattr(store, "record_turn"):
            store.record_turn(user_text=user_text, bot_text=bot_text, lang=lang)
    except Exception:
        # conversation.py API may differ across audit rounds — skip
        # quietly if signature doesn't match.
        pass

    # 3. LLM extractor — async, opt-in, only if we have a model + URL.
    # R34-S39 — DEFERRED execution to avoid VRAM/GPU contention with the
    # NEXT user turn. Without this, the extractor LLM call lands on the
    # same Ollama instance ~200ms after the reply finishes, exactly when
    # the user starts their follow-up — that next reply then waits for
    # the extractor to finish, doubling perceived latency.
    # Strategy: delay 30s + jitter; if a fresh record_turn happens
    # before the extractor fires, the new one supersedes (we keep the
    # most recent user message only).
    if enable_llm_extract and ollama_url and chat_model:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            global _PENDING_EXTRACT_TEXT, _PENDING_EXTRACT_TASK
            _PENDING_EXTRACT_TEXT = user_text
            # Cancel any in-flight extract for an earlier user message
            # — only the most recent turn deserves the LLM-fact pass.
            prev = _PENDING_EXTRACT_TASK
            if prev is not None and not prev.done():
                try:
                    prev.cancel()
                except Exception:
                    pass

            async def _runner():
                try:
                    # 30s delay so we don't race the next user turn.
                    await asyncio.sleep(30.0)
                    snapshot = _PENDING_EXTRACT_TEXT
                    if not snapshot or snapshot != user_text:
                        return  # superseded
                    extra = await extract_facts_llm(
                        snapshot, ollama_url=ollama_url, model=chat_model,
                    )
                    if extra:
                        _store_facts(extra)
                except asyncio.CancelledError:
                    return
                except Exception:
                    pass
                finally:
                    # Remove from the pending set so we don't accumulate
                    # references to completed tasks.
                    try:
                        _PENDING_TASKS.discard(asyncio.current_task())
                    except Exception:
                        pass

            task = loop.create_task(_runner())
            _PENDING_EXTRACT_TASK = task
            # Keep a hard reference so Python's GC can't reap the task
            # mid-flight (PEP 3156 — tasks must be referenced).
            _PENDING_TASKS.add(task)
            task.add_done_callback(_PENDING_TASKS.discard)


def _store_facts(facts: Iterable[FactCandidate]) -> int:
    """Push candidates to the FTS5 store. Returns count of writes."""
    count = 0
    try:
        from .facts import get_facts_store
        store = get_facts_store()
    except Exception:
        return 0
    for c in facts:
        try:
            store.add(
                key=c.key, value=c.value,
                source="voice_self_learn", confidence=c.confidence,
            )
            count += 1
        except Exception:
            continue
    return count


# ---------- conversation-summary digest (used by system prompt) ------

def recent_conversation_digest(limit: int = 5, max_chars: int = 400) -> str:
    """Return a compact digest of recent user↔bot exchanges.

    Used by the system-prompt builder to give the LLM short-term
    memory of the last few turns ("you JUST said X 30s ago"). Falls
    back to an empty string if the conversation store isn't
    available, so the voice loop is unaffected.
    """
    try:
        from .conversation import get_conversation_store  # type: ignore[attr-defined]
        store = get_conversation_store()
    except Exception:
        return ""
    if not hasattr(store, "recent"):
        return ""
    try:
        rows = store.recent(limit=limit)
    except Exception:
        return ""
    if not rows:
        return ""
    lines: List[str] = ["Recent turns:"]
    total = 0
    for row in rows:
        u = (getattr(row, "user_text", "") or "").strip()
        b = (getattr(row, "bot_text", "")  or "").strip()
        if not u and not b:
            continue
        snippet = f"- U:{u[:90]} | A:{b[:90]}"
        if total + len(snippet) > max_chars:
            break
        lines.append(snippet)
        total += len(snippet)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


__all__ = [
    "detect_user_language",
    "language_lock_block",
    "extract_facts_heuristic",
    "extract_facts_llm",
    "record_turn",
    "recent_conversation_digest",
    "FactCandidate",
]

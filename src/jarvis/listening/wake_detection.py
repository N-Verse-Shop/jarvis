"""Wake word and stop command detection logic."""

from typing import List, Optional
import difflib

from ..debug import debug_log


def is_wake_word_detected(text_lower: str, wake_word: str, aliases: List[str], fuzzy_ratio: float = 0.78) -> bool:
    """
    Check if text contains wake word using exact, fuzzy, and prefix matching.

    Args:
        text_lower: Lowercase text to check
        wake_word: Primary wake word
        aliases: List of wake word aliases
        fuzzy_ratio: Threshold for fuzzy matching (0.0-1.0)

    Returns:
        True if wake word detected
    """
    if not text_lower or not text_lower.strip():
        return False

    # Combine wake word and aliases
    all_aliases = set(aliases) | {wake_word}

    # Check exact match first
    if wake_word in text_lower:
        return True

    # Check aliases exact match
    for alias in aliases:
        if alias in text_lower:
            return True

    # Tokenize once for both fuzzy and prefix checks.
    try:
        heard_tokens = [t.strip(".,!?;:()[]{}\"'`).-_/") for t in text_lower.split() if t.strip()]
    except Exception:
        heard_tokens = []

    # Fuzzy matching for close variations.
    #
    # Tokens shorter than 4 characters are skipped here — they generate too
    # many false positives ("yes" ~ "yarves" hits ratio 0.67, "are" ~
    # "yarves" 0.67). Genuine wake mishearings ("jogs", "charly", "ярвіс")
    # are always 4+ chars in our observed log data, so we don't lose recall.
    #
    # LENGTH-DELTA GUARD (May 17): when alias and token differ in length by
    # ≥2 chars, the fuzzy ratio doesn't mean what we want — e.g.
    # "джервус"(7) ~ "жертву"(6) hits 0.77 but they're semantically
    # unrelated; "царвис"(6) ~ "цари"(4) hits 0.80; "ярви"(4) ~ "ярки"(4)
    # hits 0.75. Require |len(alias)-len(token)| ≤ 1 for fuzzy match.
    # Real wake mishearings ("джарвис"→"джагус") are usually within ±1
    # char of an alias. This eliminates the family-conversation false-fires
    # without dropping any real wake recall.
    try:
        for token in heard_tokens:
            if len(token) < 4:
                continue
            for alias in all_aliases:
                # Length-delta guard — see comment block above.
                # Audit round 6: bumped to ≤2 for LONG aliases (≥5 chars)
                # so legitimate declension variants like "джарвисом"(9)
                # against "джарвис"(7) (delta=2) match. Short aliases
                # stay at ≤1 to keep "царвис"(6)~"цари"(4) blocked.
                _delta = abs(len(alias) - len(token))
                _min_len = min(len(alias), len(token))
                if _min_len >= 5:
                    if _delta > 2:
                        continue
                else:
                    if _delta > 1:
                        continue
                ratio = difflib.SequenceMatcher(a=alias, b=token).ratio()
                if ratio >= fuzzy_ratio:
                    debug_log(f"wake word fuzzy match: '{alias}' ~ '{token}' (ratio: {ratio:.3f})", "wake")
                    print(f"🟢 wake fuzzy: '{token}' ~ '{alias}' ({ratio:.2f})", flush=True)
                    return True
    except Exception:
        pass

    # Prefix safety net — designed for cases where Whisper mangles "Jarvis"
    # into a low-confidence token that doesn't quite reach `fuzzy_ratio`
    # but is clearly a phonetic neighbour (it starts with the same lead
    # sound and has plausible length). Examples we've seen in real logs:
    #   "jogs", "jox", "yox", "charly", "джоргус", "жарюс", "жогс".
    #
    # We trigger if ANY heard token:
    #   • is 3-9 characters long (rules out long noise words like "burning")
    #   • starts with one of the wake-sound prefixes we expect after a UA
    #     or RU speaker says "Jarvis" — j / y / ch (Latin), дж / ж / ч / я
    #     (Cyrillic). One-or-two-character prefixes, no overlap with common
    #     filler words ("yes", "you", "ya", "че" — caught by length>=3 +
    #     trailing-consonant heuristic).
    #   • contains at least one consonant past position 1 (rules out
    #     "ya", "you", "yes" — which have V-C-V but no second consonant
    #     before position 3).
    # "я" was removed in audit round 6 — it matched "яскравий", "ярмарок",
    # "являється", "ярли", "я не…" (extremely common phrases in
    # UA/RU). Even with the 0.65 soft-ratio guard against canonical
    # wake words, "ярвіс"(5) ~ "джарвіс"(7) hit 0.67 — the false-fire
    # rate from "я"-prefixed words far exceeded the recall benefit of
    # catching one rare mishearing ("ярвіс"). Russian/Ukrainian wake
    # mishearings always start with дж/ж/ч, never plain "я".
    _WAKE_PREFIXES = ("j", "y", "ch", "дж", "ж", "ч")
    _FILLER = {
        "yes", "you", "your", "yeah", "ya", "yep", "yo", "yay",
        "ya", "ja",
        # UA (kept for code-switching)
        "так", "та", "ти", "ту", "те", "що", "як", "ще", "це",
        "че", "чи", "чо", "чу",
        # RU (primary after May 15 uk→ru lang migration)
        "да", "ты", "ту", "те", "что", "как", "еще", "это",
        "чё", "что-то", "чем", "чё-то", "чего",
        # Common RU filler tokens that start with "я"
        "я", "яс", "ясно", "ясно?",
    }
    for token in heard_tokens:
        if token in _FILLER:
            continue
        if not (3 <= len(token) <= 9):
            continue
        if not any(token.startswith(p) for p in _WAKE_PREFIXES):
            continue
        # Looks like a J/Y/Ч-fronted ~jarvis-shaped token. Confirm via a
        # softer (0.45) fuzzy match against the canonical wake word — this
        # filters out completely unrelated j-words like "joke", "yoga".
        # Compare against BOTH RU "джарвис" (no diacritics, primary post
        # May 16 migration) and UA "джарвіс" (legacy) so we don't bias
        # against either spelling.
        soft = max(
            difflib.SequenceMatcher(a=wake_word, b=token).ratio(),
            difflib.SequenceMatcher(a="джарвис", b=token).ratio(),
            difflib.SequenceMatcher(a="джарвіс", b=token).ratio(),
        )
        # Raised from 0.45 → 0.65 — 0.45 was triggering on everyday UA words
        # like "жарить" (cooking), "жалить", "чарівний". 0.65 still catches
        # mishearings ("джавіс" 0.77, "жарюс" 0.71, "джоргус" 0.66) but cuts
        # ambient false-positives.
        if soft >= 0.65:
            debug_log(f"wake word prefix match: '{token}' (soft ratio: {soft:.3f})", "wake")
            print(f"🟡 wake prefix: '{token}' (soft {soft:.2f})", flush=True)
            return True

    return False


def extract_query_after_wake(text_lower: str, wake_word: str, aliases: List[str]) -> str:
    """
    Extract the query portion after removing wake word.
    
    Args:
        text_lower: Lowercase text containing wake word
        wake_word: Primary wake word
        aliases: List of wake word aliases
    
    Returns:
        Query text with wake word removed
    """
    if not text_lower:
        return ""

    import re
    all_aliases = set(aliases) | {wake_word}
    fragment = text_lower

    # Remove all aliases from the text using word-boundary regex.
    # Plain str.replace was too greedy — alias "ja" would chew into
    # "japan" → " pan", "ya" → "" + "rly" inside "yarly", etc.
    # Sort longest-first so "джарвіс" matches before "дж" alias.
    sorted_aliases = sorted((a for a in all_aliases if a), key=len, reverse=True)
    for alias in sorted_aliases:
        # \b doesn't work reliably across the Latin/Cyrillic boundary,
        # so we use look-around against alphanumerics in both alphabets.
        pattern = r"(?:(?<=^)|(?<=[^\wа-щьюяїієґА-ЩЬЮЯЇІЄҐ]))" + re.escape(alias) + r"(?:(?=$)|(?=[^\wа-щьюяїієґА-ЩЬЮЯЇІЄҐ]))"
        try:
            fragment = re.sub(pattern, " ", fragment, flags=re.IGNORECASE)
        except re.error:
            # Fallback to plain replace if regex compile fails on weird input.
            fragment = fragment.replace(alias, " ")

    # Clean up punctuation that might be left after wake word removal,
    # including em-dash / en-dash often produced by Whisper on long pauses.
    fragment = re.sub(r"\s+", " ", fragment)
    fragment = fragment.strip().lstrip(",.!?;:–—-")
    fragment = fragment.strip()

    return fragment if fragment else ""


def is_stop_command(text_lower: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """
    Check if text contains a stop command.

    Audit round 18 fix: previously ``if cmd in text_lower`` matched
    ANYWHERE in the utterance, so "Jarvis stop the kitchen timer" or
    "Can you stop talking?" were silently classified as stop commands
    and dropped before the wake handler could route them. The new
    contract: the stop command must be the WHOLE utterance (modulo
    punctuation / wake-word stripping) OR the text must be short
    enough (≤2 words) for a fuzzy match. A multi-word transcript with
    a "stop" word embedded is NOT a stop command — it's a query about
    stopping something.

    Args:
        text_lower: Lowercase text to check
        stop_commands: List of stop command phrases
        fuzzy_ratio: Threshold for fuzzy matching short inputs

    Returns:
        True if stop command detected
    """
    if not text_lower or not text_lower.strip():
        return False

    # Normalise: strip leading/trailing punctuation + whitespace so
    # "stop." / "stop!" / " stop " all reach the standalone check.
    stripped = text_lower.strip().strip(",.!?;:-—–")
    stripped = " ".join(stripped.split())  # collapse internal whitespace

    detected_commands: list[str] = []
    # Standalone-match: the whole utterance equals a stop command. This
    # is the dominant path; short, deliberate "stop" / "тихо" / etc.
    for cmd in stop_commands:
        cmd_norm = cmd.strip().lower()
        if not cmd_norm:
            continue
        if stripped == cmd_norm:
            detected_commands.append(cmd_norm)

    # Fuzzy match (token-level) for short inputs only. The previous
    # "≤2 words" gate is preserved, but we now also require the FUZZY
    # match to be against a STANDALONE token — not a substring.
    if not detected_commands and len(stripped.split()) <= 2:
        try:
            for word in stripped.split():
                for cmd in stop_commands:
                    cmd_norm = cmd.strip().lower()
                    if not cmd_norm:
                        continue
                    ratio = difflib.SequenceMatcher(a=cmd_norm, b=word).ratio()
                    if ratio >= fuzzy_ratio:
                        detected_commands.append(f"{cmd_norm}~{word}")
        except Exception:
            pass

    if detected_commands:
        debug_log(f"stop command detected: {detected_commands[0]} in '{text_lower}'", "voice")
        return True

    return False

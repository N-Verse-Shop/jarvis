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
    try:
        for token in heard_tokens:
            if len(token) < 4:
                continue
            for alias in all_aliases:
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
    _WAKE_PREFIXES = ("j", "y", "ch", "дж", "ж", "ч", "я")
    _FILLER = {
        "yes", "you", "your", "yeah", "ya", "yep", "yo", "yay",
        "ya", "ja",
        "так", "та", "ти", "ту", "те", "що", "як", "ще", "це",
        "че", "чи", "чо", "чу",
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
        soft = max(
            difflib.SequenceMatcher(a=wake_word, b=token).ratio(),
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
    
    all_aliases = set(aliases) | {wake_word}
    fragment = text_lower
    
    # Remove all aliases from the text
    for alias in all_aliases:
        fragment = fragment.replace(alias, " ")
    
    # Clean up punctuation that might be left after wake word removal
    fragment = fragment.strip().lstrip(",.!?;:")
    fragment = fragment.strip()
    
    return fragment if fragment else ""


def is_stop_command(text_lower: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """
    Check if text contains a stop command.
    
    Args:
        text_lower: Lowercase text to check
        stop_commands: List of stop command phrases
        fuzzy_ratio: Threshold for fuzzy matching short inputs
    
    Returns:
        True if stop command detected
    """
    if not text_lower or not text_lower.strip():
        return False
    
    # Check for exact matches
    detected_commands = []
    for cmd in stop_commands:
        if cmd in text_lower:
            detected_commands.append(cmd)
    
    # Check fuzzy matches for short inputs (2 words or less)
    if len(text_lower.split()) <= 2:
        try:
            for word in text_lower.split():
                for cmd in stop_commands:
                    ratio = difflib.SequenceMatcher(a=cmd, b=word).ratio()
                    if ratio >= fuzzy_ratio:
                        detected_commands.append(f"{cmd}~{word}")
        except Exception:
            pass
    
    if detected_commands:
        debug_log(f"stop command detected: {detected_commands[0]} in '{text_lower}'", "voice")
        return True
    
    return False

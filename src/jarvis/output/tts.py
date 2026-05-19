from __future__ import annotations
import platform
import subprocess
import threading
import queue
import shutil
import signal
import tempfile
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

from ..debug import debug_log


# ============================================================================
# Piper TTS Model Configuration
# ============================================================================
# Default voice model for automatic download
# en_GB-alan-medium: Good quality, ~60MB, British English male
PIPER_DEFAULT_VOICE = "en_GB-alan-medium"
PIPER_VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


def _get_piper_models_dir() -> Path:
    """Get the directory for storing Piper voice models."""
    base = Path.home() / ".local" / "share" / "jarvis" / "models" / "piper"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_default_piper_model_path() -> str:
    """Get the path to the default Piper voice model."""
    return str(_get_piper_models_dir() / f"{PIPER_DEFAULT_VOICE}.onnx")


def _download_piper_voice(voice_name: str, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Download a Piper voice model from HuggingFace.

    Args:
        voice_name: Voice name like "en_US-lessac-medium"
        progress_callback: Optional callback for progress messages

    Returns:
        Path to the downloaded model, or None if download failed
    """
    import requests

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        debug_log(msg, "tts")

    # Parse voice name to construct URL
    # Format: {lang}_{region}-{name}-{quality}
    # Example: en_US-lessac-medium -> en/en_US/lessac/medium/en_US-lessac-medium.onnx
    parts = voice_name.split("-")
    if len(parts) < 3:
        log(f"Invalid voice name format: {voice_name}")
        return None

    lang_region = parts[0]  # e.g., "en_US"
    name = parts[1]         # e.g., "lessac"
    quality = parts[2]      # e.g., "medium"

    lang = lang_region.split("_")[0]  # e.g., "en"

    # Construct URLs
    base_path = f"{lang}/{lang_region}/{name}/{quality}/{voice_name}"
    onnx_url = f"{PIPER_VOICE_BASE_URL}/{base_path}.onnx"
    json_url = f"{PIPER_VOICE_BASE_URL}/{base_path}.onnx.json"

    # Target paths
    models_dir = _get_piper_models_dir()
    onnx_path = models_dir / f"{voice_name}.onnx"
    json_path = models_dir / f"{voice_name}.onnx.json"

    # Download with progress
    try:
        for url, target_path, desc in [
            (onnx_url, onnx_path, "model"),
            (json_url, json_path, "config"),
        ]:
            if target_path.exists():
                log(f"  {desc} already exists: {target_path.name}")
                continue

            log(f"  Downloading {desc}...")

            # Stream download with retry on rate limiting (HTTP 429)
            max_retries = 4
            response = None
            for attempt in range(max_retries + 1):
                response = requests.get(url, stream=True, timeout=60)
                try:
                    response.raise_for_status()
                    break  # Success
                except requests.exceptions.HTTPError as http_err:
                    response.close()
                    status = getattr(http_err.response, "status_code", None)
                    if status == 429 and attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        log(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                        continue
                    raise  # Non-429 or retries exhausted

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Write to temp file first, then rename (atomic)
            temp_path = target_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        pct = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                            log(f"  Downloading {desc}... {pct:.0f}%")

            # Rename temp to final
            temp_path.rename(target_path)
            log(f"  Downloaded {desc}: {target_path.name}")

        return str(onnx_path)

    except requests.RequestException as e:
        log(f"  Download failed: {e}")
        # Clean up partial downloads
        for p in [onnx_path, json_path]:
            tmp = p.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
        return None
    except Exception as e:
        log(f"  Download error: {e}")
        return None


# Default speaking rates for TTS estimation
DEFAULT_WPM = 200  # Default rate used in config (words per minute)
AUDIO_BUFFER_DELAY_SEC = 0.5  # Extra delay for audio buffer latency


def _estimate_tts_duration(text: str, wpm: int) -> float:
    """
    Estimate how long TTS audio will take to play.

    Args:
        text: The text being spoken
        wpm: Words per minute rate

    Returns:
        Estimated duration in seconds
    """
    # Count words (simple split on whitespace)
    words = len(text.split())

    # Calculate duration based on WPM
    if wpm <= 0:
        wpm = DEFAULT_WPM

    duration_sec = (words / wpm) * 60.0

    # Add buffer for audio latency
    return duration_sec + AUDIO_BUFFER_DELAY_SEC


def _extract_domain_description(url: str) -> tuple[str, bool]:
    """
    Extract a readable domain description from a URL.

    Returns:
        Tuple of (domain_description, is_homepage)
        - domain_description: e.g., "google.com"
        - is_homepage: True if URL points to homepage (no meaningful path)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]

        # Remove common prefixes
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check if it's a homepage (no path or just /)
        path = parsed.path.rstrip('/')
        is_homepage = not path or path == ''

        return domain, is_homepage
    except Exception:
        return url, True


_NUMBERED_MARKER_RE = re.compile(r"^\s*(\d+)[.)]\s+")


def _strip_markdown_for_speech(text: str) -> str:
    """Strip markdown formatting so TTS doesn't read syntax characters aloud.

    Small models often produce markdown (``**bold**``, bullet lists, headings)
    even when told to be conversational. Piper and similar engines read the
    syntax characters literally ("asterisk asterisk bold asterisk asterisk").
    This function removes the markup while preserving the words inside it.

    Handled:
    - Fenced code blocks ``` ```lang\\ncode\\n``` ``` → inner text only
    - Inline code ``` `x` ``` → ``x``
    - Bold ``**x**`` / ``__x__`` → ``x``
    - Italic ``*x*`` / ``_x_`` → ``x``
    - Strikethrough ``~~x~~`` → ``x``
    - Word-internal underscores (e.g. ``my_function``) are preserved so
      identifiers aren't mangled into concatenated words.
    - HTML tags ``<b>x</b>`` → ``x``
    - Leading heading markers ``# ``, ``## `` … at line start → removed
    - Setext heading underlines (``===`` / ``---`` beneath a title line) → removed
    - Leading blockquote markers ``> `` at line start → removed
    - Leading bullet markers ``- ``, ``* ``, ``+ `` at line start → removed
    - Leading numbered-list markers ``1. ``, ``2) ``: stripped only when the
      line is part of a real list — detected as ≥2 adjacent lines whose
      numbers are each ≤ 99. Prevents eating prose like "2024. The year...".
    """
    if not text:
        return text

    # Fenced code blocks: keep inner content, drop fences and language tag.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n?([\s\S]*?)```", r"\1", text)

    # Inline code: keep inner content.
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Bold / strikethrough (before italic so the double-char form matches first).
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Italic with asterisk: single * not flanked by another *.
    text = re.sub(r"(?<!\*)\*([^*\s][^*]*?)\*(?!\*)", r"\1", text)
    # Italic with underscore: require word boundaries so we don't eat
    # underscores inside identifiers like "some_variable_name".
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", text)

    # HTML tags: drop tags, keep inner text. Audit round 11 fix H5: the
    # previous `<[^>]+>` was too broad — comparison expressions in narrated
    # code ("set y < 5 if z > 0") matched as `<` to next `>` and the whole
    # span vanished. Require a letter or `/` immediately after `<` so real
    # tag syntax matches but math/code doesn't.
    text = re.sub(r"<(/?[A-Za-z][^>]*)>", "", text)

    # True list detection: a numbered line is a list item only if it's part
    # of a contiguous group of ≥2 such lines whose numbers are each ≤ 99.
    # This preserves prose like "2024. The year..." and "2023.\n2024." pairs
    # that are clearly years, not list markers.
    lines = text.split("\n")
    numbers = [
        int(m.group(1)) if (m := _NUMBERED_MARKER_RE.match(line)) else None
        for line in lines
    ]
    strip_numbered = [False] * len(lines)
    run_start: Optional[int] = None
    for i in range(len(lines) + 1):
        in_run = i < len(lines) and numbers[i] is not None and numbers[i] <= 99
        if in_run and run_start is None:
            run_start = i
        elif not in_run and run_start is not None:
            if i - run_start >= 2:
                for k in range(run_start, i):
                    strip_numbered[k] = True
            run_start = None

    cleaned: list[str] = []
    for i, line in enumerate(lines):
        # Setext heading underline: a line of only = or - (≥3 chars) directly
        # beneath a non-empty title line. Drop the underline; keep the title.
        if (
            i > 0
            and lines[i - 1].strip()
            and re.fullmatch(r"\s*(=+|-+)\s*", line)
            and len(line.strip()) >= 3
        ):
            continue
        stripped = re.sub(r"^\s*#{1,6}\s+", "", line)        # headings
        stripped = re.sub(r"^\s*>\s?", "", stripped)         # blockquotes
        stripped = re.sub(r"^\s*[-*+]\s+", "", stripped)     # bullets
        if strip_numbered[i]:
            stripped = _NUMBERED_MARKER_RE.sub("", stripped)
        cleaned.append(stripped)
    return "\n".join(cleaned)


# ────────────────────────────────────────────────────────────────────────
# Ukrainian Piper phoneme sanitizer
# ────────────────────────────────────────────────────────────────────────
# The uk_UA-ukrainian_tts-medium model's phoneme_id_map covers exactly:
#   а б в г д е ж з и й к л м н о п р с т у ф х ц ч ш щ ь ю я є і ї ґ
#   + space, punctuation: ! $ ' , - . : ; ? ^ _ — (accents)
# Anything outside is silently skipped — capital letters, Russian-only
# letters (ё ы э ъ), Latin alphabet, digits, emoji, etc. — and the result
# sounds like gurgling because half the syllables are missing.
#
# We pre-transform text into a representation the model CAN pronounce:
#   1. lowercase everything (caps not in map)
#   2. transliterate Russian-only letters to UA equivalents
#   3. transliterate Latin letters phonetically (a → а, e → е, …)
#   4. expand digits 0-9 into words (бо цифри не озвучуються)
#   5. drop anything else that's not in the model alphabet
#
# This is targeted at the Ukrainian Piper model specifically; for other
# Piper voices (en_GB, de_DE) the function is a no-op since the input
# already matches their alphabet.

_PIPER_UK_ALPHABET = set("абвгдежзийклмнопрстуфхцчшщьюяєіїґ ")
_PIPER_UK_PUNCT = set("!$',-.:;?^_—")

# Russian-only Cyrillic → closest Ukrainian letter for the model.
_RU_TO_UK = {
    "ё": "йо", "ы": "и", "э": "е", "ъ": "",
}

# Latin → Ukrainian phonetic transliteration (mainly for tech terms like
# CEO/API/GmbH appearing in mid-text). Two-letter combos handled first.
_LATIN_DIGRAPHS = {
    "ch": "ч", "sh": "ш", "th": "т", "ph": "ф", "kh": "х",
    "zh": "ж", "ts": "ц", "ya": "я", "yu": "ю", "ye": "є", "yo": "йо",
}
_LATIN_TO_UK = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф",
    "g": "ґ", "h": "г", "i": "і", "j": "й", "k": "к", "l": "л",
    "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "и", "z": "з",
}

_DIGITS_UK = {
    "0": "нуль", "1": "один", "2": "два", "3": "три", "4": "чотири",
    "5": "пять", "6": "шість", "7": "сім", "8": "вісім", "9": "девять",
}

# Hours 0–23 spoken naturally in Ukrainian (ordinal feminine — "котра година?").
_UK_HOUR_WORDS = [
    "нульова", "перша", "друга", "третя", "четверта", "пята",
    "шоста", "сьома", "восьма", "девята", "десята", "одинадцята",
    "дванадцята", "тринадцята", "чотирнадцята", "пятнадцята",
    "шістнадцята", "сімнадцята", "вісімнадцята", "девятнадцята",
    "двадцята", "двадцять перша", "двадцять друга", "двадцять третя",
]
# Minutes 0–59 — cardinal forms; reuse where possible.
_UK_TENS = ["", "", "двадцять", "тридцять", "сорок", "пятдесят"]
_UK_ONES = ["", "одна", "дві", "три", "чотири", "пять",
            "шість", "сім", "вісім", "девять",
            "десять", "одинадцять", "дванадцять", "тринадцять",
            "чотирнадцять", "пятнадцять", "шістнадцять",
            "сімнадцять", "вісімнадцять", "девятнадцять"]


def _uk_minutes_to_words(mm: int) -> str:
    if mm < 20:
        return _UK_ONES[mm] or "нуль"
    tens = _UK_TENS[mm // 10]
    ones = _UK_ONES[mm % 10]
    return (tens + " " + ones).strip()


def _uk_int_to_words(n: int) -> str:
    """Convert 0–9999 to natural Ukrainian words.
    Handles dates (14 → 'чотирнадцять'), years (2026 → 'двi тисячi двадцять шiсть'),
    and ages."""
    if n < 0 or n > 9999:
        return str(n)  # leave out-of-range as digits
    if n == 0:
        return "нуль"
    if n < 20:
        return _UK_ONES[n] or "нуль"
    if n < 100:
        tens = _UK_TENS[n // 10]
        ones = _UK_ONES[n % 10]
        return (tens + " " + ones).strip()
    if n < 1000:
        # Hundreds
        hundreds_map = ["", "сто", "двісті", "триста", "чотириста",
                        "пятсот", "шістсот", "сімсот", "вісімсот", "девятсот"]
        h = hundreds_map[n // 100]
        rest = n % 100
        return (h + " " + _uk_int_to_words(rest)).strip() if rest else h
    # 1000–9999
    thousands_word_map = ["", "одна тисяча", "дві тисячі", "три тисячі",
                          "чотири тисячі", "пять тисяч", "шість тисяч",
                          "сім тисяч", "вісім тисяч", "девять тисяч"]
    t_div = n // 1000
    rest = n % 1000
    out = thousands_word_map[t_div]
    if rest:
        out += " " + _uk_int_to_words(rest)
    return out


def _expand_time_hhmm_uk(match) -> str:
    """Convert `HH:MM` patterns to Ukrainian words.
    07:00  → 'сьома година рівно'
    14:30  → 'чотирнадцята тридцять'
    06:53  → 'шоста пятдесят три'"""
    try:
        hh = int(match.group(1))
        mm = int(match.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return match.group(0)
    except (ValueError, IndexError):
        return match.group(0)
    hour_word = _UK_HOUR_WORDS[hh]
    if mm == 0:
        return f"{hour_word} рівно"
    return f"{hour_word} {_uk_minutes_to_words(mm)}"


def _sanitize_for_piper_uk(text: str) -> str:
    """Make `text` fully pronounceable by the uk_UA Piper model.

    No-op if there's no Cyrillic in the input (assumes another voice handles
    that text)."""
    if not any('Ѐ' <= ch <= 'ӿ' for ch in text):
        return text

    s = text.lower()

    # Expand HH:MM time patterns to natural Ukrainian words BEFORE digit
    # tokenisation strips the colon. "06:53" → "шоста пятдесят три".
    import re as _re
    s = _re.sub(r'\b(\d{1,2}):(\d{2})\b', _expand_time_hhmm_uk, s)

    # Expand standalone integers 0-9999 to natural Ukrainian. Avoids
    # "14 травня" → "один чотири травня" (digit-by-digit). Matches whole-word
    # numbers only; HH:MM already consumed above.
    def _int_repl(m):
        try:
            return _uk_int_to_words(int(m.group(0)))
        except (ValueError, OverflowError):
            return m.group(0)
    s = _re.sub(r'\b\d{1,4}\b', _int_repl, s)

    # Russian-only letters → UA equivalents.
    out = []
    for ch in s:
        out.append(_RU_TO_UK.get(ch, ch))
    s = "".join(out)

    # Latin digraphs before single letters, longest first.
    for k, v in _LATIN_DIGRAPHS.items():
        s = s.replace(k, v)

    # Single-letter Latin → UA.
    s = "".join(_LATIN_TO_UK.get(ch, ch) for ch in s)

    # Digits → words (with surrounding spaces).
    s = "".join(f" {_DIGITS_UK[ch]} " if ch in _DIGITS_UK else ch for ch in s)

    # Final filter: keep only what the model alphabet supports. Collapse
    # consecutive spaces and trim.
    allowed = _PIPER_UK_ALPHABET | _PIPER_UK_PUNCT
    s = "".join(ch if ch in allowed else " " for ch in s)
    s = " ".join(s.split())
    return s


def _preprocess_for_speech(text: str) -> str:
    """
    Preprocess text for TTS by converting links to readable descriptions and
    stripping markdown formatting.

    Handles:
    - Markdown links: [text](url) → "Link to domain.com with the text 'text'" or
      "Link to a page under domain.com with the text 'text'"
    - Raw URLs: https://domain.com → "domain.com homepage" or
      https://domain.com/path → "a page under domain.com"
    - Markdown formatting (bold, italic, code, headings, lists) → stripped so
      TTS engines don't read syntax characters (``**``, ``#``, ``-``) aloud.
    """
    # Pattern for markdown links: [text](url)
    markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    def replace_markdown_link(match: re.Match) -> str:
        link_text = match.group(1)
        url = match.group(2)
        domain, is_homepage = _extract_domain_description(url)

        if is_homepage:
            return f"Link to {domain} homepage with the text '{link_text}'"
        else:
            return f"Link to a page under {domain} with the text '{link_text}'"

    # Replace markdown links first
    result = re.sub(markdown_link_pattern, replace_markdown_link, text)

    # Pattern for raw URLs (not already processed as markdown)
    # Matches http://, https://, and www. prefixed URLs
    raw_url_pattern = r'(?<!\()(https?://[^\s<>\[\]()]+|www\.[^\s<>\[\]()]+)(?!\))'

    def replace_raw_url(match: re.Match) -> str:
        url = match.group(1)
        # Ensure URL has protocol for parsing
        if url.startswith('www.'):
            url = 'https://' + url
        domain, is_homepage = _extract_domain_description(url)

        if is_homepage:
            return f"{domain} homepage"
        else:
            return f"a page under {domain}"

    # Replace raw URLs
    result = re.sub(raw_url_pattern, replace_raw_url, result)

    # Strip any remaining markdown so TTS doesn't read syntax aloud.
    result = _strip_markdown_for_speech(result)

    return result


class ChatterboxTTS:
    """Experimental TTS implementation using Resemble AI's Chatterbox model."""

    def __init__(self, enabled: bool = True, voice: Optional[str] = None, rate: Optional[int] = None,
                 device: str = "cuda", audio_prompt_path: Optional[str] = None,
                 exaggeration: float = 0.5, cfg_weight: float = 0.5) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Chatterbox, kept for interface compatibility
        self.rate = rate    # Not directly supported in Chatterbox, kept for interface compatibility
        self.device = device
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

        # Threading and queue setup (same as TextToSpeech)
        self._q: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._should_interrupt = threading.Event()

        # Audit round 21 (F02) — interrupt-epoch (see PiperTTS / SystemTTS
        # docstrings). Same race: ``interrupt()`` sets the bare flag, then
        # a fresh ``speak()`` races between the set and the worker's
        # ``clear``. Bumping an epoch on every interrupt and snapshotting
        # in ``speak`` makes stale sentences detectable in
        # ``_speak_once``.
        self._interrupt_epoch: int = 0
        self._epoch_lock = threading.Lock()

        # Chatterbox model (eagerly loaded during initialization)
        self._model = None
        self._model_error = None
        # Lazy initialization flags
        self._initialized = False
        self._init_lock = threading.Lock()

    def _initialize_with_logging(self) -> None:
        """Initialize Chatterbox with proper logging."""
        import sys

        print("🔧 [TTS] Initializing Chatterbox neural voice synthesis...", file=sys.stderr)

        try:
            print("📦 [TTS] Loading Chatterbox dependencies...", file=sys.stderr)

            # Import dependencies
            import torch
            import torchaudio as ta
            from chatterbox.tts import ChatterboxTTS as ChatterboxModel

            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                print("⚠️  [TTS] CUDA requested but not available, falling back to CPU", file=sys.stderr)
                actual_device = "cpu"
            else:
                actual_device = self.device

            print(f"🚀 [TTS] Loading Chatterbox model on {actual_device.upper()}...", file=sys.stderr)

            # Load model with proper device specification
            self._model = ChatterboxModel.from_pretrained(device=actual_device)

            print("✅ [TTS] Chatterbox neural voice synthesis ready!", file=sys.stderr)

        except ImportError as e:
            self._model_error = f"Chatterbox dependencies not available: {e}"
            print(f"❌ [TTS] Missing dependencies: {self._model_error}", file=sys.stderr)
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")
        except Exception as e:
            self._model_error = f"Failed to load Chatterbox model: {e}"
            print(f"❌ [TTS] Model loading failed: {self._model_error}", file=sys.stderr)
            warnings.warn(f"ChatterboxTTS initialization failed: {self._model_error}")

    def _ensure_initialized(self) -> None:
        """Initialize heavy dependencies only once, when actually needed."""
        if self._initialized or not self.enabled:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._initialize_with_logging()
            self._initialized = True

    def _ensure_model(self) -> bool:
        """Check if Chatterbox model is loaded. Returns True if successful."""
        # Ensure lazy initialization happens before checking model
        self._ensure_initialized()
        if self._model is not None:
            return True
        if self._model_error is not None:
            return False
        return False

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Initialize on first actual start
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        # Ensure any active speech is interrupted immediately
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None) -> None:
        if not self.enabled or not text.strip():
            return
        # Lazy start the worker thread and lazy init on first speak
        if self._thread is None:
            self.start()
        self._completion_callback = completion_callback
        self._duration_callback = duration_callback
        # Audit round 17 fix: hard cap text length BEFORE preprocessing —
        # see SystemTTS.speak for full rationale (ChatterboxTTS shares
        # the same risk: pygame mixer plays the entire synthesised
        # buffer with no interruption window).
        _TTS_MAX_CHARS = 4000
        if len(text) > _TTS_MAX_CHARS:
            ellipsis = " … (truncated)"
            text = text[: _TTS_MAX_CHARS - len(ellipsis)] + ellipsis
        # Preprocess text for speech (convert links to readable descriptions)
        processed_text = _preprocess_for_speech(text)
        # Audit round 21 (F02) — snapshot epoch into the queue tuple.
        with self._epoch_lock:
            current_epoch = self._interrupt_epoch
        try:
            self._q.put_nowait((current_epoch, processed_text))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately AND drain queued sentences.

        Audit round 7 fix C4: previously only set `_should_interrupt`,
        leaving sentences 2,3,4... from a streaming reply still queued.
        The current sentence would stop mid-word and the next queued
        sentence would immediately pull off `_q` and play — "interrupt"
        became "pause and continue". Now we drain the queue so the
        interrupt is final.

        Audit round 21 (F02): also bump ``_interrupt_epoch`` so a
        sentence racing through ``put_nowait`` past the drain is
        recognised as stale and dropped by ``_speak_once``.
        """
        self._should_interrupt.set()
        with self._epoch_lock:
            self._interrupt_epoch += 1
        try:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not item:
                continue
            # Round 21 — items are (epoch, text) tuples; tolerate
            # plain strings for legacy callers / tests.
            if isinstance(item, tuple) and len(item) == 2:
                enq_epoch, text = item
            else:
                enq_epoch, text = 0, item
            if not text:
                continue
            try:
                self._speak_once(text, enq_epoch=enq_epoch)
            except Exception:
                continue

    def _speak_once(self, text: str, enq_epoch: int = 0) -> None:
        # Audit round 21 (F02): drop stale-epoch sentences.
        with self._epoch_lock:
            current_epoch = self._interrupt_epoch
        if enq_epoch < current_epoch:
            debug_log(
                f"Chatterbox TTS: dropping stale sentence (enq_epoch={enq_epoch} < {current_epoch})",
                "tts",
            )
            return
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        interrupted = False

        # Signal speaking state to face widget
        self._notify_speaking_state(True)

        try:
            # Check if model is available
            if not self._ensure_model():
                # Fall back to system TTS if Chatterbox fails
                warnings.warn("Chatterbox TTS not available, skipping speech synthesis")
                return

            # Generate audio using Chatterbox
            import tempfile
            import pygame
            import os

            # Generate speech
            wav = self._model.generate(
                text,
                audio_prompt_path=self.audio_prompt_path,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight
            )

            # Calculate exact duration from audio samples
            exact_duration = wav.shape[-1] / self._model.sr
            debug_log(f"Chatterbox TTS synthesis complete: {exact_duration:.2f}s", "tts")

            # Notify listener of exact duration for precise echo detection
            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Chatterbox TTS duration callback error: {e}", "tts")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Save audio
                import torchaudio as ta
                ta.save(tmp_path, wav, self._model.sr)

                # Play audio using pygame (cross-platform)
                pygame.mixer.init(frequency=self._model.sr, size=-16, channels=1, buffer=1024)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

                # Wait for playback to complete or interruption
                while pygame.mixer.music.get_busy():
                    if self._should_interrupt.is_set():
                        pygame.mixer.music.stop()
                        interrupted = True
                        break
                    pygame.time.wait(100)  # Check every 100ms

            finally:
                # Cleanup
                pygame.mixer.quit()
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            warnings.warn(f"Chatterbox TTS error: {e}")
        finally:
            self._is_speaking.clear()
            
            # Signal speaking stopped to face widget
            self._notify_speaking_state(False)
            
            # Call completion callback if set and not interrupted.
            # Audit round 8 regression fix: always clear the callback
            # ref at the end of `_speak_once` regardless of interrupt
            # outcome. Previously `_completion_callback = None` lived
            # INSIDE the `if not interrupted` branch, so on interrupt
            # the stale callback persisted across the next `speak()`
            # call — a fresh speak that didn't pass its own callback
            # would inherit the dead one. SystemTTS already does this
            # correctly; aligning PiperTTS/ChatterboxTTS here.
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception:
                    pass
            self._completion_callback = None

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Notify the face widget of speaking state changes.

        Uses file-based approach to work across processes:
        - Dev mode runs daemon as subprocess (different process)
        - File-based state works across process boundaries
        """
        # Import here to avoid circular dependencies
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            if is_speaking:
                debug_log("setting face state to SPEAKING (chatterbox)", "tts")
                state_manager.set_state(JarvisState.SPEAKING)
            # Note: When speaking ends, we don't change state here - let daemon manage transitions
        except ImportError:
            debug_log("face widget not available (ImportError) (chatterbox)", "tts")
        except Exception as e:
            # Don't let face widget errors affect TTS
            debug_log(f"failed to set face state to SPEAKING (chatterbox): {e}", "tts")

    # Loopback guard helpers (same interface as TextToSpeech)
    def is_speaking(self) -> bool:
        return self._is_speaking.is_set()

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


class PiperTTS:
    """TTS implementation using Piper (local neural TTS with exact duration).

    Piper generates actual audio samples, enabling precise duration calculation
    instead of WPM-based estimation. Uses sounddevice for streaming playback
    with responsive interruption support.
    """

    def __init__(
        self,
        enabled: bool = True,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        model_path: Optional[str] = None,
        speaker: Optional[int] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.2,
    ) -> None:
        self.enabled = enabled
        self.voice = voice  # Not used in Piper, kept for interface compatibility
        self.rate = rate    # Not directly supported, use length_scale instead
        self.model_path = model_path
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence

        # Threading and queue setup (same pattern as other TTS engines)
        self._q: queue.Queue[str] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._last_spoken_text: str = ""
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._should_interrupt = threading.Event()

        # Audit round 21 fix (F02): mirror the ``SystemTTS`` epoch
        # pattern. The bare ``_should_interrupt`` flag had a race —
        # ``interrupt()`` set it just before a fresh ``speak()``
        # enqueued a new sentence. The worker popped the new sentence,
        # saw the still-set flag, and silently returned. User
        # presses stop, then asks a new question, and the new reply
        # never gets spoken. ``_interrupt_epoch`` solves it: every
        # ``speak()`` snapshots the current epoch, every
        # ``_speak_once`` re-reads it before doing work — if the
        # epoch advanced since enqueue, the sentence is stale and
        # we drop it (and only it). New sentences with the new
        # epoch sail through cleanly.
        self._interrupt_epoch: int = 0
        self._epoch_lock = threading.Lock()

        # Piper voice (lazy loaded)
        self._voice = None
        self._sample_rate: int = 22050  # Piper default, updated on model load
        self._initialized = False
        self._init_lock = threading.Lock()
        self._init_error: Optional[str] = None

        # Audio stream for interruption
        self._audio_stream = None
        self._audio_lock = threading.Lock()

    def _ensure_initialized(self) -> bool:
        """Initialize Piper voice model. Returns True if successful.

        If no model is configured, automatically downloads the default voice.
        """
        if self._initialized:
            return self._voice is not None
        if not self.enabled:
            return False

        with self._init_lock:
            if self._initialized:
                return self._voice is not None

            try:
                # Use configured path or default
                model_path = self.model_path
                if not model_path:
                    model_path = _get_default_piper_model_path()
                    debug_log(f"No model configured, using default: {model_path}", "tts")

                # Expand user path (e.g., ~/models/voice.onnx)
                model_path = os.path.expanduser(model_path)
                config_path = model_path + ".json"

                # Auto-download if model doesn't exist
                if not os.path.exists(model_path) or not os.path.exists(config_path):
                    # Extract voice name from path for download
                    voice_name = os.path.basename(model_path).replace(".onnx", "")

                    print(f"🔊 Downloading Piper voice: {voice_name}", file=sys.stderr, flush=True)
                    print("   This is a one-time download (~60MB)...", file=sys.stderr, flush=True)

                    def progress(msg):
                        print(msg, file=sys.stderr, flush=True)

                    downloaded_path = _download_piper_voice(voice_name, progress_callback=progress)

                    if not downloaded_path:
                        self._init_error = f"Failed to download voice: {voice_name}"
                        debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
                        self._initialized = True
                        return False

                    model_path = downloaded_path
                    config_path = model_path + ".json"
                    print("✓ Voice downloaded successfully!", file=sys.stderr, flush=True)

                # Final check that files exist
                if not os.path.exists(model_path):
                    self._init_error = f"Model file not found: {model_path}"
                    debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
                    self._initialized = True
                    return False

                if not os.path.exists(config_path):
                    self._init_error = f"Model config not found: {config_path}"
                    debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
                    self._initialized = True
                    return False

                debug_log(f"Piper TTS loading model: {model_path}", "tts")

                # Import piper and load model
                from piper.voice import PiperVoice

                self._voice = PiperVoice.load(model_path, config_path)
                self._sample_rate = self._voice.config.sample_rate

                debug_log(f"Piper TTS initialized: sample_rate={self._sample_rate}", "tts")

            except ImportError as e:
                self._init_error = f"piper-tts not installed: {e}"
                debug_log(f"Piper TTS init failed: {self._init_error}", "tts")
            except Exception as e:
                self._init_error = f"Failed to load Piper model: {e}"
                debug_log(f"Piper TTS init failed: {self._init_error}", "tts")

            self._initialized = True
            return self._voice is not None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        # Initialize model eagerly at startup (downloads if needed)
        # This provides better UX - download happens during startup, not first speech
        self._ensure_initialized()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            self.interrupt()
        except Exception:
            pass
        self._stop.set()
        try:
            self._q.put_nowait("")
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        self._stop.clear()

    def speak(self, text: str, completion_callback: Optional[Callable[[], None]] = None,
              duration_callback: Optional[Callable[[float], None]] = None) -> None:
        if not self.enabled or not text.strip():
            return
        # Lazy start the worker thread
        if self._thread is None:
            self.start()
        self._completion_callback = completion_callback
        self._duration_callback = duration_callback
        # Audit round 17 fix: cap text length BEFORE Piper synthesis —
        # see SystemTTS.speak for full rationale. Piper synthesises
        # the entire string offline before playback begins, so a
        # 100 KB reply can block the queue for many minutes with no
        # interruption window.
        _TTS_MAX_CHARS = 4000
        if len(text) > _TTS_MAX_CHARS:
            ellipsis = " … (truncated)"
            text = text[: _TTS_MAX_CHARS - len(ellipsis)] + ellipsis
        # Preprocess text for speech
        processed_text = _preprocess_for_speech(text)
        # Audit round 21 (F02): snapshot the epoch when ``speak`` is
        # called, NOT when the worker pops the item. If ``interrupt``
        # bumps the epoch between enqueue and pop, the worker
        # recognises the staleness and drops the sentence — without
        # this, a stale ``_should_interrupt.set()`` from a prior
        # interrupt would silently kill the first sentence of the
        # next reply.
        with self._epoch_lock:
            current_epoch = self._interrupt_epoch
        try:
            self._q.put_nowait((current_epoch, processed_text))
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop current speech immediately AND drain queued sentences.

        Audit round 7 fix C4: drain `_q` so streaming-reply sentence #2
        doesn't pull off the queue and play right after sentence #1 was
        aborted. Without the drain, "interrupt" was effectively
        "pause-current-and-play-next".

        Audit round 21 (F02): also bump ``_interrupt_epoch`` so any
        sentence that was enqueued just BEFORE this interrupt and is
        already past the drain (still sitting in another caller's
        ``put_nowait`` between python-level lock acquire and queue
        commit) gets dropped by the worker's epoch check.
        """
        self._should_interrupt.set()
        with self._epoch_lock:
            self._interrupt_epoch += 1
        with self._audio_lock:
            if self._audio_stream is not None:
                try:
                    self._audio_stream.abort()
                except Exception:
                    pass
        try:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not item:
                continue
            # Round 21 — items are now (epoch, text) tuples. Tolerate
            # plain strings for forward-/back-compat with any test or
            # internal caller that bypasses speak().
            if isinstance(item, tuple) and len(item) == 2:
                enq_epoch, text = item
            else:
                enq_epoch, text = 0, item
            if not text:
                continue
            try:
                self._speak_once(text, enq_epoch=enq_epoch)
            except Exception as e:
                debug_log(f"Piper TTS error in _speak_once: {e}", "tts")
                continue

    def _speak_once(self, text: str, enq_epoch: int = 0) -> None:
        # Audit round 21 (F02): drop stale sentences whose epoch is
        # behind the current ``_interrupt_epoch`` — they were queued
        # before the most recent interrupt and must NOT be spoken.
        with self._epoch_lock:
            current_epoch = self._interrupt_epoch
        if enq_epoch < current_epoch:
            debug_log(
                f"Piper TTS: dropping stale sentence (enq_epoch={enq_epoch} < {current_epoch})",
                "tts",
            )
            return
        self._is_speaking.set()
        self._last_spoken_text = text
        self._should_interrupt.clear()
        interrupted = False

        # Signal speaking state to face widget
        self._notify_speaking_state(True)

        try:
            # Initialize on first use
            if not self._ensure_initialized():
                if self._init_error:
                    print(f"  ⚠️ Piper TTS: {self._init_error}", flush=True)
                return

            import sounddevice as sd
            import numpy as np

            start_time = time.time()

            debug_log(f"Piper TTS starting synthesis: {len(text.split())} words", "tts")

            # Check for interruption before synthesis
            if self._should_interrupt.is_set():
                debug_log("Piper TTS interrupted before synthesis", "tts")
                return

            # Synthesize audio - synthesize() returns an iterable of AudioChunks
            from piper.config import SynthesisConfig
            syn_config = SynthesisConfig(
                speaker_id=self.speaker,
                length_scale=self.length_scale,
                noise_scale=self.noise_scale,
                noise_w_scale=self.noise_w,
            )
            # Phoneme-map sanitizer. The uk_UA-ukrainian_tts-medium model
            # (and most community Cyrillic Piper voices) ships a phoneme_id_map
            # with ONLY lowercase Ukrainian letters + a few punctuation marks.
            # Anything outside that set — capitals, Russian-specific letters
            # (ё ы э ъ), Latin letters (CEO, API, GmbH), emoji — is silently
            # dropped during phonemization, leaving holes in the audio that
            # sound like "bulk-bulk" gurgling.
            #
            # Pre-flight transform: lowercase, transliterate Russian/Latin
            # into closest Ukrainian phonetic equivalents, drop anything else.
            # Model-aware sanitization. Two camps of Piper voices:
            #   • TEXT phoneme_type (uk_UA-ukrainian_tts) — has a hand-rolled
            #     char-level phoneme map covering only lowercase UA letters.
            #     We MUST pre-sanitize: lowercase + transliterate non-UA
            #     chars to UA equivalents (the "_sanitize_for_piper_uk" path).
            #   • espeak phoneme_type (ru_RU-dmitri, de_DE-thorsten, en_*) —
            #     route through espeak-ng which handles all native chars,
            #     mixed scripts, digits, etc. Pre-sanitization would BREAK it
            #     (e.g. converting ё→йо on a Russian model produces wrong audio).
            try:
                ptype = getattr(self._voice.config, "phoneme_type", None)
                ptype_name = getattr(ptype, "name", str(ptype)).lower() if ptype else ""
            except Exception:
                ptype_name = ""
            if "text" in ptype_name:
                synth_text = _sanitize_for_piper_uk(text)
            else:
                # espeak / IPA / None path → pass text through unchanged.
                synth_text = text
            print(f"🎤 Piper synth ({ptype_name or 'unknown'}): {synth_text!r}  (from {text[:80]!r})", flush=True)
            audio_chunks = []
            for chunk in self._voice.synthesize(synth_text, syn_config):
                if self._should_interrupt.is_set():
                    debug_log("Piper TTS interrupted during synthesis", "tts")
                    return
                audio_chunks.append(chunk.audio_int16_array)

            # Check for interruption after synthesis
            if self._should_interrupt.is_set():
                debug_log("Piper TTS interrupted after synthesis", "tts")
                return

            # Concatenate all audio chunks
            if not audio_chunks:
                debug_log("Piper TTS: no audio chunks generated", "tts")
                return

            full_audio = np.concatenate(audio_chunks)

            if len(full_audio) == 0:
                debug_log("Piper TTS: no audio generated", "tts")
                return

            # Calculate exact duration from actual samples
            exact_duration = len(full_audio) / self._sample_rate
            debug_log(f"Piper TTS synthesis complete: {exact_duration:.2f}s, {len(full_audio)} samples", "tts")

            # Notify listener of exact duration for precise echo detection
            if self._duration_callback is not None:
                try:
                    self._duration_callback(exact_duration)
                except Exception as e:
                    debug_log(f"Piper TTS duration callback error: {e}", "tts")

            # Play audio with streaming for interruption support
            play_position = [0]
            # Bumped from 1024→2048 frames: at 22050 Hz the old 1024 was
            # ~46ms per buffer, which on macOS Core Audio sometimes drops
            # the FINAL partial buffer when CallbackStop fires (causes
            # "не договорює" — last syllable cut off). 2048 = ~93ms
            # buffers give Core Audio enough lead time to flush cleanly
            # while still being responsive to interruption (~93ms latency
            # is imperceptible).
            blocksize = 2048

            def audio_callback(outdata, frames, time_info, status):
                if self._should_interrupt.is_set():
                    raise sd.CallbackAbort()

                start = play_position[0]
                end = start + frames
                chunk = full_audio[start:end]

                if len(chunk) < frames:
                    # Pad with zeros if we're at the end
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk

                play_position[0] = end

            with self._audio_lock:
                self._audio_stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=1,
                    dtype='int16',
                    blocksize=blocksize,
                    # 'low' latency keeps responsiveness; 'high' would
                    # buffer 200-500ms which is too laggy for a voice
                    # assistant. macOS default 'low' = ~10-15ms latency.
                    latency='low',
                    callback=audio_callback,
                )
                self._audio_stream.start()

            # Wait for playback to complete
            try:
                while self._audio_stream is not None and self._audio_stream.active:
                    if self._should_interrupt.is_set():
                        interrupted = True
                        with self._audio_lock:
                            if self._audio_stream is not None:
                                self._audio_stream.abort()
                        break
                    time.sleep(0.05)
            finally:
                # CRITICAL FIX: pause briefly before closing the stream so
                # macOS Core Audio has time to play out the LAST buffered
                # samples. Without this, calling stream.close() immediately
                # after the wait loop exits can cause the final ~50-100ms
                # of audio to be dropped (Core Audio aborts on close()).
                # Trade-off: adds ~150ms to every TTS reply, but ensures
                # the entire utterance is heard.
                if not interrupted:
                    try:
                        time.sleep(0.15)
                    except Exception:
                        pass
                with self._audio_lock:
                    if self._audio_stream is not None:
                        try:
                            # stop() flushes pending samples cleanly;
                            # close() alone may discard them.
                            self._audio_stream.stop()
                        except Exception:
                            pass
                        try:
                            self._audio_stream.close()
                        except Exception:
                            pass
                        self._audio_stream = None

            actual_duration = time.time() - start_time
            debug_log(f"Piper TTS complete: actual={actual_duration:.2f}s (audio={exact_duration:.2f}s)", "tts")

        except Exception as e:
            debug_log(f"Piper TTS error: {e}", "tts")
            print(f"  ⚠️ Piper TTS error: {e}", flush=True)
        finally:
            self._is_speaking.clear()
            self._notify_speaking_state(False)

            # Call completion callback if set and not interrupted.
            # Audit round 8 regression fix: always clear at end (was
            # only clearing inside the `if not interrupted` branch,
            # which leaked stale callback refs on interrupt).
            if self._completion_callback is not None and not interrupted:
                try:
                    self._completion_callback()
                except Exception as e:
                    print(f"  ⚠️ Piper TTS completion callback error: {e}", flush=True)
            self._completion_callback = None

    def _notify_speaking_state(self, is_speaking: bool) -> None:
        """Notify the face widget of speaking state changes."""
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            if is_speaking:
                debug_log("setting face state to SPEAKING (piper)", "tts")
                state_manager.set_state(JarvisState.SPEAKING)
        except ImportError:
            debug_log("face widget not available (ImportError) (piper)", "tts")
        except Exception as e:
            debug_log(f"failed to set face state to SPEAKING (piper): {e}", "tts")

    # Loopback guard helpers (same interface as TextToSpeech)
    # Audit round 23 fix (F46): queue-aware. See SystemTTS.is_speaking
    # for full rationale — same race in PiperTTS streaming path.
    def is_speaking(self) -> bool:
        if self._is_speaking.is_set():
            return True
        try:
            if not self._q.empty():
                return True
        except Exception:
            pass
        return False

    def get_last_spoken_text(self) -> str:
        return self._last_spoken_text


# ════════════════════════════════════════════════════════════════════════
# SystemTTS — macOS native `say` engine, multilingual out of the box
# ════════════════════════════════════════════════════════════════════════
#
# Why this exists: Piper voices ship single-language phoneme maps. The
# uk_UA model silently drops capital letters, Russian-only Cyrillic,
# Latin alphabet, digits and emoji — anything outside its 33-char UA
# alphabet — which produces audible "gurgling" gaps in the output. Even
# with aggressive pre-sanitization, the moment LLM mixes UA + RU + EN
# (a normal суржик utterance for Danylo: "Зараз я check the GitLab CI")
# the synthesizer breaks.
#
# macOS Speech Synthesis (the `say` command + AVSpeechSynthesizer
# underneath) has high-quality voices for UA, RU, EN, DE, IT, ES, FR
# and a long tail, all in one engine. It transparently handles mixed
# scripts and tracks individual word boundaries for prosody. Zero
# install footprint — built into macOS 13+.
#
# Strategy:
#   • Detect the dominant language of each sentence (Cyrillic UA/RU/
#     mixed, Latin EN/DE)
#   • Map to a personality-matched voice (male serious for UA/RU/DE,
#     British male for EN — "billionaire CEO" tone Danylo asked for)
#   • Run `say -v <voice> -r <rate>` as a subprocess; interrupt by
#     killing the subprocess
#
# Implementation mirrors PiperTTS's public interface (start/stop/speak/
# interrupt/is_speaking) so the daemon doesn't care which engine it has.


# Voice catalogue per language. Goal: male, mature, "billionaire CEO" tone
# in every language. macOS only ships female UA/RU voices (Lesya, Milena),
# so for those we shell out to Piper neural TTS with a male speaker; for
# DE/EN we use stock macOS male voices.
#
# The `piper:` prefix is a magic value handled by SystemTTS._speak_once —
# it routes to an internal PiperTTS instance instead of `say`.
_SYSTEM_VOICES_BY_LANG = {
    "uk": "piper:mykyta",  # Piper uk_UA mykyta — male, serious, billionaire-tone
    "ru": "piper:dmitri",  # Piper ru_RU dmitri (will fall back to Milena if not on disk)
    "de": "Markus",        # macOS DE male
    "en": "Daniel",        # macOS en_GB male — gravitas, founder-cockpit tone
}

# Cyrillic letter sets that DISTINGUISH UA vs RU. If we see UA-only
# letters → Ukrainian. If we see RU-only letters → Russian. If both
# overlap (cyrillic-common), we tie-break by frequency of distinctive
# tokens, otherwise default to UA (user's primary working language).
_UA_ONLY_CHARS = set("іїєґІЇЄҐ'")
_RU_ONLY_CHARS = set("ёыэъЁЫЭЪ")


# RU-distinctive words (rough but effective) — these appear in most Russian
# sentences and rarely (if ever) in Ukrainian. Order doesn't matter, the
# detector counts hits across the whole text.
_RU_WORDS = frozenset([
    # Greetings / common
    "здравствуйте", "привет", "пожалуйста", "спасибо", "сейчас", "сегодня",
    "конечно", "извините", "пока", "хорошо", "слушаю", "слушай", "встречи",
    # Function words / pronouns
    "что", "это", "очень", "только", "чем", "если", "когда", "тебя", "меня",
    "вас", "нас", "ещё", "уже", "тоже", "также", "никак", "потому",
    # Verbs - RU forms ending in -ть, -лять, -ать
    "понимаю", "слышу", "помочь", "помогать", "знать", "делать", "делаю",
    "готов", "готовый", "буду", "будет", "был", "была", "сделать", "хочу",
    "могу", "должен", "должна", "надо", "нужно", "нравится", "ответ",
    "ответить", "помочь", "помогу", "сделаю", "найду", "открою",
    # Nouns / verbs distinctive (RU-specific morphology)
    "дела", "делами", "дело", "вопрос", "ответ", "русск", "москв", "правда",
    "место", "время", "день", "час", "минут", "секунд",
    # Genitive/accusative endings — strong RU signal vs UA
    "компании", "компания", "компанию", "компанией",
    "директор", "сотрудник", "владелец", "основатель", "президент",
    " я ", " ты ", " мы ", " вы ",  # RU subject pronouns (UA uses "ти/ми/ви")
    "которого", "которая", "которые", "потому что", "поэтому",
    "ничего", "всё", "все", "что-то", "как-то",
])
# UA-distinctive words (a complement set for tie-breaking).
_UA_WORDS = frozenset([
    # Greetings / common
    "привіт", "будь ласка", "дякую", "зараз", "сьогодні", "звичайно",
    "вибач", "поки", "добре", "гаразд",
    # Function words / pronouns
    "що", "це", "дуже", "тільки", "чим", "якщо", "коли", "тебе", "мене",
    "вас", "нас", "ще", "вже", "також", "ніяк", "тому",
    # Verbs - UA forms ending in -ти, -ляти, -ати
    "розумію", "чую", "допомогти", "допомагати", "знати", "робити", "роблю",
    "готовий", "буду", "буде", "був", "була", "зробити", "хочу",
    "можу", "повинен", "повинна", "треба", "потрібно", "подобається",
    # Nouns distinctive (UA morphology)
    "справи", "справами", "справа", "питання", "відповідь", "україн", "київ",
    "місце", "час", "день", "година", "хвилин", "секунд",
    "компанії", "компанія", "компанію", "компанією",
    " ти ", " ми ", " ви ",  # UA subject pronouns
    "якого", "яка", "які", "тому що", "тому-то",
    "нічого", "все", "щось", "якось",
])


def _detect_language(text: str) -> str:
    """Return one of 'uk'|'ru'|'de'|'en' for the dominant language of `text`.

    Strategy:
      1. If text has UA-only chars (іїєґ) — UA. If RU-only (ёыэъ) — RU.
      2. Otherwise count distinctive common-word hits. Whichever wins, wins.
      3. Tied / zero Cyrillic-Latin: check German markers, else English.
    """
    ua_chars = sum(1 for ch in text if ch in _UA_ONLY_CHARS)
    ru_chars = sum(1 for ch in text if ch in _RU_ONLY_CHARS)
    cyr_total = sum(1 for ch in text if 'Ѐ' <= ch <= 'ӿ')

    if cyr_total > 0:
        if ua_chars and not ru_chars:
            return "uk"
        if ru_chars and not ua_chars:
            return "ru"

        # Ambiguous Cyrillic — score common words.
        # Tie-break: 'ru' (post May 16 uk→ru migration). Previously 'uk'
        # default caused short RU phrases like "Слушаю, Данило" (no
        # _RU_ONLY_CHARS, no _UA_ONLY_CHARS, no scoring word match) to be
        # routed to UA Piper voice → spoken with UA accent. User report:
        # "вимовляє він неправильно тепер слова з українським акцентом".
        lower = text.lower()
        ua_score = sum(1 for w in _UA_WORDS if w in lower)
        ru_score = sum(1 for w in _RU_WORDS if w in lower)
        if ua_score > ru_score:
            return "uk"
        return "ru"

    # No Cyrillic — Latin script. Distinguish EN vs DE.
    # Umlauts/ß alone = definitely German.
    if any(ch in text for ch in "äöüÄÖÜß"):
        return "de"
    # Otherwise need ≥2 distinctive DE words (single "ist" is ambiguous —
    # it appears in English text too, e.g. "the date is X").
    lower = text.lower()
    # Strong DE markers — function words + weekday/month names that Whisper-
    # generated replies routinely contain. "heute ist donnerstag, 14. mai"
    # used to fail because old list missed temporal words.
    de_strong_words = (
        # Function words
        "ich ", "und ", "nicht ", "sind ", "haben ", "auch ", "wir ",
        "du bist", "ich bin", "sie ist", "kann ich", " der ", " die ",
        " das ", " mit ", "wie ", "geht es", "heute", "gestern", "morgen",
        "jetzt", "sehr", "gut", "schlecht", "wirklich", "natürlich",
        # Weekdays
        "montag", "dienstag", "mittwoch", "donnerstag", "freitag",
        "samstag", "sonntag", "wochenende",
        # Months
        "januar", "februar", "märz", "april", " mai ", " mai.", " mai,",
        "juni", "juli", "august", "september", "oktober", "november", "dezember",
        # Common verbs/states
        " ist ", "ist.", "ist,", "war ", "wird ", "sein ", "hat ", "kann ",
    )
    de_hits = sum(1 for w in de_strong_words if w in lower)
    if de_hits >= 2:
        return "de"
    return "en"


class SystemTTS:
    """macOS-native multilingual TTS via the `say` subprocess.

    Mirrors PiperTTS public interface for drop-in replacement."""

    def __init__(
        self,
        enabled: bool = True,
        voice: Optional[str] = None,
        rate: Optional[int] = None,
        voice_map: Optional[dict] = None,
    ):
        self.enabled = enabled
        # `voice` here is a default-override (forces one voice regardless of
        # language detection). Leave None to use the per-language mapping.
        self.voice_override = voice
        self.rate = rate or 190  # words per minute; 175-200 sounds natural
        # User-customisable per-language voice mapping (config.json):
        #   "tts_system_voice_map": {"uk": "Lesya", "ru": "Milena", ...}
        self.voice_map = {**_SYSTEM_VOICES_BY_LANG, **(voice_map or {})}

        # Audit round 15 fix: queue items are now (epoch, text) tuples.
        # Each ``speak()`` snapshots the current ``_interrupt_epoch``,
        # ``interrupt()`` bumps it. The worker drops any popped item
        # whose epoch is less than the current one — that's a sentence
        # enqueued before the most-recent interrupt. The previous
        # design used a global ``_should_interrupt`` event which had a
        # race window: a fresh ``speak("new")`` arriving AFTER
        # ``interrupt()`` drained the queue but BEFORE the worker
        # popped would see ``_should_interrupt.set()`` and silently
        # drop the new sentence (streaming-reply tail truncation).
        self._q: "queue.Queue[tuple[int, str]]" = queue.Queue()
        self._interrupt_epoch: int = 0
        self._epoch_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_speaking = threading.Event()
        self._should_interrupt = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._last_spoken_text: Optional[str] = None
        self._completion_callback: Optional[Callable[[], None]] = None
        self._duration_callback: Optional[Callable[[float], None]] = None
        self._notify_speaking_cb: Optional[Callable[[bool], None]] = None

    # ── Interface methods (mirroring PiperTTS) ──────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SystemTTS", daemon=True)
        self._thread.start()
        print("✓ SystemTTS (macOS say) ready — UA/RU/EN/DE multilingual", flush=True)

    def stop(self) -> None:
        self._stop.set()
        self.interrupt()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def interrupt(self) -> None:
        """Stop ALL TTS playback immediately.

        Audit round 7 fixes:
          C3: ALSO interrupt every cached PiperTTS sub-engine — UA/RU
              paths route through `_speak_via_piper` → `engine._speak_once`
              which uses `sounddevice.OutputStream` (not a subprocess),
              so killing only `_current_proc` (the `say` process) left
              Piper audio playing for 1-3 more seconds. The mic captured
              the tail of "Слухаю, Данило..." as a fresh user utterance.
          C4: DRAIN the pending sentence queue. Streaming replies queue
              sentences 2,3,4... via _flush_sentence → speak(). Without
              draining, the current sentence stops mid-word but the
              next queued sentence pulls off the queue and plays right
              away. The "interrupt" thus became an unintended pause-
              and-continue.
        """
        self._should_interrupt.set()
        # Audit round 15 fix: bump the epoch. Anything currently in
        # the queue (or popped but not yet checked) carries the OLD
        # epoch and will be dropped on the in-worker check below.
        # Future ``speak()`` calls snapshot the new epoch and survive.
        with self._epoch_lock:
            self._interrupt_epoch += 1
        # Stop the `say` subprocess (LV/EN paths).
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
        # C3 — also stop every cached Piper sub-engine.
        # `_piper_engines` is a class attribute on SystemTTS, so iterate
        # via the class (works regardless of instance).
        # Audit round 15 fix: take the class-level lock when iterating
        # the shared engine cache (previously a bare ``list(...)`` call
        # raced with concurrent ``_get_piper_for`` insertions — GIL-
        # safe in CPython today but explicitly NOT guaranteed by the
        # language spec; breaks under free-threaded 3.13+).
        try:
            with SystemTTS._piper_engines_lock:
                _engines = list(SystemTTS._piper_engines.values())
            for _engine in _engines:
                try:
                    _engine.interrupt()
                except Exception:
                    pass
        except Exception:
            pass
        # C4 — drain queued sentences so the streaming reply that's
        # being interrupted doesn't continue speaking sentence 2.
        try:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    break
        except Exception:
            pass

    def speak(
        self,
        text: str,
        completion_callback: Optional[Callable[[], None]] = None,
        duration_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        if not self.enabled or not text.strip():
            return
        if self._thread is None or not self._thread.is_alive():
            self.start()
        self._completion_callback = completion_callback
        self._duration_callback = duration_callback
        # Audit round 17 fix: hard cap text length BEFORE preprocessing.
        # A pathological LLM reply (e.g. an unbounded chain-of-thought,
        # a tool-output that bypassed the response_text 50 000-char
        # cap, or an injection payload that asks for the assistant to
        # echo a megabyte) used to be queued in full — Piper would
        # synthesise it offline for minutes with no interruption
        # window, and the macOS ``say`` subprocess would hit
        # ARG_MAX and fail silently because the entire string is one
        # argv element. 4000 chars is well above any natural spoken
        # response (~6 minutes of speech) and short enough that argv
        # limits never apply.
        _TTS_MAX_CHARS = 4000
        if len(text) > _TTS_MAX_CHARS:
            ellipsis = " … (truncated)"
            text = text[: _TTS_MAX_CHARS - len(ellipsis)] + ellipsis
        processed = _preprocess_for_speech(text)
        # Audit round 15 fix: snapshot the current epoch under lock
        # and enqueue together with the text. The worker drops popped
        # tuples whose epoch is older than the current one — i.e.
        # anything queued before the most-recent ``interrupt()``.
        with self._epoch_lock:
            current_epoch = self._interrupt_epoch
        try:
            self._q.put_nowait((current_epoch, processed))
        except Exception:
            pass

    # NOTE: kept as a method (not @property) to match PiperTTS interface —
    # the listener thread calls `self.tts.is_speaking()` and a property
    # would crash with "TypeError: 'bool' object is not callable".
    #
    # Audit round 23 fix (F46): also report True when sentences are
    # waiting in the queue. Background: a streaming LLM reply
    # enqueues 3-6 sentences via ``speak()`` rapid-fire. Between
    # sentences ``_is_speaking`` is briefly cleared (in the
    # ``_speak_once`` finally block) before the worker picks up the
    # next item. The listener's ``_wait_and_activate`` thread polls
    # ``is_speaking()`` to know when TTS is finished — if it polls
    # during that gap it sees False and activates the hot window
    # PREMATURELY, while sentences 2-N are still pending. The
    # premature LISTENING state transition then competes with the
    # SPEAKING transition of sentence 2 — visible as coin flicker.
    # Queue-aware check fixes both: orchestrator only fires
    # ``_on_tts_complete`` once everything is truly drained.
    def is_speaking(self) -> bool:
        if self._is_speaking.is_set():
            return True
        try:
            if not self._q.empty():
                return True
        except Exception:
            pass
        return False

    def get_last_spoken_text(self) -> Optional[str]:
        return self._last_spoken_text

    def _notify_speaking_state(self, speaking: bool) -> None:
        if self._notify_speaking_cb:
            try:
                self._notify_speaking_cb(bool(speaking))
            except Exception:
                pass
        # Audit round 23 fix (F41 + F45): NEVER emit IDLE from the
        # TTS engine. State ownership now lives entirely with the
        # listener orchestrator (state_manager.py + _on_tts_complete).
        # Round 22's F31 queue-peek attempt was structurally broken
        # for single-sentence replies: by the time the worker called
        # ``_notify_speaking_state(False)`` in the finally clause,
        # the only enqueued sentence had already been popped — queue
        # was empty — so F31 still emitted IDLE. Result: visible 2ms
        # SPEAKING→IDLE flicker on every single-sentence reply, and
        # the HUD coin's lerp damping never caught up. Live evidence
        # from events.jsonl seq=291 (SPEAKING) → seq=292 (IDLE) 2ms
        # apart for "Нило, ты где?" reply.
        #
        # The new contract:
        #   * TTS engine emits ONLY SPEAKING when audio starts.
        #   * IDLE / LISTENING is the listener's job, triggered after
        #     ``activate_hot_window`` + ``track_tts_finish`` fire from
        #     ``_on_tts_complete`` once the queue actually drains AND
        #     the playback callback finishes.
        if speaking:
            try:
                from desktop_app.face_widget import JarvisState, get_jarvis_state
                get_jarvis_state().set_state(JarvisState.SPEAKING)
            except Exception:
                pass
        # speaking == False: no-op. Don't touch face state from here.
        # If you need a "TTS finished, no follow-up" signal, hook it
        # via _completion_callback which propagates to _on_tts_complete.

    # ── Worker thread ───────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            # Audit round 15 fix: queue now holds (epoch, text) tuples.
            # A bare string from legacy code paths is tolerated as
            # epoch-0 (i.e. always-stale once any interrupt fires) so
            # an import-order accident doesn't crash the worker.
            if isinstance(item, tuple) and len(item) == 2:
                enq_epoch, text = item
            else:
                enq_epoch, text = 0, item
            if not text:
                continue
            try:
                self._speak_once(text, enq_epoch=enq_epoch)
            except Exception as e:
                print(f"⚠️ SystemTTS error: {e}", flush=True)

    def _speak_once(self, text: str, enq_epoch: int = 0) -> None:
        # Audit round 11 fix C4: previously cleared `_should_interrupt`
        # AFTER `_is_speaking.set()`. If `interrupt()` fired in the tiny
        # window between the two, the clear obliterated the interrupt
        # signal and playback ran to completion. Reordering: snapshot
        # the flag, clear it, then set speaking — any new interrupt
        # after clear() is captured by the in-loop checks.
        #
        # Audit round 15 fix: epoch check replaces the old
        # ``_should_interrupt.is_set()`` short-circuit. A sentence
        # enqueued at epoch=N is stale iff the engine has been
        # interrupted since (i.e. ``_interrupt_epoch > N``). Without
        # this, a fresh ``speak()`` call that arrived AFTER
        # ``interrupt()`` drained the queue but BEFORE the worker
        # popped would see ``_should_interrupt.set()`` and silently
        # drop the user's new sentence — streaming reply truncated.
        with self._epoch_lock:
            current_epoch = self._interrupt_epoch
        if enq_epoch < current_epoch:
            # Stale sentence (queued before the most-recent interrupt).
            self._completion_callback = None
            return
        self._should_interrupt.clear()
        self._is_speaking.set()
        self._last_spoken_text = text
        self._notify_speaking_state(True)

        try:
            lang = _detect_language(text)
            voice = self.voice_override or self.voice_map.get(lang, "Daniel")

            est_dur = _estimate_tts_duration(text, self.rate)
            if self._duration_callback:
                try:
                    self._duration_callback(est_dur)
                except Exception:
                    pass

            print(f"🎤 SystemTTS [{lang}/{voice}]: {text[:100]!r}", flush=True)

            # Route: `piper:<speaker>` → use neural Piper TTS for a male
            # voice macOS lacks. Anything else → macOS `say` subprocess.
            if isinstance(voice, str) and voice.startswith("piper:"):
                self._speak_via_piper(text, lang, voice.split(":", 1)[1])
            else:
                self._speak_via_say(text, voice)
        finally:
            # Check interrupt flag BEFORE clearing speaking state so the
            # callback decision uses the playback's actual outcome.
            # Audit round 7 fix C2: previously fired completion_callback
            # unconditionally — even on interrupt — which would call
            # `activate_hot_window()` from the ack path and open a
            # listening window while the speaker was still settling.
            # Now: skip callback on interrupt to match PiperTTS /
            # ChatterboxTTS semantics (they already do this).
            interrupted = self._should_interrupt.is_set()
            self._is_speaking.clear()
            self._notify_speaking_state(False)
            if self._completion_callback and not interrupted:
                try:
                    self._completion_callback()
                except Exception:
                    pass
            # Always clear the callback ref so the next speak() doesn't
            # accidentally reuse a stale one if it's called without a
            # new callback (PiperTTS does this too).
            self._completion_callback = None

    def _speak_via_say(self, text: str, voice: str) -> None:
        # Audit round 11 fix C1: stderr was PIPE'd but NEVER drained.
        # If `say` emits warnings (font/voice fallback notes on macOS
        # 14+ are common) the OS pipe buffer (typically 16-64KB) fills
        # and the child blocks on write. `poll()` then returns None
        # forever, the loop spins, and only `interrupt()` clears it —
        # the daemon stops speaking but the user thinks Jarvis is mute.
        # DEVNULL is what we want: `say` errors are not actionable here.
        # Also drop NUL bytes from text — Popen raises
        # `ValueError: embedded null byte` on argv with \x00, which
        # would crash the worker thread silently (M1 from audit).
        safe_text = (text or "").replace("\x00", "")
        cmd = ["say", "-v", voice, "-r", str(self.rate), safe_text]
        # Quick-exit check before spawning a process the interrupt
        # would only kill in 50ms — saves a syscall round-trip and
        # closes the "interrupt fires during fallback Popen" race
        # called out in M4 of round 11.
        if self._should_interrupt.is_set():
            return
        with self._proc_lock:
            try:
                self._current_proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except (ValueError, OSError) as e:
                debug_log(f"_speak_via_say: Popen failed: {e}", "tts")
                self._current_proc = None
                return
        try:
            while True:
                # Snapshot proc under lock to avoid TOCTOU (interrupt
                # could clear it between poll() and the kill path).
                with self._proc_lock:
                    proc = self._current_proc
                if proc is None or proc.poll() is not None:
                    break
                if self._should_interrupt.is_set():
                    # Two-stage termination: SIGTERM with 1s grace, then SIGKILL.
                    # Without the kill fallback, a hung `say` process pinned by
                    # the audio framework would keep _proc_lock held forever and
                    # block the next TTS call (silent daemon, no recovery).
                    try: proc.terminate()
                    except Exception: pass
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        try: proc.kill()
                        except Exception: pass
                        try: proc.wait(timeout=1.0)
                        except Exception: pass
                    except Exception: pass
                    break
                time.sleep(0.05)
            # Ensure the natural-completion path also reaps the process
            # (was missing — zombie risk on long-running daemon).
            with self._proc_lock:
                proc = self._current_proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
        finally:
            with self._proc_lock:
                self._current_proc = None

    # ── Piper sub-engine (lazily initialised for male UA/RU) ────────────
    # Audit round 11 fix C3: class-level dict shared across all SystemTTS
    # instances; the check-then-build pattern below was unlocked, so two
    # concurrent first-uses of the same language could both pass the
    # `cache_key in self._piper_engines` check and instantiate two
    # PiperTTS engines. Each engine spins up an audio stream thread and
    # loads a ~60MB ONNX model — the leaked instance stayed alive for
    # the daemon lifetime. Class-level lock guards the get-or-build.
    _piper_engines: dict = {}  # speaker_name → cached PiperTTS instance
    _piper_engines_lock = threading.Lock()

    def _get_piper_for(self, lang: str, speaker_name: str):
        """Get or build a cached PiperTTS engine for the given language.

        Speaker name maps to:
          uk/mykyta  → uk_UA-ukrainian_tts-medium + speaker_id=1
          uk/lada    → uk_UA-ukrainian_tts-medium + speaker_id=0
          ru/dmitri  → ru_RU-dmitri-medium (downloaded on first use)
                       fallback: ru_RU-irina-medium with notice if no male model
        """
        cache_key = f"{lang}:{speaker_name}"
        # Fast path — read without lock (CPython dict reads are atomic).
        existing = SystemTTS._piper_engines.get(cache_key)
        if existing is not None:
            return existing

        models_dir = _get_piper_models_dir()
        if lang == "uk":
            model_path = str(models_dir / "uk_UA-ukrainian_tts-medium.onnx")
            speaker_map = {"lada": 0, "mykyta": 1, "tetiana": 2}
            speaker_id = speaker_map.get(speaker_name, 1)
        elif lang == "ru":
            # Honor speaker_name from tts_system_voice_map: "ruslan" /
            # "dmitri" / "irina". Pre-May16 this branch was hardcoded to
            # dmitri and ignored speaker — so when user switched config to
            # "piper:ruslan" (deeper voice), daemon silently still loaded
            # dmitri. User report: "голос потрібно зробити більш реалістич-
            # ним... грубішого тона". The fix: actually use the speaker
            # name when picking the .onnx file.
            ru_voice_files = {
                "ruslan": "ru_RU-ruslan-medium.onnx",  # deepest, most masculine
                "dmitri": "ru_RU-dmitri-medium.onnx",  # softer male
                "irina":  "ru_RU-irina-medium.onnx",   # female fallback
            }
            requested = ru_voice_files.get(speaker_name, "ru_RU-ruslan-medium.onnx")
            requested_path = models_dir / requested
            if requested_path.exists():
                model_path = str(requested_path)
            else:
                # Fallback chain: ruslan → dmitri → irina (in order of preference).
                for fname in ("ru_RU-ruslan-medium.onnx",
                              "ru_RU-dmitri-medium.onnx",
                              "ru_RU-irina-medium.onnx"):
                    p = models_dir / fname
                    if p.exists():
                        model_path = str(p)
                        debug_log(
                            f"Piper RU: requested '{speaker_name}' not found, "
                            f"falling back to {fname}",
                            "tts",
                        )
                        break
                else:
                    return None
            speaker_id = None
        else:
            return None

        # Per-language tuning. User feedback:
        #   UA — keep fast & crisp (current 0.80) — voice daemon dialog.
        #   RU — "буквально трішки повільніше, більше грубого тону,
        #        більше виразності" → length 0.92 (slower 15%),
        #        noise_w 1.2 (more prosody variation = expressive),
        #        sentence_silence 0.25 (longer pauses = gravitas).
        #   DE/EN — same as UA defaults.
        if lang == "ru":
            length, noise, noise_w, silence = 0.92, 0.45, 1.2, 0.25
        else:  # uk / default
            length, noise, noise_w, silence = 0.80, 0.4, 1.0, 0.15
        # Slow path under lock — double-check then build.
        with SystemTTS._piper_engines_lock:
            existing = SystemTTS._piper_engines.get(cache_key)
            if existing is not None:
                return existing
            engine = PiperTTS(
                enabled=True, voice=None, rate=self.rate,
                model_path=model_path, speaker=speaker_id,
                length_scale=length,
                noise_scale=noise,
                noise_w=noise_w,
                sentence_silence=silence,
            )
            engine.start()
            SystemTTS._piper_engines[cache_key] = engine
            return engine

    def _speak_via_piper(self, text: str, lang: str, speaker: str) -> None:
        engine = self._get_piper_for(lang, speaker)
        if engine is None:
            # No Piper voice for this language → fall back to macOS say.
            fallback = "Milena" if lang == "ru" else "Daniel"
            print(f"⚠️ no Piper voice for {lang!r}, falling back to {fallback}", flush=True)
            self._speak_via_say(text, fallback)
            return

        # Direct synchronous call — bypass the engine's own queue/thread
        # since SystemTTS already owns the speech worker. We invoke the
        # internal _speak_once which runs synth+playback inline.
        try:
            engine._speak_once(text)
        except Exception as e:
            print(f"⚠️ Piper sub-engine error: {e} — falling back to say", flush=True)
            fallback = "Milena" if lang == "ru" else "Daniel"
            self._speak_via_say(text, fallback)


# Audit round 11 fix L3: duplicate `import subprocess` removed
# (already imported at line 3).


def create_tts_engine(
    engine: str = "piper",
    enabled: bool = True,
    voice: Optional[str] = None,
    rate: Optional[int] = None,
    # Chatterbox parameters
    device: str = "cuda",
    audio_prompt_path: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    # Piper parameters
    piper_model_path: Optional[str] = None,
    piper_speaker: Optional[int] = None,
    piper_length_scale: float = 1.0,
    piper_noise_scale: float = 0.667,
    piper_noise_w: float = 0.8,
    piper_sentence_silence: float = 0.2,
    # System TTS per-language voice map (Lesya/Milena/Markus/Daniel etc.)
    system_voice_map: Optional[dict] = None,
):
    """Factory function to create the appropriate TTS engine.

    Supported engines:
    - "system" (recommended): macOS native `say`, multilingual UA/RU/EN/DE
    - "piper": Neural TTS with single-language phoneme map (legacy)
    - "chatterbox": AI voice with emotion control (requires PyTorch)
    """
    if engine.lower() == "system":
        return SystemTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            voice_map=system_voice_map or None,
        )
    if engine.lower() == "chatterbox":
        return ChatterboxTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            device=device,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    else:
        # Default to Piper TTS
        return PiperTTS(
            enabled=enabled,
            voice=voice,
            rate=rate,
            model_path=piper_model_path,
            speaker=piper_speaker,
            length_scale=piper_length_scale,
            noise_scale=piper_noise_scale,
            noise_w=piper_noise_w,
            sentence_silence=piper_sentence_silence,
        )


def json_escape_ps(s: str) -> str:
    # For PowerShell, use double quotes and escape internal double quotes
    # This avoids issues with apostrophes in contractions like "you're"
    escaped = s.replace('"', '""')
    return '"' + escaped + '"'

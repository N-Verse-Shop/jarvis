"""
Voice Listener - Main orchestrator for voice capture and processing.

Coordinates audio capture, speech recognition, echo detection, and state management.
"""

from __future__ import annotations
import functools
import os
import re
import threading
import time
import queue
import sys
import platform
from collections import deque
from typing import Optional, TYPE_CHECKING, Any
from datetime import datetime

import requests  # round 25 (F51): module-level import for _start_llm_keepalive
from rapidfuzz import fuzz
from .echo_detection import EchoDetector
from .state_manager import StateManager, ListeningState
from .wake_detection import is_wake_word_detected, extract_query_after_wake, is_stop_command
from .transcript_buffer import TranscriptBuffer
from .intent_judge import IntentJudge, create_intent_judge, warm_up_ollama_model
from ..debug import debug_log
from ..utils.location import is_location_available


# Audit round 17 fix: voice queries are printed to stdout for the
# user-facing log viewer AND piped into the desktop_app log file.
# A misheard "Jarvis, remember my password Hunter2..." used to land
# verbatim in plaintext logs that survive process restart. Wrap every
# user-print of a transcript through this helper to scrub credentials
# AND truncate at 80 chars so we don't slip back into ungated logging
# the next time a new print-site lands. Truncation matches the
# already-truncated debug sites elsewhere in this file.
def _safe_user_text(text: Optional[str], limit: int = 80) -> str:
    if not text:
        return ""
    try:
        from ..utils.redact import scrub_secrets as _scrub
        scrubbed = _scrub(str(text))
    except Exception:
        scrubbed = str(text)
    flat = scrubbed.replace("\n", " ").strip()
    if len(flat) > limit:
        return flat[: limit - 1] + "…"
    return flat


# Round 30 (F93): module-level PII-aware print helper. All voice-loop
# prints that include user transcripts MUST route through this rather
# than calling print() directly — otherwise the text leaks to
# ~/Library/Logs/jarvis-assistant.out.log, which is world-readable
# (mode 644) and persists indefinitely (rotation only at 8MB).
# Live evidence (R30 audit): out.log carried multiple "📝 Heard:
# <family conversation>" lines for ambient mic capture before the
# flag-gate was added.
def _voice_debug_on() -> bool:
    """Cached check; mirrors debug._is_debug_enabled but cheap-fast.

    We don't want the audio thread paying for a config-load on every
    transcribe. Cache for 2s — matches the debug.py TTL.
    """
    try:
        from ..debug import _is_debug_enabled as _check
        return bool(_check())
    except Exception:
        return False


def _vprint(*args, **kwargs) -> None:
    """Gated print: emits to stdout ONLY when voice_debug is enabled."""
    if _voice_debug_on():
        print(*args, **kwargs)

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory


def is_whisper_hallucination(no_speech_prob: float, threshold: float) -> bool:
    """Shared Whisper no-speech gate.

    Whisper can report high `avg_logprob` confidence on hallucinated phrases
    when the audio is silent or noise. `no_speech_prob` is an independent
    signal and must be checked first. Used by both the faster-whisper path
    (`_filter_noisy_segments`) and the MLX path (`_finalize_utterance`) so
    both backends apply identical policy.
    """
    return no_speech_prob >= threshold

# Audio processing imports (optional)
try:
    import sounddevice as sd
    import webrtcvad
    import numpy as np
except ImportError as e:
    sd = None
    webrtcvad = None
    np = None
    # Log import error for debugging
    print(f"  ⚠️  Audio import error: {e}", flush=True)
    print("     This may indicate PortAudio is not found", flush=True)
    import sys as _sys
    if _sys.platform == 'linux':
        print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
    del _sys
except OSError as e:
    # PortAudio loading errors appear as OSError
    sd = None
    webrtcvad = None
    np = None
    print(f"  ❌ PortAudio initialisation failed: {e}", flush=True)
    print("     Please reinstall the application or check audio drivers", flush=True)
    import sys as _sys
    if _sys.platform == 'linux':
        print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
    del _sys

# Whisper `initial_prompt` — pseudo "previous segment" that primes the
# decoder. Whisper conditions on this text the same way it conditions
# on prior decoded segments: it learns spellings, names, technical
# terms, and language register from it. WITHOUT this, on UA the medium
# model routinely garbled:
#   "Hydrogen"   → "гідроген" / "хідроджин"
#   "Cloudflare" → "клавдфлеер"
#   "Shopify"    → "шопіфай" / "шоп ефай"
#   "Hetzner"    → "гетцнер" / "хетзнер"
#   "TTFB"       → "тіті ефбі"
#   "DACH"       → "дач" / "дах"
#   "qwen"       → "квен" / "коен"
# Listing brand names, project names, tech terms and personal names
# here fixes 80%+ of these. Keep it under ~200 tokens — longer prompts
# slow decode and hurt non-listed words. The prompt is in Ukrainian
# to anchor the language model on UA pronunciation rules.
VOICE_WHISPER_INITIAL_PROMPT = (
    # CRITICAL — Whisper REGURGITATES initial_prompt content verbatim
    # when audio is unclear. Previous version had explicit phrases like
    # "Джарвіс відкрий, Джарвіс закрий, Джарвіс заблокуй" — Whisper
    # decoded quiet audio as that exact text. User report: "почув один
    # раз і фіксується на тому і надалі не слухає більше". Per Whisper
    # docs: keep initial_prompt to vocabulary list, NO example sentences.
    # Just bare brand/tech terms + the wake word. No commands, no
    # templates, no full phrases.
    # R34-S54.1 Phase 7a: was "Українська мова. …" — biased Whisper toward
    # UA decoding under the R34-S48/S51 RU-only policy. Whisper's
    # ``initial_prompt`` controls the LM bias on the decode lattice, so
    # a "Українська мова" primer forced UA token selection on quiet
    # audio. Switched primer to RU. The brand / tech vocab is identical
    # in both languages — only the leading phrase + a few morphology
    # endings change (бекенд→бэкенд, лендінг→лендинг, конверсія→конверсия).
    "Русский язык. Данило, Джарвис, Nexus Studio, IBONS, Hydrogen, "
    "Shopify, Cloudflare, Hetzner, Tailscale, Render, Ollama, "
    "Founder Cockpit, Telegram, Linear, GitHub, Notion, Obsidian, "
    "macOS, Safari, Chrome, Slack, DACH, API, JSON, MLX, Whisper, "
    "qwen, llama, mistral, deploy, frontend, бэкенд, лендинг, "
    "конверсия, кеш, токен, промпт, streaming, latency, CDN, edge."
)


# Whisper backend imports - try MLX first on Apple Silicon, fall back to faster-whisper
MLX_WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


# ─── SINGLE SOURCE OF TRUTH for "bare junk" tokens ──────────────────────
# Audit round 6 unified three drifting copies: the collection-mode
# BARE_JUNK (around line 2181), `_canned_voice_reply`'s _STRICT_BARE_JUNK
# (~2875), and `_persist_memory_pair`'s _MEMORY_BARE_JUNK (~703). They
# had different entries, which meant a query like "продолжение следует"
# could be persisted as memory even though the canned-reply path rejected
# it. Now there's ONE constant referenced everywhere — touch this set
# once and the change applies to every gate. All tokens are lowercase,
# exact-match strings stripped of trailing punctuation.
BARE_JUNK_SET: frozenset = frozenset({
    # Pleasantries
    "спасибо", "благодарю", "дякую", "дяки", "спасибі", "дякуємо",
    "thanks", "thank you", "thx", "danke", "пожалуйста",
    # Farewells
    "пока", "па", "па-па", "бувай", "bye", "goodbye",
    # Acknowledgements
    "ага", "угу", "ок", "окей", "ok", "okay", "так", "yes", "yep", "yeah",
    "ладно", "хорошо", "понятно", "добре", "гаразд", "зрозумів",
    "ну", "это я",
    # Test / probing phrases
    "тест", "test", "проверка", "перевірка",
    # Compliments
    "круто", "супер", "молодец", "молодець", "класс", "класно",
    # Greetings (when arriving WITHOUT wake context they're ambient)
    "доброе утро", "добрый день", "добрый вечер",
    "здравствуйте", "всем привет",
    # Whisper hallucinations frequently seen in archived polluted dialogs
    "субтитры", "продолжение следует", "продолжения следует",
    "продовження буде", "редактор субтитров", "удачи", "удачи!",
    "корректор", "субтитры от", "перевод субтитров",
})


# Single source of truth for the voice-mode system prompt. Used by
# both _voice_direct_chat() and _start_llm_warmup() so the KV-cache
# prefix HITS on every real voice turn (prompt-eval drops from
# 15-25s cold to 1-3s warm on qwen2.5:7b @ Hetzner CCX23 CPU).
#
# Keep this STATIC. Dynamic content (language directive, history,
# query) is appended in separate messages — they evaluate fast
# without re-evaluating this 500-token prefix.
VOICE_STATIC_SYSTEM_PROMPT = (
    # SLIM voice system prompt — May 15 rewrite. Previous version was
    # 820 tokens of grammar rules + Mac action templates → cost ~16s
    # of prompt-eval per query on CPU CCX23 (50 tok/s). User reported
    # "first token in 62.51s" cold cache. New version is ~140 tokens.
    #
    # Rationale for removals:
    # - UA grammar rules: qwen3:8b has native UA training (it's not
    #   qwen2.5 which needed hand-holding). It does NOT need a 250-tok
    #   manual on case endings. Spot-checked on test queries — grammar
    #   quality is preserved.
    # - Mac action templates: handled by listener._direct_command path
    #   BEFORE the LLM sees the query. Chat model never needs the full
    #   list, just the single rule "if user requests an action, say
    #   one trigger phrase".
    # - 10 numbered response rules: collapsed into 3 essentials.
    # MIGRATED uk → ru (May 15). Користувач попросив повний перехід на
    # російську для голосової взаємодії — Whisper medium має ~10x більше
    # RU training даних, тому розпізнавання значно точніше. qwen3:8b
    # відповідає RU природно. UA-фрази в коді залишаються тільки для
    # коментарів і документації.
    "Ты — Джарвис, голосовой AI-ассистент Данила Молянко (Nexus Studio, "
    "B2B agency, DACH-рынок; стек: Shopify Hydrogen, React Native, Render). "
    "ВАЖНО: обращайся к нему ПОЛНЫМ ИМЕНЕМ 'Данило' (украинская форма, как он "
    "себя называет). НИКОГДА не сокращай до 'Нило', 'Данил', 'Данилу', "
    "'Даня' — только 'Данило'.\n"
    "ПРАВИЛА:\n"
    "1. КОРОТКО. Голосовые ответы — 1-3 предложения. Если запрос сложный, "
    "дай суть в 2 предложениях, потом спроси раскрыть ли детали.\n"
    "2. По-русски, естественно. Без markdown, emoji, ссылок, формул — "
    "текст для голоса.\n"
    "3. Без восклицаний ('Угу', 'Конечно', 'Сейчас подумаю').\n"
    "4. Не знаешь — начни с 'Не знаю наверняка' (триггер веб-поиска).\n"
    "5. Действие Mac (открыть app, скриншот, громкость, заметка, "
    "буфер обмена, iMessage, set volume): СНАЧАЛА скажи фразу-триггер "
    "С ИМЕНЕМ ОБЪЕКТА ('Сейчас открою Safari', 'Сейчас открою YouTube "
    "в Safari', 'Сейчас сделаю скриншот', 'Сейчас прочитаю буфер'), "
    "ПОТОМ сразу добавь 'Подтверди моё действие, пожалуйста' и ЖДИ "
    "слова 'подтверждаю' или 'выполняй'. БЕЗ имени объекта триггер "
    "НЕ сработает — обязательно укажи ЧТО ИМЕННО открываешь."
)


def _get_mic_permission_hint() -> str:
    """Return platform-appropriate microphone permission guidance."""
    if sys.platform == 'win32':
        return "Windows Settings > Privacy > Microphone > Allow apps to access"
    elif sys.platform == 'darwin':
        return "System Settings > Privacy & Security > Microphone"
    else:
        return "`pactl list sources` or audio settings for your desktop environment"

def _resample(audio, src_rate: int, dst_rate: int):
    """Resample a 1-D float32 numpy array from *src_rate* to *dst_rate*.

    Uses linear interpolation — fast and good enough for speech going into Whisper.
    """
    if src_rate == dst_rate or np is None:
        return audio
    ratio = dst_rate / src_rate
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _setup_nvidia_dll_path() -> None:
    """Add NVIDIA CUDA DLL directories to PATH on Windows.

    The pip packages nvidia-cublas-cu12 and nvidia-cudnn-cu12 install DLLs
    under site-packages/nvidia/*/bin/ which isn't on PATH by default.
    PyInstaller bundles place them in {app}/cuda/. This function finds
    both locations and prepends them to PATH so ctypes.CDLL can find them.
    """
    import os

    dirs_to_add = []

    # 1. Check for NVIDIA pip packages in site-packages
    try:
        import nvidia.cublas  # type: ignore[import-untyped]
        for pkg_path in nvidia.cublas.__path__:
            bin_dir = os.path.join(pkg_path, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    except (ImportError, AttributeError):
        pass

    try:
        import nvidia.cudnn  # type: ignore[import-untyped]
        for pkg_path in nvidia.cudnn.__path__:
            bin_dir = os.path.join(pkg_path, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    except (ImportError, AttributeError):
        pass

    # 2. Check for CUDA DLLs in app directory (installed by install_cuda.ps1)
    # For frozen apps: check next to the executable (not _MEIPASS, since
    # CUDA libs are downloaded post-install, not bundled in the archive)
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = None

    if app_dir:
        cuda_dir = os.path.join(app_dir, "cuda")
        if os.path.isdir(cuda_dir):
            dirs_to_add.append(cuda_dir)

    # 3. Register DLL directories (must happen before ctypes.CDLL probes)
    # Use both os.add_dll_directory (for ctypes.CDLL) and PATH (for
    # subprocess/child processes). On Windows, PATH changes after process
    # start don't affect ctypes.CDLL search — add_dll_directory is needed.
    if dirs_to_add:
        current_path = os.environ.get("PATH", "")
        new_entries = os.pathsep.join(dirs_to_add)
        os.environ["PATH"] = new_entries + os.pathsep + current_path
        for d in dirs_to_add:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            debug_log(f"added NVIDIA DLL path: {d}", "voice")


@functools.lru_cache(maxsize=None)
def _probe_cuda_available() -> tuple[bool, list[str]]:
    """Probe cuBLAS + cuDNN availability once per process and cache the result.

    The version ranges intentionally span more than the currently pinned
    versions in `installer/windows/install_cuda.ps1` (`cublas64_12.dll`,
    `cudnn_ops64_9.dll`) so a future installer bump doesn't silently fall
    back to CPU until this probe is updated too. A bump outside the
    existing range still requires widening these ranges — the relationship
    is by convention, not enforced.

    Cached because DLLs don't appear or disappear while the process is
    running, and the scan does up to 18 `LoadLibrary` calls on a miss.
    """
    _setup_nvidia_dll_path()

    missing_libs: list[str] = []
    cublas_found = False
    cudnn_found = False
    try:
        import ctypes

        for ver in range(20, 10, -1):
            try:
                ctypes.CDLL(f"cublas64_{ver}.dll")
                cublas_found = True
                debug_log(f"cuBLAS found (cublas64_{ver}.dll)", "voice")
                break
            except OSError:
                continue
        if not cublas_found:
            missing_libs.append("cuBLAS")

        for ver in range(15, 7, -1):
            try:
                ctypes.CDLL(f"cudnn_ops64_{ver}.dll")
                cudnn_found = True
                debug_log(f"cuDNN found (cudnn_ops64_{ver}.dll)", "voice")
                break
            except OSError:
                continue
        if not cudnn_found:
            missing_libs.append("cuDNN")
    except Exception as e:
        debug_log(f"CUDA library probe failed: {e}", "voice")

    return cublas_found and cudnn_found, missing_libs


def _probe_windows_cuda_libraries(device: str) -> tuple[str, list[str]]:
    """Return the device to use and any missing CUDA lib names.

    Short-circuits on non-Windows or non-CUDA device strings. Otherwise
    delegates to the cached `_probe_cuda_available()` so the expensive DLL
    scan only runs once per process lifetime.
    """
    if sys.platform != "win32" or device not in ("auto", "cuda"):
        return device, []

    available, missing_libs = _probe_cuda_available()
    if not available:
        return "cpu", missing_libs
    return device, []


def _print_cuda_unavailable_hint(missing_libs: list[str]) -> None:
    """Print the user-facing CUDA-missing message and recovery hint.

    The hint deliberately points at the tray action, not at "reinstall the
    app". The Inno Setup task only fires once and skips on stale marker
    files, so reinstalling without first deleting `{app}\\cuda` rarely
    fixes the underlying problem. The tray action re-runs install_cuda.ps1
    directly with UAC, which is the actual recovery path.
    """
    debug_log(f"CUDA libraries missing: {missing_libs}, forcing CPU mode", "voice")
    print("  ℹ️  CUDA not available, using CPU mode", flush=True)
    if missing_libs:
        print(f"     Missing: {', '.join(missing_libs)}", flush=True)
    print(
        "  💡 For GPU acceleration, click 'Reinstall GPU libraries' in the Jarvis tray menu",
        flush=True,
    )


try:
    if _is_apple_silicon():
        import mlx_whisper
        MLX_WHISPER_AVAILABLE = True
except Exception:
    mlx_whisper = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    # Catch broad: the faster-whisper import chain can raise ValueError
    # (e.g. "psutil.__spec__ is not set") in some environments.
    WhisperModel = None


def _is_faster_whisper_turbo_supported() -> bool:
    """Check if the installed faster-whisper supports the large-v3-turbo model."""
    try:
        import faster_whisper
        from packaging.version import Version
        return Version(faster_whisper.__version__) >= Version("1.1.0")
    except Exception:
        return False


def _get_mlx_model_repo(model_name: str) -> str:
    """Get the MLX Community HuggingFace repo for a Whisper model."""
    # Map standard model names to MLX Community repos
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "tiny.en": "mlx-community/whisper-tiny.en-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "base.en": "mlx-community/whisper-base.en-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "small.en": "mlx-community/whisper-small.en-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    return model_map.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def _clear_corrupted_whisper_cache(error_message: str) -> bool:
    """Clear a corrupted Whisper model cache directory.

    Parses the CTranslate2 error message to find the snapshot directory,
    then deletes the parent ``models--`` directory so the model can be
    re-downloaded cleanly (including blobs that may also be corrupt).

    Returns ``True`` if a cache directory was found and deleted.
    """
    import re
    import shutil

    # CTranslate2 error format:
    #   "Unable to open file 'model.bin' in model '/path/to/snapshots/hash'"
    match = re.search(
        r"unable to open file\s+'[^']+'\s+in model\s+'([^']+)'",
        error_message,
        re.IGNORECASE,
    )
    if not match:
        debug_log("could not parse cache path from error message", "voice")
        return False

    snapshot_path = match.group(1)

    # Walk up to the models-- directory
    # snapshot_path is e.g. .../models--Org--Name/snapshots/<hash>
    # We want to delete .../models--Org--Name entirely
    from pathlib import Path
    path = Path(snapshot_path)
    model_dir = None
    for parent in [path] + list(path.parents):
        if parent.name.startswith("models--"):
            model_dir = parent
            break

    if model_dir is None or not model_dir.is_dir():
        debug_log(f"could not locate models-- cache directory from: {snapshot_path}", "voice")
        return False

    try:
        shutil.rmtree(model_dir)
        debug_log(f"cleared corrupted Whisper cache: {model_dir}", "voice")
        return True
    except OSError as e:
        debug_log(f"failed to clear corrupted cache: {e}", "voice")
        return False


class VoiceListener(threading.Thread):
    """Main voice listening thread that orchestrates all voice processing."""

    def __init__(self, db: "Database", cfg, tts: Optional[Any],
                 dialogue_memory: "DialogueMemory"):
        """
        Initialise voice listener.

        Args:
            db: Database instance for storage
            cfg: Configuration object
            tts: Text-to-speech engine (optional)
            dialogue_memory: Dialogue memory instance
        """
        super().__init__(daemon=True)

        self.db = db
        self.cfg = cfg
        self.tts = tts
        self.dialogue_memory = dialogue_memory
        self._should_stop = False
        self._dictation_active = False  # Pause flag set by dictation engine
        # Round 25 (F51): timestamp of last user activity (dispatch,
        # wake, command). Used by the LLM keepalive thread to skip
        # pings when the user is actively engaging — avoids queuing
        # prompt-eval work behind a real query. Bumped at every
        # ``_dispatch_query`` and wake-word detection site.
        self._last_user_activity_ts: float = 0.0
        # Round 28 (F68): unsanitized LLM reply used for parse_action().
        # _voice_direct_chat stashes the raw response here (with Latin
        # app names intact like "Safari"/"YouTube") BEFORE running
        # _sanitize_for_piper_uk which strips all Latin words ≥2 chars
        # for the Piper TTS engine. _dispatch_query reads this for
        # action parsing so it sees the original LLM intent.
        self._last_raw_reply: str = ""
        self._first_utterance = True  # Suppress turn separator before the very first transcription
        # ISO-639-1 code Whisper detected for the most recent utterance.
        # Updated at every successful transcription site (MLX + faster-
        # whisper) and consumed by `_dispatch_query` so downstream tools
        # can pick locale-appropriate resources (e.g. tr.wikipedia.org).
        # One-utterance-at-a-time voice flow means the read in
        # `_dispatch_query` always matches the write from the Whisper
        # call that produced the transcript.
        self._last_detected_language: Optional[str] = None

        # Audio processing components
        self._whisper_backend: Optional[str] = None  # "mlx" or "faster-whisper"
        self._whisper_device: Optional[str] = None  # "cpu" or "cuda" (resolved from CTranslate2)
        self._mlx_model_repo: Optional[str] = None  # For MLX backend
        self.model: Optional[Any] = None  # WhisperModel for faster-whisper, None for MLX
        self.transcribe_lock = threading.Lock()  # Shared lock for Whisper model access
        self._audio_q: queue.Queue = queue.Queue(maxsize=64)
        self._pre_roll: deque = deque()

        # Audio callback monitoring (for debugging)
        self._callback_count = 0
        self._last_callback_log_time = 0

        # Voice activity detection
        self.is_speech_active = False
        self._silence_frames = 0
        self._voice_run = 0  # Run of consecutive voiced frames; see endpoint loop hysteresis
        self._utterance_frames: list = []
        self._frame_samples = 0
        # Live transcription bookkeeping — sample buffer counter so we
        # only emit a vad/level event every N frames (avoid event flood).
        self._vad_emit_counter = 0
        # Last time we kicked off a partial transcribe (per-utterance).
        self._last_partial_ts = 0.0
        # Set to True by force_finalize control action — voice loop
        # immediately calls _finalize_utterance() on next iteration.
        self._force_finalize_requested = False
        # Set to True by cancel_utterance control action — voice loop
        # drops accumulated frames without dispatching.
        self._cancel_utterance_requested = False
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        self._vad: Optional = None

        # Initialise VAD if available
        if webrtcvad is not None and bool(getattr(self.cfg, "vad_enabled", True)):
            try:
                self._vad = webrtcvad.Vad(int(getattr(self.cfg, "vad_aggressiveness", 2)))
            except Exception:
                self._vad = None

        # Initialise modular components
        self.echo_detector = EchoDetector(
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            energy_spike_threshold=float(getattr(self.cfg, "echo_energy_threshold", 2.0)),
            # Pass persistent-mode flag so echo_detector skips its
            # 3.5s stale-clear timer in persistent hot-window — see
            # `EchoDetector.track_tts_finish` for the bug history.
            hot_window_persistent=bool(getattr(self.cfg, "hot_window_persistent", False)),
        )

        self.state_manager = StateManager(
            hot_window_seconds=float(getattr(self.cfg, "hot_window_seconds", 3.0)),
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            voice_collect_seconds=float(getattr(self.cfg, "voice_collect_seconds", 3.0)),
            max_collect_seconds=float(getattr(self.cfg, "voice_max_collect_seconds", 60.0)),
            hot_window_persistent=bool(getattr(self.cfg, "hot_window_persistent", False)),
            # Persistent-session safety nets — kill switch for runaway
            # echo loops that "talk to themselves" forever. These two
            # ceilings are the only way out of a persistent session
            # besides a stop command or HUD End-session click.
            hot_window_max_session_seconds=float(
                getattr(self.cfg, "hot_window_max_session_seconds", 1800.0)
            ),
            hot_window_max_idle_seconds=float(
                getattr(self.cfg, "hot_window_max_idle_seconds", 180.0)
            ),
        )

        # Energy tracking for echo detection
        self._recent_audio_energy: deque = deque(maxlen=50)

        # Audio-level wake word detection timestamp
        self._wake_timestamp: Optional[float] = None

        # Streaming-reply abort flag — set by any tts.interrupt() call site
        # AND by stop-command handlers. Checked at the top of
        # `_flush_sentence` and inside the `iter_lines` loop of
        # `_voice_direct_chat` so that interrupting Jarvis mid-reply also
        # stops the LLM stream from queuing more sentences to speak.
        # Cleared at the start of every new `_voice_direct_chat` call.
        self._stream_abort = threading.Event()

        # Provenance of the next dispatch — used by `_persist_memory_pair`
        # to decide whether to write the exchange to disk AND to enrich
        # the memory record with where the query came from. Set by the
        # call sites of `_dispatch_query` BEFORE the call:
        #   "wake"             → explicit wake word in transcript
        #   "wake_collection"  → collection started by wake, completed
        #                        on a later transcript (timeout/finalise)
        #   "hot_window"       → persistent hot-window follow-up gated
        #                        by intent judge directed=True
        # `_dispatch_source` is the "next dispatch's source", set by
        # call sites BEFORE invoking `_dispatch_query`. Cleared
        # ATOMICALLY at the top of `_dispatch_query` into a local stored
        # as `_active_dispatch_source` for the persist helper to read.
        # This split prevents the cross-turn leak that happened pre-
        # audit-round-6 (early-return paths in `_dispatch_query`
        # skipped `_persist_memory_pair`, so the old write-side clear
        # never fired → stale "wake" tag poisoned the next turn).
        self._dispatch_source: Optional[str] = None
        self._active_dispatch_source: Optional[str] = None

        # Rolling transcript buffer for context-aware processing
        # Used for both retention and context passed to intent judge
        self._buffer_duration = float(getattr(self.cfg, "transcript_buffer_duration_sec", 120.0))
        self._transcript_buffer = TranscriptBuffer(max_duration_sec=self._buffer_duration)
        debug_log(f"transcript buffer initialised ({self._buffer_duration}s)", "voice")

        # Intent judge (full context, larger model) - always used when available
        self._intent_judge = create_intent_judge(self.cfg)
        if self._intent_judge is not None:
            debug_log(f"intent judge initialised (model: {self._intent_judge.config.model})", "voice")
        else:
            debug_log("intent judge unavailable, using simple wake word detection", "voice")

        # Thinking tune player
        self._tune_player: Optional = None

        # Active reply language. Defaults to RU (post May 16 uk→ru migration).
        # Switched explicitly via detect_language_switch() + user confirmation.
        # Resets to RU on force_end_session().
        # CRITICAL — this is the main control of which language the LLM is
        # asked to reply in. Was "uk" → caused "speaks UA still" bug after
        # whisper_language was migrated to "ru".
        self._active_language: str = "ru"
        self._pending_lang_switch: Optional[str] = None
        self._pending_action = None
        self._pending_upgrade = None
        self._pending_confirmation = None
        # Lock around the cross-thread pending-* state. HUD watcher
        # thread mutates `_pending_confirmation`/`_pending_action`;
        # dispatch thread reads + clears them. Pre-existing risk per
        # round-11 audit: read-then-clear sequence without lock can
        # double-fire on simultaneous HUD click + voice arrival.
        self._pending_lock = threading.Lock()

        # Persistent dialog memory — JSONL file in ~/.config/jarvis/memory/.
        # User asked: "нехай зробиться окремий мозочок ... він там записує
        # всі наші розмови в зжатому форматі щоб він завжди міг памятати
        # контекст за нашу сесію та між сесіями".
        # Each user/assistant exchange is appended as one JSON line, and
        # the most recent N pairs are auto-loaded at startup so Jarvis
        # remembers across daemon restarts. Compression: we keep only the
        # text (no metadata bloat) and the per-line tail-load is O(1).
        self._init_persistent_memory()

    def _consume_pending_confirmation_from_hud(self) -> None:
        """Execute or cancel a pending action when the HUD set the flag.

        Audit round 11 fix C1: the HUD ✓/✗ buttons only set
        ``_pending_confirmation`` and the dispatch path consumed the
        flag — but the dispatch path only fires on voice arrival. So a
        user clicking ✓ without speaking would see no effect. This
        helper runs from the HUD watcher thread immediately after the
        flag is set; it fires the pending fn synchronously and speaks
        the outcome, matching what the voice path would do.

        The function is intentionally tolerant: missing pending action,
        TTS not initialised, or fn raising are all swallowed with a
        debug log — better silent skip than a watcher-thread crash.
        """
        try:
            with self._pending_lock:
                choice = self._pending_confirmation
                pending = self._pending_action
                if choice not in ("yes", "no") or pending is None:
                    # Lang switch / upgrade flow is consumed by the
                    # voice dispatch path; HUD direct-execute only
                    # covers the action-fn flow.
                    return
                # Clear flags BEFORE running fn so a slow fn doesn't
                # get double-fired by a fast second click.
                self._pending_confirmation = None
                self._pending_action = None
            if choice == "yes":
                debug_log(f"executing pending action via HUD direct: {pending.name}", "voice")
                try:
                    ok, msg = pending.fn()
                except Exception as e:
                    # R34-S51 — RU-only TTS policy.
                    ok, msg = False, f"Ошибка при выполнении действия: {e}"
                self._speak_and_continue(msg if ok else f"Не получилось. {msg}")
            else:
                debug_log(f"cancelled pending action via HUD direct: {pending.name}", "voice")
                self._speak_and_continue("Отменено.")
        except Exception as e:
            debug_log(f"_consume_pending_confirmation_from_hud failed: {e}", "voice")

    def _init_persistent_memory(self) -> None:
        """Load recent dialog history from the persistent memory file.

        Writes go to one JSONL per calendar day so we can rotate easily.
        On startup we load the most recent 8 exchanges (16 messages) so
        the model has context immediately — no cold start required.
        """
        import os
        import json
        from datetime import datetime
        mem_dir = os.path.expanduser("~/.config/jarvis/memory")
        # Audit round 21 fix (F28): protect ``_dialog_history`` with a
        # dedicated lock. The list is touched from at least three
        # threads — the voice direct-chat reply path (append + slice
        # trim), the HUD control watcher (``clear()`` on end_session
        # and confirmation flows), and the TTS completion callback
        # (also can reach back via the pending-action mechanism).
        # Under GIL the individual ``append``/``pop``/slice ops are
        # safe, but the "trim to last 16" pattern is two distinct
        # operations — a concurrent reader sees a 17-element history
        # briefly and ``messages.extend(self._dialog_history[-2:])``
        # in ``_voice_direct_chat`` can grab two messages of mixed
        # vintage. The lock makes the invariant atomic.
        self._dialog_history_lock = threading.RLock()
        try:
            os.makedirs(mem_dir, exist_ok=True)
        except Exception as e:
            debug_log(f"could not create memory dir: {e}", "voice")
            self._dialog_history: list[dict] = []
            self._memory_file = None
            return
        # One file per day — keeps individual files manageable.
        # Audit round 12 fix: also remember the directory so
        # `_persist_memory_pair` can recompute the per-day path on
        # every write. The previous implementation froze today's date
        # at startup; a daemon running across midnight kept appending
        # to yesterday's file until restart (data ends up in the
        # wrong dialog-YYYY-MM-DD.jsonl and the per-file size grows
        # unbounded since there's no rotation logic for it).
        self._memory_dir = mem_dir
        today = datetime.now().strftime("%Y-%m-%d")
        self._memory_file = os.path.join(mem_dir, f"dialog-{today}.jsonl")
        # Also remember yesterday's file for cross-day context loading.
        loaded: list[dict] = []
        # Load most recent across last 2 days (today + yesterday).
        files_to_load: list[str] = []
        try:
            all_files = sorted(
                [f for f in os.listdir(mem_dir) if f.startswith("dialog-") and f.endswith(".jsonl")],
                reverse=True,
            )
            files_to_load = [os.path.join(mem_dir, f) for f in all_files[:2]]
        except Exception:
            pass
        # Audit round 7 fix M2: use a bounded deque so we never load
        # tens of thousands of lines into a Python list just to slice
        # off the last 16. With `maxlen=200` we keep enough headroom
        # for the lang-filter to drop wrong-language records and still
        # hand `recent = loaded[-16:]` a meaningful tail.
        from collections import deque
        rolling: deque = deque(maxlen=200)
        for path in reversed(files_to_load):  # oldest first so latest end up at tail
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if obj.get("role") in ("user", "assistant") and obj.get("content"):
                                # Audit round 7 fix M1: preserve `lang`
                                # and `source` metadata when present.
                                # The filter below now prefers explicit
                                # `lang` over char-heuristic so a record
                                # tagged `lang:"uk"` from a UA session
                                # is correctly dropped under active=ru.
                                rec = {"role": obj["role"], "content": obj["content"]}
                                if obj.get("lang"):
                                    rec["lang"] = obj["lang"]
                                if obj.get("source"):
                                    rec["source"] = obj["source"]
                                rolling.append(rec)
                        except Exception:
                            continue
            except Exception as e:
                debug_log(f"failed to load memory from {path}: {e}", "voice")
        loaded = list(rolling)
        # Keep only most recent 16 messages (8 pairs).
        recent = loaded[-16:]
        # Filter contaminated entries from the WRONG language — they pollute
        # the model's language anchor (it sees prior UA replies in history
        # and continues UA even when current system prompt asks for RU).
        # User report (May 16): "він всерівно спілкується українською" even
        # after full uk→ru migration. Root cause: old UA dialog files were
        # loaded as history, model mimicked their language.
        #
        # Strategy: drop assistant messages whose dominant language differs
        # from the active language (whisper_language config). User messages
        # are kept regardless (user is free to speak any language).
        ua_only = set("іїєґІЇЄҐ")
        ru_only = set("ёыэъЁЫЭЪ")
        active_lang = getattr(self.cfg, "whisper_language", "ru") or "ru"
        def _msg_is_active_lang(msg: dict) -> bool:
            """Return True if `msg` matches the current active language.

            Audit round 7 fix M1: prefer the explicit `lang` field
            (set by `_persist_memory_pair` since round 5) over the
            char-heuristic. Old records without `lang` still fall
            back to char-counting for backwards compatibility.
            """
            # Prefer explicit metadata when present.
            explicit_lang = msg.get("lang")
            if explicit_lang:
                # EN/DE replies kept regardless of active (rare but
                # legitimate — user may have asked something in EN).
                if explicit_lang not in ("uk", "ru"):
                    return True
                return explicit_lang == active_lang
            text = msg.get("content") or ""
            if not text:
                return False
            ua_hits = sum(1 for c in text if c in ua_only)
            ru_hits = sum(1 for c in text if c in ru_only)
            if active_lang == "ru":
                # RU: drop if has UA-only chars (іїєґ) and no RU-only chars
                if ua_hits > 0 and ru_hits == 0:
                    return False
                # No conflict — accept (could be RU, or generic Cyrillic that's
                # safe to keep)
                return True
            elif active_lang == "uk":
                # UA: drop if has RU-only chars (ёыэъ) and no UA-only chars
                if ru_hits > 0 and ua_hits == 0:
                    return False
                return True
            return True  # other languages (en/de) — no filter
        cleaned: list[dict] = []
        dropped = 0
        for m in recent:
            if m["role"] == "user":
                # User messages: keep regardless of language (user is free
                # to speak any language); strip metadata for prompt size.
                cleaned.append({"role": m["role"], "content": m["content"]})
            elif _msg_is_active_lang(m):
                # Assistant message in the right language — strip
                # metadata fields before passing to the LLM (model
                # doesn't need `lang`/`source` in the chat history).
                cleaned.append({"role": m["role"], "content": m["content"]})
            else:
                dropped += 1
        self._dialog_history: list[dict] = cleaned
        debug_log(
            f"persistent memory loaded: {len(self._dialog_history)} messages "
            f"from {len(files_to_load)} file(s) (dropped {dropped} wrong-lang, active={active_lang})",
            "voice",
        )

    # `_MEMORY_BARE_JUNK` removed in audit round 6 — replaced by the
    # module-level `BARE_JUNK_SET` so all three gates (collection-mode,
    # canned-reply, persist-memory) reference one source of truth.

    def _persist_memory_pair(self, query: str, reply: str) -> None:
        """Append a (user, assistant) pair to today's memory file.

        Quality filters — applied BEFORE writing so junk never enters
        the long-term context. The polluted 144-line dialog file we
        had to archive on 2026-05-16 came from skipping these checks:
          • bare-junk queries ("спасибо", "ок", "пока") got persisted
            with canned-reply pleasantries → re-anchored the LLM on
            restart into content-free responses
          • Whisper hallucinations ("Субтитры от...") got persisted
            as user turns → the model would later "remember" them
            and reference "subtitles" in unrelated replies
          • dispatches without wake confirmation (`_dispatch_source`
            unset or "collection_timeout") slip through as user turns
            even though they're often misheard ambient speech

        Metadata block records provenance so a future audit can
        distinguish wake-confirmed turns from low-confidence ones and
        we can selectively scrub by source rather than nuking the
        whole file. The `ts` field is per-record (paired) so a manual
        scrub can use file mtime + content to identify suspect runs.

        Always clears `self._dispatch_source` on return so a missed
        call site can't leak provenance from a prior turn into the
        next persist.
        """
        # ── Read provenance set at top of `_dispatch_query` ───────────
        # `_active_dispatch_source` is set once per dispatch and lives
        # for the lifetime of the dispatch (read-only here). The "next
        # dispatch source" intent (`self._dispatch_source`) is cleared
        # at the top of `_dispatch_query`. Falls back to "unknown" if
        # called from outside `_dispatch_query` (defence in depth).
        source = getattr(self, "_active_dispatch_source", None) or "unknown"

        if not self._memory_file:
            return
        if not query or not reply:
            debug_log(f"persist-memory: refused empty pair (q={bool(query)} r={bool(reply)})", "voice")
            return

        q_strip = query.strip().lower().rstrip("?.!,;:")
        # ── Quality filter 1: bare-junk user query ────────────────────
        if q_strip in BARE_JUNK_SET:
            debug_log(
                f"persist-memory: refused bare-junk query '{q_strip}' "
                f"(source={source})",
                "voice",
            )
            return
        # ── Quality filter 2: too-short user query (<3 chars / 1 char alpha) ─
        if len(q_strip) < 3:
            debug_log(
                f"persist-memory: refused too-short query '{q_strip}' "
                f"(source={source})",
                "voice",
            )
            return
        # ── Quality filter 3: dispatch came from low-confidence source ─
        # `collection_timeout` means the user opened collection (often
        # via misheard fuzzy wake) and then said nothing or said
        # ambient noise that timed out. Reply went out (canned ack),
        # but the user-turn text is rarely something we should
        # remember. `unknown` is the fail-safe — any new dispatch
        # site we add must explicitly set `_dispatch_source` to opt-in.
        if source in ("collection_timeout", "unknown"):
            debug_log(
                f"persist-memory: refused low-confidence source '{source}' "
                f"query='{q_strip[:40]}'",
                "voice",
            )
            return

        # ── Quality filter 4: reply is a hard-fail placeholder ────────
        # Only refuse persistence for "I tried and failed" failure
        # responses — those would train the rolling history to expect
        # more failures. Pre-audit-round-6 this list also included
        # "не понял" / "не зрозумів" / "не понимаю" / "не розумію" —
        # but those are legitimate ask-for-clarification responses
        # that anchor the model to be a good listener
        # ("Не зрозумів, повтори будь ласка. Що саме?"). Dropping them
        # made Jarvis re-ask the same clarification over and over
        # because he never "remembered" he just asked. Kept ONLY
        # operational-failure prefixes here.
        r_strip = reply.strip().lower()
        _APOLOGY_PREFIXES = (
            "не вийшло", "не зміг", "не получилось", "не смог",
            "вибач, не вийшло", "извини, не получилось",
            "sorry, i couldn't",
            # The system speaks this when LLM call times out — pure
            # placeholder, never useful as memory.
            "вибач, не встиг подумати",
            "извини, не успел подумать",
        )
        if any(r_strip.startswith(p) for p in _APOLOGY_PREFIXES):
            debug_log(
                f"persist-memory: refused hard-fail reply '{r_strip[:40]}' "
                f"(source={source})",
                "voice",
            )
            return

        # ── All filters passed — write with provenance metadata ───────
        import json
        import time as _t
        ts = _t.time()
        lang = getattr(self, "_active_language", "ru")
        # Audit round 12 fix: recompute today's path on each write so
        # a daemon running across midnight rolls over into the new day's
        # file. Cheap: one strftime + os.path.join per persist.
        mem_dir = getattr(self, "_memory_dir", None)
        if mem_dir:
            today = datetime.now().strftime("%Y-%m-%d")
            self._memory_file = os.path.join(mem_dir, f"dialog-{today}.jsonl")
        try:
            with open(self._memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {
                        "role": "user",
                        "content": query,
                        "ts": ts,
                        "source": source,
                        "lang": lang,
                    },
                    ensure_ascii=False,
                ) + "\n")
                f.write(json.dumps(
                    {
                        "role": "assistant",
                        "content": reply,
                        "ts": ts,
                        "source": source,
                        "lang": lang,
                    },
                    ensure_ascii=False,
                ) + "\n")
            debug_log(
                f"persist-memory: wrote pair (source={source}, "
                f"q={len(query)}c r={len(reply)}c)",
                "voice",
            )
        except Exception as e:
            debug_log(f"failed to persist memory: {e}", "voice")
        finally:
            # Audit round 7 (regression find R4): clear the active
            # source so a future caller invoking persist outside the
            # `_dispatch_query` lifecycle can't pick up a stale value
            # from the prior turn. The defence-in-depth claim in the
            # docstring is now actually enforced.
            self._active_dispatch_source = None

    def stop(self) -> None:
        """Stop the voice listener."""
        self._should_stop = True
        self.state_manager.stop()
        self._stop_thinking_tune()

    def _start_thinking_tune(self) -> None:
        """Start the thinking tune when processing a query."""
        if (self.cfg.tune_enabled and
            self._tune_player is None and
            (self.tts is None or not self.tts.is_speaking())):
            from ..output.tune_player import TunePlayer
            self._tune_player = TunePlayer(enabled=True)
            self._tune_player.start_tune()

    def _stop_thinking_tune(self) -> None:
        """Stop the thinking tune. Caller decides what state to set next.

        Previously this forcibly set state to IDLE, which caused the HUD
        to collapse captions for ~50ms between THINKING and SPEAKING
        (or LISTENING after hot window activation). User report:
        "після відповіді закривається". The IDLE flicker was the trigger
        — daemonActive went false → captions fade timer fired → next
        SPEAKING state didn't re-show captions until a sentence event
        arrived (which is too late for short responses).
        """
        if self._tune_player is not None:
            self._tune_player.stop_tune()
            self._tune_player = None

    def _is_thinking_tune_active(self) -> bool:
        """Check if thinking tune is currently active."""
        return self._tune_player is not None and self._tune_player.is_playing()

    def _set_face_state_listening(self) -> None:
        """Set the desktop face widget to LISTENING state."""
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.LISTENING)
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"failed to set face state to LISTENING: {e}", "voice")

    def track_tts_start(self, tts_text: str) -> None:
        """Called when TTS starts speaking."""
        if self.tts and self.tts.enabled:
            # Calculate baseline energy from recent audio samples
            baseline_energy = 0.0045  # default
            if self._recent_audio_energy:
                baseline_energy = sum(self._recent_audio_energy) / len(self._recent_audio_energy)

            self.echo_detector.track_tts_start(tts_text, baseline_energy)

    def _interrupt_tts(self, reason: str = "") -> None:
        """Stop current TTS AND abort any in-flight streaming LLM reply.

        Every caller that previously did `self.tts.interrupt()` standalone
        was leaking the rest of a streamed reply: the iter_lines loop in
        `_voice_direct_chat` kept pulling tokens, `_flush_sentence` kept
        queueing them into Piper, and Jarvis kept talking over the user
        for another 5-15 seconds after the "interrupt". This helper sets
        `self._stream_abort` so both the iter_lines loop and the next
        `_flush_sentence` call short-circuit immediately.

        Safe to call when there's no TTS engine, no active reply, or
        no streaming in flight — it's a no-op in those cases. `reason`
        is logged only when debug is enabled.
        """
        # Always set the abort flag first so that if `tts.interrupt()`
        # races with `_flush_sentence` reading the flag (different
        # threads — TTS callback vs HTTP stream consumer), the flag is
        # already visible by the time interrupt unblocks the audio.
        try:
            self._stream_abort.set()
        except Exception:
            pass
        if self.tts:
            try:
                self.tts.interrupt()
            except Exception as e:
                debug_log(f"tts.interrupt failed: {e}", "voice")
        if reason:
            debug_log(f"_interrupt_tts: {reason}", "voice")

    def activate_hot_window(self) -> None:
        """Activate hot window after TTS completion."""
        debug_log("TTS completed, checking hot window activation", "voice")

        if not self.cfg.hot_window_enabled:
            debug_log("hot window disabled in config, skipping", "voice")
            return

        # Track TTS finish time for echo detection
        self.echo_detector.track_tts_finish()

        # Push HUD to LISTENING immediately — don't wait for the
        # echo_tolerance delay (which would flash IDLE in between).
        # User complaint: "після короткої відповіді він всерівно
        # згортається і відкривається по новому". The state_manager's
        # delayed _activate() would set LISTENING after echo_tolerance,
        # but in the meantime SPEAKING→IDLE→LISTENING was visible as
        # a coin pop. Setting LISTENING here closes the gap.
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.LISTENING)
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"failed to pre-set HUD to LISTENING after TTS: {e}", "voice")

        # Schedule delayed hot window activation
        debug_log(f"scheduling hot window activation (echo_tolerance={self.state_manager.echo_tolerance}s, hot_window={self.state_manager.hot_window_seconds}s)", "voice")
        self.state_manager.schedule_hot_window_activation(self.cfg.voice_debug)

    def _process_transcript(self, text: str, utterance_energy: float = 0.0, utterance_start_time: float = 0.0, utterance_end_time: float = 0.0) -> None:
        """
        Process a transcript from speech recognition.

        Args:
            text: Transcribed text from audio
            utterance_energy: Pre-calculated energy from the utterance frames
        """
        if not text or not text.strip():
            # Check for timeouts
            if self.state_manager.check_collection_timeout():
                query = self.state_manager.clear_collection()
                if query.strip():
                    # Collection was started by an earlier wake, so the
                    # collected query is wake-confirmed even though the
                    # final dispatch fires on a silent transcript that
                    # triggers the timeout. Tag accordingly — but only
                    # if a more-specific source (e.g. "hot_window")
                    # wasn't already set by the path that opened the
                    # collection. Without the None-check, a hot-window
                    # follow-up would be re-tagged "wake_collection"
                    # and lose its hot-window provenance in memory.
                    if self._dispatch_source is None:
                        self._dispatch_source = "wake_collection"
                    self._dispatch_query(query)

            # Check hot window expiry
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        text_lower = text.strip().lower()

        # Block known-bad Whisper hallucination patterns BEFORE any further
        # processing — they spam intent-judge / fuzzy passes with garbage.
        if self._is_known_hallucination(text):
            debug_log(f"rejected known hallucination: '{text[:80]}...'", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Min-duration filter for AMBIENT (not in hot-window, no TTS).
        #
        # Original threshold was 1.2s — too aggressive after RU migration.
        # Single-word "Джарвис" (3 syllables, ~0.4-0.7s) was being dropped
        # as ambient. Real failing case from log:
        #   'ambient drop (<1.2s, no wake): 'Джай' dur=0.62s'
        # — Whisper heard real "Джарвис" as "Джай", fuzzy ratio
        # ('джарвис' ~ 'джай') = 0.545 < 0.78 threshold → not recognized
        # as wake → ambient drop. So the user's actual wake call gets
        # silently swallowed.
        #
        # Tighter threshold (0.4s) catches single-frame breath / clicks
        # but lets short wake-words through. Family-chatter false-positive
        # defence is now carried by voice_min_energy + RU hallucination
        # pattern blocklist below.
        utterance_duration = (
            (utterance_end_time - utterance_start_time)
            if (utterance_end_time and utterance_start_time)
            else 0.0
        )
        try:
            _in_hot = self.state_manager.was_speech_during_hot_window(
                utterance_start_time, utterance_end_time
            )
        except Exception:
            _in_hot = False
        _tts_active = bool(self.tts and self.tts.is_speaking())
        if (
            utterance_duration > 0
            and utterance_duration < 0.4
            and not _in_hot
            and not _tts_active
            and len(text.strip()) < 25  # very short text + very short audio
        ):
            # Quick wake-word check — let real isolated "Джарвіс" through.
            # USE THE CONFIG'D fuzzy ratio (default 0.78, user-tuned 0.68).
            # The hardcoded 0.78 here was the user's exact "doesn't fire"
            # bug: Whisper outputs "Джагуис"/"Драйвис" (ratio 0.71) which
            # passes everywhere ELSE in the codebase using cfg 0.68, but
            # was silently dropped here.
            _ww_quick = getattr(self.cfg, "wake_word", "jarvis")
            _wal_quick = list(set(getattr(self.cfg, "wake_aliases", [])) | {_ww_quick})
            _wfr_quick = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
            if not is_wake_word_detected(text.lower(), _ww_quick, _wal_quick, _wfr_quick):
                debug_log(
                    f"ambient drop (<0.4s, no wake): '{text.strip()[:60]}' dur={utterance_duration:.2f}s",
                    "voice",
                )
                try:
                    self._transcript_buffer.mark_segment_processed(text.strip().lower())
                except Exception:
                    pass
                return

        # Reset wake timestamp — it must reflect only the current utterance.
        # If this utterance contains a wake word, the early-beep check below
        # will set it. Without this reset, a prior rejected wake-worded
        # utterance would vouch for subsequent unrelated utterances via the
        # `_wake_timestamp is not None` guard in the intent-judge accept path.
        self._wake_timestamp = None

        start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
        end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3] if utterance_end_time > 0 else "N/A"
        debug_log(f"heard: '{text}' (utterance from {start_time_str} to {end_time_str})", "voice")

        # Track if this input was received during TTS.
        # CRITICAL: check both (a) TTS currently speaking AND (b) whether
        # the utterance start time falls within a recent TTS window.
        # Previously this only checked is_speaking() at processing time,
        # which is False by the time Whisper finishes — so echo audio
        # captured during TTS leaked past the gate and was processed as
        # a real query. User report: TTS said "Слухаю, Даниле", mic
        # picked it up, Whisper transcribed it as "Привінжер", echo
        # check failed (text didn't match TTS), and "Привінжер" got
        # dispatched as a query.
        received_during_tts = bool(self.tts and self.tts.is_speaking())
        if not received_during_tts:
            try:
                # Audit round 14 fix C3: snapshot the TTS window
                # atomically — reading start_time and finish_time
                # separately races with track_tts_start/_finish.
                tts_start_t, tts_finish_t, _, _ = (
                    self.echo_detector.snapshot_tts_window()
                )
                tts_finish_t = tts_finish_t or 0.0
                tts_start_t = tts_start_t or 0.0
                tol = float(self.echo_detector.echo_tolerance or 0.3)
                # Utterance overlaps a TTS window if it started before
                # TTS finished (+tolerance) AND ended after TTS started.
                if (
                    tts_finish_t > 0
                    and utterance_start_time > 0
                    and utterance_start_time < (tts_finish_t + tol)
                    and (utterance_end_time <= 0 or utterance_end_time > tts_start_t - tol)
                ):
                    received_during_tts = True
                    debug_log(
                        f"utterance overlapped recent TTS window — flagging as during-TTS",
                        "voice",
                    )
            except Exception:
                pass

        # ─── PRE-EMPTIVE WAKE-WORD INTERRUPT ────────────────────────────
        # User asked: "коли я знову говорю йому Джарвіс він повинен
        # зупинятися і починати слухати". Previous wake-interrupt check
        # ran AFTER echo-rejection — but echo-rejection sometimes
        # classified "Джарвіс" as part of the TTS echo and dropped it
        # before the wake check ever ran. So we now check FIRST, before
        # any echo logic, whenever TTS is speaking. If we see a wake
        # word, abort TTS and treat the remainder as the new query.
        if received_during_tts:
            _ww = getattr(self.cfg, "wake_word", "jarvis")
            _wal = list(set(getattr(self.cfg, "wake_aliases", [])) | {_ww})
            _wfr = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
            # HEAD CHECK (May 16): pre-emptive interrupt only fires when
            # the wake word is in the FIRST 4 tokens. Whisper sometimes
            # mishears TTS echo as "...джарвіс..." mid-sentence (e.g.
            # "Добавил субтитры джарвіс джарвіс джарвіс") and used to
            # spuriously interrupt TTS + dispatch garbage as a new query.
            _head_text = " ".join(text_lower.split()[:2])
            if is_wake_word_detected(_head_text, _ww, _wal, _wfr):
                debug_log(f"pre-emptive wake-interrupt during TTS: '{text_lower[:60]}'", "voice")
                # R34-S55.1 Phase 8a (P1 security): _vprint+_safe_user_text
                # so the raw user transcript only lands in ``out.log`` when
                # voice-debug is explicitly enabled.
                _vprint(f"  ⏸  Wake interrupt: \"{_safe_user_text(text_lower, 60)}\"", flush=True)
                # Confirmed wake → reset persistent-session idle timer.
                try:
                    self.state_manager.mark_user_wake()
                except Exception:
                    pass
                # Use the helper so any streaming reply from the previous
                # turn aborts cleanly — without this, sentence #2/#3 of
                # the prior answer would keep playing after wake.
                self._interrupt_tts(reason=f"wake re-fire: {text_lower[:40]}")
                try:
                    while not self._audio_q.empty():
                        self._audio_q.get_nowait()
                except Exception:
                    pass
                # Clear echo-detector's "last_tts" so the next utterance
                # isn't classified as echo of the interrupted TTS.
                # Audit round 14 fix C3: use helper so the write goes
                # through _tts_state_lock instead of racing track_tts_start.
                try:
                    self.echo_detector.clear_last_tts_text()
                except Exception:
                    pass
                from .wake_detection import extract_query_after_wake
                extracted = extract_query_after_wake(text_lower, _ww, _wal).strip()
                if extracted and len(extracted) > 2:
                    # Explicit wake + inline query — highest-confidence
                    # source. Tag for persist-memory provenance.
                    self._dispatch_source = "wake"
                    self._dispatch_query(extracted)
                else:
                    # Just the wake word — open collection for follow-up.
                    self.state_manager.start_collection("")
                    self._set_face_state_listening()
                return

            # No wake word and we DID overlap TTS — this is echo OR
            # user speaking simultaneously over Jarvis. Reject hard so
            # garbled Whisper output of TTS echo ("Привінжер" from
            # "Слухаю, Даниле") doesn't reach intent-judge → dispatch.
            # User can always re-say "Джарвіс" to interrupt cleanly.
            session_stop_words_quick = getattr(
                self.cfg, "session_stop_commands",
                ["стоп", "досить", "тиша", "замовкни"],
            )
            if not is_stop_command(text_lower, session_stop_words_quick):
                debug_log(
                    f"rejected utterance overlapping TTS window (no wake, no stop): '{text_lower[:80]}'",
                    "voice",
                )
                # R34-S55.1 Phase 8a: gate + scrub raw transcript.
                _vprint(f"  🔇 Dropped TTS-overlap audio: \"{_safe_user_text(text_lower, 50)}\"", flush=True)
                # CRITICAL: also mark as processed in transcript_buffer.
                # Without this, the rejected TTS-echo segment stays as
                # context for the NEXT utterance's intent judge, which
                # then "inlines references" from the garbage echo. User
                # report: "відбувається привязка розмови до старих запитів".
                try:
                    self._transcript_buffer.mark_segment_processed(text_lower)
                except Exception:
                    pass
                return

        # Audit round 23 fix (F39): POST-TTS extended echo check.
        # Round 22 (F35) extended ``EchoDetector.should_reject_as_echo``
        # after-TTS window from 3s to 12s — but the listener only
        # called the detector with ``is_during_tts=True`` (line ~1501
        # below), so the extended window was unreachable dead code.
        # Live evidence (events.jsonl seq=288):
        #   Jarvis TTS "Джарвис здесь." finished at 12:00:06
        #   Whisper stt_final "здесь." at 12:00:16  (Δ=10s)
        # That "здесь." (single common word, lifted from Jarvis-own
        # TTS) reached the LLM and triggered the "Нило, ты где?"
        # nonsense reply. The 10s gap fits within F35's 12s window
        # but no caller invoked it.
        #
        # Now we explicitly call it here, ONLY when the utterance
        # didn't already overlap the TTS window (the overlap path
        # above handles its own rejection). Strict threshold 92
        # via in_hot_window=True avoids false rejections of
        # legitimate user follow-ups that share a single common
        # word with TTS.
        if not received_during_tts:
            try:
                if self.echo_detector.should_reject_as_echo(
                    text_lower,
                    utterance_energy,
                    is_during_tts=False,
                    tts_rate=getattr(self.cfg, "tts_rate", 200),
                    utterance_start_time=utterance_start_time,
                    in_hot_window=True,
                ):
                    debug_log(
                        f"rejected delayed echo (post-TTS extended window): '{text_lower[:80]}'",
                        "echo",
                    )
                    print(
                        f"  🔇 Dropped delayed echo: \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"",
                        flush=True,
                    )
                    try:
                        self._transcript_buffer.mark_segment_processed(text_lower)
                    except Exception:
                        pass
                    return
            except Exception as _echo_exc:
                debug_log(f"post-TTS echo check raised: {_echo_exc}", "echo")

        # --- Early echo check + early beep ---
        # Check for echo BEFORE starting beep and BEFORE intent judge.
        # This prevents: false beeps on echo, intent judge blocking the audio
        # loop for seconds on echo, and hot window extending from echo resets.
        if not received_during_tts and not self._is_thinking_tune_active():
            in_hot_window = self.state_manager.was_speech_during_hot_window(
                utterance_start_time, utterance_end_time
            )
            if in_hot_window:
                # Fuzzy echo check — instant, no intent judge needed.
                # Only catches pure echo (transcript ≈ TTS text). Mixed
                # echo+speech chunks (user spoke over echo) go to the
                # intent judge which can extract the user's speech.
                # Audit round 14 fix C3: snapshot under lock.
                _, _, last_tts_text, _ = self.echo_detector.snapshot_tts_window()
                last_tts_text = last_tts_text or ""
                if last_tts_text:
                    echo_score = fuzz.partial_ratio(
                        text_lower, last_tts_text.lower()
                    )
                    tts_words = len(last_tts_text.split())
                    text_words = len(text_lower.split())
                    is_pure_echo = (
                        echo_score >= 70
                        and text_words <= max(tts_words * 1.3, tts_words + 3)
                    )
                    if is_pure_echo:
                        # Before rejecting, try to salvage user speech appended
                        # after the echo prefix. Whisper commonly merges the tail
                        # of TTS echo with the user's follow-up into a single
                        # transcript; without salvage, the user's real speech
                        # would be dropped before the intent judge ever sees it.
                        # Try exact-word cleanup first (cheapest, most precise),
                        # then fall back to the rightmost-boundary scan which
                        # handles Whisper mis-transcriptions at the echo/speech
                        # join ("explores" → "laws") that exact matching can't.
                        salvaged = self.echo_detector.cleanup_leading_echo(text_lower)
                        if salvaged == text_lower:
                            salvaged_alt = self.echo_detector.salvage_after_echo_tail(text_lower)
                            if salvaged_alt:
                                salvaged = salvaged_alt
                        # Require ≥ min_salvage_words to avoid treating Whisper's
                        # echo-tail hallucinations ("…regions like Steneti") as
                        # genuine user speech. The threshold lives on the echo
                        # detector so every salvage site shares one policy.
                        min_words = self.echo_detector.min_salvage_words
                        if (salvaged != text_lower
                                and len(salvaged.split()) >= min_words):
                            debug_log(
                                f"salvaged user speech from hot-window echo+speech "
                                f"chunk: '{salvaged}'",
                                "voice",
                            )
                            print(
                                f"  ✂️ Stripped echo prefix, kept: \"{salvaged[:60]}"
                                f"{'...' if len(salvaged) > 60 else ''}\"",
                                flush=True,
                            )
                            self._transcript_buffer.update_last_segment_text(salvaged)
                            # text_lower now carries the salvaged query — the rest
                            # of _process_transcript reads from this variable.
                            text_lower = salvaged
                        else:
                            debug_log(f"🔇 Early echo rejection (score={echo_score}): \"{text_lower}\"", "voice")
                            _vprint(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                            return

                # Non-echo (or salvaged) in hot window — start beep
                self._start_thinking_tune()
                self._set_face_state_listening()
                debug_log("early beep: hot window active", "voice")
            else:
                # Not in hot window — check for wake word
                wake_word = getattr(self.cfg, "wake_word", "jarvis")
                aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
                # Wake-position guard — wake word must be in FIRST 2
                # tokens of the utterance (was 4, tightened May 16).
                # Defends against Whisper hallucinations like "Добавил
                # субтитры джарвіс джарвіс джарвіс" where the wake word
                # appears after content words. Real users say "Джарвис,
                # ..." or "Ну, Джарвис, ..." — never with 3+ content
                # words before the wake.
                all_aliases_list = list(set(aliases) | {wake_word})
                tokens = text_lower.split()
                head = " ".join(tokens[:2]).lower()
                if (
                    is_wake_word_detected(text_lower, wake_word, aliases, fuzzy_ratio)
                    and is_wake_word_detected(head, wake_word, aliases, fuzzy_ratio)
                ):
                    self._wake_timestamp = utterance_start_time
                    self._start_thinking_tune()
                    self._set_face_state_listening()
                    debug_log("early beep: wake word detected (in head)", "voice")
                elif is_wake_word_detected(text_lower, wake_word, aliases, fuzzy_ratio):
                    debug_log(
                        f"wake word found mid-sentence (not in first 4 tokens) — ignoring as background speech: '{text_lower[:80]}'",
                        "voice",
                    )
                    # R34-S55.1 Phase 8a: gate + scrub raw transcript.
                    _vprint(f"  🔇 Ignored mid-sentence wake: \"{_safe_user_text(text_lower, 60)}\"", flush=True)
                    self._transcript_buffer.mark_segment_processed(text_lower)
                    return

        # Session-end stop commands — UA "стоп"/"досить"/"тиша"/"завершити сесію".
        # These work BOTH during TTS (interrupt + end session) AND during
        # hot window (just end session). They're the only voice path out
        # of a persistent hot window.
        session_stop_words = getattr(
            self.cfg, "session_stop_commands",
            [
                "стоп", "досить", "тиша", "замовкни",
                "завершити сесію", "вимкнись", "відбій",
                "stop session", "end session",
            ],
        )
        if is_stop_command(text_lower, session_stop_words):
            debug_log(f"session-stop command detected: '{text_lower}'", "voice")
            # R34-S55.1 Phase 8a: gate + scrub raw transcript.
            _vprint(f"  🛑 Session stop: \"{_safe_user_text(text_lower, 60)}\"", flush=True)
            if self.tts and self.tts.enabled and self.tts.is_speaking():
                self._interrupt_tts(reason="session-stop command")
            # Wipe dialog history — new session starts with a clean slate
            # so old context doesn't leak into the next wake.
            # Round 29 (F75): under _dialog_history_lock — the F28 lock
            # made append+trim atomic, but session-stop wiped history
            # outside the lock. Concurrent _voice_direct_chat (line
            # 2651) could read a torn dialog_history slice during the
            # millisecond between clear()-pre and clear()-post.
            if hasattr(self, "_dialog_history"):
                with self._dialog_history_lock:
                    self._dialog_history.clear()
            # Reset language to RU default and clear any pending states.
            #
            # R34-S55.1 Phase 8b (P2 concurrency): the four pending
            # slots are also touched by the HUD watcher and the
            # dispatch path at line ~3884 — clearing them WITHOUT the
            # ``_pending_lock`` (as the prior version did) left a
            # half-cleared state when force_end_session races with
            # a concurrent HUD click or voice dispatch. E.g. we null
            # ``_pending_action`` here while the dispatch path is mid-
            # snapshot, so the dispatch sees the old reference and
            # fires the action even though the user just stopped the
            # session. Lock makes the four-field reset atomic w.r.t.
            # every other reader/writer of these slots.
            self._active_language = "ru"
            with self._pending_lock:
                self._pending_lang_switch = None
                self._pending_action = None
                self._pending_upgrade = None
                self._pending_confirmation = None
            self.state_manager.force_end_session()
            try:
                while not self._audio_q.empty():
                    self._audio_q.get_nowait()
            except Exception:
                pass
            return

        # Echo rejection & stop commands — only while TTS is actively playing.
        # After TTS finishes, the intent judge handles everything (echo detection,
        # hot window follow-ups, etc.) using full transcript context + last TTS text.
        if self.tts and self.tts.enabled and self.tts.is_speaking():
            # Stop command detection (fast, text-based)
            stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
            if is_stop_command(text_lower, stop_commands):
                debug_log(f"stop command detected during TTS: {text_lower} (energy: {utterance_energy:.4f})", "voice")
                self._interrupt_tts(reason=f"stop command: {text_lower[:30]}")
                try:
                    while not self._audio_q.empty():
                        self._audio_q.get_nowait()
                except Exception:
                    pass
                return

            # Echo rejection during active TTS
            should_reject = self.echo_detector.should_reject_as_echo(
                text_lower, utterance_energy, True,
                getattr(self.cfg, 'tts_rate', 200), utterance_start_time
            )
            if should_reject:
                # Try to salvage user speech appended after echo
                salvaged = self.echo_detector.cleanup_leading_echo_during_tts(
                    text_lower,
                    getattr(self.cfg, 'tts_rate', 200),
                    utterance_start_time,
                )
                min_words = self.echo_detector.min_salvage_words
                if (salvaged and salvaged.strip() and salvaged != text_lower
                        and len(salvaged.split()) >= min_words):
                    debug_log(f"salvaged user speech from echo during TTS: '{salvaged}'", "voice")
                    self._transcript_buffer.update_last_segment_text(salvaged)
                    text_lower = salvaged
                else:
                    debug_log(f"echo rejected during TTS: '{text_lower[:50]}'", "echo")
                    _vprint(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                    return

        # Salvage user speech from merged echo+speech chunks.
        # When Whisper delivers a single transcript containing TTS echo followed by
        # user speech (e.g. "I can only provide... Well you can search for it"), the
        # echo portion was captured during TTS but the transcript arrives after TTS
        # finishes. Try to strip the leading echo and use just the user's speech.
        # Skip entirely if there's no prior TTS — nothing to match against.
        # Audit round 14 fix C3: snapshot the TTS window atomically.
        _, last_tts_finish, last_tts_text_for_salvage, _ = (
            self.echo_detector.snapshot_tts_window()
        )
        last_tts_text_for_salvage = last_tts_text_for_salvage or ""
        last_tts_finish = last_tts_finish or 0.0
        # Use echo_tolerance as buffer — speaker/mic latency means the utterance
        # may start slightly after TTS finish yet still contain the echo.
        echo_tol = self.echo_detector.echo_tolerance
        if (last_tts_text_for_salvage and last_tts_finish > 0
                and utterance_start_time > 0
                and utterance_start_time < last_tts_finish + echo_tol):
            salvaged = self.echo_detector._salvage_suffix_from_echo(
                text_lower,
                getattr(self.cfg, 'tts_rate', 200),
                utterance_start_time,
            )
            # If the prefix-based salvage fails or truncates too aggressively
            # (Whisper-mangled echo boundary → exact cleanup misses; fuzzy
            # prefix iteration prefers shortest suffix), fall through to the
            # rightmost-boundary scan which recovers the full follow-up.
            boundary_salvaged = self.echo_detector.salvage_after_echo_tail(text_lower)
            if boundary_salvaged and (
                salvaged is None or salvaged == text_lower
                or len(boundary_salvaged.split()) > len(salvaged.split())
            ):
                salvaged = boundary_salvaged
            min_words = self.echo_detector.min_salvage_words
            if (salvaged and salvaged.strip() and salvaged != text_lower
                    and len(salvaged.split()) >= min_words):
                debug_log(f"salvaged user speech from merged echo+speech chunk: '{salvaged}'", "voice")
                self._transcript_buffer.update_last_segment_text(salvaged)
                text_lower = salvaged

        # Check hot window expiry
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

        # Intent judge — the single decision-maker for all post-TTS input.
        # Gets full transcript context, last TTS text, and hot window state.
        # Handles: echo detection, wake word queries, hot window follow-ups.
        # During active TTS, skip short utterances (<=3 words) as those are
        # handled by stop command detection above.
        is_speaking_now = self.tts and self.tts.is_speaking()
        intent_judgment = None

        # Determine if this could be a hot window follow-up.
        # Only use formal hot window state — no time-based grace period.
        # The state manager already handles the timing (echo_tolerance
        # delay before activation, hot_window_seconds before expiry).
        # A generous grace period caused false hot window claims after
        # the user had already seen "Returning to wake word mode".
        could_be_hot_window = self.state_manager.was_speech_during_hot_window(
            utterance_start_time, utterance_end_time
        )

        # Use the upgraded intent judge if available (with full transcript context)
        # Allow during TTS for longer utterances (>3 words) that might be user responses
        word_count = len(text_lower.split())
        skip_intent_judge_during_tts = is_speaking_now and word_count <= 3

        # R34-S58.3 (P2-C3.4): fast-path stop commands during TTS.
        # An utterance like "стоп Джарвіс заткнись будь ласка" is >3
        # words so it normally falls through to the intent judge —
        # which during TTS pays a 5-15 s timeout just to confirm what
        # any human would call obvious. If the utterance LEADS with a
        # stop word, treat it as immediate stop and skip the judge.
        # Matches the prefixes in the session-end stop list (line ~1562)
        # plus a few Russian variants the user actually says.
        _STOP_PREFIXES = (
            "стоп", "стой", "досить", "хватит",
            "тихо", "замолчи", "замовкни", "прекрати",
        )
        if (
            is_speaking_now
            and not skip_intent_judge_during_tts
            and text_lower.strip().startswith(_STOP_PREFIXES)
        ):
            debug_log(
                f"intent-judge bypass: stop-prefix during TTS → immediate stop "
                f"(\"{text_lower[:40]}{'...' if len(text_lower) > 40 else ''}\")",
                "voice",
            )
            skip_intent_judge_during_tts = True
            # Synthesise a STOP judgment so downstream handlers can
            # short-circuit reply playback exactly like the judge would.
            try:
                from .intent_judge import IntentJudgment
                intent_judgment = IntentJudgment(
                    directed=True,
                    query="",
                    stop=True,
                    confidence="high",
                    reasoning="stop-prefix fast-path during TTS",
                )
            except Exception:
                intent_judgment = None

        # Gate the intent judge on an engagement signal. Without this check the
        # judge was called on every ambient utterance, blocking the audio loop
        # for up to `timeout_sec` on each background chatter — which could
        # cascade into UI freezes when many utterances queued up during a slow
        # or loaded Ollama. The judge adds value only when one of:
        #   1. A wake word was detected in the current utterance
        #   2. We are in (or pending) a hot window following TTS
        #   3. TTS is currently speaking (intent judge can catch responses / stops
        #      that the fast text-based stop command check missed)
        has_engagement_signal = (
            self._wake_timestamp is not None
            or could_be_hot_window
            or is_speaking_now
        )

        if not has_engagement_signal:
            debug_log(
                f"skipping intent judge — no wake word, no hot window, no TTS "
                f"(ambient: \"{text_lower[:40]}{'...' if len(text_lower) > 40 else ''}\")",
                "voice",
            )

        # Wake-word FAST-PATH — skip the intent judge entirely when the
        # wake word fired AT THE START of the current utterance. Background:
        # on shared-model setups (qwen2.5:3b for both chat AND judge), the
        # chat warmup evicts the judge's prefix from Ollama's single
        # KV-cache slot. The first post-warmup wake-word judge call then
        # pays ~30-40s for cold-cache re-evaluation, which the user
        # experiences as "Джарвіс дуже довго включає". Skipping the judge
        # for vocative wake-word matches (where the user is calling out to
        # Jarvis at the start of their utterance) saves 30-40s.
        #
        # Critically, we MUST NOT fast-path a wake-word match buried mid-
        # sentence — that's almost always referential speech ("сказав
        # Джарвісу", "це Джарвіс був") or a household member discussing
        # the daemon, not a directed query. In those cases the leading
        # words ARE NOT a question for Jarvis. Fall through to the regular
        # intent-judge path which has the LLM context to decide.
        if (
            self._wake_timestamp is not None
            and not could_be_hot_window
            and not is_speaking_now
        ):
            from .wake_detection import extract_query_after_wake
            wake_word = getattr(self.cfg, "wake_word", "jarvis")
            aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
            stripped = extract_query_after_wake(text_lower, wake_word, aliases).strip()

            # Decide whether this is a vocative wake (Jarvis at start of
            # utterance, possibly followed by a question) or a mid-sentence
            # mention. We look at the FIRST ~30 chars of the original text:
            # if the wake-word alias appears there, treat as vocative.
            head = text_lower[:30].lower()
            is_vocative = False
            for alias in aliases:
                if alias.lower() in head:
                    is_vocative = True
                    break
            # Also catch Whisper mishearings that fuzzy/prefix matcher saw
            # but that aren't in our alias list (e.g. "джагвис", "жаріс").
            # Heuristic: if the first word is short (3-9 chars) and starts
            # with one of the wake-sound prefixes, treat as vocative.
            #
            # Audit round 11 fix H4: the prior single-char trigger string
            # `"jyчжяeх"` included "я", "е", "х" — Cyrillic letters that
            # commonly start ordinary RU/UA sentences (я хочу, є питання,
            # хочу спати, хто там, етап один). With a wake_timestamp set
            # via fuzzy match elsewhere, ANY first word ≤9 chars starting
            # with one of those letters was tagged vocative and the whole
            # utterance got dispatched as the query. `wake_detection.py`
            # already pruned "я" from its alias table for exactly the same
            # reason.
            # Audit round 14 fix H4: ``ч`` and ``ж`` were still on the
            # single-char trigger — but ordinary RU/UA queries start with
            # those letters constantly ("чому ти не…", "чекай хвилинку",
            # "жінка дзвонила", "чи можеш…"). With a stale wake_timestamp
            # the whole utterance was being treated as vocative and
            # dispatched. The dz-sound mishearings these were trying to
            # catch already match through the explicit ``startswith
            # ("дж", "ya", "ja"))`` prefix below — so dropping ч/ж from
            # the single-char list costs nothing and removes a major
            # false-positive surface. Trigger now: j (Jarvis Latin),
            # y (you / Jarvis YA-mishearing).
            if not is_vocative:
                first_word = head.split()[0] if head.split() else ""
                if (
                    (3 <= len(first_word) <= 9 and first_word[:1] in "jy")
                    or (3 <= len(first_word) <= 9 and first_word.startswith(("дж", "ya", "ja")))
                ):
                    is_vocative = True

            if not is_vocative:
                # Mid-sentence reference — let the LLM judge decide. Don't
                # consume the wake timestamp; let the judge see it.
                debug_log(
                    f"wake-word fast-path skipped: not vocative "
                    f"(head: \"{head}\"), falling through to judge",
                    "voice",
                )
            else:
                # Vocative wake. Extract follow-up:
                # 1. Take portion after the matched wake fragment.
                # 2. If the result is suspiciously long (>60 chars), the
                #    extraction probably failed to find the right boundary
                #    (fuzzy match wasn't in alias list). Treat as bare
                #    wake — enter empty collection mode.
                extracted = stripped
                if len(extracted) > 60:
                    debug_log(
                        f"wake-word fast-path: extracted text too long "
                        f"({len(extracted)} chars), treating as bare wake "
                        f"and entering listen mode",
                        "voice",
                    )
                    extracted = ""

                debug_log(
                    f"wake-word fast-path: judge skipped, extracted='{extracted}'",
                    "voice",
                )
                # R34-S55.1 Phase 8a (P1 security): gate + scrub.
                # ``extracted`` is the post-wake user query — gating
                # through ``_vprint`` keeps it out of world-readable
                # out.log unless voice-debug is enabled.
                _vprint(f"  ⚡ Wake fast-path: \"{_safe_user_text(extracted, 60)}\"" if extracted
                        else "  ⚡ Wake fast-path: (listening for query)", flush=True)
                self.state_manager.cancel_hot_window_activation()
                self._transcript_buffer.mark_segment_processed(text_lower)
                self._clear_audio_buffers()
                self._wake_timestamp = None  # consumed
                if extracted:
                    # We have a full query already ("Джарвіс, привіт" →
                    # "привіт"). Dispatch IMMEDIATELY — skipping the 1s
                    # collection wait that's only useful when we need to
                    # collect a follow-up utterance. This is the path
                    # that gives sub-1s perceived response on canned
                    # greetings.
                    # Wake fast-path = highest confidence; tag for persist.
                    self._dispatch_source = "wake"
                    self._dispatch_query(extracted)
                else:
                    # User said just "Джарвіс" with no follow-up. Enter
                    # collection mode so the NEXT utterance becomes the
                    # query. Collection timeout (configurable, default
                    # 1.0s) decides when to give up and TTS an ack.
                    self.state_manager.start_collection(extracted)
                    self._start_thinking_tune()
                return

        if (
            not skip_intent_judge_during_tts
            and has_engagement_signal
            and self._intent_judge is not None
            and self._intent_judge.available
        ):
            # Get recent transcript segments for context (full buffer)
            context_segments = self._transcript_buffer.get_last_seconds(self._buffer_duration)

            # Get TTS context for echo detection
            # Audit round 14 fix C3: snapshot under lock.
            _, last_tts_finish_time, last_tts_text, _ = (
                self.echo_detector.snapshot_tts_window()
            )
            last_tts_text = last_tts_text or ""
            last_tts_finish_time = last_tts_finish_time or 0.0

            intent_judgment = self._intent_judge.judge(
                segments=context_segments,
                wake_timestamp=self._wake_timestamp,
                last_tts_text=last_tts_text,
                last_tts_finish_time=last_tts_finish_time,
                in_hot_window=could_be_hot_window,
                current_text=text_lower,
            )

            if intent_judgment is not None:
                # Log intent judge decision for user visibility
                mode_str = "hot window" if could_be_hot_window else "wake word"
                if intent_judgment.directed:
                    _vprint(f"  🧠 Intent ({mode_str}): directed → \"{_safe_user_text(intent_judgment.query or text_lower)}\"", flush=True)
                else:
                    _vprint(f"  🧠 Intent ({mode_str}): not directed ({intent_judgment.reasoning})", flush=True)
            else:
                reason = self._intent_judge.last_failure_reason or "no segments or unavailable"
                _vprint(f"  🧠 Intent judge: unavailable ({reason})", flush=True)
                debug_log(f"intent judge returned None — falling back ({reason})", "voice")
                # Hot window fallback: if the early echo check already cleared
                # this text, accept it even without the judge's verdict.
                if could_be_hot_window:
                    # Audit round 14 fix C3: snapshot under lock.
                    _, _, last_tts_text_fb, _ = self.echo_detector.snapshot_tts_window()
                    last_tts_text_fb = last_tts_text_fb or ""
                    is_pure_echo = False
                    if last_tts_text_fb:
                        echo_score = fuzz.partial_ratio(
                            text_lower, last_tts_text_fb.lower()
                        )
                        tts_words = len(last_tts_text_fb.split())
                        text_words = len(text_lower.split())
                        is_pure_echo = (
                            echo_score >= 70
                            and text_words <= max(tts_words * 1.3, tts_words + 3)
                        )
                    if not is_pure_echo:
                        _vprint(f"  🧠 Intent fallback: accepting hot window speech", flush=True)
                        debug_log(f"✅ Hot window fallback (judge unavailable): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        # Hot-window fallback path → tag for persist.
                        self._dispatch_source = "hot_window"
                        self.state_manager.start_collection(text_lower)
                        self._start_thinking_tune()
                        try:
                            _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                        except Exception:
                            pass
                        return

            if intent_judgment is not None:
                # HOISTED (May 17): define wake_word/wake_aliases/fuzzy_ratio
                # at the top of this block so all downstream branches
                # (hot-window guard, override guard, accept path) can use
                # them without re-defining or hitting UnboundLocalError.
                # Previously these were only defined inside the
                # `not could_be_hot_window` sub-branch → my hot-window
                # guard at line ~1806 crashed the audio thread.
                wake_word = getattr(self.cfg, "wake_word", "jarvis")
                wake_aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
                aliases = wake_aliases  # alias for legacy code below

                # If judge says stop command, interrupt TTS
                if intent_judgment.stop and self.tts and self.tts.is_speaking():
                    debug_log(f"🛑 Intent judge detected stop command", "voice")
                    self._interrupt_tts(reason="intent judge stop")
                    return

                # If directed with query, process it
                if intent_judgment.directed and intent_judgment.query:
                    # In wake word mode, verify the wake word is actually present
                    # The LLM sometimes hallucinates wake words that don't exist
                    if not could_be_hot_window:
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
                            _vprint(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed but no wake word found in '{text_lower[:50]}...' "
                                f"(reasoning: {intent_judgment.reasoning})",
                                "voice"
                            )
                            # Don't accept - fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{intent_judgment.query}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            # Wake word was confirmed present and judge accepted
                            # → bump persistent-session idle timer.
                            try:
                                self.state_manager.mark_user_wake()
                            except Exception:
                                pass
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            # Wake-confirmed via judge → tag for persist.
                            self._dispatch_source = "wake"
                            self.state_manager.start_collection(intent_judgment.query)
                            self._start_thinking_tune()
                            try:
                                _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window mode - no wake word needed, but check for echo.
                        # The mic can pick up Jarvis's own TTS output and Whisper
                        # transcribes it as user speech. Check fuzzy similarity.
                        # Only reject PURE echo — if the heard text is significantly
                        # longer than TTS, it contains user speech mixed with echo
                        # and the intent judge's extraction should be used instead.
                        if last_tts_text:
                            echo_score = fuzz.partial_ratio(
                                text_lower, last_tts_text.lower()
                            )
                            tts_words = len(last_tts_text.split())
                            text_words = len(text_lower.split())
                            is_pure_echo = (
                                echo_score >= 70
                                and text_words <= max(tts_words * 1.3, tts_words + 3)
                            )
                            if is_pure_echo:
                                # Also check judge's extracted query — if it matches
                                # TTS too, it's genuinely pure echo. If the query is
                                # different, the judge extracted real user speech.
                                query_echo_score = fuzz.partial_ratio(
                                    intent_judgment.query.lower(),
                                    last_tts_text.lower()
                                )
                                if query_echo_score >= 70:
                                    debug_log(f"🔇 Echo in hot window (directed, score={echo_score}): \"{text_lower}\"", "voice")
                                    _vprint(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                                    self._stop_thinking_tune()
                                    return
                                else:
                                    debug_log(
                                        f"echo in text (score={echo_score}) but judge extracted "
                                        f"non-echo query: \"{intent_judgment.query}\"", "voice"
                                    )

                        # The intent judge is explicitly designed to prune echo
                        # and extract the actual user query. But it ALSO tries
                        # to "inline vague references" by pulling context from
                        # earlier transcript segments (rule #6 in its prompt).
                        # When Whisper garbles current utterance, judge may
                        # invent a query by combining old segments with new
                        # — e.g. heard "Є кейві бой" + prior TTS echo →
                        # judge synthesizes "якщо так, є кейві бой". User
                        # report: "відбувається привязка розмови до старих
                        # запитів". Safety check: if the judge query is
                        # significantly LONGER than the heard text AND adds
                        # words not present in the heard text, treat it as
                        # over-eager context inlining and prefer raw text.
                        judge_query = (intent_judgment.query or "").strip()
                        # Vague pronouns / wh-words that legitimately need
                        # context expansion. If heard text contains any of
                        # these, judge is allowed to inline context.
                        VAGUE_TOKENS = {
                            # RU (primary post May 16)
                            "это", "этот", "эта", "то", "тот", "та", "те",
                            "она", "он", "оно", "они", "его", "её", "ее",
                            "такой", "такая", "такие", "такое",
                            "что-то", "кто-то",
                            # UA (legacy)
                            "це", "цей", "ця", "те", "той", "ті",
                            "вона", "він", "воно", "вони", "її",
                            # EN
                            "that", "it", "this", "they", "them",
                            "what about", "how about",
                        }
                        heard_lower = text_lower.lower()
                        has_vague_ref = any(t in heard_lower.split() or t in heard_lower for t in VAGUE_TOKENS)
                        heard_words = set(heard_lower.split())
                        judge_words = set(judge_query.lower().split())
                        new_words_added = judge_words - heard_words
                        # Hardened guard (May 16): floor 40 → 20 chars so 1-2
                        # word heard fragments ("сообщений", "функций") don't
                        # bypass the check just because judge_query is short.
                        # Also: ABSOLUTE rule — if heard is <3 words AND judge
                        # adds ANY new words → reject (no legitimate inlining
                        # case for such a short heard).
                        heard_word_count = len(heard_lower.split())
                        too_much_expansion = (
                            len(judge_query) > max(20, int(len(text_lower) * 1.6))
                            and len(new_words_added) >= 2
                        )
                        heard_too_short = (
                            heard_word_count < 3
                            and len(new_words_added) >= 1
                        )
                        if (
                            judge_query
                            and not has_vague_ref
                            and (too_much_expansion or heard_too_short)
                        ):
                            debug_log(
                                f"REJECTED judge expansion (no vague ref + too short/too inflated): "
                                f"judge='{judge_query}' heard='{text_lower[:80]}'",
                                "voice",
                            )
                            judge_query = ""
                        hot_query = judge_query or text_lower
                        if judge_query and judge_query.lower() != text_lower:
                            debug_log(
                                f"using judge query over heard text: "
                                f"\"{judge_query}\" (heard: \"{text_lower[:80]}\")",
                                "voice",
                            )
                        # P0 GUARD (May 16 night): in hot-window dispatches
                        # the judge's prompt unconditionally returns DIRECTED
                        # — so we need a per-dispatch sanity filter HERE.
                        # Reject ambient TV/family speech ("какие дворцы",
                        # "опа Газих") that don't continue the conversation.
                        #
                        # Audit round 23 fix (F40): the original guard
                        # rejected ANYTHING under 6 words without a head-
                        # wake or vague-continuation. That dropped ~90% of
                        # legitimate user commands — "Какие твои
                        # возможности?" (4w), "Который час?" (2w), "Запусти
                        # Telegram" (2w), "Открой Notion" (2w). Live log
                        # at 12:00:30 showed exactly this false reject.
                        # Fix: ALSO accept short text when it starts with
                        # a question word OR an imperative command verb.
                        # That covers virtually all real voice commands
                        # while still rejecting ambient noise ("какие
                        # дворцы" doesn't start with a question word —
                        # "дворцы" is a noun, "какие" is an adjective
                        # masquerading as one, so it still gets caught).
                        _hot_head = " ".join(text_lower.split()[:2])
                        _hot_head_wake = is_wake_word_detected(
                            _hot_head, wake_word, wake_aliases, fuzzy_ratio
                        )
                        _hot_starts_vague = any(
                            text_lower.startswith(t + " ") or text_lower == t
                            for t in VAGUE_TOKENS
                        )
                        _hot_words_n = len(hot_query.split())
                        # F40: question-word / command-verb detection.
                        # Russian primary + UA + EN. First-word check —
                        # captures the natural "Что нового?" / "Open
                        # Notion" pattern. Common imperatives have a
                        # distinctive shape — typically a verb in second-
                        # person form. Allowlist covers ~95% of practical
                        # voice commands without being so broad it
                        # accepts noise.
                        _QUESTION_OR_CMD_FIRST_WORDS = {
                            # RU questions
                            "что", "чем", "какой", "какая", "какие", "какое",
                            "где", "куда", "откуда", "когда", "почему", "зачем",
                            "как", "сколько", "кто", "чей", "чья", "чьё",
                            # UA questions
                            "що", "чому", "де", "куди", "звідки", "коли",
                            "як", "скільки", "хто",
                            # EN questions
                            "what", "where", "when", "why", "how", "who",
                            "which", "whose", "do", "does", "is", "are",
                            # RU imperatives (Jarvis commands)
                            "найди", "открой", "запусти", "включи", "выключи",
                            "покажи", "расскажи", "скажи", "спой", "сделай",
                            "напиши", "отправь", "позвони", "переведи",
                            "переключи", "поставь", "поищи", "проверь",
                            "удали", "создай", "сохрани", "закрой",
                            "напомни", "запиши", "продиктуй", "повтори",
                            "посчитай", "сравни", "объясни", "помоги",
                            # Round 25 fix (F52): RU infinitives. The
                            # intent judge frequently normalises
                            # imperatives to infinitives:
                            #   user said "открой Telegram"
                            #   judge returned "открыть telegram"
                            # Without these the F40 guard rejected the
                            # judge's own normalisation.
                            "открыть", "запустить", "включить", "выключить",
                            "найти", "показать", "рассказать", "сказать",
                            "спеть", "сделать", "написать", "отправить",
                            "позвонить", "перевести", "переключить",
                            "поставить", "проверить", "удалить", "создать",
                            "сохранить", "закрыть", "напомнить",
                            "продиктовать", "повторить", "посчитать",
                            "сравнить", "объяснить", "помочь",
                            # UA imperatives
                            "знайди", "відкрий", "запусти", "увімкни",
                            "вимкни", "покажи", "розкажи", "скажи", "заспівай",
                            "зроби", "напиши", "надішли", "зателефонуй",
                            "переклади", "переключи", "постав", "перевір",
                            "видали", "створи", "збережи", "закрий",
                            "нагадай", "запиши", "повтори", "порахуй",
                            "поясни", "допоможи",
                            # UA infinitives
                            "відкрити", "запустити", "увімкнути", "вимкнути",
                            "знайти", "показати", "розказати", "сказати",
                            "зробити", "написати", "надіслати",
                            "перекласти", "перевірити", "видалити",
                            "створити", "зберегти", "закрити", "нагадати",
                            # EN imperatives
                            "find", "open", "launch", "start", "stop",
                            "show", "tell", "say", "sing", "do", "make",
                            "write", "send", "call", "translate", "switch",
                            "set", "search", "check", "delete", "create",
                            "save", "close", "remind", "record", "repeat",
                            "calculate", "compare", "explain", "help",
                            "play", "pause", "resume", "shutdown", "restart",
                            "screenshot", "copy", "paste", "cut",
                        }
                        # Round 25 fix (F52): check BOTH the raw heard
                        # text AND the judge-normalised hot_query. The
                        # judge may have rewritten the user's command
                        # ("открой ютуб" → "открыть youtube"); if the
                        # heard-text first word didn't match but the
                        # normalised one does, accept.
                        def _first_word_clean(s: str) -> str:
                            parts = s.split()
                            return parts[0].strip(",.!?;:") if parts else ""
                        _heard_first = _first_word_clean(text_lower)
                        _query_first = _first_word_clean(hot_query.lower())
                        _hot_starts_question_or_cmd = (
                            _heard_first in _QUESTION_OR_CMD_FIRST_WORDS
                            or _query_first in _QUESTION_OR_CMD_FIRST_WORDS
                        )
                        # Also accept when EITHER heard or judge ends with "?"
                        _hot_ends_question = (
                            text_lower.rstrip().endswith("?")
                            or hot_query.rstrip().endswith("?")
                        )
                        if (
                            not _hot_head_wake
                            and not _hot_starts_vague
                            and not _hot_starts_question_or_cmd
                            and not _hot_ends_question
                            and _hot_words_n < 4
                        ):
                            debug_log(
                                f"🚫 hot-window dispatch rejected — short ({_hot_words_n}w) "
                                f"+ no head-wake + no vague + no question/cmd: \"{hot_query}\"",
                                "voice",
                            )
                            print(
                                f"  🚫 Hot-window: ignored ambient speech \"{text_lower[:60]}\"",
                                flush=True,
                            )
                            self._stop_thinking_tune()
                            try:
                                self._transcript_buffer.mark_segment_processed(text_lower)
                            except Exception:
                                pass
                            return

                        debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{hot_query}\"", "voice")
                        # Hot-window judge-accept counts as real user activity
                        # ONLY when we have STRONG signal (head-wake OR vague
                        # continuation). Plain high-confidence accepts of
                        # ambient speech (e.g. "какие дворцы") used to reset
                        # the idle timer → safety-net never expired → infinite
                        # self-talk. Now bump idle only on real engagement.
                        if _hot_head_wake or _hot_starts_vague:
                            try:
                                self.state_manager.mark_user_wake()
                            except Exception:
                                pass
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()

                        # Tag provenance BEFORE start_collection. The
                        # eventual dispatch flows through the collection-
                        # timeout path (see `_check_query_timeout` /
                        # `process_transcription` empty-text branch),
                        # both of which only set `_dispatch_source` if
                        # it's still None — so this tag survives.
                        self._dispatch_source = "hot_window"
                        self.state_manager.start_collection(hot_query)

                        # Start thinking tune and show processing message
                        self._start_thinking_tune()
                        try:
                            _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                        except Exception:
                            pass
                        return

                # If directed with high confidence but no extracted query, use actual text
                # Per spec: "Hot window input should reflect what the user actually said"
                # This handles cases where intent judge correctly identifies directed speech
                # but fails to extract/synthesize a query (e.g., conversational follow-ups)
                if intent_judgment.directed and intent_judgment.confidence == "high":
                    # In wake word mode, verify the wake word is actually present
                    if not could_be_hot_window:
                        wake_word = getattr(self.cfg, "wake_word", "jarvis")
                        aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
                            _vprint(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed (no query) but no wake word in '{text_lower[:50]}...'",
                                "voice"
                            )
                            # Fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted (directed, high confidence, using actual text): \"{text_lower}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            # Wake-confirmed dispatch via judge → tag "wake"
                            # so the eventual collection-timeout flush
                            # carries the right provenance.
                            self._dispatch_source = "wake"
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window — echo check before accepting
                        # Only reject pure echo (similar word count to TTS)
                        if last_tts_text:
                            echo_score = fuzz.partial_ratio(
                                text_lower, last_tts_text.lower()
                            )
                            tts_words = len(last_tts_text.split())
                            text_words = len(text_lower.split())
                            is_pure_echo = (
                                echo_score >= 70
                                and text_words <= max(tts_words * 1.3, tts_words + 3)
                            )
                            if is_pure_echo:
                                debug_log(f"🔇 Echo in hot window (directed/no-query, score={echo_score}): \"{text_lower}\"", "voice")
                                _vprint(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                                self._stop_thinking_tune()
                                return

                        debug_log(f"✅ Intent judge accepted (directed, high confidence, using actual text): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        # Hot-window directed-high path → tag "hot_window"
                        # for persist-memory provenance.
                        self._dispatch_source = "hot_window"
                        self.state_manager.start_collection(text_lower)
                        self._start_thinking_tune()
                        try:
                            _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                        except Exception:
                            pass
                        return

                # If not directed with high confidence, check reasoning before rejecting
                if not intent_judgment.directed and intent_judgment.confidence == "high":
                    # Surgical fix: If intent judge claims "echo" but echo system already cleared
                    # this utterance (we reached here, meaning Priority 2 didn't reject), don't
                    # trust the LLM's echo reasoning - fall through to wake word detection instead.
                    # The echo system does actual text similarity matching; the LLM sometimes
                    # hallucinates echo matches that don't exist.
                    reasoning_lower = (intent_judgment.reasoning or "").lower()
                    if "echo" in reasoning_lower:
                        debug_log(
                            f"⚠️ Intent judge claimed echo but echo system cleared - "
                            f"checking if near hot window: \"{text_lower}\"",
                            "voice"
                        )
                        # Check if utterance started shortly after hot window expired
                        # This catches cases where user started speaking just as hot window expired
                        # Use a 2-second grace period after the 3-second hot window
                        hot_window_grace = 2.0
                        # Audit round 14 fix C3: snapshot under lock.
                        _, last_tts_finish, _, _ = self.echo_detector.snapshot_tts_window()
                        last_tts_finish = last_tts_finish or 0.0
                        hot_window_end = last_tts_finish + self.state_manager.hot_window_seconds
                        time_after_hot_window = utterance_start_time - hot_window_end if utterance_start_time > 0 and hot_window_end > 0 else float('inf')

                        if 0 <= time_after_hot_window < hot_window_grace:
                            # Utterance started within grace period after hot window
                            debug_log(
                                f"✅ Accepting as directed: started {time_after_hot_window:.2f}s after hot window expired",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()

                            # Mark the current segment as processed to prevent re-extraction
                            self._transcript_buffer.mark_segment_processed(text_lower)

                            self._clear_audio_buffers()
                            # Grace-period accept = treat as hot-window follow-up.
                            self._dispatch_source = "hot_window"
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                            except Exception:
                                pass
                            return

                        # Check could_be_hot_window (handles overlap: utterance
                        # started during TTS but extended into hot window span).
                        # The grace period above only checks utterance_start_time
                        # which is negative for overlapping utterances.
                        if could_be_hot_window:
                            # Verify it's not pure echo before overriding
                            echo_score = 0
                            is_pure_echo = False
                            if last_tts_text:
                                echo_score = fuzz.partial_ratio(
                                    text_lower, last_tts_text.lower()
                                )
                                tts_words = len(last_tts_text.split())
                                text_words = len(text_lower.split())
                                is_pure_echo = (
                                    echo_score >= 70
                                    and text_words <= max(tts_words * 1.3, tts_words + 3)
                                )
                            if is_pure_echo:
                                debug_log(f"🔇 Echo in hot window (echo reasoning confirmed, score={echo_score}): \"{text_lower}\"", "voice")
                                self._stop_thinking_tune()
                                return
                            # Mixed echo+speech — override the echo reasoning
                            _vprint(f"  🧠 Intent override: accepting hot window speech (mixed echo+speech)", flush=True)
                            debug_log(
                                f"⚡ Overriding echo reasoning in hot window "
                                f"(echo_score={echo_score}, text longer than TTS): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            # Mixed echo+speech in hot window → hot_window source.
                            self._dispatch_source = "hot_window"
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                            except Exception:
                                pass
                            return

                        # Otherwise fall through to wake word detection
                        debug_log(f"⏭️ Not near hot window ({time_after_hot_window:.2f}s after), falling through to wake word check", "voice")
                        # Continue to wake word detection below
                    else:
                        # Check if text is pure echo of TTS output
                        echo_score = 0
                        is_pure_echo = False
                        if last_tts_text:
                            echo_score = fuzz.partial_ratio(
                                text_lower, last_tts_text.lower()
                            )
                            tts_words = len(last_tts_text.split())
                            text_words = len(text_lower.split())
                            is_pure_echo = (
                                echo_score >= 70
                                and text_words <= max(tts_words * 1.3, tts_words + 3)
                            )

                        if could_be_hot_window and is_pure_echo:
                            # Confirmed pure echo — early check should have caught
                            # this, but handle as safety net.
                            debug_log(f"🔇 Echo in hot window (score={echo_score}): \"{text_lower}\"", "voice")
                            self._stop_thinking_tune()
                            return

                        if could_be_hot_window:
                            # Hot window + non-echo speech → user is talking to us.
                            # Override the intent judge rejection — small models
                            # sometimes reject valid follow-ups like "don't you
                            # already know that?" as not directed.
                            #
                            # SAFETY GUARD (May 16): refuse override on suspect
                            # fragments. User report "відповідає сам собі"
                            # traced to Whisper hallucinating short phrases
                            # ("или спросить", "это я", etc.) during quiet
                            # post-TTS gaps. Judge correctly marked them
                            # NOT DIRECTED but override still fired, creating
                            # an echo loop: hallucination → dispatch → LLM
                            # reply → new TTS → new hallucination.
                            #
                            # Trust HIGH-confidence judge rejection for short
                            # fragments (<4 words) that don't contain the
                            # wake word. Override only kicks in when:
                            #   - judge confidence is low/medium, OR
                            #   - text is long enough to be a real query (≥4 words), OR
                            #   - text contains wake-shaped tokens (means user
                            #     repeated wake mid-window).
                            jconf = str(getattr(intent_judgment, "confidence", "low")).lower()
                            text_words_n = len(text_lower.split())
                            # Wake-token check on HEAD only — same logic
                            # as trump-card guard. Wake mid-sentence
                            # ("я тоже джарвіс думаю") is NOT a real
                            # re-wake; only vocative-position counts.
                            _head_text_hw = " ".join(text_lower.split()[:2])
                            has_wake_token = is_wake_word_detected(
                                _head_text_hw, wake_word, wake_aliases, fuzzy_ratio
                            )
                            # TIGHTENED (May 16 evening): with hot_window_persistent=True
                            # the override is the main "talks to itself" path. Refuse
                            # override unless BOTH conditions hold:
                            #   - text is substantial (≥4 words), OR has head-wake
                            #   - AND judge is NOT high-confidence rejection
                            # Old code accepted short fragments at jconf=low/medium
                            # without wake → Whisper hallucinations slipped through.
                            if jconf == "high" and not has_wake_token:
                                debug_log(
                                    f"🚫 Refusing override — judge HIGH confidence rejection + no head-wake: "
                                    f"\"{text_lower}\"",
                                    "voice"
                                )
                                self._stop_thinking_tune()
                                return
                            if text_words_n < 4 and not has_wake_token:
                                debug_log(
                                    f"🚫 Refusing override — short ({text_words_n}w) + no head-wake: "
                                    f"\"{text_lower}\"",
                                    "voice"
                                )
                                self._stop_thinking_tune()
                                return
                            _vprint(f"  🧠 Intent override: accepting hot window speech", flush=True)
                            debug_log(
                                f"⚡ Overriding intent judge in hot window "
                                f"(echo_score={echo_score}, conf={jconf}, words={text_words_n}, reasoning={intent_judgment.reasoning}): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            # Intent-judge override in hot window → hot_window source.
                            self._dispatch_source = "hot_window"
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
                            except Exception:
                                pass
                            return

                        # Outside hot window — trust rejection
                        debug_log(f"🚫 Intent judge rejected (not directed, high confidence): \"{text_lower}\"", "voice")
                        self._stop_thinking_tune()
                        return
                else:
                    # For inconclusive results, fall through to wake word detection
                    debug_log(f"⏭️ Intent judge inconclusive ({intent_judgment.confidence}), checking wake word", "voice")

        # Priority 4: Wake word detection (fallback when intent judge unavailable/inconclusive)
        wake_word = getattr(self.cfg, "wake_word", "jarvis")
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
        fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))

        # HEAD CHECK (May 16): wake word must be in the FIRST 4 tokens.
        # Whisper hallucinations like "Май щеперска мови джарвіс ниво
        # джарвіс" used to bypass this gate and dispatch as a query.
        # Real users always put the wake word at the start.
        _head_text_wake = " ".join(text_lower.split()[:2])
        wake_detected = is_wake_word_detected(_head_text_wake, wake_word, list(aliases), fuzzy_ratio)
        debug_log(f"wake word check (head): '{wake_word}' in '{_head_text_wake}' → {wake_detected}", "voice")

        if wake_detected:
            # Cancel any pending hot window activation when new query starts
            self.state_manager.cancel_hot_window_activation()

            # Confirmed wake → reset persistent-session idle timer.
            try:
                self.state_manager.mark_user_wake()
            except Exception:
                pass

            # Mark the current segment as processed to prevent re-extraction
            self._transcript_buffer.mark_segment_processed(text_lower)

            # Clear audio buffers to prevent concatenation issues
            self._clear_audio_buffers()

            query_fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
            # Priority-4 wake fallback (judge unavailable/inconclusive)
            # — text contained a wake word, so the eventual dispatch is
            # wake-confirmed even though it'll fire from the collection-
            # timeout path. Tag for persist-memory provenance.
            self._dispatch_source = "wake"
            self.state_manager.start_collection(query_fragment)

            # Start thinking tune and show processing message
            self._start_thinking_tune()
            try:
                _vprint(f"\n✨ Working on it: {_safe_user_text(self.state_manager.get_pending_query())}")
            except Exception:
                pass
            return

        # Priority 5: Collection mode handling
        if self.state_manager.is_collecting():
            # DEFENCE-IN-DEPTH (May 16): never let hallucinations / echo
            # leak into the collection buffer just because we're past
            # the top-of-function gate. Without this, Whisper junk
            # ("Спасибо.", "Дякую.", echoes of our own TTS) silently
            # piled up and dispatched on collection timeout, triggering
            # canned replies to nobody — the self-talk loop.
            _stripped = text_lower.strip().rstrip(".!?,;:…»\"'")

            # (a) Re-run the canonical hallucination filter. Cheap, idempotent.
            if self._is_known_hallucination(text):
                debug_log(f"collection: dropped known hallucination: '{text_lower[:60]}'", "voice")
                try:
                    self._transcript_buffer.mark_segment_processed(text_lower)
                except Exception:
                    pass
                return

            # (b) Bare-token Whisper-junk blocklist (punctuation-stripped).
            # Now references the module-level `BARE_JUNK_SET` — audit
            # round 6 unified what used to be three drifting copies.
            if _stripped in BARE_JUNK_SET:
                debug_log(f"collection: dropped bare junk: '{_stripped}'", "voice")
                try:
                    self._transcript_buffer.mark_segment_processed(text_lower)
                except Exception:
                    pass
                return

            # (c) TTS-echo guard: if collection text closely matches the
            # last thing the daemon SAID, it's our own speaker leaking
            # into the mic, not a user follow-up.
            try:
                # Audit round 14 fix C3: snapshot under lock.
                _, _, _last_tts_raw, _ = self.echo_detector.snapshot_tts_window()
                last_tts = (_last_tts_raw or "").lower().strip()
                if last_tts and len(_stripped) >= 3:
                    echo_score = fuzz.partial_ratio(_stripped, last_tts) / 100.0
                    if echo_score >= 0.85:
                        debug_log(
                            f"collection: dropped TTS echo (ratio={echo_score:.2f}): "
                            f"'{_stripped[:50]}' vs TTS '{last_tts[:50]}'",
                            "voice",
                        )
                        try:
                            self._transcript_buffer.mark_segment_processed(text_lower)
                        except Exception:
                            pass
                        return
            except Exception:
                pass

            self.state_manager.add_to_collection(text_lower)
            return

        # Priority 6: Non-wake input (ignore)
        # Provide clear debug info about why input was ignored
        intent_info = ""
        if intent_judgment is not None:
            intent_info = f", intent={intent_judgment.directed}/{intent_judgment.confidence}"

        # Stop any early-started beep since we're not processing this input
        self._stop_thinking_tune()

        if received_during_tts:
            # User spoke during TTS but it wasn't a stop command - this is likely a response
            # to a TTS question that arrived before hot window activated
            debug_log(f"input ignored (during TTS, not a stop command{intent_info}): {text_lower}", "voice")
            try:
                _vprint(f"  ⏳ Heard during TTS (waiting for hot window): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
            except Exception:
                pass
        else:
            debug_log(f"input ignored (no wake word{intent_info}): {text_lower}", "voice")

    def _voice_direct_chat(self, query: str) -> str:
        """Direct LLM chat call with minimal system prompt for voice speed.

        Bypasses the full reply engine (which loads persona + memory digest +
        graph context + tool descriptions = ~3000-token system prompt that
        takes 120-240s prompt-eval on qwen2.5:3b CPU). Uses a tiny system
        prompt + a rolling history of the last N turns so Джарвіс actually
        remembers what we just discussed within a session.

        Returns the assistant reply, or "" on failure (caller falls back to
        an apology TTS).
        """
        if not query or not query.strip():
            return ""

        import requests
        import time as _time

        # NOTE: `_stream_abort` is cleared at the TOP of `_dispatch_query`,
        # not here. Audit round 6 finding: clearing it at the start of
        # _voice_direct_chat created a TOCTOU race — if a HUD interrupt
        # fired during the brief window between dispatch deciding to
        # call this function and the `.clear()` above, the interrupt
        # was silently lost. Now the clear happens before any chat path
        # is selected, so an interrupt that arrives AFTER dispatch
        # starts is preserved through to the stream loop.

        base_url = getattr(self.cfg, "ollama_base_url", "")
        model = getattr(self.cfg, "ollama_chat_model", "qwen2.5:3b")
        if not base_url:
            return ""

        # Voice-mode system prompt. KV-cache OPTIMIZATION: this whole
        # block is STATIC (byte-identical across calls and matches the
        # warmup in _start_llm_warmup). Dynamic content (lang switch,
        # history, query) goes into separate messages BELOW the static
        # system prompt, so the prefix cache keeps hitting on every
        # voice turn — cuts prompt-eval from 15-25s to 1-3s.
        #
        # Module-level constant VOICE_STATIC_SYSTEM_PROMPT keeps this
        # the single source of truth across direct-chat and warmup.
        system_prompt = VOICE_STATIC_SYSTEM_PROMPT

        # Language is dynamic (RU default after May 16 migration; UA/EN/DE
        # on explicit switch) — passed as a SEPARATE system message so it
        # doesn't bust the cache on the main prompt. ~30 tokens, evals in <1s.
        # CRITICAL DEFAULT: 'ru' here is the ROOT cause of "speaks UA still"
        # bug — the previous 'uk' default forced every chat call to ask the
        # model for a UA reply, regardless of whisper_language="ru" config.
        lang_directive = self._language_directive(getattr(self, "_active_language", "ru"))

        # Build messages with rolling dialog history so the model has
        # context across turns AND across sessions (persistent memory
        # loaded at startup from ~/.config/jarvis/memory/dialog-YYYY-MM-DD.jsonl).
        # We pass the last 10 messages (5 pairs) — fits easily in
        # num_ctx=2560 along with system prompts + current query.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": lang_directive},
        ]
        # Audit round 21 (F28): snapshot history under the lock — the
        # mutation site holds the lock for append+trim, so the snapshot
        # here is guaranteed to be a consistent prefix (no torn 17-
        # element view). Keep the read short — just a slice into a
        # local list so messages.extend() works on a frozen copy.
        with self._dialog_history_lock:
            _history_snapshot = list(self._dialog_history[-2:])
        # Last 2 messages (1 turn) — was 4. KV-cache prefix scan on
        # CPU at ~50 tok/s prompt-eval rate means every 100 tokens of
        # history = 2s added to TTFT. 2 turns of avg 100-tok replies
        # was costing ~8s per query. 1 turn (100-150 tok) costs ~2-3s.
        # The model still has the immediate previous exchange for
        # follow-up coherence ("а ще?", "так само") — that's the only
        # legitimate use of history in voice mode anyway.
        # Audit round 21 (F28): use the lock-protected snapshot taken
        # above instead of re-reading ``self._dialog_history`` here.
        messages.extend(_history_snapshot)
        messages.append({"role": "user", "content": query})

        try:
            # R34-S54.1 Phase 7b: monotonic for elapsed math. The same
            # NTP-step pitfall S52 H / S53.1 I-3 fixed for the echo-gate
            # applies to every elapsed-time anchor in this file. ``t0``
            # is paired with the elapsed read 200+ lines below; both
            # must use the same clock.
            t0 = _time.monotonic()
            # STREAMING mode: ollama emits NDJSON, each line is a partial
            # `{"message":{"content":"..."}}`. We accumulate content
            # token-by-token and SPEAK the first complete sentence the
            # moment it arrives — drops perceived latency from 15-25s
            # (whole reply) to 3-6s (first sentence). The remaining
            # sentences are returned in `content` for the caller to
            # speak through the normal completion-callback path.
            import json as _json
            import re as _re
            response = requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": "24h",
                    # qwen3 thinking mode spends 50-300 tokens on
                    # `<think>...</think>` before producing the user-
                    # facing answer, blowing the num_predict budget
                    # and often returning empty content. Top-level
                    # `think: false` disables it natively (cleaner
                    # than `/no_think` directive in the prompt). For
                    # non-thinking models (gemma2, qwen2.5, llama3.x)
                    # Ollama silently ignores this field — safe default.
                    "think": False,
                    "options": {
                        "temperature": 0.4,
                        # VOICE-OPTIMIZED — cut from 600 → 220 tokens.
                        # User report: "Джарвіс викликається супер довго
                        # очікував приблизно 5хв". At qwen3:8b @ 6.55 tok/s
                        # on CCX23, 600 tok = 92s, 220 tok = 34s. Streaming
                        # TTS speaks sentence-1 at ~5-8s anyway, but the
                        # FULL response completes ~3× faster. Voice replies
                        # rarely need more than 100-150 tokens (~3 short
                        # UA sentences) — capping at 220 cuts long-tail
                        # latency without losing useful content. If user
                        # explicitly asks for long explanation, model
                        # learns to be more concise; downstream tool-use
                        # path uses a separate larger budget.
                        "num_predict": 220,
                        "repeat_penalty": 1.2,
                        "repeat_last_n": 192,
                        "presence_penalty": 0.3,
                        "frequency_penalty": 0.2,
                        # 2048 ctx (was 4096) — KV-cache scales linearly
                        # with context length on CPU, and 4096 was wasting
                        # ~50% of prompt-eval time on slots never used.
                        # Real voice dialogs measured at 500-1500 tokens:
                        #   system prompt:    ~300 tok
                        #   language directive: ~30 tok
                        #   last 5-7 turns:   ~400-800 tok
                        #   current query:    ~30-150 tok
                        #   reply budget:     up to num_predict (600)
                        # Headroom: 2048 - 1500 = 548 tok safety margin.
                        # If history overflows we drop oldest turns
                        # (listener already does this), so the worst case
                        # is "Jarvis forgets a turn from 10min ago" —
                        # an acceptable trade for ~40% faster prompt-eval.
                        "num_ctx": 2048,
                        # 4 worker threads (Hetzner CCX23 = 4 vCPU).
                        "num_thread": 4,
                        # Larger batch = more tokens per prompt-eval pass
                        # = better CPU throughput.
                        "num_batch": 256,
                    },
                },
                # R34-S58.0 perf: (connect, read) split. Stale-NAT
                # socket fails fast at 5 s connect deadline instead
                # of hanging for the 240 s request timeout. Tracks
                # the same fix in llm.py:call_llm_streaming.
                # 240s read = gemma2:9b cold-cache rebuild can hit
                # 60-90 s on first call; warm subsequent calls return
                # in 8-30 s. 600 num_predict at 6 tok/s = 100 s.
                timeout=(5.0, 240.0),
                stream=True,
                # Connection: close removed — see llm.py for the full
                # rationale (closing on a 30+ s streaming reply just
                # wastes the keep-alive between intent-judge → chat
                # within the same voice turn).
            )
            if response.status_code != 200:
                debug_log(
                    f"voice direct-chat HTTP {response.status_code}",
                    "voice",
                )
                return ""

            content_parts = []
            buf = ""
            sentences_spoken: list[str] = []
            sent_end_re = _re.compile(r"[.!?…]")
            first_tok_t: Optional[float] = None

            def _flush_sentence(sent: str) -> None:
                """Sanitize, strip lazy prefix, queue into TTS."""
                # ABORT GUARD — if the user interrupted (wake re-fire,
                # stop command, intent-judge cut), every tts.interrupt()
                # call site also sets `self._stream_abort`. Drop the
                # remaining sentences silently instead of speaking over
                # the user. Without this guard, a long reply keeps
                # finishing its queued sentences even after interrupt —
                # the very "talks over user" bug we're fixing.
                if self._stream_abort.is_set():
                    debug_log(
                        f"stream flush aborted: '{sent[:40]}' (stream_abort set)",
                        "voice",
                    )
                    return
                cleaned = self._strip_lazy_prefix(
                    self._sanitize_for_piper_uk(sent)
                ).strip()
                if not cleaned:
                    return
                # Round 26 fix (F63), refined Round 29 (F77).
                # The LLM occasionally output truncations like "Нило"
                # or "Даня" instead of the canonical "Данило".
                # User reported: "він каже до мене нило а не Данило".
                #
                # F77 narrows the rewrite — previous version included
                # the dative "Данилу", instrumental "Данилом", and
                # vocative "Даниле" which are CORRECT grammatical
                # forms; replacing them with the nominative produced
                # ungrammatical TTS like "Поздоровляю Данило з днем
                # народження!" (should be "Данила"). We now only fix
                # genuine mishearings/truncations: "Нило", "Даня",
                # bare "Данил" stem, and the diminutive "Дани". The
                # proper declensions Данила/Данилу/Данилом/Даниле/
                # Данилові are left alone.
                try:
                    cleaned = _re.sub(
                        r"\b(Нило|Даня|Дани|Данил)\b(?![а-яёІіЇїЄєҐґ])",
                        "Данило",
                        cleaned,
                    )
                except Exception:
                    pass
                # Emit typed sentence event regardless of TTS being on
                # (consumers like Telegram bridge may want text only).
                try:
                    from ..ipc import get_stream
                    get_stream().emit(
                        "sentence",
                        text=cleaned,
                        idx=len(sentences_spoken),
                    )
                except Exception:
                    pass
                if not (self.tts and self.tts.enabled):
                    return
                try:
                    # Audit round 22 fix (F36): register EVERY streamed
                    # sentence with the echo detector, not just the
                    # first. Background: a multi-sentence streaming
                    # reply ("Джарвіс. / Що нужно, Даніло?") used to
                    # seed echo detector with only sentence #1
                    # ("Джарвіс."). When the speaker→mic loop later
                    # caught a fragment of sentence #2 ("что нужно,
                    # данил?") the echo detector compared against
                    # "Джарвіс." → mismatch → accepted as user input
                    # → LLM responded to its own echo → "бред" loop.
                    # Now every sentence updates ``_last_tts_text``
                    # via ``track_tts_start`` so echo comparisons
                    # always reference the most recent TTS output.
                    self.track_tts_start(cleaned)
                    debug_log(
                        f"stream sent #{len(sentences_spoken)+1}: {cleaned[:60]}",
                        "voice",
                    )
                    self.tts.speak(cleaned)
                    sentences_spoken.append(cleaned)
                except Exception as e:
                    debug_log(f"stream-speak failed: {e}", "voice")

            try:
                for raw in response.iter_lines(decode_unicode=False):
                    # ABORT GUARD — break the LLM stream as soon as a
                    # `tts.interrupt()` call sets the flag. Without this
                    # we keep pulling tokens off the HTTP stream (and
                    # `_flush_sentence` would queue them) even after the
                    # user has spoken over Jarvis. Closing the response
                    # connection via `break` also tells Ollama to stop
                    # generation server-side (it cancels the prediction
                    # job when the client hangs up).
                    if self._stream_abort.is_set():
                        debug_log(
                            "stream iter aborted (stream_abort set)",
                            "voice",
                        )
                        try:
                            response.close()
                        except Exception:
                            pass
                        break
                    if not raw:
                        continue
                    try:
                        obj = _json.loads(raw)
                    except Exception:
                        continue
                    chunk = ""
                    msg = obj.get("message") if isinstance(obj, dict) else None
                    if isinstance(msg, dict):
                        chunk = msg.get("content", "") or ""
                    if chunk:
                        if first_tok_t is None:
                            # R34-S54.1 Phase 7b: monotonic — paired
                            # with t0 above.
                            first_tok_t = _time.monotonic() - t0
                            debug_log(
                                f"first token in {first_tok_t:.2f}s",
                                "voice",
                            )
                        content_parts.append(chunk)
                        buf += chunk
                        # Flush EVERY complete sentence to TTS as soon
                        # as it's ready. This is the core latency win:
                        # user hears the reply unfolding token-by-token
                        # instead of waiting for the full 15-25s
                        # generation to finish.
                        while True:
                            m = sent_end_re.search(buf)
                            if not m:
                                break
                            sentence = buf[: m.end()].strip()
                            buf = buf[m.end():]
                            if len(sentence) >= 4:
                                _flush_sentence(sentence)
                    if isinstance(obj, dict) and obj.get("done"):
                        break
            except Exception as e:
                debug_log(f"stream iter failed: {e}", "voice")
            finally:
                # Round 29 (F78): always close the response. The abort
                # path at line 2853 already calls response.close(), but
                # the natural `done`-break path didn't — leaving the
                # HTTP socket in CLOSE_WAIT until GC. Under heavy turn
                # count the requests connection pool to Ollama got
                # exhausted, manifesting as gradual session slowdown.
                try:
                    response.close()
                except Exception:
                    pass

            # Flush any trailing fragment (no terminal punctuation).
            # `_flush_sentence` itself respects `self._stream_abort`,
            # so an aborted reply silently drops the tail too — no
            # extra check needed here, but reading the abort flag
            # directly avoids the function-call overhead in the hot
            # path when an interrupt has fired.
            tail = buf.strip()
            if tail and len(tail) >= 2 and not self._stream_abort.is_set():
                _flush_sentence(tail)

            # R34-S54.1 Phase 7b: monotonic — paired with t0 above.
            elapsed = _time.monotonic() - t0
            content = "".join(content_parts).strip()
            # Stash for _dispatch_query so it knows the reply has
            # ALREADY been queued into TTS — it must NOT speak the
            # whole thing again. We pass the count so dispatch can
            # decide whether to schedule a completion-callback poll.
            self._streamed_sentence_count = len(sentences_spoken)
            self._streamed_first_sentence = (
                sentences_spoken[0] if sentences_spoken else ""
            )
            self._streamed_full_text = "".join(sentences_spoken)
            if sentences_spoken:
                debug_log(
                    f"voice direct-chat streamed {len(sentences_spoken)} "
                    f"sentence(s) in {elapsed:.1f}s",
                    "voice",
                )
            if content:
                # Round 28 fix (F68): preserve the RAW LLM reply with
                # Latin app names intact so the action_dispatcher's
                # parse_action() can extract "Safari"/"YouTube"/etc.
                # BEFORE _sanitize_for_piper_uk strips them out.
                #
                # Live evidence (events.jsonl): LLM produced "Сейчас
                # открою YouTube в Safari." which became "Сейчас
                # открою  в  ." after Latin-strip → parse_action saw
                # "в" and called _open_app("в") → AppleScript failure.
                # User report: "сафарі та ютуб не відкрив такі!"
                #
                # _dispatch_query reads this attribute right after
                # _voice_direct_chat returns. The sanitized `content`
                # below is what TTS speaks and what history stores.
                self._last_raw_reply = content
                # Sanitize Latin words / mixed-script content — Piper UA
                # model has no English phoneme map and silently TRUNCATES
                # audio when it hits a Latin word ("Jarvis допоможе" plays
                # only "Jarvis " worth of silence then cuts the whole rest
                # of the line). Replace common Latin tokens with Cyrillic
                # equivalents AFTER the LLM call as a safety net for when
                # the strict-Cyrillic instruction in the system prompt is
                # ignored.
                content = self._sanitize_for_piper_uk(content)
                content = self._strip_lazy_prefix(content)

                # Web-search fallback: if the model admitted ignorance,
                # do a DuckDuckGo search and re-prompt with fresh
                # snippets. This is the "always-internet-access" feature
                # the user explicitly requested.
                #
                # TIGHTENED (May 16): trigger ONLY on explicit search-request
                # phrases ("нужен поиск" / "поищи", etc.), NOT on common
                # "не знаю" responses. The fallback was firing on every
                # benign "не зрозумів" reply, doubling LLM call → 9.6s TTFT
                # × 2 = 19s perceived response time. User report: "дуже
                # довго та повільно працює". Now needs explicit user intent.
                lc = content.lower()
                if (
                    "нужен поиск" in lc
                    or "нужен веб-поиск" in lc
                    or "поищи в интернете" in lc
                    or "потрібен пошук" in lc
                    or "потрібен веб-пошук" in lc
                ) and len(query) > 4:
                    debug_log(f"web-search fallback for: '{query[:60]}'", "voice")
                    _vprint(f"  🌐 Шукаю в інтернеті: \"{query[:50]}\"", flush=True)
                    results = self._web_search(query, max_results=4)
                    if results:
                        ctx = "\n".join(
                            f"- {r['title']}: {r['snippet']}"
                            for r in results if r.get("snippet")
                        )
                        retry_msgs = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content":
                                f"{query}\n\n[Контекст из веб-поиска — используй эти факты:]\n{ctx}\n\n"
                                f"Дай СОДЕРЖАТЕЛЬНЫЙ ответ на русском языке на основе этого контекста."},
                        ]
                        try:
                            # R34-S54.1 Phase 7b: monotonic for elapsed.
                            t1 = _time.monotonic()
                            r2 = requests.post(
                                f"{base_url.rstrip('/')}/api/chat",
                                json={
                                    "model": model,
                                    "messages": retry_msgs,
                                    "stream": False,
                                    "keep_alive": "24h",
                                    "think": False,  # qwen3 — see _voice_direct_chat above
                                    "options": {
                                        "temperature": 0.4,
                                        "num_predict": 220,  # Matches voice budget — see _voice_direct_chat
                                        "repeat_penalty": 1.3,
                                        "repeat_last_n": 128,
                                        "presence_penalty": 0.5,
                                        "frequency_penalty": 0.4,
                                        "num_ctx": 2048,  # web-search retry — same rationale as _voice_direct_chat
                                    },
                                },
                                # R34-S58.0 perf: (connect, read)
                                # split mirrors the direct-chat path
                                # above. Stale-NAT fails at 5 s.
                                timeout=(5.0, 90.0),
                            )
                            if r2.status_code == 200:
                                d2 = r2.json()
                                m2 = d2.get("message", {}) if isinstance(d2, dict) else {}
                                c2 = (m2.get("content") or "").strip()
                                if c2 and len(c2) > 20:
                                    # F68: keep raw web-search retry reply for
                                    # action_dispatcher (Latin app names intact).
                                    self._last_raw_reply = c2
                                    content = self._sanitize_for_piper_uk(c2)
                                    content = self._strip_lazy_prefix(content)
                                    debug_log(
                                        # R34-S54.1 Phase 7b: monotonic
                                        # paired with t1 above.
                                        f"web-search retry ok in {_time.monotonic()-t1:.1f}s",
                                        "voice",
                                    )
                        except (requests.exceptions.ConnectTimeout,
                                requests.exceptions.ConnectionError) as e:
                            # R34-S58.3 (A1.2): ConnectTimeout is the
                            # stale-NAT signal; treat it as a connection
                            # error rather than honour-the-budget timeout.
                            debug_log(f"web-search retry: connect error ({e})", "voice")
                        except requests.exceptions.ReadTimeout:
                            # R34-S58.3 (A1.2): read budget elapsed.
                            debug_log("web-search retry: read-timeout (Ollama slow)", "voice")
                        except Exception as e:
                            debug_log(f"web-search retry error: {e}", "voice")
                # Append to rolling dialog history AND persist to disk
                # so Jarvis remembers across daemon restarts.
                # Audit round 21 (F28): the append + slice "trim to 16"
                # was a two-step operation that briefly left a 17- or
                # 18-element list visible to other threads (HUD
                # control watcher, TTS completion callback). The lock
                # makes the invariant atomic — append, append, trim
                # all happen under a single critical section so
                # concurrent reads either see the pre-append state
                # or the post-trim state, never an intermediate one.
                with self._dialog_history_lock:
                    self._dialog_history.append({"role": "user", "content": query})
                    self._dialog_history.append({"role": "assistant", "content": content})
                    if len(self._dialog_history) > 16:
                        self._dialog_history = self._dialog_history[-16:]
                # Persist asynchronously — disk write is fast (~1ms) but
                # we still don't want to block the voice loop on it.
                self._persist_memory_pair(query, content)
                debug_log(
                    f"voice direct-chat ok in {elapsed:.1f}s, history={len(self._dialog_history)}: '{content[:80]}{'...' if len(content) > 80 else ''}'",
                    "voice",
                )
            return content
        except requests.exceptions.ConnectTimeout as e:
            # R34-S58.3 (A1.2): stale-NAT signal — no socket established
            # within the connect deadline. Distinct from a slow Ollama
            # reply. Voice is single-shot here (no retry loop) so the
            # next user turn will get a fresh socket via urllib3's pool.
            debug_log(f"voice direct-chat: connect-timeout ({e})", "voice")
            return ""
        except requests.exceptions.ReadTimeout:
            # R34-S58.3 (A1.2): Ollama actually started responding but
            # the read budget elapsed. Honour caller's deadline.
            debug_log("voice direct-chat: read-timeout (Ollama slow)", "voice")
            return ""
        except requests.exceptions.ConnectionError as e:
            debug_log(f"voice direct-chat: connection error ({e})", "voice")
            return ""
        except requests.exceptions.Timeout:
            debug_log("voice direct-chat: timeout", "voice")
            return ""
        except Exception as e:
            debug_log(f"voice direct-chat error: {e}", "voice")
            return ""

    def _sanitize_for_piper_uk(self, text: str) -> str:
        """Replace Latin words/letters with Cyrillic equivalents.

        Piper's uk_UA-ukrainian_tts-medium model has a single-language
        phoneme map. When it encounters a Latin letter inside a Ukrainian
        sentence (a wake-word brand name "Jarvis", a typo with mixed-script
        "бude" where the u is actually U+0075, a stray "OK" / "AI"), the
        synth either drops the word silently or — worse — cuts the rest of
        the audio buffer mid-sentence. The user perceives this as Jarvis
        "не договорює і закривається відразу" (cuts off mid-reply).

        We patch over the most common offenders post-LLM. The LLM is also
        instructed in its system prompt to avoid Latin, but small models
        leak through under load.
        """
        if not text:
            return text
        import re
        # Common brand/word substitutions — case-insensitive whole word.
        # Round 28 (F69): EXPANDED with common macOS app names so the
        # TTS engine actually says "Сафарі"/"Ютуб"/"Хром" instead of
        # eating the whole word (the line ~3098 pure-Latin stripper).
        #
        # Round 30 (F90): language-aware transliteration. F69 originally
        # hard-coded Ukrainian spellings ("Сафарі" with і, "Ютуб") for
        # every app. The system has been running in RU mode since the
        # May UA→RU migration — embedding UA-script chars (і, ї, є, ґ)
        # inside a Russian sentence flipped the Piper UA model into
        # Ukrainian phoneme mode mid-utterance. User report:
        # "переключається на український голос часом!!!". Live evidence
        # (events.jsonl): "Сейчас открою Сафарі и перейду на Ютуб." —
        # RU stem + UA-spelled app names → mixed-accent TTS.
        # We now pick the spelling based on self._active_language.
        lang = getattr(self, "_active_language", "ru")
        # Common substitutions that don't depend on language.
        common_subs = [
            (r"\bJarvis\b", "Джарвіс"),
            (r"\bJARVIS\b", "Джарвіс"),
            (r"\bAI\b", "ШІ" if lang == "uk" else "ИИ"),
            (r"\bOK\b", "добре" if lang == "uk" else "хорошо"),
            (r"\bok\b", "добре" if lang == "uk" else "хорошо"),
            (r"\bGPT\b", "Джі-Пі-Ті"),
            (r"\bLLM\b", "мовна модель" if lang == "uk" else "языковая модель"),
            (r"\bAPI\b", "АПІ" if lang == "uk" else "АПИ"),
            (r"\bCEO\b", "генеральний директор" if lang == "uk" else "генеральный директор"),
            (r"\bUI\b", "інтерфейс" if lang == "uk" else "интерфейс"),
            (r"\bUX\b", "юзабіліті" if lang == "uk" else "юзабилити"),
            (r"\bNexus\b", "Нексус"),
            (r"\bStudio\b", "Студіо" if lang == "uk" else "Студио"),
        ]
        # App-name spellings split by target language. RU uses "и"
        # everywhere; UA uses "і". Piper's UA model handles both but
        # using the wrong vowel triggers the wrong phoneme cluster
        # mid-sentence and the user perceives an accent switch.
        if lang == "uk":
            app_subs = [
                (r"\bSafari\b", "Сафарі"),
                (r"\bYouTube\b", "Ютуб"), (r"\bYoutube\b", "Ютуб"),
                (r"\bChrome\b", "Хром"),
                (r"\bFirefox\b", "Файрфокс"),
                (r"\bTelegram\b", "Телеграм"),
                (r"\bSlack\b", "Слек"),
                (r"\bDiscord\b", "Діскорд"),
                (r"\bZoom\b", "Зум"),
                (r"\bSpotify\b", "Спотіфай"),
                (r"\bNotion\b", "Ноушн"),
                (r"\bFigma\b", "Фігма"),
                (r"\bGitHub\b", "Гітхаб"), (r"\bGithub\b", "Гітхаб"),
                (r"\bWhatsApp\b", "Вотсап"), (r"\bWhatsapp\b", "Вотсап"),
                (r"\bInstagram\b", "Інстаграм"),
                (r"\bTwitter\b", "Твіттер"),
                (r"\bFacebook\b", "Фейсбук"),
                (r"\bGmail\b", "Джімейл"),
                (r"\bGoogle\b", "Гугл"),
                (r"\bMicrosoft\b", "Майкрософт"),
                (r"\bWindows\b", "Віндовс"),
                (r"\bmacOS\b", "макОС"),
                (r"\bLinux\b", "Лінукс"),
                (r"\bTerminal\b", "Термінал"),
                (r"\bFinder\b", "Файндер"),
                (r"\bMail\b", "Мейл"),
                (r"\bCalendar\b", "Календар"),
                (r"\bNotes\b", "Нотатки"),
                (r"\bSettings\b", "Налаштування"),
                (r"\bSystem\s+Settings\b", "Системні налаштування"),
                (r"\bVS\s*Code\b", "Вс Код"), (r"\bVSCode\b", "Вс Код"),
                (r"\bXcode\b", "Ікс Код"),
                (r"\bClaude\b", "Клод"),
                (r"\bChatGPT\b", "ЧатДжіПіТі"), (r"\bChatgpt\b", "ЧатДжіПіТі"),
                (r"\bMaps\b", "Карти"),
                (r"\bPhotos\b", "Фото"),
                (r"\bMusic\b", "Музика"),
                (r"\bWeather\b", "Погода"),
                (r"\bReminders\b", "Нагадування"),
                (r"\bPreview\b", "Перегляд"),
                (r"\bTextEdit\b", "ТекстЕдіт"),
                (r"\bKeynote\b", "Кейноут"),
                (r"\bPages\b", "Пейджес"),
                (r"\bNumbers\b", "Намберс"),
            ]
        else:
            # Russian — use "и" (U+0438) not "і" (U+0456), no UA-only
            # letters (ї, є, ґ). This is what Piper UA hears as
            # pure-Russian phonemes; no accent-flip mid-sentence.
            app_subs = [
                (r"\bSafari\b", "Сафари"),
                (r"\bYouTube\b", "Ютуб"), (r"\bYoutube\b", "Ютуб"),
                (r"\bChrome\b", "Хром"),
                (r"\bFirefox\b", "Файрфокс"),
                (r"\bTelegram\b", "Телеграм"),
                (r"\bSlack\b", "Слэк"),
                (r"\bDiscord\b", "Дискорд"),
                (r"\bZoom\b", "Зум"),
                (r"\bSpotify\b", "Спотифай"),
                (r"\bNotion\b", "Ноушн"),
                (r"\bFigma\b", "Фигма"),
                (r"\bGitHub\b", "Гитхаб"), (r"\bGithub\b", "Гитхаб"),
                (r"\bWhatsApp\b", "Вотсап"), (r"\bWhatsapp\b", "Вотсап"),
                (r"\bInstagram\b", "Инстаграм"),
                (r"\bTwitter\b", "Твиттер"),
                (r"\bFacebook\b", "Фейсбук"),
                (r"\bGmail\b", "Джимейл"),
                (r"\bGoogle\b", "Гугл"),
                (r"\bMicrosoft\b", "Майкрософт"),
                (r"\bWindows\b", "Виндовс"),
                (r"\bmacOS\b", "макОС"),
                (r"\bLinux\b", "Линукс"),
                (r"\bTerminal\b", "Терминал"),
                (r"\bFinder\b", "Файндер"),
                (r"\bMail\b", "Мейл"),
                (r"\bCalendar\b", "Календарь"),
                (r"\bNotes\b", "Заметки"),
                (r"\bSettings\b", "Настройки"),
                (r"\bSystem\s+Settings\b", "Системные настройки"),
                (r"\bVS\s*Code\b", "Вс Код"), (r"\bVSCode\b", "Вс Код"),
                (r"\bXcode\b", "Икс Код"),
                (r"\bClaude\b", "Клод"),
                (r"\bChatGPT\b", "ЧатДжиПиТи"), (r"\bChatgpt\b", "ЧатДжиПиТи"),
                (r"\bMaps\b", "Карты"),
                (r"\bPhotos\b", "Фото"),
                (r"\bMusic\b", "Музыка"),
                (r"\bWeather\b", "Погода"),
                (r"\bReminders\b", "Напоминания"),
                (r"\bPreview\b", "Просмотр"),
                (r"\bTextEdit\b", "ТекстЭдит"),
                (r"\bKeynote\b", "Кейноут"),
                (r"\bPages\b", "Пейджес"),
                (r"\bNumbers\b", "Намберс"),
            ]
        replacements = common_subs + app_subs
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        # Mixed-script word fix: any token that contains BOTH Cyrillic
        # and Latin letters is almost certainly a homoglyph typo from
        # the LLM (e.g. "бude" → "буде", "Дanил" → "Данил"). For these,
        # replace each Latin char with its closest Cyrillic look-alike.
        latin_to_cyr = {
            "a": "а", "A": "А",
            "o": "о", "O": "О",
            "e": "е", "E": "Е",
            "c": "с", "C": "С",
            "p": "р", "P": "Р",
            "x": "х", "X": "Х",
            "y": "у", "Y": "У",
            "u": "и", "U": "И",   # "u" mid-word ≈ "и" (closest sound)
            "i": "і", "I": "І",
            "k": "к", "K": "К",
            "H": "Н", "h": "г",
            "T": "Т", "t": "т",
            "B": "В", "b": "б",
            "M": "М", "m": "м",
            "n": "н", "N": "Н",
            "d": "д", "D": "Д",
            "g": "г", "G": "Г",
            "s": "с", "S": "С",
            "r": "р", "R": "Р",
            "l": "л", "L": "Л",
            "f": "ф", "F": "Ф",
            "v": "в", "V": "В",
            "z": "з", "Z": "З",
            "j": "й", "J": "Й",
            "w": "в", "W": "В",
            "q": "к", "Q": "К",
        }
        cyr_re = re.compile(r"[Ѐ-ӿ]")
        lat_re = re.compile(r"[a-zA-Z]")
        def _fix_word(m):
            w = m.group(0)
            has_cyr = bool(cyr_re.search(w))
            has_lat = bool(lat_re.search(w))
            if has_cyr and has_lat:
                return "".join(latin_to_cyr.get(c, c) for c in w)
            return w
        text = re.sub(r"\S+", _fix_word, text)

        # Drop any remaining PURE Latin words >= 2 chars — they're
        # untranslatable proper nouns, better to silently omit than
        # crash the synth.
        text = re.sub(r"\b[a-zA-Z]{2,}\b", "", text)
        # Tidy double spaces left by removals.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _web_search(self, query: str, max_results: int = 5) -> list[dict]:
        """Quick keyless web search via DuckDuckGo HTML endpoint.

        Returns a list of {title, snippet, url} dicts. Used as a fallback
        when the LLM admits ignorance ("не знаю напевно", "потрібен
        пошук") — we then re-prompt the model with the snippets as
        fresh context so the reply has actual web knowledge.

        DDG HTML is keyless and rate-limited only per IP. No registration,
        no API key. We use a desktop User-Agent so we get the full results
        page (mobile UA returns a stripped layout).
        """
        if not query or not query.strip():
            return []
        import requests as _rq
        from bs4 import BeautifulSoup as _BS
        try:
            r = _rq.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "uk-ua"},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                        "Version/17.0 Safari/605.1.15"
                    ),
                },
                timeout=8.0,
            )
            if r.status_code != 200:
                debug_log(f"web search HTTP {r.status_code}", "voice")
                return []
            soup = _BS(r.text, "html.parser")
            results = []
            for div in soup.select("div.result")[:max_results]:
                a = div.select_one("a.result__a")
                snip = div.select_one(".result__snippet")
                if not a:
                    continue
                results.append({
                    "title": a.get_text(strip=True),
                    "snippet": snip.get_text(strip=True) if snip else "",
                    "url": a.get("href", ""),
                })
            debug_log(f"web search '{query[:40]}' → {len(results)} results", "voice")
            return results
        except Exception as e:
            debug_log(f"web search error: {e}", "voice")
            return []

    def _strip_lazy_prefix(self, text: str) -> str:
        """Strip filler prefixes like 'Добре, ...' that 1.5b model adds.

        The qwen2.5:1.5b model frequently violates system prompt rule #2
        and starts replies with conversational fillers — 'Добре,',
        'Зрозумів,', 'Так,', 'Звичайно,', 'Звісно,', 'Гаразд,',
        'Окей,', 'Авжеж,' — even though we explicitly forbid these.
        With num_predict=40, that filler eats 4-6 tokens of generation
        budget, leaving the actual answer truncated. Worse, when the
        prefix happens to be ALL the model output (model finishes a
        sentence like 'Добре. Я не знаю.'), the user perceives a
        useless 'Добре' reply.

        We strip these prefixes after sanitization. If the entire
        response was just filler, we return empty so the caller can
        fall back to a different path (canned reply / engine).
        """
        if not text:
            return text
        import re
        # Filler patterns we want to strip from the START of replies.
        # Each is matched case-insensitively, optionally followed by
        # punctuation + whitespace. Include the bare word too (no
        # trailing comma) for cases where the LLM stops at just the
        # filler word.
        fillers = (
            # RU fillers (primary after uk→ru migration)
            r"хорошо", r"понятно", r"понял", r"ладно",
            r"да(?:,\s*я)?", r"конечно", r"разумеется",
            r"окей", r"ок", r"безусловно", r"я могу",
            r"я знаю", r"я готов", r"я здесь",
            # UA fillers kept for code-switching safety — user
            # occasionally drops UA words mid-sentence.
            r"добре", r"зрозуміло", r"зрозумів", r"гаразд",
            r"звичайно", r"звісно", r"авжеж", r"я готовий", r"я тут",
        )
        # Build alternation. Allow up to 3 fillers chained
        # ('Добре, зрозумів, я...') by running the strip 3×.
        pattern = re.compile(
            r"^\s*(?:" + "|".join(fillers) + r")\s*[,.\!\?:;—–-]*\s*",
            re.IGNORECASE,
        )
        prev = None
        for _ in range(3):
            stripped = pattern.sub("", text)
            if stripped == prev:
                break
            prev = stripped
            text = stripped
            if not text.strip():
                break
        # Capitalize the new first letter if we ate the original cap.
        text = text.strip()
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    def _canned_voice_reply(self, query: str) -> str:
        """Return a canned reply for very short voice queries, or ''.

        This is a latency hack for qwen2.5:3b CPU. The Jarvis reply engine
        builds a 3000+ token system prompt (persona + memory digest + tool
        descriptions); even with a primed KV-cache, evaluation on a
        4-vCPU Hetzner CCX23 takes 30-180s depending on cache hits. For
        the small set of voice queries that don't actually need any
        reasoning (greetings, thanks, farewells), it's better UX to skip
        the LLM entirely.

        We deliberately keep the trigger list narrow — anything that
        could plausibly need the LLM (e.g. "як справи з проектом?")
        must fall through to the real engine. Match is on exact
        normalized query equality or membership in a small alias set.

        Returns reply text on match, or empty string on miss.
        """
        if not query:
            return ""
        # Normalize: lower, strip punctuation/spaces, collapse common
        # Whisper mishearings ("певіт" instead of "привіт", "джавіс"
        # at the start) so the same canned reply fires regardless of
        # transcription quality.
        q = query.lower().strip()
        q = re.sub(r"[!?,.;:\"'`()\[\]{}«»…]+", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        # Drop any leading wake-word fragment if intent-judge missed it.
        for wake in ("джарвіс", "джарвис", "ярвіс", "ярвис", "jarvis",
                     "джарвес", "джарваз", "джавіс", "ярвес"):
            if q.startswith(wake + " "):
                q = q[len(wake) + 1:].strip()
                break

        # P0 SAFETY (May 17): refuse to emit a canned reply for bare
        # 1-2 word fragments that are almost always misheard family /
        # TV speech (Whisper hallucinates these on silence and fuzzy
        # aliases catch single tokens). User report: "відповідає сам
        # собі" — daemon was answering "Пожалуйста. Обращайтесь." to
        # transcriptions of family's "спасибо" said in another room.
        # The canned-reply path is the loudest single source of unwanted
        # TTS because it bypasses the LLM entirely and speaks instantly.
        # Audit round 6: now references the module-level BARE_JUNK_SET
        # so the three gates (collection / canned / persist) stay in sync.
        if q in BARE_JUNK_SET:
            debug_log(
                f"canned-reply: refused for bare-junk query '{q}' "
                f"(likely misheard family/TV speech)", "voice"
            )
            return ""

        # Greetings — Ukrainian first (primary language), then RU/EN.
        greetings = {
            "привіт", "превіт", "певіт", "приіт",
            "здоров", "здорова", "здоровенькі", "здоровеньки", "здоровенький були",
            "вітаю", "вітання", "доброго дня", "добрий день", "доброго ранку",
            "добрий вечір", "добрий ранок", "добридень",
            "привет", "здравствуй", "здравствуйте", "доброе утро",
            "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
            "hallo", "guten tag", "guten morgen",
        }
        if q in greetings:
            return "Привет, Данило! Чем могу помочь?"

        # Acknowledgements/thanks.
        thanks = {
            "дякую", "дяки", "спасибі", "вдячний",
            "спасибо", "благодарю",
            "thanks", "thank you", "thx",
            "danke", "vielen dank",
        }
        if q in thanks:
            return "Пожалуйста. Обращайтесь."

        # Farewells.
        farewells = {
            "па", "пока", "бувай", "до побачення", "побачимось", "па-па",
            "bye", "goodbye", "see you", "cya",
            "tschüss", "auf wiedersehen",
        }
        if q in farewells:
            return "До встречи, Данило."

        # Status check — "як справи?" / "як ти?" / "як ся маєш?"
        status = {
            "як справи", "як ти", "як ся маєш", "як ся маєте", "як настрій",
            "як дела", "как дела", "как ты",
            "how are you", "how's it going", "what's up",
        }
        if q in status:
            # R34-S54.1 Phase 7a: was UA "Все добре, працюю. А у вас?"
            return "Всё хорошо, работаю. А у вас?"

        # Single-word "test"/"перевірка"/"чуєш мене" — confirm liveness.
        liveness = {
            "тест", "перевірка", "тестування", "чуєш", "чуєш мене", "ти тут",
            "ти мене чуєш", "чути", "ти живий", "ти працюєш",
            "test", "testing", "are you there", "can you hear me",
            "проверка", "слышишь меня",
        }
        if q in liveness:
            # R34-S54.1 Phase 7a: was UA "Чую вас. Працюю."
            return "Слышу вас. Работаю."

        # Confirmations / brief acks — bypass intent judge entirely.
        # These phrases ALONE shouldn't trigger the LLM at all.
        # NOTE: these are also CONFIRM_WORDS for pending actions — that
        # path handles them in _dispatch_query BEFORE we reach canned.
        # Here we only hit them as standalone utterances when no action
        # is pending — in which case we acknowledge and stay quiet.
        bare_acks = {
            "ок", "окей", "ага", "так", "добре", "гаразд", "зрозумів",
            "ok", "okay", "yep", "yeah", "sure",
            "хорошо", "понятно", "ладно",
        }
        if q in bare_acks:
            # R34-S54.1 Phase 7a: was UA "Гаразд."
            return "Хорошо."

        # Identity / who-are-you — common test phrases.
        identity = {
            "хто ти", "як тебе звати", "ти хто", "представся",
            "who are you", "what are you", "what's your name",
            "кто ты", "как тебя зовут",
        }
        if q in identity:
            # R34-S54.1 Phase 7a: was UA "Я — Джарвіс, ваш AI-асистент. Готовий допомогти."
            return "Я — Джарвис, ваш AI-ассистент. Готов помочь."

        # Sorry / apology — symmetric ack.
        apologies = {
            "вибач", "вибачте", "пробач", "прости", "перепрошую",
            "sorry", "my bad", "excuse me",
            "извини", "извините",
        }
        if q in apologies:
            # R34-S54.1 Phase 7a: was UA "Нічого, працюємо далі."
            return "Ничего, работаем дальше."

        # Quick yes/no questions about Jarvis capability are misleading
        # to canned — leave to LLM.

        # Common short questions that don't really need LLM — answer
        # straight away.
        # R34-S54.1 Phase 7a: weekday + month arrays and the formatted
        # strings migrated UA→RU. Match-keys (q in {…}) still recognise
        # UA utterances because Whisper continues to transcribe UA when
        # the user code-switches; the RETURN is what gets spoken.
        if q in {"котра година", "котра зараз година", "скільки часу", "який час",
                 "сколько времени", "сколько сейчас времени", "который час",
                 "what time is it"}:
            from datetime import datetime
            now = datetime.now()
            weekday = ["понедельник", "вторник", "среда", "четверг",
                       "пятница", "суббота", "воскресенье"][now.weekday()]
            return f"Сейчас {now.strftime('%H:%M')}, {weekday}."

        if q in {"яке сьогодні число", "яка дата", "який сьогодні день",
                 "який день тижня", "сьогодні який день",
                 "какое сегодня число", "какая дата", "какой сегодня день",
                 "какой день недели", "сегодня какой день"}:
            from datetime import datetime
            now = datetime.now()
            months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                      "июля", "августа", "сентября", "октября", "ноября", "декабря"]
            weekday = ["понедельник", "вторник", "среда", "четверг",
                       "пятница", "суббота", "воскресенье"][now.weekday()]
            return f"Сегодня {weekday}, {now.day} {months[now.month-1]} {now.year} года."

        # Compliment / nice — friendly echo.
        compliments = {
            "ти класний", "молодець", "круто", "супер", "відмінно",
            "good job", "well done", "nice",
            "молодец", "класс",
        }
        if q in compliments:
            return "Спасибо! Рад помочь."

        return ""

    def _lang_name(self, code: str) -> str:
        """Human-readable Russian name of a language code (post uk→ru migration)."""
        return {
            "ru": "по-русски",
            "uk": "українською",
            "en": "по-английски",
            "de": "по-немецки",
        }.get(code, code)

    def _language_directive(self, code: str) -> str:
        """Return the system-prompt clause that locks reply language.

        Default (May 15 2026 onwards) is RU — Whisper transcribes RU
        significantly better than UA, so the voice path runs in RU and
        the assistant replies in kind. UA / EN / DE are still available
        if the user explicitly switches mid-conversation.
        """
        directives = {
            "ru": (
                "ЯЗЫК ОТВЕТА: Русский (кириллица). Всегда по-русски, "
                "даже если в истории есть UA/EN/DE — не повторяй тот выбор. "
                "Вместо 'Jarvis' пиши 'Джарвис'."
            ),
            "uk": (
                "МОВА ВІДПОВІДІ: Українська (кирилиця). Користувач явно попросив "
                "перейти на українську — відповідай тільки українською. "
                "Замість 'Jarvis' пиши 'Джарвіс'."
            ),
            "en": (
                "REPLY LANGUAGE: English. User explicitly requested English — "
                "respond only in English. Use 'Jarvis' for the assistant name."
            ),
            "de": (
                "ANTWORTSPRACHE: Deutsch. Der Nutzer hat explizit Deutsch "
                "verlangt — antworte nur auf Deutsch."
            ),
        }
        return directives.get(code, directives["ru"])

    def _run_self_upgrade_async(self, user_request: str) -> None:
        """Spawn Claude Code in a background thread to self-upgrade."""
        import threading as _t
        def _runner():
            # Audit round 15 fix F8: pair every spawn_claude (which
            # acquires the concurrency lock) with a release in the
            # finally — otherwise a crash anywhere in the runner
            # leaves the lock held and future upgrades silently
            # refused.
            from ..agent.self_upgrade import release_upgrade_lock
            acquired = False
            try:
                from ..agent.self_upgrade import (
                    write_upgrade_brief, spawn_claude,
                    wait_for_completion_and_restart,
                )
                brief = write_upgrade_brief(user_request)
                proc = spawn_claude(brief)
                if proc is None:
                    # spawn_claude either failed to find the binary OR
                    # rejected the spawn (concurrent run / invalid
                    # REPO). In the binary-missing case the lock was
                    # never acquired; in the others it was acquired
                    # but already released by spawn_claude on its
                    # error path. Either way no release needed here.
                    # R34-S54.1 Phase 7a: was UA.
                    self._speak_and_continue(
                        "Не нашёл Claude Code CLI или другое самообновление уже идёт."
                    )
                    return
                # spawn_claude returned a live process → it acquired
                # the lock and we own the release responsibility.
                acquired = True
                ok, summary = wait_for_completion_and_restart(proc)
                # After kickstart we'll be dead — speak BEFORE restart.
                if ok and "Перезапускаюсь" in summary:
                    self._speak_and_continue(summary)
                    # Daemon will be killed by kickstart; new instance loads.
                else:
                    self._speak_and_continue(summary)
            except Exception as e:
                debug_log(f"self-upgrade thread error: {e}", "voice")
                self._speak_and_continue(f"Ошибка самообновления: {e}")
            finally:
                if acquired:
                    release_upgrade_lock()
        _t.Thread(target=_runner, daemon=True, name="self-upgrade").start()

    def _speak_and_continue(self, text: str) -> None:
        """Speak `text` via TTS and keep the hot window active.

        Used after action execution / cancellation so the user can
        immediately continue the conversation without re-saying the
        wake word.
        """
        if not text:
            return
        if not (self.tts and self.tts.enabled):
            print(f"📢 {text}", flush=True)
            return
        def _done():
            self.activate_hot_window()
        self.track_tts_start(text)
        self.tts.speak(text, completion_callback=_done)

    def _dispatch_query(self, query: str) -> None:
        """
        Dispatch a complete query to the reply engine.

        Args:
            query: Complete user query to process
        """
        # Round 25 (F51): bump activity timestamp so the LLM
        # keepalive thread suppresses its next ping — we're already
        # about to pile prompt-eval work onto Ollama, no need to
        # queue an extra dummy generation behind it.
        # R34-S54.1 Phase 7b: monotonic — this anchor is read at the
        # LLM-keepalive idle check (line ~5097) to compute "how long
        # since last activity". NTP step would either freeze the
        # keepalive forever (clock jumped forward) or trigger spurious
        # pings (clock jumped back). Paired conversion below.
        self._last_user_activity_ts = time.monotonic()
        # PROVENANCE LIFECYCLE — atomically read-and-clear `_dispatch_source`
        # at the top of dispatch. Previously cleared ONLY inside
        # `_persist_memory_pair`, so early-return paths (lang switch /
        # upgrade / action confirmation / direct user-command) left a stale
        # "wake" tag dangling for the NEXT turn — a low-confidence
        # collection-timeout dispatch would then get persisted as if it
        # were wake-confirmed. We stash into a local now, pass to the
        # persist helper below if/when we get there. Audit round 6 finding.
        # P0 also fixes the `_voice_direct_chat` early-return leak path
        # (empty query / HTTP non-200 / timeout) — those paths now also
        # have their source cleared because we cleared it here at the top.
        _dispatch_source_local = self._dispatch_source
        self._dispatch_source = None
        # Stash on the instance for `_persist_memory_pair` to read. We
        # use a separate name (`_active_dispatch_source`) so that any
        # NEW dispatch starting mid-flow (shouldn't happen, but defence
        # in depth) doesn't trample the value the persist helper reads.
        self._active_dispatch_source = _dispatch_source_local

        # STREAM-ABORT LIFECYCLE — clear at the START of every dispatch.
        # Was previously cleared inside `_voice_direct_chat` which
        # opened a TOCTOU race (HUD interrupt set the flag during the
        # brief window between dispatch start and the LLM call → the
        # later `.clear()` wiped the interrupt → user heard Jarvis
        # ignore their stop click). Clearing here means an interrupt
        # that arrives AFTER this point is preserved through to the
        # stream loop. Any interrupt BEFORE this point is for a
        # previous reply that's already finished — safe to drop.
        try:
            self._stream_abort.clear()
        except Exception:
            pass

        debug_log(f"dispatching query: '{query}' (source={_dispatch_source_local or 'unknown'})", "voice")
        # Emit typed STT-final event for HUD + any other consumers.
        try:
            from ..ipc import get_stream
            get_stream().emit(
                "stt_final",
                text=query,
                lang=getattr(self, "_active_language", "ru"),
                confidence=1.0,
            )
        except Exception:
            pass

        # Language switch check: "говори російською" / "switch to English"
        # → stage with confirmation. Active language defaults to UA on
        # daemon start and after force_end_session.
        try:
            from .action_dispatcher import detect_language_switch
            lang_req = detect_language_switch(query)
            if lang_req is not None and not getattr(self, "_pending_lang_switch", None):
                lang_code, ack = lang_req
                current = getattr(self, "_active_language", "ru")
                if lang_code != current:
                    self._pending_lang_switch = lang_code
                    # R34-S54.1 Phase 7a: was UA. RU-only persona policy
                    # means even the lang-switch confirmation prompt
                    # speaks RU. The lang_code is preserved for
                    # downstream policy.
                    self._speak_and_continue(
                        f"Хочешь чтобы я говорил {self._lang_name(lang_code)}? "
                        f"Подтверди — скажи 'выполняй'."
                    )
                    return
            if getattr(self, "_pending_lang_switch", None):
                from .action_dispatcher import is_confirmation as _ic, is_denial as _id
                if _ic(query):
                    new_lang = self._pending_lang_switch
                    self._pending_lang_switch = None
                    self._active_language = new_lang
                    # CRITICAL: flush dialog history when switching
                    # language. Otherwise the model keeps seeing the
                    # last 4 messages in the OLD language and copies
                    # that style — producing mixed output ("конфліктують").
                    # F75: under _dialog_history_lock for atomicity.
                    try:
                        with self._dialog_history_lock:
                            self._dialog_history.clear()
                    except Exception:
                        pass
                    debug_log(f"language switched to {new_lang}; history cleared", "voice")
                    # Acknowledge in the new language so user hears the
                    # voice immediately.
                    acks = {
                        "uk": "Перейшов на українську.",
                        "ru": "Перешёл на русский.",
                        "en": "Switched to English.",
                        "de": "Auf Deutsch gewechselt.",
                    }
                    self._speak_and_continue(acks.get(new_lang, "OK"))
                    return
                if _id(query):
                    self._pending_lang_switch = None
                    # R34-S51 — RU-only TTS policy. Was UA "залишаюсь на
                    # українській" — the legacy listener path supported
                    # voice-driven lang switch but the user pinned to RU
                    # via R34-S48, so this branch now always speaks RU.
                    self._speak_and_continue("Отменено, остаюсь на русском.")
                    return
                self._pending_lang_switch = None
        except Exception as e:
            debug_log(f"language-switch check error: {e}", "voice")

        # Self-upgrade check: "Джарвіс, я хочу щоб ти краще ..." →
        # stage an upgrade request, ask for confirmation, then spawn
        # Claude Code under user's Max subscription.
        try:
            from ..agent.self_upgrade import is_upgrade_request
            if is_upgrade_request(query) and not getattr(self, "_pending_upgrade", None):
                self._pending_upgrade = query
                # R34-S54.1 Phase 7a: was UA.
                self._speak_and_continue(
                    "Запускаю самообновление через Claude Code. "
                    "Это может занять 5-15 минут. Подтверди — скажи 'выполняй'."
                )
                return
            if getattr(self, "_pending_upgrade", None):
                from .action_dispatcher import is_confirmation as _is_conf, is_denial as _is_den
                if _is_conf(query):
                    req = self._pending_upgrade
                    self._pending_upgrade = None
                    # R34-S54.1 Phase 7a: was UA "Починаю самооновлення."
                    self._speak_and_continue("Начинаю самообновление.")
                    self._run_self_upgrade_async(req)
                    return
                if _is_den(query):
                    self._pending_upgrade = None
                    self._speak_and_continue("Отменено.")
                    return
                self._pending_upgrade = None
        except Exception as e:
            debug_log(f"upgrade-check error: {e}", "voice")

        # Confirmation flow: if there's a pending action and the user
        # is confirming/denying, run/cancel BEFORE going to LLM.
        from .action_dispatcher import (
            parse_action, is_confirmation, is_denial,
        )
        # R34-S55.1 Phase 8b (P2 concurrency double-fire): snapshot
        # ``pending`` UNDER ``_pending_lock`` and treat it as null if
        # the slot was cleared between the lock entry and the snapshot.
        # The prior version read ``_pending_action`` outside the lock,
        # then re-entered the lock to read ``_pending_confirmation``.
        # In the race window between those two ops the HUD-watcher path
        # (``_consume_pending_confirmation_from_hud``) could acquire the
        # lock, NULL ``_pending_action``, AND execute ``pending.fn()``.
        # Our captured ``pending`` reference survives the null and we
        # would then double-fire via the ``is_confirmation`` branch.
        # Holding the lock for the whole read+clear closes the window;
        # we release before calling ``pending.fn()`` so the action runs
        # without holding the lock (preserves the original perf shape
        # and avoids any cross-thread blocking).
        with self._pending_lock:
            pending = self._pending_action
            hud_choice = self._pending_confirmation
            # Take ownership of BOTH slots atomically. From this point
            # on, the HUD watcher and any concurrent dispatch will see
            # ``_pending_action = None`` and the dispatch path below
            # uniquely owns ``pending``. This is what makes the double-
            # fire impossible — without clearing here, the downstream
            # is_confirmation / is_denial / "user-said-something-new"
            # branches each naked-clear ``self._pending_action`` AFTER
            # the lock release, which leaves a window for HUD to
            # execute against the same pending in parallel.
            if pending is not None:
                self._pending_action = None
            if hud_choice in ("yes", "no"):
                self._pending_confirmation = None
        if pending is not None:
            if hud_choice in ("yes", "no"):
                if hud_choice == "yes":
                    debug_log(f"executing pending action via HUD: {pending.name}", "voice")
                    try:
                        ok, msg = pending.fn()
                    except Exception as e:
                        # R34-S51 — RU-only TTS policy.
                        ok, msg = False, f"Ошибка при выполнении действия: {e}"
                    self._speak_and_continue(msg if ok else f"Не получилось. {msg}")
                    return
                else:  # "no"
                    debug_log(f"cancelled pending action via HUD: {pending.name}", "voice")
                    self._speak_and_continue("Отменено.")
                    return
            if is_confirmation(query):
                debug_log(f"executing pending action: {pending.name}", "voice")
                ok, msg = pending.fn()
                self._pending_action = None
                self._speak_and_continue(msg if ok else f"Не получилось. {msg}")
                return
            if is_denial(query):
                debug_log(f"cancelled pending action: {pending.name}", "voice")
                self._pending_action = None
                self._speak_and_continue("Отменено.")
                return
            # Otherwise — user said something new (not confirm/deny).
            # Drop the pending action AND remove the "Зараз відкрию X.
            # Підтверди..." proposal from dialog history, otherwise the
            # LLM keeps anchoring the next reply to that abandoned
            # action and ignores the actual new query.
            self._pending_action = None
            try:
                # F75: under _dialog_history_lock so the read of
                # _dialog_history[-1] and the paired pop()s are atomic
                # with respect to _voice_direct_chat's snapshot read.
                with self._dialog_history_lock:
                    if (
                        self._dialog_history
                        and self._dialog_history[-1].get("role") == "assistant"
                        and "підтверди" in self._dialog_history[-1].get("content", "").lower()
                    ):
                        # Pop the orphan assistant proposal + its paired user msg
                        self._dialog_history.pop()
                        if self._dialog_history and self._dialog_history[-1].get("role") == "user":
                            self._dialog_history.pop()
                    debug_log("removed abandoned action proposal from history", "voice")
            except Exception:
                pass

        # Clear audio buffers to prevent stale audio from next query
        self._clear_audio_buffers()

        # ── DIRECT USER COMMAND (no LLM) ─────────────────────────────
        # If the user just said a clear imperative ("відкрий Safari",
        # "вимкни звук", "котра година"), execute IMMEDIATELY and skip
        # the LLM round-trip. ~50ms latency vs 15-25s LLM path. User
        # complained: "він всерівно неможе навіть відкрити сафарі" —
        # the 7b CPU model was either inventing JSON or mis-phrasing
        # so ACTION_PATTERNS never matched on its reply. Direct parser
        # solves that by skipping the LLM for unambiguous commands.
        try:
            from .action_dispatcher import (
                parse_user_command, _run_async, SYNC_ACTIONS,
            )
            direct = parse_user_command(query)
            if direct is not None:
                debug_log(
                    f"DIRECT command bypassing LLM: {direct.name} — {direct.description}",
                    "voice",
                )
                try:
                    from desktop_app.face_widget import get_jarvis_state, JarvisState
                    get_jarvis_state().set_state(JarvisState.SPEAKING)
                except Exception:
                    pass
                # SYNC actions (battery/time/clipboard) return data the
                # user needs to hear — must run inline.
                # ASYNC actions (open_app/lock/mute/etc) — submit to
                # worker pool and speak the description immediately so
                # voice loop doesn't block on 10s subprocess timeout.
                if direct.name in SYNC_ACTIONS:
                    ok, msg = direct.fn()
                    spoken = msg if ok else f"Не вийшло. {msg}"
                else:
                    _run_async(direct.fn, direct.name)
                    # User hears the description ("Відкриваю Safari") —
                    # action completes in background, status is logged.
                    spoken = direct.description
                    msg = spoken  # for dialog history
                self._speak_and_continue(spoken)
                # Store in dialog history so context stays coherent.
                # Audit round 12 fix: the voice direct-chat path caps
                # `_dialog_history` to 16 entries (lines ~2721-2722) but
                # THIS direct-action path used to append without any
                # cap. Every "стоп"/"час"/"open url" command leaked two
                # dict entries forever; engine.py:2408 reads the whole
                # list per turn so per-turn latency grew linearly.
                # R34-S55.1 Phase 8a (P1 concurrency regression): wrap
                # the direct-action dialog-history mutation in
                # ``_dialog_history_lock`` to match the invariant F28/F75
                # established at lines 3091-3091 (the cousin path).
                # Without this lock, a concurrent ``_voice_direct_chat``
                # snapshot can observe a torn 17/18-element history; and
                # the ``= self._dialog_history[-16:]`` REBIND swaps the
                # attribute out from under any reader inside the lock
                # at that instant — the reader operates on the OLD list
                # while we point ``self._dialog_history`` at a new one.
                try:
                    with self._dialog_history_lock:
                        self._dialog_history.append({"role": "user", "content": query})
                        self._dialog_history.append({"role": "assistant", "content": msg})
                        if len(self._dialog_history) > 16:
                            self._dialog_history = self._dialog_history[-16:]
                except Exception:
                    pass
                return
        except Exception as e:
            debug_log(f"direct user-command parse failed: {e}", "voice")

        # Set face state to THINKING
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            state_manager = get_jarvis_state()
            state_manager.set_state(JarvisState.THINKING)
            debug_log("face state set to THINKING (dispatch_query)", "voice")
        except Exception as e:
            debug_log(f"failed to set face state to THINKING: {e}", "voice")

        # Voice fast-path: short canned replies for greetings/acks/farewells.
        #
        # On qwen2.5:3b CPU Hetzner CCX23 the full reply engine pays ~120-240s
        # for prompt-eval on cold cache (3000-token Jarvis system prompt +
        # tool descriptions). Users perceive this as a hang — the HUD coin
        # spins for 3+ minutes with no audible response. For the most common
        # voice queries — "привіт" / "hi" / "як справи?" / "дякую" — we
        # bypass the engine entirely and return a canned response. Saves the
        # full LLM round-trip and gives <500ms voice feedback.
        canned = self._canned_voice_reply(query)
        if canned:
            debug_log(f"voice fast-path: canned reply for '{query}' → '{canned}'", "voice")
            reply = canned
        else:
            # NO interjections at all — user explicitly demanded
            # ("прибери повністю всі вигуки!!"). All ack words removed:
            # no "Угу/Хм/Зрозумів/Шукаю відповідь." Streaming TTS
            # (sentence-by-sentence) already provides perceived
            # feedback at ~3-5s — that's our only "I heard you" signal.
            # Try direct chat (bypass full reply engine) for SPEED. The full
            # engine builds a 3000+ token system prompt (persona + memory
            # digest + tool descriptions) that takes ~120-240s prompt-eval
            # on qwen2.5:3b CPU. Voice users want <3s response. The direct
            # path uses a minimal ~80-token system prompt and skips tools,
            # memory, and graph context — giving ~2-5s responses on warm
            # cache. If it fails or times out, fall back to the engine.
            reply = self._voice_direct_chat(query)
            if not reply:
                debug_log(
                    "voice direct-chat returned empty/timeout — apology reply",
                    "voice",
                )
                reply = "Вибач, не встиг подумати. Спробуй ще раз."

        # Check if the reply contains a PC-control action plan.
        # If user already said "виконуй" in the original query, execute
        # immediately without asking. Otherwise stage the action and
        # ask for confirmation.
        #
        # Round 28 (F68): parse the RAW reply (Latin app names intact)
        # instead of the sanitized `reply` which has all ≥2-char Latin
        # words stripped. Without this, "Сейчас открою Safari" became
        # "Сейчас открою " in `reply`, the regex captured the lone "в"
        # preposition between "открою" and ".", and _open_app("в")
        # failed silently — user's "сафарі не відкрив" complaint.
        raw_for_action = getattr(self, "_last_raw_reply", "") or reply
        # One-shot read; clear so a subsequent canned/non-LLM dispatch
        # doesn't accidentally re-use a stale unsanitized reply.
        self._last_raw_reply = ""
        action = parse_action(raw_for_action)
        if action is not None:
            if is_confirmation(query):
                # Inline confirm — user said "виконуй ..." right away.
                debug_log(f"inline-confirmed action: {action.name}", "voice")
                ok, msg = action.fn()
                reply = f"{action.description}. {msg}" if ok else f"Не вийшло: {msg}"
            else:
                # Stage for the next utterance's confirmation.
                self._pending_action = action
                # R34-S54.1 Phase 7a: was UA "Підтверди — скажи 'виконуй'."
                # Match BOTH UA and RU "підтверди"/"подтверди" stems so we
                # don't double-append the suffix when the LLM already
                # produced its own RU confirmation prompt.
                if "підтверди" not in reply.lower() and "подтверди" not in reply.lower():
                    reply = reply.rstrip(".!?") + ". Подтверди — скажи 'выполняй'."

        # Handle TTS with proper callbacks
        if reply and self.tts and self.tts.enabled:
            # Stop thinking tune when TTS starts
            self._stop_thinking_tune()

            # TTS completion callback for hot window
            def _on_tts_complete():
                import time as _time
                debug_log(f"TTS completion callback triggered at {_time.time():.3f}", "voice")
                self.activate_hot_window()

            # Duration callback to update echo detector with exact timing (Piper only)
            def _on_duration_known(duration: float):
                debug_log(f"TTS exact duration: {duration:.2f}s", "voice")
                if self.echo_detector:
                    # Audit round 14 fix C3: route through helper so the
                    # write goes under _tts_state_lock instead of racing
                    # _matches_tts_segment readers.
                    self.echo_detector.set_tts_exact_duration(duration)

            # If _voice_direct_chat already streamed the WHOLE reply
            # sentence-by-sentence to TTS, we just need to wait for
            # the queue to drain and then activate the hot window.
            # No additional speak() — that would dupe everything.
            streamed_count = int(getattr(self, "_streamed_sentence_count", 0) or 0)
            self._streamed_sentence_count = 0
            self._streamed_first_sentence = ""
            self._streamed_full_text = ""

            if streamed_count > 0:
                debug_log(
                    f"reply already streamed ({streamed_count} sentences) "
                    "— scheduling hot-window activation",
                    "voice",
                )
                # Activate hot window after the TTS queue drains.
                def _wait_and_activate():
                    import time as _t
                    while self.tts and self.tts.is_speaking():
                        _t.sleep(0.1)
                    _t.sleep(0.2)  # echo settle
                    _on_tts_complete()
                threading.Thread(
                    target=_wait_and_activate,
                    daemon=True,
                    name="jarvis-hotwin-wait",
                ).start()
            else:
                # Non-streamed path: canned reply / action result /
                # web-search fallback. Speak the full reply normally.
                self.track_tts_start(reply)
                debug_log(
                    f"starting TTS for reply ({len(reply)} chars)",
                    "voice",
                )
                self.tts.speak(
                    reply,
                    completion_callback=_on_tts_complete,
                    duration_callback=_on_duration_known,
                )
        else:
            debug_log(f"no TTS output: reply={bool(reply)}, tts={bool(self.tts)}, enabled={getattr(self.tts, 'enabled', False) if self.tts else False}", "voice")
            # Stop thinking tune if no TTS response
            self._stop_thinking_tune()

    def _calculate_audio_energy(self, frames: list) -> float:
        """Calculate RMS energy from audio frames."""
        if not frames or np is None:
            return 0.0
        try:
            audio_data = np.concatenate(frames)
            rms = float(np.sqrt(np.mean(np.square(audio_data))))
            return rms
        except Exception:
            return 0.0

    def _clear_audio_buffers(self) -> None:
        """Clear all audio buffers and reset speech state.

        Call this on state transitions to prevent old audio from being
        incorrectly concatenated with new input.
        """
        self._utterance_frames = []
        self._pre_roll.clear()
        self.is_speech_active = False
        self._silence_frames = 0
        self._voice_run = 0

        # Clear wake detection state
        self._wake_timestamp = None

        # Drain the audio queue
        try:
            while not self._audio_q.empty():
                self._audio_q.get_nowait()
        except Exception:
            pass

        debug_log("audio buffers cleared", "voice")

    def _is_speech_frame(self, frame) -> bool:
        """Determine if audio frame contains speech.

        REVERTED (May 16 evening): the OR-combination with energy_floor
        broke endpoint detection — with `voice_min_energy=0.0025` +
        normal room ambience, EVERY frame became "voice" → utterances
        maxed out at 350 frames (7s) without ever endpointing → Whisper
        decoded 5.7s of silence as UA YouTube hallucinations on loop.

        Back to webrtcvad as the single source of truth when present.
        Energy floor only used when VAD is unavailable (degraded mode).
        For whisper-quiet wake words, the right answer is to TIGHTEN
        `vad_aggressiveness` to capture quiet speech (1 → 2), not to
        widen the speech gate with an OR.
        """
        if np is None:
            return True

        # Track energy for echo detection (used elsewhere)
        rms = float(np.sqrt(np.mean(np.square(frame))))
        self._recent_audio_energy.append(rms)

        if self._vad is None:
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))

        try:
            pcm16 = np.clip(frame.flatten() * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return bool(self._vad.is_speech(
                pcm16, getattr(self, "_stream_samplerate", self._samplerate)
            ))
        except Exception:
            return False

    def _filter_noisy_segments(self, segments):
        """Filter out low-confidence Whisper segments."""
        min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.3)
        marginal_threshold = min_confidence / 3  # Show user-visible log for marginal confidence
        # Threshold above which a segment is considered non-speech (hallucination during silence).
        # Checked independently of avg_logprob because Whisper can be confident about a
        # hallucinated phrase even when no real speech is present.
        no_speech_threshold = getattr(self.cfg, "whisper_no_speech_threshold", 0.5)
        filtered = []

        for seg in segments:
            # Hard filter: high no_speech_prob means no real speech regardless of logprob.
            if hasattr(seg, 'no_speech_prob') and is_whisper_hallucination(seg.no_speech_prob, no_speech_threshold):
                debug_log(
                    f"segment filtered (no_speech_prob={seg.no_speech_prob:.2f}): '{seg.text[:50]}'",
                    "voice",
                )
                continue

            confidence = None
            if hasattr(seg, 'avg_logprob'):
                confidence = min(1.0, max(0.0, (seg.avg_logprob + 1.0)))
            elif hasattr(seg, 'no_speech_prob'):
                confidence = 1.0 - seg.no_speech_prob

            if confidence is not None and confidence < min_confidence:
                if confidence >= marginal_threshold:
                    # Marginal confidence — F93 gated. Don't dump
                    # third-party speech to world-readable .out.log.
                    if getattr(self.cfg, "voice_debug", False):
                        print(f"🔇 Low confidence ({confidence:.2f}): \"{seg.text.strip()[:50]}...\"", flush=True)
                    else:
                        debug_log(f"segment filtered (low conf {confidence:.2f}, len={len(seg.text)})", "voice")
                else:
                    # Very low confidence - debug only
                    debug_log(f"segment filtered (confidence={confidence:.2f}): '{seg.text}'", "voice")
                continue

            filtered.append(seg)

        return filtered

    def _is_repetitive_hallucination(self, text: str) -> bool:
        """
        Detect repetitive hallucinations that Whisper produces on quiet/ambiguous audio.

        Common patterns include repeated single words like "don't don't don't..."
        or repeated short phrases. Also detects character-level repetition patterns
        like "Jろ Jろ Jろ..." which may appear with or without spaces.

        Args:
            text: Transcribed text to check

        Returns:
            True if the text appears to be a hallucination
        """
        import re
        from collections import Counter

        if not text:
            return False

        text_stripped = text.strip()
        if len(text_stripped) < 6:
            return False

        # --- Character-level repetition detection ---
        # Remove all whitespace to detect patterns like "Jろ Jろ Jろ" or "JろJろJろ"
        text_no_space = re.sub(r'\s+', '', text_stripped.lower())

        # Look for repeating patterns of 1-5 characters appearing 3+ times consecutively
        # This catches "JろJろJろJろ" (pattern "Jろ" repeating)
        for pattern_len in range(1, 6):
            if len(text_no_space) < pattern_len * 3:
                continue

            # Check if text is mostly composed of a repeating pattern
            for start in range(pattern_len):
                pattern = text_no_space[start:start + pattern_len]
                if not pattern:
                    continue

                # Count how many times this pattern repeats consecutively from this start position
                remaining = text_no_space[start:]
                repeat_count = 0
                pos = 0
                while pos + pattern_len <= len(remaining) and remaining[pos:pos + pattern_len] == pattern:
                    repeat_count += 1
                    pos += pattern_len

                # If pattern repeats 4+ times and covers most of the string, it's a hallucination
                covered_chars = repeat_count * pattern_len
                coverage = covered_chars / len(text_no_space) if text_no_space else 0

                if repeat_count >= 4 and coverage >= 0.6:
                    debug_log(f"char-level repetition detected: pattern '{pattern}' repeats {repeat_count}x, coverage={coverage:.0%}", "voice")
                    return True

        # --- Word-level repetition detection (existing logic) ---
        words = text_stripped.lower().split()
        if len(words) < 4:
            return False

        # Strip punctuation from words for comparison (handles "word..." vs "word")
        clean_words = [re.sub(r'[^\w]', '', w) for w in words]
        clean_words = [w for w in clean_words if w]  # Remove empty strings

        if len(clean_words) < 4:
            return False

        word_counts = Counter(clean_words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]

        # If a single word makes up more than 50% of all words and appears 4+ times
        if most_common_count >= 4 and most_common_count / len(clean_words) > 0.5:
            debug_log(f"repetitive hallucination detected: '{most_common_word}' repeated {most_common_count}x in '{text[:50]}...'", "voice")
            return True

        # Check for repeated consecutive sequences (e.g., "don don don" or "stop stop stop")
        # Look for any word repeated 3+ times consecutively
        consecutive_count = 1
        for i in range(1, len(clean_words)):
            if clean_words[i] == clean_words[i-1]:
                consecutive_count += 1
                if consecutive_count >= 3:
                    debug_log(f"consecutive repetition detected: '{clean_words[i]}' repeated {consecutive_count}+ times", "voice")
                    return True
            else:
                consecutive_count = 1

        return False

    def _is_known_hallucination(self, text: str) -> bool:
        """Block Whisper's known-bad hallucination outputs on Cyrillic silence.

        Whisper consistently regurgitates the same UA-tokenized noise patterns
        when the audio is mostly silence — the model was likely trained on
        webscrapes where these phrases repeat heavily. They never appear in
        legitimate user speech, so we can hard-block them upstream of
        repetitive-detection (saves CPU on intent-judge / fuzzy passes).

        TRUMP-CARD GUARD (May 16): if the user actually said the wake word
        (or any of its aliases) ANYWHERE in the heard text, we never reject
        it as a hallucination. Earlier versions matched substrings like
        "это я", "ладно", "ну" — which then ate legitimate commands like
        "Джарвис, нужно открыть финдер" (substring "ну" in "нужно") or
        "Джарвис, это я попросил X". The hallucination filter must never
        outrank a real wake utterance.

        WORD-BOUNDARY MATCHING (May 16): patterns are now matched against
        the heard text via regex word-boundaries (`\\bPATTERN\\b`). This
        stops short fragments like "ну"/"бо"/"ага" from matching inside
        unrelated words ("нужно", "тебе", "магазин").
        """
        if not text:
            return False
        lower = text.lower()

        # Trump-card guard: real wake word AT THE START of utterance →
        # never reject. The wake must be in the FIRST 4 tokens (vocative
        # position) — exactly where a real user puts it. This prevents
        # Whisper hallucinations like "Добавил субтитры, джарвіс,
        # джарвіс, джарвіс" or "Май щеперска мови джарвіс ниво джарвіс"
        # from bypassing the hallucination filter and reaching dispatch.
        # Real users say "Джарвис, ..." — they don't bury the wake word
        # mid-sentence after 4+ unrelated words.
        try:
            _ww = getattr(self.cfg, "wake_word", "jarvis")
            _wal = list(set(getattr(self.cfg, "wake_aliases", [])) | {_ww})
            _wfr = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
            head_tokens = lower.split()[:2]
            head_text = " ".join(head_tokens)
            if is_wake_word_detected(head_text, _ww, _wal, _wfr):
                return False
        except Exception:
            # Fail open — better to occasionally let a hallucination
            # through than to drop a real wake word due to a guard bug.
            pass
        # Known idle-noise patterns. Each phrase observed multiple times in
        # production logs with `whisper_no_speech_threshold` ≥ 0.85.
        # NEW (May 15) patterns from user logs after large-v3-turbo switch:
        # Whisper trained heavily on UA YouTube outros/intros, so silence
        # often decodes as the "thanks for watching" template in UA.
        KNOWN_PATTERNS = (
            # Personal-name salads (training-data scraping artifacts)
            "дмитро павловський",
            "білян ліна керівськ",
            "білян ліпчук",
            "хіднось продавав",
            # UA YouTube-outro hallucinations (NEW — seen on user's mic)
            "дякую за просвіт",
            "дякую за просмак",
            "дякую за перегляд",
            "дякую за увагу",
            "напиши умови",   # ambient-noise-as-imperative artifact
            "додай нотатку",  # NEW — observed 8+ times in May 15 logs
            "продовження буде",
            "продовження слідує",
            "напиши коментар",
            "натисни лайк",
            "підпишись на канал",
            "до зустрічі",
            "до побачення",  # only as hallucination — real goodbyes
            "усім бувай",    # are routed via direct user-command path
            # English silence-hallucinations (Whisper-default training)
            "thank you",
            "thanks for watching",
            "subscribe to",
            "see you next",
            "subtitles by",
            "продолжение следует",
            # NEW (May 15 evening): observed mishearings of TTS echo
            # ("Слухаю, Даниле" → "Привінжер") and ambient room noise.
            "кінець брифінгу",
            "кінець брифінга",
            "брифінг закінчено",
            "привінжер",          # TTS-echo mishearing
            "пописано",            # ambient-noise artifact
            "ошикається",          # ambient-noise artifact
            "роберт",              # repeated-name silence hallucination
            "ярсь",                # silence noise artifact
            "яксь",                # silence noise artifact (with question)
            "чисте ж, даром",     # silence noise artifact
            "посмотри, свій",     # silence noise artifact
            # NEW (May 15 night): quiet-audio hallucinations observed 4+
            # times in single session. User: "видумує сам слова і пише
            # їх і дає відповідь". These are short common UA YouTube
            # phrases Whisper produces on near-silence.
            "дякуємо",             # Whisper's default "thanks" hallucination
            "дякуємо!",
            "дякуємо за",
            "є кейві бой",        # garbage Whisper output
            "якщо є питання",     # YouTube outro hallucination
            "якщо є запитання",
            "якщо є питання, пишіть",
            "пишіть в коментар",
            # NEW (round 2): unambiguous Whisper noise artifacts only.
            # Avoid filtering real UA words (так/ні/дякую) since user
            # may legitimately say them. Background-speech defence is
            # handled separately via min-duration gate + wake-position.
            # REMOVED (May 16): bare "бо" — word-boundary regex would
            # match the Cyrillic conjunction "бо" too, which is a real
            # UA word ("джарвіс йди, бо я зайнятий"). Keep only the
            # ellipsis-stuffed forms that are pure noise tokens.
            "бо...",
            # RU YouTube-outro hallucinations (May 15, after lang switch
            # uk → ru). Whisper trained heavily on RU YouTube scrapes —
            # silence consistently decodes as these template phrases.
            "спасибо за просмотр",
            "спасибо за внимание",
            "спасибо за просмотр!",
            "спасибо за внимание!",
            "подпишитесь на канал",
            "подписывайтесь на канал",
            "ставьте лайки",
            "ставьте лайк",
            "пишите в комментариях",
            "пишите комментарии",
            "до новых встреч",
            "до новых видео",
            "увидимся в следующем",
            "продолжение следует",
            "редактор субтитров",
            "субтитры подготовил",
            "субтитры сделал",
            "субтитры:",
            "корректор:",
            # Audit round 22 fix (F34): RU subtitle-loop hallucinations
            # that survived round 21's filter. Live evidence from
            # ~/Library/Application Support/jarvis/events.jsonl:
            #   stt_final: "смотрите продолжение в следующей серии."
            #   sentence (Jarvis reply!): "Следующая серия — в
            #                              следующем эпизоде."
            # The LLM was responding to its own echo of a Whisper
            # subtitle hallucination — full "бред" loop the user
            # complained about. These patterns kill the loop at the
            # input gate.
            "смотрите продолжение",
            "следующая серия",
            "следующей серии",
            "в следующем эпизоде",
            "следующем эпизоде",
            "следующий эпизод",
            "в следующем выпуске",
            "следующий выпуск",
            "продолжение в следующ",
            "конец первой серии",
            "конец серии",
            "конец эпизода",
            "до встречи в следующ",
            "увидимся в следующей",
            # MBC/КBC Korean broadcasting watermark hallucinations
            # — Whisper picks these up from training corpus when fed
            # broadband white noise (HVAC, fan).
            "мбц ньюс",
            "мбц новости",
            "kbs ньюс",
            # Audit round 23 fix (F43): single/short words that are
            # Jarvis-own TTS output picked up as echo through the
            # speakers. Live evidence (events.jsonl seq=288):
            #   Jarvis TTS: "Джарвис здесь."
            #   Whisper transcribed echo as: "здесь."  →  passed
            #     through every guard  →  LLM responded with "Нило,
            #     ты где?" → user hears "бред".
            # These bare words have no business arriving as user
            # input — they're either echo or noise. The trump-card
            # guard above checks the FIRST 2 tokens for the wake
            # word, so legitimate "Джарвис, я здесь" still gets
            # through (wake in head). Bare standalone forms below
            # are rejected.
            "здесь",
            "здесь.",
            "я здесь",
            "я здесь.",
            "данило",
            "данило.",
            "данил",
            "данил.",
            "что нужно",
            "что нужно?",
            "что нужно.",
            "нило",
            "нило.",
            "ты где",
            "ты где?",
            "слушаю",
            "слушаю.",
            # Common RU silence-noise artifacts.
            # RE-ADDED bare "спасибо" (May 16 evening): the trump-card
            # guard now restricts to head-2-tokens, so "Джарвис,
            # спасибо тебе" still passes (wake is in head). Bare
            # "спасибо." alone is the #1 Whisper silence hallucination
            # and was flooding logs every 1-3 seconds. Word-boundary
            # regex below ensures it doesn't match inside "спасибочки".
            "спасибо",
            "спасибо.",
            "спасибо!",
            "продолжаем",
            # NEW (May 15 evening): RU YouTube-intro greetings — Whisper
            # outputs these AS PURE SILENCE. Observed 10+ times in
            # 5 minutes when user wasn't talking at all:
            #   Heard: "Добрый вечер!"
            #   Heard: "Добрый день!"
            #   Heard: "Доброе утро!"
            #   Heard: "Добро пожаловать!"
            #   Heard: "Благодарю"
            #   Heard: "Добрый день, друзья!"
            # These are the RU counterpart to the UA "Дякую за перегляд"
            # YouTube outros. Block at source.
            "добрый вечер",
            "добрый день",
            "доброе утро",
            "добро пожаловать",
            "благодарю",
            "добрый день, друзья",
            "добрый вечер, друзья",
            "приветствую вас",
            "приветствую",
            "здравствуйте",
            "здравствуйте, друзья",
            "здравствуйте всем",
            "всем привет",
            "доброго времени суток",
            # Whisper noise-token outputs (RU)
            # PURGED (May 16): bare "ну"/"ага"/"угу" — even with
            # word-boundary regex these match user fillers in legitimate
            # speech ("джарвис, ну скажи как дела"). Only ellipsis forms
            # and reduplicated stutters survive — those are pure noise.
            "м-м-м",
            "ну-ну",
            # NEW (May 16 evening): echo-loop hallucinations. THE WAKE-SHAPED
            # VARIANTS (жарвис/джарвис/харвис/гарвис) WERE REMOVED — they
            # ARE wake attempts, and the trump-card guard at the top of
            # this function lets them through. The remaining patterns are
            # daemon's own TTS-echo phrases that Whisper sometimes catches
            # and routes back as user input.
            #
            # PURGED also: bare "ладно"/"хорошо"/"это я"/"или спросить"
            # — these are normal user follow-ups in conversation
            # ("джарвис, ладно, давай дальше"). The trump-card guard
            # already protects them when wake is present. When no wake
            # is present, the wake-position check + intent judge handle
            # rejection.
            "хорошо понял",       # 2-word phrase — pattern intact
            "хорошо, понял",
            "что-нибудь еще",
            "что-нибудь ещё",
            "чем еще могу",
            "чем ещё могу",
            "задавай вопрос",     # TTS echo of daemon's own follow-up prompt
            "задавайте вопрос",
            "постараюсь ответить",
            "нужно что-то сделать",
            # NEW (May 16): user has TV/radio playing biblical content in
            # the room. Whisper transcribed 48s of ambient noise as a long
            # passage about Solomon and the Ark of the Covenant. These
            # phrases NEVER appear in legitimate user voice queries:
            "соломон",
            "иерусалим",
            "ковчег",
            "ковчега",
            "ковчегу",
            "господь",
            "господа",
            "господнего",
            "господней",
            "израиля",
            "израильт",
            "израильск",
            "священник",
            "ветхий завет",
            "новый завет",
            "евангели",
            "молитв",
            "благослов",
            "церков",
            "храм",
            "архангел",
            # Whisper bracket-annotations for non-speech audio (sirens,
            # music, applause). These are training-data artifacts where
            # transcribers labeled non-vocal sounds in caps.
            "[музыка]",
            "[аплодисменты]",
            "[смех]",
            "[аплодисмент",
            "полицейская сирена",
            "сирена",
            "звук сирены",
            "playing music",
            "*music*",
        )
        # Word-boundary matching: stop short patterns like "бо"/"ну"
        # from matching inside real words ("тебе"/"нужно"). Python's
        # `\b` works on word chars (incl. Cyrillic via UNICODE flag,
        # which is the default in Python 3). For patterns that contain
        # punctuation/spaces (e.g. "[музыка]", "*music*", "продолжение
        # следует") the `\b` rule still anchors against the alnum edges
        # of those tokens, so they still match.
        #
        # Round 25 fix (F57): cache compiled regex patterns so we don't
        # re-compile 80+ regexes on every utterance. With every
        # utterance hitting this hot path 4-6 times (process_transcription,
        # collection guard, intent_judge entry, dispatch entry), the
        # savings are ~3-5ms per utterance on real Whisper output.
        # Substring-only patterns (brackets/asterisks) cached as None
        # so the dispatch logic below keeps its single shape.
        if not hasattr(self.__class__, "_HALL_RE_CACHE"):
            cache: dict[str, "re.Pattern | None"] = {}
            for pat in KNOWN_PATTERNS:
                if any(ch in pat for ch in "[]*"):
                    cache[pat] = None
                    continue
                try:
                    cache[pat] = re.compile(r"\b" + re.escape(pat) + r"\b")
                except re.error:
                    cache[pat] = None
            self.__class__._HALL_RE_CACHE = cache
        cache = self.__class__._HALL_RE_CACHE
        for pat in KNOWN_PATTERNS:
            compiled = cache.get(pat)
            if compiled is None:
                if pat in lower:
                    return True
                continue
            if compiled.search(lower):
                return True
        return False

    def _check_query_timeout(self) -> None:
        """Check if there's a pending query that has timed out, and check hot window expiry."""
        if self.state_manager.check_collection_timeout():
            query = self.state_manager.clear_collection()
            if query.strip():
                # Same provenance as the inline-text timeout path above
                # (`process_transcription`'s `_dispatch_source = "wake_collection"`).
                # Collection was opened by a confirmed wake — the timeout
                # just decided when to finalise. Preserve more-specific
                # sources like "hot_window" by only setting when unset.
                if self._dispatch_source is None:
                    self._dispatch_source = "wake_collection"
                self._dispatch_query(query)
            else:
                # Empty collection on timeout — user said just "Джарвіс"
                # with no follow-up question. Speak ack, then keep mic
                # open via hot window so user can give the command
                # without having to say "Джарвіс" a second time.
                # User report: "після того як він каже слухаю даниле
                # відразу закривається і не можна нечого зробити".
                debug_log("collection timed out with empty query — speaking ack + opening hot window", "voice")
                self._stop_thinking_tune()
                if self.tts and self.tts.enabled:
                    # ECHO TRACKING — track BEFORE speak so the echo
                    # detector knows what we're about to play. Without
                    # this, the mic captures "Слушаю, Данило" as if it
                    # were a user follow-up command → fed to intent
                    # judge → judge says directed=true → loop. Audit
                    # round 7 finding C1 (this was the smoking gun for
                    # "Jarvis talks to itself after empty wake").
                    ack_text = "Слушаю, Данило."
                    self.track_tts_start(ack_text)
                    # Hot-window activation deferred to the TTS completion
                    # callback so it fires AFTER the audio settles, not
                    # before. Previously `activate_hot_window()` ran
                    # synchronously, which opened the listening window
                    # while the speaker was still playing the ack →
                    # immediate self-capture.
                    def _ack_done():
                        try:
                            self.activate_hot_window()
                        except Exception as e:
                            debug_log(f"failed to activate hot window after ack: {e}", "voice")
                    try:
                        self.tts.speak(ack_text, completion_callback=_ack_done)
                    except TypeError:
                        # Some TTS backends don't accept the kwarg —
                        # fall back to sync speak + immediate activate.
                        self.tts.speak(ack_text)
                        try:
                            self.activate_hot_window()
                        except Exception as e:
                            debug_log(f"failed to activate hot window after ack (fallback): {e}", "voice")
                else:
                    # No TTS available — open hot window immediately.
                    try:
                        self.activate_hot_window()
                    except Exception as e:
                        debug_log(f"failed to activate hot window after ack (no TTS): {e}", "voice")

        # Also check hot window expiry - this ensures the timeout is enforced
        # even when there's no audio being processed
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

    def _on_audio(self, indata, frames, time_info, status):
        """Audio callback from sounddevice."""
        try:
            if self._should_stop or self._dictation_active:
                return
            self._callback_count += 1
            chunk = (indata.copy() if hasattr(indata, "copy") else indata)
            try:
                self._audio_q.put_nowait(chunk)
            except Exception:
                pass
            # Energy-spike interrupt — DISABLED BY DEFAULT.
            #
            # The original idea: monitor mic RMS during TTS, fire when
            # it exceeds a baseline. PROBLEM: when TTS plays through
            # external speakers (or even Mac speakers with low volume),
            # the mic never picks up TTS echo. So baseline stays at
            # room-noise level (~0.004 RMS). Any user speech (~0.04
            # RMS) then triggers — but instantly, on the FIRST frame.
            # Worse: the interrupt fires before TTS gets to say a
            # single word, so the assistant looks like it can't speak.
            # User report: "у нього дуже короткий час прослуховування
            # і він відразу… переходить в жовтий вигляд".
            #
            # Cleaner approaches that DO work:
            #   1. HUD coin click → write control.json "interrupt_tts"
            #   2. Wake-word "Джарвіс" mid-TTS — handled in
            #      _process_transcript pre-emptive check
            #   3. Stop-word "стоп"/"досить" — same path
            # The energy-spike code stays in place but gated by
            # cfg.voice_interrupt_energy_enabled (default False).
            try:
                if (
                    self.tts is not None
                    and self.tts.is_speaking()
                    and np is not None
                    and bool(getattr(self.cfg, "voice_interrupt_energy_enabled", False))
                ):
                    arr = chunk if hasattr(chunk, "shape") else None
                    if arr is None:
                        return
                    if arr.ndim > 1:
                        arr = arr[:, 0]
                    rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float32)))))
                    # Initialise the rolling state lazily.
                    if not hasattr(self, "_tts_rms_baseline"):
                        self._tts_rms_baseline = rms
                        self._tts_spike_frames = 0
                        self._tts_last_interrupt_ts = 0.0
                    # Exponential moving average of TTS-echo baseline
                    # (slow update — only when we're NOT spiking, so
                    # spikes don't pollute the baseline).
                    spike_ratio = float(getattr(self.cfg, "voice_interrupt_spike_ratio", 1.5))
                    if rms < self._tts_rms_baseline * spike_ratio:
                        # Calm period — update baseline.
                        self._tts_rms_baseline = 0.95 * self._tts_rms_baseline + 0.05 * rms
                        self._tts_spike_frames = max(0, self._tts_spike_frames - 1)
                    else:
                        # Spike! count consecutive elevated frames.
                        self._tts_spike_frames += 1
                    # ABSOLUTE threshold — when user yells or just speaks
                    # at a normal volume while TTS plays through external
                    # speakers (mic doesn't hear TTS echo, so the
                    # relative-baseline check never triggers). 0.025 is
                    # ~normal speaking voice; below that mic baseline noise.
                    abs_rms = float(getattr(self.cfg, "voice_interrupt_absolute_rms", 0.025))
                    abs_spike = rms >= abs_rms
                    if abs_spike:
                        self._tts_spike_frames = max(self._tts_spike_frames, 8)
                    # 20ms per frame × 8 = 160ms of sustained spike
                    spike_threshold = int(getattr(self.cfg, "voice_interrupt_spike_frames", 8))
                    now = time_info.inputBufferAdcTime if time_info else 0.0
                    cooldown = 1.0  # don't fire more than once per 1.0s
                    if (self._tts_spike_frames >= spike_threshold
                            and (now - self._tts_last_interrupt_ts) > cooldown):
                        self._tts_last_interrupt_ts = now
                        self._tts_spike_frames = 0
                        debug_log(
                            f"⚡ TTS interrupted by mic energy spike "
                            f"(rms={rms:.4f} baseline={self._tts_rms_baseline:.4f})",
                            "voice",
                        )
                        try:
                            print("  ⏸  Перебиваю — чую тебе", flush=True)
                            self._interrupt_tts(reason="mic-energy spike")
                            # Clear echo-detector's last_tts so the next
                            # transcript isn't auto-rejected as echo.
                            # Audit round 14 fix C3: route through helper.
                            try:
                                self.echo_detector.clear_last_tts_text()
                            except Exception:
                                pass
                            # Drop queued mic audio (likely TTS echo).
                            while not self._audio_q.empty():
                                self._audio_q.get_nowait()
                            self.state_manager.start_collection("")
                            self._set_face_state_listening()
                        except Exception as e:
                            debug_log(f"interrupt-on-spike failed: {e}", "voice")
                else:
                    # Not speaking — clear spike counter so old spikes
                    # don't bleed into the next TTS session.
                    if hasattr(self, "_tts_spike_frames"):
                        self._tts_spike_frames = 0
            except Exception:
                pass
        except Exception:
            return

    def _determine_whisper_backend(self) -> str:
        """Determine which Whisper backend to use based on config and availability."""
        backend_pref = getattr(self.cfg, "whisper_backend", "auto")

        if backend_pref == "remote":
            # Caller (_init_whisper) validates URL + token and falls
            # back to MLX if either is missing.
            return "remote"

        if backend_pref == "mlx":
            if MLX_WHISPER_AVAILABLE:
                return "mlx"
            debug_log("MLX Whisper requested but not available, falling back to faster-whisper", "voice")
            return "faster-whisper"

        if backend_pref == "faster-whisper":
            return "faster-whisper"

        # Auto mode: prefer MLX on Apple Silicon
        if MLX_WHISPER_AVAILABLE and _is_apple_silicon():
            return "mlx"

        return "faster-whisper"

    def _apply_whisper_load_success(
        self, model_name: str, try_device: str, try_compute: str,
        device: str, compute: str, cpu_threads: int,
        context: str = "",
    ) -> str:
        """Record state and print diagnostics after a successful Whisper model load.

        Returns the resolved device string.
        """
        ct2_model = getattr(self.model, "model", None)
        resolved_device = str(getattr(ct2_model, "device", try_device)).lower()
        debug_log(
            f"faster-whisper initialised{context}: name={model_name}, "
            f"device={resolved_device}, compute={try_compute}, "
            f"cpu_threads={cpu_threads}",
            "voice",
        )
        self._whisper_device = resolved_device

        if try_device != device and device in ("auto", "cuda"):
            print("     ⚠️  CUDA not available, using CPU (this may be slower)", flush=True)
            print("     💡 Tip: Install NVIDIA CUDA toolkit for faster speech recognition", flush=True)
        if try_compute != compute:
            print(f"     ⚠️  Using '{try_compute}' compute type ('{compute}' not supported)", flush=True)
        if resolved_device == "cpu":
            print(f"     ⚡ CPU mode: using {cpu_threads} threads with optimised decoding", flush=True)

        suffix = f" ({context})" if context else ""
        print(f"     🎤 Whisper '{model_name}' loaded on {resolved_device}{suffix}", flush=True)
        return resolved_device

    def _start_llm_warmup(self) -> list[threading.Thread]:
        """Pre-load chat and intent judge models into Ollama memory.

        Starts up to two daemon threads concurrently so warmup overlaps
        with Whisper initialisation. When both models point at the same
        Ollama model, a single warmup covers both (Ollama loads the
        weights once; ``keep_alive`` keeps them resident for every caller).

        Results land in ``self._llm_warmup_results`` keyed by role. The
        caller joins the returned threads with a shared deadline before
        announcing "Listening!" so the ready state actually means ready.
        """
        self._llm_warmup_results: dict[str, tuple[str, bool]] = {}

        chat_model = str(getattr(self.cfg, "ollama_chat_model", "") or "").strip()
        base_url = str(getattr(self.cfg, "ollama_base_url", "") or "").strip()
        chat_timeout = max(float(getattr(self.cfg, "llm_tools_timeout_sec", 8.0)), 60.0)
        judge = self._intent_judge
        judge_model = judge.config.model if judge is not None else ""
        shared_judge = bool(chat_model) and judge_model == chat_model

        # Tool router — only warmed when the LLM selection strategy is active
        # AND the router points at a model distinct from chat/judge. An empty
        # `tool_router_model` means "reuse the intent-judge model (small, fast,
        # already loaded for wake-word paths) or the chat model as a last
        # resort". Resolve the same way the reply engine does so warmup targets
        # whatever the engine will actually call. Skipping warmup for non-LLM
        # strategies avoids loading a model that won't be used this session.
        strategy = str(getattr(self.cfg, "tool_selection_strategy", "") or "").lower()
        # Use the same resolution helper the reply engine uses so warmup
        # targets the model the engine will actually call. Keeping a single
        # source of truth prevents drift between warmup and runtime.
        from ..reply.engine import resolve_tool_router_model
        router_model_effective = resolve_tool_router_model(self.cfg)
        router_model = router_model_effective if strategy == "llm" else ""
        shared_router = bool(router_model) and router_model in {chat_model, judge_model}

        threads: list[threading.Thread] = []

        if chat_model and base_url:
            def _warm_chat() -> None:
                ok = warm_up_ollama_model(base_url, chat_model, timeout=chat_timeout)
                # Additional KV-cache prefill with a realistic-sized chat
                # system prompt. Bare weight-load (above) leaves the cache
                # cold, so the first real "Джарвіс, привіт" pays ~120-180s
                # prompt-eval on qwen2.5:3b CPU @ ~25 tok/s and busts the
                # chat timeout. Pre-feeding ~1500 tokens of representative
                # context primes the prefix cache so the first real chat
                # call is ~4-8s instead.
                if ok:
                    try:
                        import requests as _rq
                        # CRITICAL: must match _voice_direct_chat() exactly,
                        # otherwise KV-cache prefix doesn't hit and the first
                        # real voice call pays cold-cache latency. Both pull
                        # from VOICE_STATIC_SYSTEM_PROMPT (module constant)
                        # so they CAN'T drift.
                        warm_lang = self._language_directive("ru")
                        warm_resp = _rq.post(
                            f"{base_url.rstrip('/')}/api/chat",
                            json={
                                "model": chat_model,
                                "messages": [
                                    {"role": "system", "content": VOICE_STATIC_SYSTEM_PROMPT},
                                    {"role": "system", "content": warm_lang},
                                    {"role": "user", "content": "[warmup] привет"},
                                ],
                                "stream": False,
                                "keep_alive": "24h",
                                # Match _voice_direct_chat's think=False EXACTLY,
                                # otherwise qwen3 KV-cache slot is different
                                # (thinking vs non-thinking prefix) → first
                                # real call pays cold-cache rebuild.
                                "think": False,
                                "options": {
                                    # Match _voice_direct_chat options
                                    # EXACTLY so warmup primes the same
                                    # cache slot. Any drift = cache miss
                                    # on first real call = 15-25s pause.
                                    "temperature": 0.4,
                                    "num_predict": 220,  # MUST match real call exactly (KV-cache slot)
                                    "repeat_penalty": 1.2,
                                    "repeat_last_n": 192,
                                    "presence_penalty": 0.3,
                                    "frequency_penalty": 0.2,
                                    # MUST match _voice_direct_chat (2048).
                                    # Different num_ctx → different KV-cache
                                    # slot → cold rebuild on first real call.
                                    "num_ctx": 2048,
                                    "num_thread": 4,
                                    "num_batch": 256,
                                },
                            },
                            timeout=240.0,
                        )
                        chat_warm_ok = warm_resp.status_code == 200
                        debug_log(
                            f"chat KV-cache warmup: "
                            f"{'ok' if chat_warm_ok else f'failed HTTP {warm_resp.status_code}'}",
                            "voice",
                        )
                    except Exception as e:
                        debug_log(f"chat KV-cache warmup error: {e}", "voice")
                self._llm_warmup_results["chat"] = (chat_model, ok)
                # When chat and judge share a model, one warmup covers both.
                if shared_judge:
                    self._llm_warmup_results["judge"] = (chat_model, ok)
                # Router reusing chat_model is already covered.
                if router_model and router_model == chat_model:
                    self._llm_warmup_results["router"] = (chat_model, ok)

            threads.append(threading.Thread(target=_warm_chat, daemon=True, name="warmup-chat"))

        # ALWAYS run judge.warm_up() — even when shared_judge=True. The chat
        # warmup loads weights but does NOT seed the judge's KV-cache for its
        # 500-token system prompt. Without cache seeding, the first wake-word
        # judge call pays ~25s prompt-eval and busts the 45s timeout. The
        # weights are already resident at this point so this is effectively a
        # KV-cache prefill only.
        if judge is not None:
            def _warm_judge() -> None:
                ok = judge.warm_up()
                self._llm_warmup_results["judge"] = (judge_model, ok)
                if router_model and router_model == judge_model:
                    self._llm_warmup_results["router"] = (judge_model, ok)

            threads.append(threading.Thread(target=_warm_judge, daemon=True, name="warmup-judge"))

        if router_model and base_url and not shared_router:
            def _warm_router() -> None:
                ok = warm_up_ollama_model(base_url, router_model, timeout=chat_timeout)
                self._llm_warmup_results["router"] = (router_model, ok)

            threads.append(threading.Thread(target=_warm_router, daemon=True, name="warmup-router"))

        for t in threads:
            t.start()

        debug_log(
            f"LLM warmup started (chat={chat_model or 'n/a'}, "
            f"judge={judge_model or 'n/a'}, router={router_model or 'n/a'}, "
            f"shared_judge={shared_judge}, shared_router={shared_router})",
            "voice",
        )
        return threads

    def _start_llm_keepalive(self) -> None:
        """Periodic Ollama keepalive ping to prevent model eviction.

        Audit round 24 fix (F48): even though chat calls pass
        ``keep_alive=24h``, Ollama on Hetzner CCX23 occasionally
        evicts the qwen3:8b weights under memory pressure (host
        OOM-reaper, other guests on shared infrastructure). A reload
        costs ~25-35s of cold prompt-eval — the source of user's
        "дуже довго чкати відповіді" complaint. A lightweight ping
        every 30s (1-token output, ~50ms server work) is cheap and
        keeps the KV cache hot.

        We don't ping during active conversation — only when the
        daemon has been IDLE for >20s. This avoids piling extra
        load on the LLM during a streaming reply.
        """
        import threading as _t
        import time as _time

        chat_model = self._llm_warmup_results.get("chat", (None, False))[0] if hasattr(self, "_llm_warmup_results") else None
        if not chat_model:
            # Fall back to config — warmup may not have populated
            # _llm_warmup_results yet at the time this is called.
            try:
                from ..config import resolve_chat_model
                chat_model = resolve_chat_model(self.cfg)
            except Exception:
                chat_model = getattr(self.cfg, "chat_model", "qwen3:8b")
        if not chat_model:
            debug_log("LLM keepalive: no chat_model resolved — skipping", "voice")
            return

        # Round 28 (F72): also keep intent_judge model resident. Live
        # evidence (events.jsonl): "intent_judge timeout after 45.0s"
        # repeated every 60-90s because Ollama evicts qwen2.5:3b after
        # the configured intent_judge keep_alive (10m) expires between
        # voice turns. The chat-side keep_alive=24h kept the 8b chat
        # model hot but the 3b intent judge model died → every wake
        # paid a 45s cold-load tax on the very first decision.
        intent_model = getattr(self.cfg, "intent_judge_model", None)
        if not intent_model and self._intent_judge is not None:
            intent_model = getattr(self._intent_judge.config, "model", None)
        # Dedupe: if intent_judge_model == chat_model, only ping once.
        ping_models = [chat_model]
        if intent_model and intent_model != chat_model:
            ping_models.append(intent_model)

        base_url = getattr(self.cfg, "ollama_base_url", "http://127.0.0.1:11434").rstrip("/")
        ping_url = f"{base_url}/api/generate"
        ping_interval = 30.0   # seconds between pings
        idle_threshold = 20.0  # only ping when daemon idle this long

        # Round 25 fix (F51): the round-24 implementation referenced
        # ``self._stop_event`` (does not exist on Listener — only on
        # tune_player.TunePlayer) and called ``requests.post`` without
        # importing requests at module scope. Live log:
        #   ``LLM keepalive ping failed: name 'requests' is not defined``
        # — the feature was 100% dead code since round 24. Now using
        # the existing ``self._should_stop`` flag pattern (set by
        # daemon shutdown at line 1010) and the module-level
        # ``requests`` import added at the top of this file.
        def _ping_loop() -> None:
            # Stagger first ping so we don't collide with initial warmup
            _time.sleep(ping_interval)
            while not self._should_stop:
                try:
                    # Skip ping if a reply is in flight or TTS is speaking —
                    # the model is already loaded and we don't want to
                    # queue extra prompt-eval work behind the user's call.
                    if self.tts and self.tts.is_speaking():
                        _time.sleep(ping_interval)
                        continue
                    # Also skip if a hot window / collection is active —
                    # user just spoke or is about to speak.
                    try:
                        if self.state_manager.is_collecting():
                            _time.sleep(ping_interval)
                            continue
                    except Exception:
                        pass
                    # Only ping when we've been idle "long enough" — avoids
                    # double-loading model right after a real query.
                    last_activity = self._last_user_activity_ts or 0.0
                    # R34-S54.1 Phase 7b: monotonic — paired with the
                    # _last_user_activity_ts anchor in _dispatch_query.
                    if last_activity > 0 and (_time.monotonic() - last_activity) < idle_threshold:
                        _time.sleep(ping_interval)
                        continue

                    # Minimal ping: 1-token output. We use /api/generate
                    # rather than /api/chat so it doesn't pollute any
                    # chat-side KV cache prefix. F72: ping each model
                    # we care about (chat + intent_judge if distinct).
                    for _model in ping_models:
                        try:
                            requests.post(
                                ping_url,
                                json={
                                    "model": _model,
                                    "prompt": " ",
                                    "stream": False,
                                    "options": {"num_predict": 1, "temperature": 0.0},
                                    "keep_alive": "24h",
                                },
                                timeout=10.0,
                                # R34-S56.1 Phase 9b (P2): close socket
                                # so the keepalive itself doesn't
                                # cache a dead Tailscale path between
                                # idle ticks (every 60-180s).
                                headers={"Connection": "close"},
                            )
                            debug_log(f"LLM keepalive ping ok ({_model})", "voice")
                        except Exception as e_inner:
                            debug_log(f"LLM keepalive ping {_model} failed: {e_inner}", "voice")
                except Exception as e:
                    debug_log(f"LLM keepalive loop error: {e}", "voice")
                _time.sleep(ping_interval)

        t = _t.Thread(target=_ping_loop, daemon=True, name="jarvis-llm-keepalive")
        t.start()
        debug_log(
            f"LLM keepalive started (models={ping_models}, interval={ping_interval}s)",
            "voice",
        )

    def _start_hud_control_watcher(self) -> None:
        """Poll the HUD→daemon control file for session-management commands.

        The Electron HUD writes JSON like {"action":"end_session","ts":...}
        into `~/Library/Application Support/jarvis/control.json` when the
        user right-clicks the coin and picks an action from the context
        menu. This watcher polls the file once per second and dispatches:

          * end_session  → state_manager.force_end_session()
          * mute         → tts.interrupt() (handled here for symmetry)

        The file is deleted after each successful action so the same
        command never fires twice. Polling (not fs.watch) is used because
        fs.watch on macOS is unreliable across FUSE/network mounts and
        the user's ~/Library/ on iCloud Drive sometimes misses events.
        """
        import os
        import json
        import threading as _t

        ctrl_dir = os.path.expanduser("~/Library/Application Support/jarvis")
        ctrl_path = os.path.join(ctrl_dir, "control.json")
        try:
            os.makedirs(ctrl_dir, exist_ok=True)
        except Exception:
            pass

        def _watch():
            last_ts = 0.0
            # Audit round 13 fix: snapshot our own UID once so we can
            # reject control.json files written by another local user
            # (multi-user macOS). The HUD runs as the same user as
            # the daemon, so any other-UID writer is suspicious.
            our_uid = os.getuid()
            while not getattr(self.state_manager, "_should_stop", False):
                try:
                    if os.path.exists(ctrl_path):
                        # Reject other-UID writers — local privilege gap
                        # otherwise lets any local process drive Jarvis
                        # state (end_session, interrupt_tts, mute).
                        try:
                            st = os.stat(ctrl_path)
                            if st.st_uid != our_uid:
                                debug_log(
                                    f"HUD control: rejecting foreign-UID write (uid={st.st_uid})",
                                    "voice",
                                )
                                try:
                                    os.remove(ctrl_path)
                                except Exception:
                                    pass
                                time.sleep(0.1)
                                continue
                        except FileNotFoundError:
                            time.sleep(0.1)
                            continue
                        with open(ctrl_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        ts = float(data.get("ts", 0))
                        action = str(data.get("action", "")).lower().strip()
                        if ts > last_ts and action:
                            last_ts = ts
                            debug_log(f"HUD control: action='{action}' ts={ts}", "voice")
                            if action in ("end_session", "stop_session", "session_end"):
                                # Audit round 20 fix (CRITICAL — user
                                # reported "кнопки завершення не
                                # працюють"): previously this branch
                                # was guarded by ``tts.is_speaking()``,
                                # so during the THINKING window (LLM
                                # streaming, first sentence not yet
                                # dequeued by Piper) the click was a
                                # no-op. ``_interrupt_tts`` is safe to
                                # call unconditionally — it sets the
                                # ``_stream_abort`` flag (which breaks
                                # the iter_lines loop in
                                # ``_voice_direct_chat``) AND drains
                                # the TTS engine. Now end_session
                                # ALWAYS aborts the in-flight reply,
                                # mirroring the voice "стоп" path at
                                # lines 1455-1469.
                                self._interrupt_tts(reason="HUD end_session")
                                # F75: under lock so concurrent
                                # _voice_direct_chat reads stay consistent.
                                if hasattr(self, "_dialog_history"):
                                    with self._dialog_history_lock:
                                        self._dialog_history.clear()
                                # Mirror voice-stop cleanup: reset
                                # language + clear any pending
                                # confirmation/lang-switch/upgrade so
                                # the next session starts fresh.
                                try:
                                    self._active_language = "ru"
                                except Exception:
                                    pass
                                try:
                                    with self._pending_lock:
                                        self._pending_lang_switch = None
                                        self._pending_action = None
                                        self._pending_upgrade = None
                                        self._pending_confirmation = None
                                except Exception:
                                    pass
                                self.state_manager.force_end_session()
                                # Drain captured-audio queue so a
                                # half-utterance mid-click is not
                                # dispatched after the session ends.
                                try:
                                    while not self._audio_q.empty():
                                        self._audio_q.get_nowait()
                                except Exception:
                                    pass
                                # Reset state-manager collection so a
                                # half-collected query doesn't get
                                # dispatched on the next silence
                                # timeout.
                                try:
                                    self.state_manager.clear_collection()
                                except Exception:
                                    pass
                                print("  🛑 Session ended (HUD right-click)", flush=True)
                            elif action in ("interrupt_tts", "stop_speaking", "interrupt"):
                                # Audit round 20 fix: dropped the
                                # ``is_speaking()`` guard. The handler
                                # MUST fire even when no audio is
                                # playing yet — that's the THINKING
                                # window where LLM stream is running
                                # and ``_stream_abort`` is the only
                                # way to short-circuit it. Without
                                # this, clicking ⏸ in the first 1-2 s
                                # of a long reply did nothing.
                                self._interrupt_tts(reason="HUD interrupt_tts")
                                print("  ⏸  TTS interrupted (HUD click)", flush=True)
                                # Clear echo-detector last_tts so the
                                # next utterance isn't auto-rejected.
                                # Audit round 14 fix C3: route through helper.
                                try:
                                    self.echo_detector.clear_last_tts_text()
                                except Exception:
                                    pass
                                self._set_face_state_listening()
                            elif action == "confirm":
                                # User confirmed a pending action.
                                # Audit round 11 fix C1: immediately
                                # execute the action so the click works
                                # standalone; previously the flag just
                                # sat there until the next voice arrived.
                                with self._pending_lock:
                                    self._pending_confirmation = "yes"
                                print("  ✅ Confirmed (HUD)", flush=True)
                                self._consume_pending_confirmation_from_hud()
                            elif action == "deny":
                                with self._pending_lock:
                                    self._pending_confirmation = "no"
                                print("  ❌ Denied (HUD)", flush=True)
                                self._consume_pending_confirmation_from_hud()
                            elif action == "mute":
                                # Audit round 20 fix: drop the
                                # ``self.tts and self.tts.enabled``
                                # guard. ``_interrupt_tts`` is None-
                                # safe and ALSO sets ``_stream_abort``
                                # (the only way to break the LLM
                                # stream loop). Even on a config with
                                # TTS disabled, mute should still kill
                                # an in-flight LLM call so the daemon
                                # doesn't keep generating tokens after
                                # the user explicitly silenced it.
                                self._interrupt_tts(reason="HUD mute")
                                print("  🔇 TTS muted (HUD right-click)", flush=True)
                            elif action in ("force_finalize", "confirm_input"):
                                # User clicked ✓ — context-aware:
                                #
                                # IF there's a pending action awaiting
                                #   confirmation → treat as YES (confirm).
                                # ELIF speech is currently being captured
                                #   → finalise the utterance NOW (skip
                                #   VAD silence wait).
                                # ELIF a query is currently being collected
                                #   (hot window, awaiting more user input
                                #   to dispatch) → force-dispatch what we
                                #   already have. This is the path the
                                #   user complained about (round 24): a
                                #   transcript was shown in captions but
                                #   nothing happened on click — because
                                #   the daemon was waiting in COLLECTING
                                #   state for more audio. Force-dispatch
                                #   ends that wait immediately.
                                # ELIF TTS is currently speaking → treat
                                #   as "I want to interrupt and ask
                                #   something else". Aborting TTS is
                                #   handled by the wake-interrupt path
                                #   normally, but if user can't say wake
                                #   right now (background noise, hand-
                                #   held), pressing Confirm gives them
                                #   the same escape hatch.
                                # ELSE → no-op (with a log entry — the
                                #   button click DID register, just no
                                #   actionable state to act on).
                                #
                                # Round 24 fix (F47): expanded coverage
                                # so the button always does SOMETHING
                                # useful or, in the genuine no-op case,
                                # at least leaves a log line so the
                                # user knows their click was received.
                                with self._pending_lock:
                                    has_pending = (
                                        self._pending_confirmation is None and (
                                            self._pending_action is not None
                                            or self._pending_lang_switch is not None
                                            or self._pending_upgrade is not None
                                        )
                                    )
                                    if has_pending:
                                        self._pending_confirmation = "yes"
                                if has_pending:
                                    print("  ✅ Confirmed pending action (HUD ✓)", flush=True)
                                    self._consume_pending_confirmation_from_hud()
                                elif self.is_speech_active:
                                    self._force_finalize_requested = True
                                    print("  ✓ Force-finalise (HUD)", flush=True)
                                elif self.state_manager.is_collecting():
                                    # Round 24 fix (F47): user pressed
                                    # Confirm while we were collecting —
                                    # treat as "send what I have now".
                                    # Force collection timeout so the
                                    # next loop iteration dispatches.
                                    try:
                                        self.state_manager.force_collection_timeout()
                                        print("  ✓ Force-dispatch collection (HUD ✓)", flush=True)
                                    except Exception as _e:
                                        debug_log(
                                            f"force_finalize collection-dispatch failed: {_e}",
                                            "voice",
                                        )
                                elif self.tts and self.tts.is_speaking():
                                    # Round 24 fix (F47): user pressed
                                    # Confirm during TTS — treat as
                                    # "OK, stop and listen". Cleaner
                                    # than waiting through a long reply
                                    # the user no longer wants.
                                    self._interrupt_tts(reason="HUD confirm during TTS")
                                    print("  ⏸  TTS interrupted (HUD ✓ during reply)", flush=True)
                                else:
                                    debug_log(
                                        "force_finalize: no pending action / no speech / "
                                        "no collection / no TTS — click acknowledged but no-op",
                                        "voice",
                                    )
                                    print(
                                        "  ✓ HUD button click registered (no action available right now)",
                                        flush=True,
                                    )
                            elif action in ("cancel_utterance", "reset_input"):
                                # User clicked ✗ — context-aware (mirrors
                                # the confirm button logic above):
                                #
                                # IF pending action awaiting confirmation
                                #   → treat as NO (deny it).
                                # ELIF speech being captured → drop audio.
                                with self._pending_lock:
                                    has_pending = (
                                        self._pending_confirmation is None and (
                                            self._pending_action is not None
                                            or self._pending_lang_switch is not None
                                            or self._pending_upgrade is not None
                                        )
                                    )
                                    if has_pending:
                                        self._pending_confirmation = "no"
                                if has_pending:
                                    print("  ❌ Denied pending action (HUD ✗)", flush=True)
                                    self._consume_pending_confirmation_from_hud()
                                elif self.is_speech_active:
                                    self._cancel_utterance_requested = True
                                    print("  ✗ Cancel utterance (HUD)", flush=True)
                                else:
                                    debug_log(
                                        "cancel_utterance ignored — no active speech",
                                        "voice",
                                    )
                            # Delete to avoid re-firing on next poll.
                            try:
                                os.remove(ctrl_path)
                            except Exception:
                                pass
                except Exception as e:
                    debug_log(f"HUD control watcher error: {e}", "voice")
                # 100ms poll (was 500ms). User report (May 16): "перемикається
                # на підтвердження довго" — half-second click→action delay
                # felt sluggish. 100ms = 5× faster button feedback, costs ~0%
                # CPU (one stat() + 0/1 small JSON load per tick on idle).
                time.sleep(0.1)

        t = _t.Thread(target=_watch, daemon=True, name="hud-control-watcher")
        t.start()
        debug_log(f"HUD control watcher started ({ctrl_path})", "voice")

    def _weather_example(self, wake_title: str) -> str:
        """Return the weather query example for the startup banner.

        Shows the plain form when a location source is configured, or the
        [your city] placeholder form so the user knows to supply a city.
        """
        location_enabled = getattr(self.cfg, "location_enabled", True)
        location_auto_detect = getattr(self.cfg, "location_auto_detect", True)
        location_ip_address = getattr(self.cfg, "location_ip_address", None)
        location_known = (
            location_enabled
            and (location_auto_detect or bool(location_ip_address))
            and is_location_available()
        )
        if location_known:
            return f"\"How's the weather, {wake_title}?\""
        return f"\"How's the weather in [your city], {wake_title}?\""

    def run(self) -> None:
        """Main voice listening loop."""
        if sd is None:
            debug_log("sounddevice not available", "voice")
            print("  ❌ Audio system not available - sounddevice failed to load", flush=True)
            return

        # Verify PortAudio is working by querying devices (catches Windows DLL issues)
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            debug_log(f"PortAudio initialised: {len(input_devices)} input device(s) found", "voice")
            if not input_devices:
                print("  ❌ No microphone found. Please connect a microphone.", flush=True)
                return
        except Exception as e:
            debug_log(f"PortAudio device query failed: {e}", "voice")
            print(f"  ❌ Audio system error: {e}", flush=True)
            print("     PortAudio may not be properly installed", flush=True)
            if sys.platform == 'linux':
                print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
            return

        # Windows 11: Test microphone permission by attempting a brief recording
        # This catches privacy settings that silently block audio access.
        # A 5-second timeout prevents indefinite hangs when Windows blocks
        # the audio device at the system level without raising an error.
        # Uses InputStream (not sd.rec) so the stream can be explicitly closed
        # on timeout, avoiding resource leaks that could block later audio init.
        if sys.platform == 'win32':
            try:
                print("  🔐 Checking microphone permission...", flush=True)
                mic_ok = threading.Event()
                mic_error: list = [None]
                mic_stream: list = [None]

                def _mic_check():
                    try:
                        stream = sd.InputStream(
                            samplerate=self._samplerate, channels=1,
                            dtype="float32", blocksize=int(self._samplerate * 0.1),
                        )
                        mic_stream[0] = stream
                        stream.start()
                        time.sleep(0.15)
                        stream.stop()
                        stream.close()
                        mic_stream[0] = None
                        mic_ok.set()
                    except Exception as exc:
                        mic_error[0] = exc

                check_thread = threading.Thread(target=_mic_check, daemon=True)
                check_thread.start()
                check_thread.join(timeout=5.0)

                if check_thread.is_alive():
                    # Clean up the stream if the thread is still blocked
                    debug_log("microphone permission check timed out after 5s", "voice")
                    stream_ref = mic_stream[0]
                    if stream_ref is not None:
                        try:
                            stream_ref.abort()
                            stream_ref.close()
                        except Exception:
                            pass
                    print("  ⚠️  Microphone permission check timed out", flush=True)
                    print("     This may indicate Windows is blocking microphone access.", flush=True)
                    print("     Continuing anyway — voice input may not work.", flush=True)
                elif mic_error[0] is not None:
                    e = mic_error[0]
                    error_str = str(e).lower()
                    print(f"  ❌ Microphone permission check failed: {e}", flush=True)
                    if "unapproved" in error_str or "denied" in error_str or "access" in error_str or "-9999" in str(e):
                        print("", flush=True)
                        print("  ┌─────────────────────────────────────────────────────────┐", flush=True)
                        print("  │  🔒 MICROPHONE ACCESS BLOCKED BY WINDOWS               │", flush=True)
                        print("  │                                                         │", flush=True)
                        print("  │  To fix this:                                          │", flush=True)
                        print("  │  1. Open Windows Settings                              │", flush=True)
                        print("  │  2. Go to Privacy & security → Microphone              │", flush=True)
                        print("  │  3. Turn ON 'Microphone access'                        │", flush=True)
                        print("  │  4. Turn ON 'Let apps access your microphone'          │", flush=True)
                        print("  │  5. Turn ON 'Let desktop apps access your microphone'  │", flush=True)
                        print("  │                                                         │", flush=True)
                        print("  │  Then restart Jarvis.                                  │", flush=True)
                        print("  └─────────────────────────────────────────────────────────┘", flush=True)
                        print("", flush=True)
                    return
                elif mic_ok.is_set():
                    print("  ✅ Microphone permission OK", flush=True)
                else:
                    print("  ⚠️  Microphone returned empty audio", flush=True)
            except Exception as e:
                debug_log(f"microphone permission check error: {e}", "voice")
                print(f"  ⚠️  Microphone check error: {e}", flush=True)

        # Kick off LLM warmups in parallel with Whisper load so the first
        # user engagement doesn't pay cold-load cost on either model. All
        # warmup output (Whisper + LLMs) is indented under this header to
        # visually group the phase.
        print("  🔥 Warming up models...", flush=True)
        # R34-S54.1 Phase 7b: monotonic — anchor for the warmup deadline
        # math below. NTP step during warmup (very common right after
        # mac wake-from-sleep, where the LLM warmup happens) could
        # either freeze the join (deadline jumped past now) or skip the
        # wait entirely (deadline jumped behind now).
        self._llm_warmup_started_at = time.monotonic()
        self._llm_warmup_threads = self._start_llm_warmup()

        # Audit round 24 fix (F48): periodic keepalive ping. Even
        # though chat calls pass ``keep_alive=24h``, Ollama on
        # Hetzner can OOM-evict a model under memory pressure
        # (other guests on the box, OS reclaim). A reload =
        # 25-35s cold prompt-eval — the source of the user's
        # "дуже довго чекати" complaint. A 30s lightweight ping
        # is cheap (1-token output, ~50ms server work, ~50KB net)
        # and reliably keeps the model resident even under
        # transient memory pressure. The thread is daemon=True
        # so it exits with the process.
        self._start_llm_keepalive()

        # Start HUD control-file watcher — picks up "end_session" / "stop"
        # commands written by the HUD's right-click menu.
        self._start_hud_control_watcher()

        # Determine and initialise Whisper backend
        self._whisper_backend = self._determine_whisper_backend()
        model_name = getattr(self.cfg, "whisper_model", "small")

        # Remote backend — no local model loaded. We just stash the URL
        # and token; _transcribe_remote handles the HTTP POST per
        # utterance. Saves ~1.5GB Mac RAM and removes the 30s cold-load
        # on first transcription. The server (whisper-service on
        # Hetzner) has 6+ GB headroom — easily fits large-v3-turbo.
        #
        # CRITICAL: this whole inline init block lives inside run().
        # The original `return` here exited run() entirely and the
        # voice loop never started — silent dead daemon. We use a flag
        # and skip the local-load branches below instead.
        skip_local_load = False
        if self._whisper_backend == "remote":
            remote_url = getattr(self.cfg, "whisper_remote_url", "")
            remote_token = getattr(self.cfg, "whisper_remote_token", "")
            if not remote_url or not remote_token:
                print(
                    "  ❌ whisper_backend='remote' but whisper_remote_url / "
                    "whisper_remote_token are empty in config. "
                    "Falling back to MLX.", flush=True,
                )
                self._whisper_backend = "mlx" if MLX_WHISPER_AVAILABLE else "faster-whisper"
            else:
                self._remote_whisper_url = remote_url
                self._remote_whisper_token = remote_token
                # Probe /health so we fail fast at startup, not on the
                # user's first utterance.
                try:
                    import requests as _rq
                    r = _rq.get(f"{remote_url}/health", timeout=5)
                    if r.status_code == 200:
                        info = r.json()
                        srv_model = info.get("model_name", "?")
                        loaded = info.get("model_loaded")
                        print(
                            f"     🌐 Remote Whisper ready: {remote_url} "
                            f"(model={srv_model}, loaded={loaded})", flush=True,
                        )
                    else:
                        print(
                            f"  ⚠️  whisper-service /health returned HTTP "
                            f"{r.status_code} — will retry per-utterance",
                            flush=True,
                        )
                except Exception as e:
                    print(
                        f"  ⚠️  whisper-service unreachable at probe "
                        f"({remote_url}): {e}. Voice will still try at "
                        f"each utterance.", flush=True,
                    )
                # Skip the local-model-load branches below; control
                # continues into the audio-stream setup further down in
                # run() so the daemon actually starts listening.
                skip_local_load = True

        # Validate large-v3-turbo support for faster-whisper backend
        if not skip_local_load and model_name == "large-v3-turbo" and self._whisper_backend != "mlx":
            if not _is_faster_whisper_turbo_supported():
                debug_log(
                    "faster-whisper does not support large-v3-turbo, "
                    "falling back to large-v3", "voice",
                )
                print(
                    "  ⚠️  large-v3-turbo is not supported by the installed Whisper engine, "
                    "using large-v3 instead", flush=True,
                )
                model_name = "large-v3"

        if not skip_local_load and self._whisper_backend == "mlx":
            if not MLX_WHISPER_AVAILABLE:
                debug_log("MLX Whisper not available", "voice")
                print("  ❌ MLX Whisper not available. Install with: pip install mlx-whisper", flush=True)
                return

            self._mlx_model_repo = _get_mlx_model_repo(model_name)
            print(f"     🎤 Loading MLX Whisper '{model_name}' (Apple Silicon GPU)...", flush=True)

            max_retries = 4
            for attempt in range(max_retries + 1):
                try:
                    # Pre-load the model by doing a warmup transcription.
                    # Use low-amplitude noise (not silence) so the decoder actually runs —
                    # silent audio trips the no-speech short-circuit and leaves the decode
                    # path cold, so the first real utterance still pays the full cost.
                    if np is not None:
                        rng = np.random.default_rng(0)
                        warmup_audio = rng.standard_normal(self._samplerate).astype(np.float32) * 0.01
                        _ = mlx_whisper.transcribe(
                            warmup_audio,
                            path_or_hf_repo=self._mlx_model_repo,
                            language=None,
                        )
                        debug_log(f"MLX Whisper model pre-loaded: repo={self._mlx_model_repo}", "voice")

                    print(f"     🎤 MLX Whisper '{model_name}' ready (Apple Silicon GPU)", flush=True)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limited = (
                        any(x in error_str for x in ["429", "too many requests", "rate limit"])
                        or getattr(getattr(e, "response", None), "status_code", None) == 429
                    )
                    if is_rate_limited and attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        debug_log(f"rate limited loading MLX Whisper (attempt {attempt + 1}): {e}", "voice")
                        print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...", flush=True)
                        time.sleep(wait)
                        continue
                    debug_log(f"failed to initialise MLX Whisper: {e}", "voice")
                    print(f"  ❌ Failed to initialise MLX Whisper: {e}", flush=True)
                    if is_rate_limited:
                        print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                    return
        elif not skip_local_load:
            # faster-whisper backend (remote backend already returned above)
            if not FASTER_WHISPER_AVAILABLE:
                debug_log("faster-whisper not available", "voice")
                print("  ❌ faster-whisper not available. Install with: pip install faster-whisper", flush=True)
                return

            device = getattr(self.cfg, "whisper_device", "auto")
            compute = getattr(self.cfg, "whisper_compute_type", "int8")

            # On Windows, probe for CUDA runtime libraries before trying to
            # use them. faster-whisper/CTranslate2 lazily loads cuBLAS and
            # cuDNN during transcription, so without this check a model
            # that loaded fine on cuda will crash on the first audio chunk.
            resolved_device, missing_libs = _probe_windows_cuda_libraries(device)
            if missing_libs:
                _print_cuda_unavailable_hint(missing_libs)
            device = resolved_device

            # Build list of (device, compute_type) combinations to try
            # This handles both compute type fallbacks and CUDA -> CPU fallbacks
            configs_to_try = []

            # Start with preferred config
            compute_types = [compute]
            if compute == "int8":
                compute_types.extend(["float16", "float32"])
            elif compute == "float16":
                compute_types.append("float32")

            # Add preferred device with all compute types
            for ct in compute_types:
                configs_to_try.append((device, ct))

            # If device is "auto" or "cuda", add CPU fallback configs
            # This handles Windows without CUDA libraries
            if device in ("auto", "cuda"):
                for ct in compute_types:
                    configs_to_try.append(("cpu", ct))

            last_error = None
            used_device = device
            used_compute = compute
            for try_device, try_compute in configs_to_try:
                try:
                    cpu_threads = (os.cpu_count() or 4) if try_device in ("cpu", "auto") else 0
                    print(f"     🎤 Loading Whisper '{model_name}' (device={try_device}, compute={try_compute})...", flush=True)
                    self.model = WhisperModel(
                        model_name, device=try_device, compute_type=try_compute,
                        cpu_threads=cpu_threads,
                    )
                    self._apply_whisper_load_success(
                        model_name, try_device, try_compute,
                        device, compute, cpu_threads,
                    )
                    used_device = try_device
                    used_compute = try_compute
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check if this is a CUDA/GPU-related error that we should fall back from
                    is_cuda_error = any(x in error_str for x in [
                        "cuda", "cublas", "cudnn", "gpu", "nvidia",
                        ".dll is not found", "library", "ctypes"
                    ])
                    is_compute_error = any(x in error_str for x in [
                        "compute type", "int8", "float16"
                    ])

                    if is_cuda_error or is_compute_error:
                        debug_log(f"config ({try_device}, {try_compute}) failed, trying fallback: {e}", "voice")
                        continue

                    # Check for corrupted model cache (e.g. interrupted download)
                    is_corrupted_cache = "unable to open file" in error_str

                    if is_corrupted_cache:
                        debug_log(f"detected corrupted Whisper model cache: {e}", "voice")
                        print("  ⚠️  Whisper model cache appears corrupted, attempting recovery...", flush=True)

                        cache_cleared = _clear_corrupted_whisper_cache(str(e))
                        if cache_cleared:
                            try:
                                print(f"     🎤 Re-downloading Whisper '{model_name}'...", flush=True)
                                self.model = WhisperModel(
                                    model_name, device=try_device, compute_type=try_compute,
                                    cpu_threads=cpu_threads,
                                )
                                self._apply_whisper_load_success(
                                    model_name, try_device, try_compute,
                                    device, compute, cpu_threads,
                                    context="recovered",
                                )
                                used_device = try_device
                                used_compute = try_compute
                                last_error = None
                                break
                            except Exception as retry_e:
                                debug_log(f"retry after cache clear also failed: {retry_e}", "voice")
                                print(f"  ❌ Failed to load Whisper model after cache recovery: {retry_e}", flush=True)
                                return
                        else:
                            debug_log("could not clear corrupted cache automatically", "voice")
                            print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                            print("  💡 Try manually deleting the Whisper model cache directory and restarting", flush=True)
                            return
                    # Check for rate limiting (HTTP 429) — check string and response status code
                    # (HfHubHTTPError may carry the status on .response without "429" in str(e))
                    is_rate_limited = (
                        any(x in error_str for x in ["429", "too many requests", "rate limit"])
                        or getattr(getattr(e, "response", None), "status_code", None) == 429
                    )

                    if is_rate_limited:
                        _max_retries = 4
                        _backoff = 2
                        debug_log(f"rate limited loading Whisper model: {e}", "voice")
                        retry_succeeded = False
                        for retry_num in range(1, _max_retries + 1):
                            wait = _backoff ** retry_num
                            print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({retry_num}/{_max_retries})...", flush=True)
                            time.sleep(wait)
                            try:
                                self.model = WhisperModel(
                                    model_name, device=try_device, compute_type=try_compute,
                                    cpu_threads=cpu_threads,
                                )
                                self._apply_whisper_load_success(
                                    model_name, try_device, try_compute,
                                    device, compute, cpu_threads,
                                    context="rate-limit retry",
                                )
                                used_device = try_device
                                used_compute = try_compute
                                last_error = None
                                retry_succeeded = True
                                break
                            except Exception as retry_e:
                                debug_log(f"rate-limit retry {retry_num} failed: {retry_e}", "voice")
                                last_error = retry_e
                        if retry_succeeded:
                            break
                        debug_log(f"gave up after {_max_retries} rate-limit retries", "voice")
                        print(f"  ❌ Failed to load Whisper model after {_max_retries} retries: {last_error}", flush=True)
                        print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                        return
                    else:
                        # For other errors (model not found, etc.), don't try fallbacks
                        debug_log(f"failed to initialise faster-whisper: {e}", "voice")
                        print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                        return

            if last_error is not None:
                debug_log(f"failed to initialise faster-whisper with any config: {last_error}", "voice")
                print(f"  ❌ Failed to load Whisper model: {last_error}", flush=True)
                return

            # Warm up faster-whisper so the first real utterance doesn't pay
            # the cold-decode cost. Use low-amplitude noise rather than pure
            # silence — silence trips faster-whisper's no-speech short-circuit
            # and the decoder never actually runs. Mirror the real transcribe
            # parameters so beam search, language detection, and the timestamp
            # path are all exercised here instead of on the user's first word.
            if np is not None and self.model is not None:
                try:
                    cpu_mode = self._whisper_device == "cpu"
                    rng = np.random.default_rng(0)
                    warmup_audio = rng.standard_normal(self._samplerate).astype(np.float32) * 0.01
                    try:
                        segments_iter, _ = self.model.transcribe(
                            warmup_audio,
                            language=None,
                            vad_filter=False,
                            condition_on_previous_text=not cpu_mode,
                            without_timestamps=cpu_mode,
                        )
                    except TypeError:
                        segments_iter, _ = self.model.transcribe(warmup_audio, language=None)
                    for _ in segments_iter:
                        pass
                    debug_log("faster-whisper warmup transcription complete", "voice")
                except Exception as e:
                    debug_log(f"faster-whisper warmup failed: {e}", "voice")

        # Wait for LLM warmups before announcing "Listening!" so the first
        # engagement is responsive. A single 60s budget is shared across
        # all warmup threads so a slow/down Ollama can't block us from
        # listening — we'll just pay the cold-load cost on demand.
        warmup_threads = getattr(self, "_llm_warmup_threads", [])
        if warmup_threads:
            budget = 60.0
            # R34-S54.1 Phase 7b: monotonic — paired with the
            # _llm_warmup_started_at anchor set in the warm-up entry.
            deadline = getattr(self, "_llm_warmup_started_at", time.monotonic()) + budget
            for t in warmup_threads:
                remaining = max(0.0, deadline - time.monotonic())
                t.join(timeout=remaining)

            still_warming = any(t.is_alive() for t in warmup_threads)
            results = getattr(self, "_llm_warmup_results", {})

            # Trailing space after ⚠️ intentional: the warning glyph renders
            # narrower than 🧠/💬, so the pad keeps columns aligned.
            def _print_status(role_key: str, label: str, ok_icon: str) -> None:
                entry = results.get(role_key)
                if entry is None:
                    return
                name, ok = entry
                icon = ok_icon if ok else "⚠️ "
                status = "ready" if ok else "warmup failed — will load on first use"
                print(f"     {icon} {label} '{name}' {status}", flush=True)

            _print_status("chat", "Chat model", "💬")
            _print_status("judge", "Intent judge", "🧠")
            _print_status("router", "Tool router", "🔧")

            if still_warming:
                debug_log("LLM warmup still running after 60s — continuing without", "voice")
                print("     ⏳ Some models still warming — continuing anyway", flush=True)

        # Audio parameters
        frame_ms = int(getattr(self.cfg, "vad_frame_ms", 20))
        self._frame_samples = max(1, int(self._samplerate * frame_ms / 1000))
        pre_roll_ms = int(getattr(self.cfg, "vad_pre_roll_ms", 240))
        endpoint_silence_ms = int(getattr(self.cfg, "endpoint_silence_ms", 800))
        max_utt_ms = int(getattr(self.cfg, "max_utterance_ms", 12000))
        tts_max_utt_ms = int(getattr(self.cfg, "tts_max_utterance_ms", 3000))

        pre_roll_max_frames = max(1, int(pre_roll_ms / frame_ms))
        endpoint_silence_frames = max(1, int(endpoint_silence_ms / frame_ms))
        # max_utt_frames will be calculated dynamically based on TTS state
        normal_max_utt_frames = max(1, int(max_utt_ms / frame_ms))
        tts_max_utt_frames = max(1, int(tts_max_utt_ms / frame_ms))

        debug_log(f"audio params: sample_rate={self._samplerate}, frame_ms={frame_ms}, frame_samples={self._frame_samples}", "voice")
        debug_log(f"VAD: enabled={bool(self._vad is not None)}, aggressiveness={getattr(self.cfg, 'vad_aggressiveness', 2)}", "voice")

        # Audio device setup
        stream_kwargs = {}
        device_env = (self.cfg.voice_device or '').strip().lower()

        if self.cfg.voice_debug:
            debug_log("available input devices:", "voice")
            try:
                for idx, dev in enumerate(sd.query_devices()):
                    try:
                        max_in = int(dev.get("max_input_channels", 0))
                    except Exception:
                        max_in = 0
                    if max_in > 0:
                        name = dev.get("name")
                        rate = dev.get("default_samplerate")
                        debug_log(f"  [{idx}] {name} (channels={max_in}, default_sr={rate})", "voice")
            except Exception:
                pass

        # Configure audio device
        if device_env and device_env not in ("default", "system"):
            try:
                device_index = int(self.cfg.voice_device)
            except ValueError:
                device_index = None
                try:
                    for idx, dev in enumerate(sd.query_devices()):
                        if isinstance(dev.get("name"), str) and (self.cfg.voice_device or '').lower() in dev.get("name").lower():
                            device_index = idx
                            break
                except Exception:
                    device_index = None
            if device_index is not None:
                stream_kwargs["device"] = device_index

        # Log which device will be used
        try:
            if "device" in stream_kwargs:
                dev = sd.query_devices(stream_kwargs["device"])
                device_name = dev.get('name', 'Unknown')
                debug_log(f"using input device: {device_name} (index {stream_kwargs['device']})", "voice")
                print(f"  🎤 Using audio device: {device_name}", flush=True)
            else:
                debug_log("using system default input device", "voice")
                try:
                    default_dev = sd.query_devices(sd.default.device[0])
                    print(f"  🎤 Using default device: {default_dev.get('name', 'Unknown')}", flush=True)
                except Exception:
                    print("  🎤 Using system default input device", flush=True)
        except Exception:
            pass

        # Open audio stream — try configured rate first, fall back to device
        # native rate when the hardware rejects 16 kHz (common on Linux ALSA).
        self._stream_samplerate = self._samplerate
        open_error = None
        try:
            stream = sd.InputStream(
                samplerate=self._samplerate,
                channels=1,
                dtype="float32",
                blocksize=self._frame_samples,
                callback=self._on_audio,
                **stream_kwargs,
            )
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_error = "sample rate" in error_msg or "9987" in error_msg
            if is_rate_error:
                debug_log(f"device rejected {self._samplerate} Hz, querying native rate", "voice")
                try:
                    if "device" in stream_kwargs:
                        dev_info = sd.query_devices(stream_kwargs["device"])
                    else:
                        dev_info = sd.query_devices(kind="input")
                    native_rate = int(dev_info.get("default_samplerate", self._samplerate))
                    if native_rate != self._samplerate:
                        self._stream_samplerate = native_rate
                        native_frame_samples = max(1, int(native_rate * 30 / 1000))
                        print(f"  ⚠️  Device doesn't support {self._samplerate} Hz — using {native_rate} Hz with resampling", flush=True)
                        debug_log(f"retrying stream at native {native_rate} Hz", "voice")
                        stream = sd.InputStream(
                            samplerate=native_rate,
                            channels=1,
                            dtype="float32",
                            blocksize=native_frame_samples,
                            callback=self._on_audio,
                            **stream_kwargs,
                        )
                    else:
                        open_error = e
                except Exception:
                    open_error = e
            else:
                open_error = e

        if open_error is not None:
            error_msg = str(open_error).lower()
            debug_log(f"failed to open input stream: {open_error}", "voice")

            # Provide helpful error messages for common issues
            if "access" in error_msg or "permission" in error_msg:
                print(f"  ❌ Microphone access denied. Please check: {_get_mic_permission_hint()}", flush=True)
            elif "device" in error_msg and ("use" in error_msg or "busy" in error_msg):
                print("  ❌ Microphone is being used by another application", flush=True)
            elif "device" in error_msg:
                print(f"  ❌ Failed to open microphone: {open_error}", flush=True)
                print("     Try selecting a different audio device in settings", flush=True)
            else:
                print(f"  ❌ Failed to start audio recording: {open_error}", flush=True)
            return

        # Main audio processing loop
        with stream:
            # Verify stream is actually recording (helps catch permission issues)
            if not stream.active:
                try:
                    stream.start()
                except Exception as e:
                    error_msg = str(e).lower()
                    debug_log(f"failed to start audio stream: {e}", "voice")
                    if "access" in error_msg or "permission" in error_msg:
                        print(f"  ❌ Microphone access denied. Please check: {_get_mic_permission_hint()}", flush=True)
                    else:
                        print(f"  ❌ Failed to start recording: {e}", flush=True)
                    return

            # Show ready message only after stream is confirmed active
            wake_word = getattr(self.cfg, "wake_word", "jarvis").lower()
            wake_title = wake_word.title()
            print(f"\n{'─' * 50}\n🎙️  Listening! Try:", flush=True)
            print(f"      {self._weather_example(wake_title)}", flush=True)
            print(f"      \"I just ate a Big Mac, {wake_title}.\"", flush=True)
            print(f"      \"What are you thinking, {wake_title}?\"", flush=True)
            print(f"      \"What do you know about me, {wake_title}?\"", flush=True)

            # Small-model disclaimer: SMALL models can't infer your intent
            # from vague prompts, but they can still execute complex flows
            # if you spell out the steps. Assume the model is dumb and lay
            # things out for it. Classification lives in model_variants so
            # it stays in sync when supported models change.
            from ..reply.prompts.model_variants import detect_model_size, ModelSize
            chat_model_name = str(getattr(self.cfg, "ollama_chat_model", "") or "").strip()
            if chat_model_name and detect_model_size(chat_model_name) == ModelSize.SMALL:
                print(
                    f"  ⚠️  Small model in use ({chat_model_name}). Assume it can't infer — spell out the steps for anything more involved:",
                    flush=True,
                )
                print(
                    f"      \"Tell me tomorrow's weather, then find local events for tomorrow, then recommend ones that suit the weather, {wake_title}.\"",
                    flush=True,
                )

            # Chrome MCP tip: the chrome MCP exposes a `navigate` tool that
            # takes a URL. Vague phrasing like "Open YouTube" forces the model
            # to guess a URL; "Navigate to youtube.com" maps directly to the
            # tool's argument and is more reliable on small models.
            try:
                from ..tools.registry import get_cached_mcp_tools
                mcp_tool_names = list(get_cached_mcp_tools().keys())
                has_chrome_mcp = any("chrome" in name.lower() for name in mcp_tool_names)
            except Exception:
                has_chrome_mcp = False
            if has_chrome_mcp:
                print(
                    f"  🌐 Chrome MCP detected. Name the destination URL so the browser tool can act directly:",
                    flush=True,
                )
                print(
                    f"      \"Navigate to youtube.com, {wake_title}.\"",
                    flush=True,
                )

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                state_manager = get_jarvis_state()
                state_manager.set_state(JarvisState.IDLE)
            except Exception:
                pass

            # Track start time for audio health monitoring
            _audio_start_time = time.time()
            _audio_health_logged = False

            while not self._should_stop:
                # One-time audio health check after 5 seconds
                if not _audio_health_logged and time.time() - _audio_start_time > 5:
                    _audio_health_logged = True
                    if self._callback_count == 0:
                        print("  ⚠️  No audio received after 5 seconds!", flush=True)
                        print(f"     Check: {_get_mic_permission_hint()}", flush=True)
                        print("     Also check that your microphone is not muted", flush=True)

                try:
                    item = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    # Critical: Check timeouts even when no audio is being received
                    # This ensures hot window expiry fires reliably
                    self._check_query_timeout()
                    continue

                if item is None:
                    # Reset marker
                    self.is_speech_active = False
                    self._silence_frames = 0
                    self._voice_run = 0
                    self._utterance_frames = []
                    self._pre_roll.clear()
                    continue

                if np is None:
                    continue

                # Process audio buffer
                buf = item
                try:
                    mono = buf.reshape(-1, buf.shape[-1])[:, 0] if buf.ndim > 1 else buf.flatten()
                except Exception:
                    mono = buf.flatten()

                # Process frames
                offset = 0
                total = mono.shape[0]
                frame_timestamp = time.time()  # Timestamp for this batch of frames

                while offset + self._frame_samples <= total:
                    frame = mono[offset: offset + self._frame_samples]
                    offset += self._frame_samples

                    # VAD decision
                    is_voice = self._is_speech_frame(frame)

                    if not self.is_speech_active:
                        if is_voice:
                            self.is_speech_active = True

                            # Backdate start time by pre-roll duration — the
                            # actual speech onset was before VAD triggered.
                            pre_roll_sec = len(self._pre_roll) * frame_ms / 1000.0
                            utterance_start_time = time.time() - pre_roll_sec

                            # Track utterance timing for echo detection
                            self.echo_detector.track_utterance_timing(utterance_start_time, 0.0)

                            # Seed with pre-roll
                            if self._pre_roll:
                                self._utterance_frames.extend(list(self._pre_roll))
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                            self._voice_run = 1  # entering speech-active = one voiced frame so far
                            self._last_partial_ts = time.time()
                            # Emit speech-start so HUD can show "listening" UI
                            try:
                                from ..ipc import get_stream
                                get_stream().emit("vad", speaking=True, level=0.0)
                            except Exception:
                                pass
                        else:
                            # Maintain pre-roll buffer
                            self._pre_roll.append(frame.copy())
                            while len(self._pre_roll) > pre_roll_max_frames:
                                try:
                                    self._pre_roll.popleft()
                                except Exception:
                                    break
                    else:
                        # Control: force-finalize (user clicked ✓) → cut now
                        if self._force_finalize_requested:
                            self._force_finalize_requested = False
                            debug_log("force-finalize requested by user", "voice")
                            self._finalize_utterance()
                            self._pre_roll.clear()
                            continue
                        # Control: cancel (user clicked ✗) → drop frames
                        if self._cancel_utterance_requested:
                            self._cancel_utterance_requested = False
                            debug_log("cancel-utterance requested by user", "voice")
                            self._utterance_frames = []
                            self.is_speech_active = False
                            self._silence_frames = 0
                            self._pre_roll.clear()
                            try:
                                from ..ipc import get_stream
                                get_stream().emit("vad", speaking=False, level=0.0)
                                get_stream().emit("stt_partial", text="", lang=None)
                            except Exception:
                                pass
                            continue
                        if is_voice:
                            self._utterance_frames.append(frame.copy())
                            # SILENCE HYSTERESIS (May 16 critical fix):
                            # The previous logic `self._silence_frames=0`
                            # reset the counter on EVERY voiced frame.
                            # In a noisy room (family chatter, TV), a
                            # single VAD-voiced frame every <500ms kept
                            # the counter at 0 → endpoint never fired →
                            # utterance maxed out at 7s → Whisper got
                            # 5.7s of mixed audio → nonstop UA YT-outro
                            # hallucinations ("Дякую за перегляд!").
                            #
                            # Fix: require 3 consecutive voiced frames
                            # (60ms) to clear the silence counter. A
                            # single voiced blip mid-pause decays the
                            # counter by 1 instead of resetting it.
                            # Real speech stays voiced 5-50 frames in a
                            # row, so wake detection is unaffected;
                            # noise blips no longer extend utterances.
                            self._voice_run = getattr(self, "_voice_run", 0) + 1
                            if self._voice_run >= 3:
                                self._silence_frames = 0
                            else:
                                self._silence_frames = max(0, self._silence_frames - 1)
                            # SAFETY CAP — VAD-stuck-on-noise protection.
                            # If voice frames never stop (constant TV/HVAC
                            # noise above voice_min_energy, VAD aggressiveness
                            # too low), the silence-endpoint never fires and
                            # max_utt_frames check below is skipped because
                            # we're in the is_voice branch. Frames accumulate
                            # → OOM. Force finalize at the hard ceiling.
                            current_max_frames = tts_max_utt_frames if (self.tts and self.tts.is_speaking()) else normal_max_utt_frames
                            if len(self._utterance_frames) >= current_max_frames:
                                debug_log(
                                    f"max_utterance reached while still voice ({len(self._utterance_frames)} frames) — force-finalize",
                                    "voice",
                                )
                                self._finalize_utterance()
                                self._pre_roll.clear()
                        else:
                            self._silence_frames += 1
                            self._voice_run = 0
                            # Use shorter timeout during TTS for quick stop command detection
                            current_max_frames = tts_max_utt_frames if (self.tts and self.tts.is_speaking()) else normal_max_utt_frames
                            if self._silence_frames >= endpoint_silence_frames or len(self._utterance_frames) >= current_max_frames:
                                # Audit round 22 fix (F33): emit
                                # ``vad speaking=false`` on natural
                                # endpoint so the HUD captions panel's
                                # ``recordingHideTimer`` actually fires.
                                # Without this, the panel stays in
                                # ``recording`` class forever after the
                                # first utterance ends (user complaint:
                                # "вікно транскрипції не закривається").
                                # The previous code only emitted
                                # ``speaking=true`` from the voice
                                # branch above — never the falling
                                # edge — so 57k+ ``vad`` events in
                                # events.jsonl had ``speaking=true``
                                # and zero had ``speaking=false``.
                                try:
                                    from ..ipc import get_stream
                                    get_stream().emit("vad", speaking=False, level=0.0)
                                except Exception:
                                    pass
                                self._finalize_utterance()
                                self._pre_roll.clear()

                        # Live UX:
                        # 1) Emit VAD-level every ~100ms so HUD level meter
                        #    pulses with real audio energy (not the random
                        #    "thinking" shimmer it falls back to).
                        # 2) Kick a background partial-transcribe every 2s
                        #    while speech is active — gives user live caption
                        #    BEFORE final endpoint silence.
                        self._vad_emit_counter += 1
                        if self._vad_emit_counter >= 5:  # frame_ms=20 → every 100ms
                            self._vad_emit_counter = 0
                            try:
                                level = float(min(1.0, abs(frame).mean() * 8.0))
                                from ..ipc import get_stream
                                get_stream().emit("vad", speaking=True, level=level)
                            except Exception:
                                pass
                        now_ts = time.time()
                        # Round 27 fix (F65): live-caption regression.
                        # The old 1.0s gate meant short utterances
                        # (300-800ms, common for RU commands like "открой
                        # сафари") ended before the FIRST stt_partial
                        # ever emitted. User reported: "більше не показує
                        # транскрипцію мого мовлення в живому форматі".
                        # Live evidence: only 1 stt_partial out of 1120
                        # vad events after restart.
                        #
                        # New cadence: emit FIRST partial as soon as we
                        # have ~300ms of audio (15 frames @ 20ms), then
                        # every 800ms. Captures even short commands
                        # while keeping server load bounded.
                        if not getattr(self, "_first_partial_emitted", False):
                            if len(self._utterance_frames) >= 15:
                                self._first_partial_emitted = True
                                self._last_partial_ts = now_ts
                                self._schedule_partial_transcribe()
                        elif (now_ts - self._last_partial_ts) >= 0.8 and len(self._utterance_frames) > 10:
                            self._last_partial_ts = now_ts
                            self._schedule_partial_transcribe()

                    # Check for query timeouts
                    self._check_query_timeout()

                # Handle remaining audio
                if offset < total:
                    tail = mono[offset:]
                    if tail.size > 0:
                        self._pre_roll.append(tail.copy())
                        while len(self._pre_roll) > pre_roll_max_frames:
                            try:
                                self._pre_roll.popleft()
                            except Exception:
                                break

    def _transcribe_remote(self, audio_np, language: Optional[str] = None, partial: bool = False) -> dict:
        """POST audio to whisper-service on Hetzner, return MLX-shaped result.

        Args:
            audio_np: float32 1-D numpy array @ self._samplerate (16 kHz).
            language: Optional ISO-639-1 hint ("uk", "ru", etc).

        Returns:
            Dict with keys {"text", "language", "segments"} mirroring the
            shape that mlx_whisper.transcribe returns, so the downstream
            filter/segment-scoring code keeps working unchanged.
            On error returns {"text": "", "segments": [], "language": None}
            and logs — caller decides whether to retry or drop the audio.

        Implementation notes:
          - We spool a 16-bit PCM WAV into a tempfile-shaped BytesIO so
            faster-whisper / ffmpeg on the server can decode it. The
            server's /transcribe endpoint accepts any ffmpeg-readable
            format and we keep it simple with WAV (no compression cost
            on the Mac CPU — the user explicitly asked to minimize Mac
            load).
          - Authentication header is `X-Jarvis-Token: <hex>` matching the
            whisper-service auth.require_token implementation.
          - Timeout: 60s. Real transcription is typically 1-3s; the
            generous cap covers cold model load on the server side
            (only happens once after service restart).
          - Network failure or non-200 response → empty dict so the
            voice loop treats it as "no speech detected" (the user
            can repeat the utterance — far better than hanging).
        """
        import io as _io
        import wave as _wave
        import requests as _rq

        if np is None:
            return {"text": "", "segments": [], "language": None}

        # Float32 [-1, 1] → int16 PCM. Whisper expects 16-bit mono.
        pcm16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)

        buf = _io.BytesIO()
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(self._samplerate))
            wf.writeframes(pcm16.tobytes())
        wav_bytes = buf.getvalue()

        headers = {"X-Jarvis-Token": self._remote_whisper_token}
        files = {"audio": ("utt.wav", wav_bytes, "audio/wav")}
        # initial_prompt INTENTIONALLY OMITTED.
        #
        # Real-world failure (May 15): even the "vocabulary-only" prompt
        # ("Данило, Джарвіс, Nexus Studio, IBONS, Hydrogen, Shopify,
        # Cloudflare, Hetzner, ...") was regurgitated VERBATIM by Whisper
        # on uncertain audio. Heard log line:
        #   'Данило, Джарвіс, Nexus Studio, IBONS, Hydrogen, Shopify,
        #    Cloudflare, Hetzner,'
        # → wake-word fast-path fired → "Слухаю, Даниле" → fixation loop.
        #
        # Trade-off accepted: brand names ("Hydrogen", "Cloudflare",
        # "Hetzner") will revert to garbled Cyrillic transliterations
        # ("гідроген", "клавдфлеер", "гетцнер") about 20% of the time.
        # That's strictly better than a daemon that gets stuck in a
        # phantom-wake loop and won't listen to the user.
        #
        # If we want vocabulary priming back, the correct mechanism is
        # faster-whisper's `hotwords` (probability boost only, no decoder
        # context injection) — but that requires a server-side change.
        # Tracked as a follow-up.
        #
        # Empty string overrides server's WS_DEFAULT_INITIAL_PROMPT
        # fallback so we don't get the prompt back via the back door.
        data = {"initial_prompt": ""}
        if language:
            data["language"] = language
        if partial:
            # Server runs beam_size=1 with looser thresholds — ~3x faster
            # at the cost of slightly lower accuracy. Acceptable for live
            # caption preview; final pass uses full beam.
            data["partial"] = "true"

        try:
            # R34-S54.1 Phase 7b: monotonic — local elapsed measurement
            # over the HTTP round-trip. Wall-clock is unsafe across NTP
            # steps mid-request (Mac wakes from sleep, ntpd corrects).
            t0 = time.monotonic()
            # Timeout 120s — server can take 17-20s for 45s of audio
            # (medium CPU @ 2.7× realtime). Old 60s cap timed out on
            # max-length utterances and dropped real wake-word audio.
            r = _rq.post(
                f"{self._remote_whisper_url}/transcribe",
                headers=headers,
                files=files,
                data=data,
                timeout=(10.0, 120.0),
            )
            elapsed = time.monotonic() - t0
            if r.status_code != 200:
                debug_log(
                    f"remote whisper HTTP {r.status_code}: {r.text[:200]}",
                    "voice",
                )
                return {"text": "", "segments": [], "language": None}
            body = r.json()
            # Reshape server response to MLX-compatible dict. The server
            # returns segments as a list of {text, start, end, ...} dicts
            # — same shape as MLX, so it passes through cleanly.
            return {
                "text": body.get("text", ""),
                "language": body.get("language") or language,
                "segments": body.get("segments", []) or [],
                "_remote_inference_s": body.get("inference_time_s", elapsed),
            }
        except Exception as e:
            debug_log(f"remote whisper request failed: {e}", "voice")
            return {"text": "", "segments": [], "language": None}

    def _schedule_partial_transcribe(self) -> None:
        """Spawn a background thread to transcribe current speech buffer.

        Emits an stt_partial event with the partial text. Non-blocking —
        the audio loop continues capturing without waiting for the result.
        Uses a copy of _utterance_frames so the live loop can keep
        appending without racing on the buffer.
        """
        if np is None or not self._utterance_frames:
            return
        # Remote-backend partials — single-flight + last 4s only.
        # User report: "транскрипція не показується відразу під час
        # того як я говорю". So we DO send partials, but only one at
        # a time (drop if pending) and only the trailing 4s of audio
        # (not the whole buffer) so server queue doesn't saturate.
        if self._whisper_backend == "remote":
            # Round 27 fix (F67): timeout guard on _partial_in_flight.
            # If a remote whisper-service call hangs (network blip,
            # service restart, GPU thrashing), the flag could stay
            # True for the entire ``requests.post`` timeout (60s),
            # blocking ALL live captions during that window. Force-
            # reset after 8s — the partial decoded a stale tail anyway,
            # better to start a fresh one than wait out the hang.
            if getattr(self, "_partial_in_flight", False):
                in_flight_ts = getattr(self, "_partial_in_flight_ts", 0.0)
                # R34-S54.1 Phase 7b: monotonic — paired with the
                # _partial_in_flight_ts anchor below. Wall-clock could
                # either force-reset a healthy in-flight (NTP jump
                # forward) or never reset a genuinely stuck one (jump
                # back) → blocks ALL live captions indefinitely.
                if in_flight_ts > 0 and (time.monotonic() - in_flight_ts) > 8.0:
                    debug_log(
                        f"_partial_in_flight stuck for {time.monotonic() - in_flight_ts:.1f}s — force-reset",
                        "voice",
                    )
                    self._partial_in_flight = False
                else:
                    return  # previous partial still running, skip this tick
            # Tighter window — 2.5s of audio decodes in ~0.8-1.2s on
            # medium-CPU with beam_size=1. User wants live captions, so
            # latency matters more than long-context accuracy for the
            # preview (final pass still uses full utterance).
            tail_seconds = 2.5
            tail_frames = int(self._samplerate * tail_seconds)
            # CRITICAL: snapshot frames into a NEW list before iterating —
            # audio thread mutates self._utterance_frames concurrently and
            # `list(live)` is the only safe atomic-copy. Previously this
            # iterated the live list (assigned to `all_frames`) and could
            # race on `[i].shape[0]` if the audio thread appended/sliced.
            tail_audio = list(self._utterance_frames)
            total_samples = sum(arr.shape[0] for arr in tail_audio if hasattr(arr, "shape"))
            if total_samples > tail_frames:
                # Walk back from the end until we have tail_seconds of audio
                acc = 0
                start_idx = len(tail_audio)
                for i in range(len(tail_audio) - 1, -1, -1):
                    acc += tail_audio[i].shape[0]
                    if acc >= tail_frames:
                        start_idx = i
                        break
                tail_audio = tail_audio[start_idx:]
            frames_copy = tail_audio
            self._partial_in_flight = True
            # R34-S54.1 Phase 7b: monotonic — paired with the
            # 8-second stuck-detection check above. F67 timeout guard.
            self._partial_in_flight_ts = time.monotonic()

            def _remote_partial_run():
                try:
                    audio = np.concatenate(frames_copy, axis=0).flatten()
                    forced_lang = getattr(self.cfg, "whisper_language", None) or None
                    result = self._transcribe_remote(audio, language=forced_lang, partial=True)
                    text = (result.get("text") or "").strip()
                    if text and not self._is_known_hallucination(text):
                        try:
                            from ..ipc import get_stream
                            # Audit round 12 fix: stt_partial payload
                            # used to be inconsistent across the three
                            # emit sites — this one shipped without
                            # `lang`, which made HUD consumers crash
                            # when they did `payload.lang.startsWith(…)`.
                            # Always include `lang` (None when unknown).
                            get_stream().emit(
                                "stt_partial",
                                text=text,
                                lang=getattr(self, "_active_language", None),
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                finally:
                    self._partial_in_flight = False

            threading.Thread(target=_remote_partial_run, daemon=True).start()
            return
        if mlx_whisper is None or self._mlx_model_repo is None:
            return  # only MLX path supports cheap partial right now
        # Copy frames so we don't race with the audio thread mutating
        # _utterance_frames while we're concatenating.
        frames_copy = list(self._utterance_frames)

        def _run():
            try:
                audio = np.concatenate(frames_copy, axis=0).flatten()
                forced_lang = getattr(self.cfg, "whisper_language", None) or None
                if self._whisper_backend == "remote":
                    # Remote service handles everything; no local model
                    # lock needed because each request is independent.
                    result = self._transcribe_remote(audio, language=forced_lang)
                else:
                    # Local MLX — cheap single-temp pass, no beam search,
                    # no logprob filtering. Just a UX preview.
                    with self.transcribe_lock:
                        result = mlx_whisper.transcribe(
                            audio,
                            path_or_hf_repo=self._mlx_model_repo,
                            language=forced_lang,
                            temperature=0.0,
                            condition_on_previous_text=False,
                            # initial_prompt OMITTED — Whisper regurgitates
                            # it verbatim on uncertain audio (see remote
                            # path comment above for the full incident).
                        )
                text = (result.get("text") or "").strip()
                if not text:
                    return
                # Drop known Whisper silence-hallucinations. Mirror the
                # canonical filter in _is_known_hallucination so the
                # partial line doesn't preview UA-YouTube garbage that
                # the final pass will then suppress (confusing UX).
                if self._is_known_hallucination(text):
                    return
                lang = result.get("language") or forced_lang
                try:
                    from ..ipc import get_stream
                    get_stream().emit("stt_partial", text=text, lang=lang)
                except Exception:
                    pass
            except Exception as e:
                debug_log(f"partial transcribe failed: {e}", "voice")

        threading.Thread(
            target=_run, daemon=True, name="jarvis-partial-stt"
        ).start()

    def _finalize_utterance(self) -> None:
        """Process completed utterance through speech recognition."""
        # Round 27 (F65): reset first-partial flag per utterance so
        # the next one also emits its first partial early.
        self._first_partial_emitted = False
        if np is None or not self._utterance_frames:
            self.is_speech_active = False
            self._silence_frames = 0
            self._voice_run = 0
            self._utterance_frames = []
            return

        # Track when utterance ends - but don't overwrite global timing yet
        utterance_end_time = time.time()
        utterance_start_time = self.echo_detector._utterance_start_time

        if self.cfg.voice_debug:
            utterance_duration = utterance_end_time - utterance_start_time if utterance_start_time > 0 else 0
            start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
            end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"utterance captured: duration={utterance_duration:.2f}s (started: {start_time_str}, ended: {end_time_str})", "voice")

        # Transcribe full audio - the intent judge will extract the relevant query
        try:
            audio = np.concatenate(self._utterance_frames, axis=0).flatten()
        except Exception:
            audio = None

        # PEAK NORMALIZATION (May 16, TIGHTENED rev 2; audit round 7 rev 3):
        # Original `1e-4 < peak < 0.3` amplified mic NOISE FLOOR
        # (~0.02-0.04) by 30x → Whisper hallucinated "Спасибо." on
        # every breath. Rev 2 narrowed to `0.05 < peak < 0.15`.
        # Rev 3 (round 7): CAP the gain at 4× so borderline-clipping
        # speech (peak~0.05 → 0.95 = 19× gain) doesn't introduce
        # clipping artifacts that Whisper still hallucinates on.
        # Final gain = min(0.95/peak, 4.0) → for peak=0.05 we now
        # boost to 0.20 not 0.95; for peak=0.10 we boost to 0.40.
        # Quieter speech is preserved but no longer pushed into
        # saturation territory.
        if audio is not None and len(audio) > 0:
            try:
                peak = float(np.max(np.abs(audio)))
                if 0.05 < peak < 0.15:
                    gain = min(0.95 / peak, 4.0)
                    audio = audio * gain
                    debug_log(
                        f"audio peak-normalized: {peak:.4f} × {gain:.2f} "
                        f"= {peak*gain:.4f} ({len(audio)} samples)",
                        "voice",
                    )
            except Exception:
                pass

        # Calculate energy before clearing frames for transcript processing
        utterance_energy = self._calculate_audio_energy(self._utterance_frames[-10:] if self._utterance_frames else [])

        # Reset state before processing
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames = []

        if audio is None or audio.size == 0:
            return

        # Resample to Whisper's expected rate if the stream ran at a different rate
        stream_rate = getattr(self, "_stream_samplerate", self._samplerate)
        if stream_rate != self._samplerate:
            audio = _resample(audio, stream_rate, self._samplerate)

        # Filter short audio
        audio_duration = len(audio) / self._samplerate
        min_duration = getattr(self.cfg, "whisper_min_audio_duration", 0.3)
        if audio_duration < min_duration:
            debug_log(f"audio too short ({audio_duration:.2f}s < {min_duration}s), ignoring", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # NO VAD-finalize ack — user explicitly asked to remove all
        # interjections. The streaming-first-sentence path provides
        # fast feedback (~3-5s) by speaking the LLM reply as it
        # generates, without adding extra "Угу/Хм" sounds that the
        # user found annoying and confused them with confirmation
        # words.

        # Speech recognition with appropriate backend
        try:
            # Forced language hint — Whisper auto-detect on short wake-word
            # audio routinely guesses English (training imbalance) and
            # hallucinates "Thank you" / "Charlie's" over real Ukrainian.
            # Forcing 'uk' keeps it on the right phonetic map and still
            # transcribes RU and most EN wake-word mishearings acceptably.
            forced_lang = getattr(self.cfg, "whisper_language", None) or None

            if self._whisper_backend == "remote":
                # Server-side STT via whisper-service on Hetzner. We lose
                # the local temperature-fallback / compression-threshold
                # knobs (server uses its own faster-whisper defaults) but
                # get zero local RAM use and the ability to run
                # large-v3-turbo on the server for better accuracy.
                # MUST be checked FIRST — previously this was nested
                # inside `if backend == "mlx":` and was unreachable, so
                # remote backend fell into the faster-whisper else and
                # crashed on self.model.transcribe (model was None).
                result = self._transcribe_remote(audio, language=forced_lang)
                # Capture detected language (matches MLX behaviour).
                detected = result.get("language")
                if isinstance(detected, str) and detected:
                    self._last_detected_language = detected
                # Build text from segments to mirror MLX confidence-filter
                # path; remote service applies its own VAD/no-speech
                # filtering server-side, so segments are already clean.
                segs = result.get("segments") or []
                if segs:
                    text = " ".join(s.get("text", "") for s in segs).strip()
                else:
                    text = (result.get("text") or "").strip()
            elif self._whisper_backend == "mlx":
                # MLX Whisper transcription — local Apple Silicon path.
                with self.transcribe_lock:
                    # Better recognition: temperature fallback (default
                    # only [0.0] — adding fallbacks lets Whisper retry
                    # when confidence drops), wider beam search for
                    # tricky accents/dialects. Cost: ~30% slower on
                    # marginal audio, but user explicitly asked
                    # "краще навчити джарвіса розуміти мене".
                    # TIGHTENED (May 16): user reports daemon hears family
                    # chatter as gibberish ("Я Никита Утар", "Семьи кудри",
                    # "Слышь, слой тебя") and NEVER as "Джарвис". Audit
                    # traced this to the temperature fallback ladder
                    # producing fabricated grammar-coherent RU phrases on
                    # weak audio (T=0.4-0.8 = canonical hallucination
                    # recipe). Restore Whisper defaults for thresholds
                    # and drop the ladder; on hard accents we lose a
                    # little recovery, but with `language="ru"` + MLX
                    # large-v3-turbo we get cleaner output and the wake
                    # word actually surfaces.
                    #
                    # initial_prompt OMITTED — REVERTED (May 16 evening):
                    # Even a tiny 3-word wake-word prompt caused full
                    # regurgitation: Whisper output "Добавил субтитры,
                    # джарвіс, джарвіс, джарвіс." over and over on
                    # silence. Original code comment was right —
                    # ANY initial_prompt is poison on uncertain audio.
                    # TIGHTENED (May 16 evening): compression_ratio_threshold
                    # 2.4 → 1.8 to reject repetitive hallucinations.
                    # "джарвіс, джарвіс, джарвіс" and "Духову не ет? Духову
                    # не ет? Духову не ет?" have very high compression
                    # ratios (repeated tokens). 1.8 is aggressive enough
                    # to catch these without dropping legit RU speech
                    # which sits around 1.3-1.7.
                    # TIGHTENED (May 16 rev 2): logprob_threshold -1.0 → -0.7.
                    # Amplified-noise hallucinations like "Спасибо." typically
                    # log at avg_logprob ≈ -0.6 to -0.9. -0.7 catches the
                    # worst offenders without dropping legit speech (which
                    # sits at -0.1 to -0.5).
                    result = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo=self._mlx_model_repo,
                        language=forced_lang,
                        temperature=(0.0, 0.2),
                        condition_on_previous_text=False,
                        compression_ratio_threshold=1.8,
                        logprob_threshold=-0.7,
                        no_speech_threshold=float(getattr(
                            self.cfg, "whisper_no_speech_threshold", 0.6
                        )),
                    )

                # Capture Whisper's auto-detected language (ISO-639-1) so
                # downstream tools can pick locale-appropriate resources.
                detected = result.get("language")
                if isinstance(detected, str) and detected:
                    self._last_detected_language = detected

                # Filter segments by confidence (MLX Whisper returns segments with avg_logprob)
                min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.3)
                marginal_threshold = min_confidence / 3  # Show user-visible log for marginal confidence
                no_speech_threshold = getattr(self.cfg, "whisper_no_speech_threshold", 0.5)
                segments = result.get("segments", [])

                if segments:
                    filtered_texts = []
                    for seg in segments:
                        avg_logprob = seg.get("avg_logprob", 0)
                        no_speech_prob = seg.get("no_speech_prob", 0)

                        # Convert avg_logprob to confidence (typically -1 to 0, so add 1)
                        confidence = min(1.0, max(0.0, avg_logprob + 1.0))
                        seg_text = seg.get("text", "").strip()

                        # Hard filter: high no_speech_prob means no real speech regardless of logprob.
                        if is_whisper_hallucination(no_speech_prob, no_speech_threshold):
                            debug_log(f"MLX segment filtered (no_speech_prob={no_speech_prob:.2f}): '{seg_text[:50]}'", "voice")
                            continue

                        if confidence < min_confidence:
                            if confidence >= marginal_threshold:
                                # F93 — gated print, see line ~4136.
                                if getattr(self.cfg, "voice_debug", False):
                                    print(f"🔇 Low confidence ({confidence:.2f}): \"{seg_text[:50]}...\"", flush=True)
                                else:
                                    debug_log(f"MLX segment filtered (low conf {confidence:.2f}, len={len(seg_text)})", "voice")
                            else:
                                # Very low confidence - debug only
                                debug_log(f"MLX segment filtered (confidence={confidence:.2f}): '{seg_text[:50]}'", "voice")
                            continue

                        filtered_texts.append(seg.get("text", ""))

                    text = " ".join(filtered_texts).strip()
                else:
                    # Fallback to full text if no segments
                    text = result.get("text", "").strip()
            else:
                # faster-whisper transcription
                # CPU mode: skip timestamps and disable context carry-over for speed
                cpu_mode = self._whisper_device == "cpu"
                # GUARD: if backend is "remote", self.model is None — skip
                # this path entirely. Prevented hundreds of NoneType.transcribe
                # errors in production logs when remote backend was active
                # but a code path fell through to the local branch.
                if self.model is None:
                    debug_log(
                        "local whisper model is None (backend=remote?) — skipping faster-whisper branch",
                        "voice",
                    )
                    text = ""
                    segments_list = []
                else:
                    with self.transcribe_lock:
                        try:
                            segments, _info = self.model.transcribe(
                                audio, language=None, vad_filter=False,
                                condition_on_previous_text=not cpu_mode,
                                without_timestamps=cpu_mode,
                            )
                        except TypeError:
                            segments, _info = self.model.transcribe(audio, language=None)
                        segments_list = list(segments)
                # Capture the detected language (faster-whisper exposes it
                # on the info object). Guard against older API variants
                # where the attribute may be absent. _info only exists
                # when we actually ran the model — guard accordingly.
                if self.model is not None and segments_list:
                    detected = getattr(_info, "language", None)
                    if isinstance(detected, str) and detected:
                        self._last_detected_language = detected
                filtered_segments = self._filter_noisy_segments(segments_list)
                text = " ".join(seg.text for seg in filtered_segments).strip()
        except Exception as e:
            debug_log(f"transcription error: {e}", "voice")
            if sys.platform == 'win32':
                print(f"  ❌ Whisper error: {e}", flush=True)
            text = ""

        if not text or not text.strip():
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Log successful transcription — separator omitted on the first utterance since
        # there is no prior turn to visually separate from.
        # Round 30 (F93 — privacy): print every `📝 Heard` line ONLY if
        # voice_debug is on. Live finding: jarvis-assistant.out.log
        # contained full third-party speech transcripts captured by the
        # mic ("📝 Heard: ...family conversation..."). The plist no
        # longer sets JARVIS_VOICE_DEBUG=1 (F73), but these prints
        # weren't gated — they fired unconditionally and launchd routes
        # stdout to the world-readable .out.log. We now gate every
        # PII-bearing print on the same flag and additionally truncate
        # to a hashable digest when verbose is off.
        self._first_utterance = False
        if getattr(self.cfg, "voice_debug", False):
            separator = "" if self._first_utterance else f"\n{'─' * 50}"
            print(f"{separator}\n📝 Heard: \"{text}\"", flush=True)
        # Always emit a typed STT event via IPC so the HUD can render —
        # that file is 0600-locked, not the broad .out.log.

        # Filter out known-bad Whisper hallucination outputs on silence
        # (these appear repeatedly with `whisper_no_speech_threshold` permissive
        # and would otherwise spam intent-judge / fuzzy passes).
        if self._is_known_hallucination(text):
            debug_log(f"rejected known hallucination: '{text[:80]}...'", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Filter out repetitive hallucinations (e.g., "don't don't don't...")
        if self._is_repetitive_hallucination(text):
            debug_log(f"rejected repetitive hallucination: '{text[:80]}...'", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Add to transcript buffer for context-aware processing
        # Mark as "during TTS" if utterance STARTED during TTS (not just if TTS is still speaking now)
        # This ensures mixed echo+user speech gets properly marked for intent judge
        if self.tts is not None and self.tts.is_speaking():
            is_during_tts = True
        else:
            # Audit round 14 fix C3: snapshot under lock.
            _, tts_finish_time, _, _ = self.echo_detector.snapshot_tts_window()
            tts_finish_time = tts_finish_time or 0.0
            echo_tolerance = self.echo_detector.echo_tolerance
            is_during_tts = (tts_finish_time > 0 and utterance_start_time > 0 and utterance_start_time < tts_finish_time + echo_tolerance)
        self._transcript_buffer.add(
            text=text,
            start_time=utterance_start_time,
            end_time=utterance_end_time,
            energy=utterance_energy,
            is_during_tts=is_during_tts,
        )

        # Process the transcript with pre-calculated energy and utterance timing
        self._process_transcript(text, utterance_energy, utterance_start_time, utterance_end_time)

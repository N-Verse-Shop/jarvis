"""
Voice Listener - Main orchestrator for voice capture and processing.

Coordinates audio capture, speech recognition, echo detection, and state management.
"""

from __future__ import annotations
import functools
import os
import threading
import time
import queue
import sys
import platform
from collections import deque
from typing import Optional, TYPE_CHECKING, Any
from datetime import datetime

from rapidfuzz import fuzz
from .echo_detection import EchoDetector
from .state_manager import StateManager, ListeningState
from .wake_detection import is_wake_word_detected, extract_query_after_wake, is_stop_command
from .transcript_buffer import TranscriptBuffer
from .intent_judge import IntentJudge, create_intent_judge, warm_up_ollama_model
from ..debug import debug_log
from ..utils.location import is_location_available

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

# Whisper backend imports - try MLX first on Apple Silicon, fall back to faster-whisper
MLX_WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


# Single source of truth for the voice-mode system prompt. Used by
# both _voice_direct_chat() and _start_llm_warmup() so the KV-cache
# prefix HITS on every real voice turn (prompt-eval drops from
# 15-25s cold to 1-3s warm on qwen2.5:7b @ Hetzner CCX23 CPU).
#
# Keep this STATIC. Dynamic content (language directive, history,
# query) is appended in separate messages — they evaluate fast
# without re-evaluating this 500-token prefix.
VOICE_STATIC_SYSTEM_PROMPT = (
    "Ти — Джарвіс, особистий AI-асистент Данила Молянка. "
    "Контекст: Данило Молянко, директор агентства Nexus Studio "
    "(B2B digital innovation, Rehburg-Loccum, Німеччина, DACH-ринок). "
    "Стек: Shopify Hydrogen + Cloudflare (IBONS), "
    "React Native + Expo (Nexus Mobile), Render.com (Founder Cockpit).\n\n"
    "УКРАЇНСЬКА ГРАМАТИКА — КРИТИЧНО (це найважливіше правило):\n"
    "• Кожне слово має ПРАВИЛЬНЕ закінчення для свого відмінка.\n"
    "• Прикметник + іменник у тому ж роді/числі/відмінку:\n"
    "  ✗ 'нова проєкт' → ✓ 'новий проєкт'\n"
    "  ✗ 'у українська' → ✓ 'в українській'\n"
    "  ✗ 'для клієнт' → ✓ 'для клієнта'\n"
    "  ✗ 'на сервер' (місцевий) → ✓ 'на сервері'\n"
    "• Дієслова в потрібному часі/особі/числі:\n"
    "  ✗ 'я зробить' → ✓ 'я зроблю'\n"
    "  ✗ 'ти можу' → ✓ 'ти можеш'\n"
    "• 'в/у' перед голосною/приголосною: 'в Україні', 'у Львові'.\n"
    "• Числівники узгоджуй: 'два проєкти' (не 'два проєктів'), "
    "'п'ять годин' (не 'п'ять години').\n"
    "• НЕ калькуй з російської: ✗'находжуся' → ✓'перебуваю', "
    "✗'получається' → ✓'виходить', ✗'каждий' → ✓'кожен'.\n"
    "• Перед видачею ПРОЧИТАЙ свою відповідь подумки і перевір "
    "кожне закінчення. Якщо сумніваєшся — використовуй простіше слово.\n\n"
    "Як відповідати (голосовий формат — слухаються вголос):\n"
    "1. ЗМІСТОВНО і КОНКРЕТНО: факти, числа, кроки.\n"
    "2. Без слів-наповнювачів ('Добре,', 'Зрозумів,', 'Звичайно,', 'Угу', 'Хм').\n"
    "3. БЕЗ ВИГУКІВ ВЗАГАЛІ — ніколи не починай з 'Угу/Ага/М-м/Зараз думаю'.\n"
    "4. Не знаєш точно — почни з 'Не знаю напевно' (сигнал для веб-пошуку).\n"
    "5. Без markdown, emoji, формул, посилань.\n"
    "6. Без повторів вже сказаного.\n"
    "7. Тон: професійний, дружній технічний радник, лаконічний.\n"
    "8. КОЖНА НОВА РЕПЛІКА КОРИСТУВАЧА — НОВА ТЕМА. Відповідай на ПОТОЧНИЙ "
    "запит. Не повертайся до попередньої теми (наприклад: пропонував "
    "відкрити Safari, а тепер питають про погоду — відповідай про погоду, "
    "забудь Safari). Контекст потрібен лише якщо новий запит явно "
    "посилається на попередній ('а ще…', 'так само…', 'а той файл').\n"
    "9. РЕЧЕННЯ КОРОТКІ (5-12 слів). Голос складно слухати довгі — "
    "розбивай на 2-3 короткі замість одного довгого.\n\n"
    "ТВОЇ MAC-ДІЇ (для команд керування ПК):\n"
    "Коли користувач просить дію — скажи рівно одну фразу-тригер, "
    "потім чекай слова 'виконуй'. Якщо запит вже містить 'виконуй' — дій одразу.\n"
    "• 'Зараз відкрию <Назва>' (Safari, Telegram, Notion, Mail, VS Code…)\n"
    "• 'Зараз відкрию <URL>' (наприклад github.com)\n"
    "• 'Шукаю в Google: <запит>'\n"
    "• 'Зроблю скріншот'\n"
    "• 'Запишу нотатку: <текст>'\n"
    "• 'Поставлю гучність <0-100>' / 'Вимкну звук' / 'Увімкну звук'\n"
    "• 'Запущу музику' / 'Спаузу музику' / 'Наступний трек'\n"
    "• 'Прочитаю буфер обміну' / 'Скопіюю в буфер: <текст>'\n"
    "• 'Знайду <запит>' (Spotlight)\n"
    "• 'Заблокую екран'\n"
    "• 'Напишу листа' / 'Напишу листа до <email>'\n"
    "• 'Подивлюся котра година' / 'Перевірю рівень батареї'\n"
    "Поза цим списком — 'Поки що я цього не вмію'."
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
        self._utterance_frames: list = []
        self._frame_samples = 0
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
            energy_spike_threshold=float(getattr(self.cfg, "echo_energy_threshold", 2.0))
        )

        self.state_manager = StateManager(
            hot_window_seconds=float(getattr(self.cfg, "hot_window_seconds", 3.0)),
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            voice_collect_seconds=float(getattr(self.cfg, "voice_collect_seconds", 2.0)),
            max_collect_seconds=float(getattr(self.cfg, "voice_max_collect_seconds", 60.0)),
            hot_window_persistent=bool(getattr(self.cfg, "hot_window_persistent", False)),
        )

        # Energy tracking for echo detection
        self._recent_audio_energy: deque = deque(maxlen=50)

        # Audio-level wake word detection timestamp
        self._wake_timestamp: Optional[float] = None

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

        # Active reply language. Defaults to UA. Switched explicitly via
        # detect_language_switch() + user confirmation. Resets to UA on
        # force_end_session().
        self._active_language: str = "uk"
        self._pending_lang_switch: Optional[str] = None
        self._pending_action = None
        self._pending_upgrade = None
        self._pending_confirmation = None

        # Persistent dialog memory — JSONL file in ~/.config/jarvis/memory/.
        # User asked: "нехай зробиться окремий мозочок ... він там записує
        # всі наші розмови в зжатому форматі щоб він завжди міг памятати
        # контекст за нашу сесію та між сесіями".
        # Each user/assistant exchange is appended as one JSON line, and
        # the most recent N pairs are auto-loaded at startup so Jarvis
        # remembers across daemon restarts. Compression: we keep only the
        # text (no metadata bloat) and the per-line tail-load is O(1).
        self._init_persistent_memory()

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
        try:
            os.makedirs(mem_dir, exist_ok=True)
        except Exception as e:
            debug_log(f"could not create memory dir: {e}", "voice")
            self._dialog_history: list[dict] = []
            self._memory_file = None
            return
        # One file per day — keeps individual files manageable.
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
                                loaded.append({"role": obj["role"], "content": obj["content"]})
                        except Exception:
                            continue
            except Exception as e:
                debug_log(f"failed to load memory from {path}: {e}", "voice")
        # Keep only most recent 16 messages (8 pairs).
        recent = loaded[-16:]
        # Filter out contaminated non-Ukrainian entries — they pollute
        # the model's language anchor (it sees RU/EN in history and
        # switches modes). We're strict: drop any assistant message
        # without at least 3 distinct Cyrillic UA characters
        # (а-я,А-Я + ї/є/і/ґ). User messages we keep regardless (the
        # user is allowed to speak any language).
        ua_chars = set("абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ")
        def _is_ua(text: str) -> bool:
            if not text:
                return False
            hits = sum(1 for c in text if c in ua_chars)
            return hits >= 3
        cleaned: list[dict] = []
        dropped = 0
        for m in recent:
            if m["role"] == "user":
                cleaned.append(m)
            elif _is_ua(m["content"]):
                cleaned.append(m)
            else:
                dropped += 1
        self._dialog_history: list[dict] = cleaned
        debug_log(
            f"persistent memory loaded: {len(self._dialog_history)} messages "
            f"from {len(files_to_load)} file(s) (dropped {dropped} non-UA)",
            "voice",
        )

    def _persist_memory_pair(self, query: str, reply: str) -> None:
        """Append a (user, assistant) pair to today's memory file.

        One JSON object per line, atomic append. Failure is logged but
        non-fatal — we never block the voice loop on disk errors.
        """
        if not self._memory_file:
            return
        import json
        import time as _t
        try:
            with open(self._memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"role": "user", "content": query, "ts": _t.time()},
                    ensure_ascii=False,
                ) + "\n")
                f.write(json.dumps(
                    {"role": "assistant", "content": reply, "ts": _t.time()},
                    ensure_ascii=False,
                ) + "\n")
        except Exception as e:
            debug_log(f"failed to persist memory: {e}", "voice")

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
        """Stop the thinking tune and revert face state to IDLE."""
        if self._tune_player is not None:
            self._tune_player.stop_tune()
            self._tune_player = None
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                get_jarvis_state().set_state(JarvisState.IDLE)
            except ImportError:
                pass
            except Exception:
                pass

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

        # Reset wake timestamp — it must reflect only the current utterance.
        # If this utterance contains a wake word, the early-beep check below
        # will set it. Without this reset, a prior rejected wake-worded
        # utterance would vouch for subsequent unrelated utterances via the
        # `_wake_timestamp is not None` guard in the intent-judge accept path.
        self._wake_timestamp = None

        start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
        end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3] if utterance_end_time > 0 else "N/A"
        debug_log(f"heard: '{text}' (utterance from {start_time_str} to {end_time_str})", "voice")

        # Track if this input was received during TTS (for logging purposes)
        received_during_tts = self.tts and self.tts.is_speaking()

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
            if is_wake_word_detected(text_lower, _ww, _wal, _wfr):
                debug_log(f"pre-emptive wake-interrupt during TTS: '{text_lower[:60]}'", "voice")
                print(f"  ⏸  Wake interrupt: \"{text_lower[:60]}\"", flush=True)
                if self.tts:
                    self.tts.interrupt()
                try:
                    while not self._audio_q.empty():
                        self._audio_q.get_nowait()
                except Exception:
                    pass
                # Clear echo-detector's "last_tts" so the next utterance
                # isn't classified as echo of the interrupted TTS.
                try:
                    self.echo_detector._last_tts_text = ""
                except Exception:
                    pass
                from .wake_detection import extract_query_after_wake
                extracted = extract_query_after_wake(text_lower, _ww, _wal).strip()
                if extracted and len(extracted) > 2:
                    self._dispatch_query(extracted)
                else:
                    # Just the wake word — open collection for follow-up.
                    self.state_manager.start_collection("")
                    self._set_face_state_listening()
                return

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
                last_tts_text = self.echo_detector._last_tts_text or ""
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
                            print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
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
                if is_wake_word_detected(text_lower, wake_word, aliases, fuzzy_ratio):
                    self._wake_timestamp = utterance_start_time
                    self._start_thinking_tune()
                    self._set_face_state_listening()
                    debug_log("early beep: wake word detected", "voice")

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
            print(f"  🛑 Session stop: \"{text_lower[:60]}\"", flush=True)
            if self.tts and self.tts.enabled and self.tts.is_speaking():
                self.tts.interrupt()
            # Wipe dialog history — new session starts with a clean slate
            # so old context doesn't leak into the next wake.
            if hasattr(self, "_dialog_history"):
                self._dialog_history.clear()
            # Reset language to UA default and clear any pending states.
            self._active_language = "uk"
            self._pending_lang_switch = None
            self._pending_action = None
            self._pending_upgrade = None
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
                self.tts.interrupt()
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
                    print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                    return

        # Salvage user speech from merged echo+speech chunks.
        # When Whisper delivers a single transcript containing TTS echo followed by
        # user speech (e.g. "I can only provide... Well you can search for it"), the
        # echo portion was captured during TTS but the transcript arrives after TTS
        # finishes. Try to strip the leading echo and use just the user's speech.
        # Skip entirely if there's no prior TTS — nothing to match against.
        last_tts_text_for_salvage = self.echo_detector._last_tts_text or ""
        last_tts_finish = self.echo_detector._last_tts_finish_time or 0.0
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
            if not is_vocative:
                first_word = head.split()[0] if head.split() else ""
                if (3 <= len(first_word) <= 9
                        and first_word[0] in "jyчжяeх"
                        or first_word.startswith(("дж", "ya", "ja"))):
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
                print(f"  ⚡ Wake fast-path: \"{extracted}\"" if extracted
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
            last_tts_text = self.echo_detector._last_tts_text or ""
            last_tts_finish_time = self.echo_detector._last_tts_finish_time or 0.0

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
                    print(f"  🧠 Intent ({mode_str}): directed → \"{intent_judgment.query or text_lower}\"", flush=True)
                else:
                    print(f"  🧠 Intent ({mode_str}): not directed ({intent_judgment.reasoning})", flush=True)
            else:
                reason = self._intent_judge.last_failure_reason or "no segments or unavailable"
                print(f"  🧠 Intent judge: unavailable ({reason})", flush=True)
                debug_log(f"intent judge returned None — falling back ({reason})", "voice")
                # Hot window fallback: if the early echo check already cleared
                # this text, accept it even without the judge's verdict.
                if could_be_hot_window:
                    last_tts_text_fb = self.echo_detector._last_tts_text or ""
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
                        print(f"  🧠 Intent fallback: accepting hot window speech", flush=True)
                        debug_log(f"✅ Hot window fallback (judge unavailable): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self.state_manager.start_collection(text_lower)
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

            if intent_judgment is not None:
                # If judge says stop command, interrupt TTS
                if intent_judgment.stop and self.tts and self.tts.is_speaking():
                    debug_log(f"🛑 Intent judge detected stop command", "voice")
                    self.tts.interrupt()
                    return

                # If directed with query, process it
                if intent_judgment.directed and intent_judgment.query:
                    # In wake word mode, verify the wake word is actually present
                    # The LLM sometimes hallucinates wake words that don't exist
                    if not could_be_hot_window:
                        wake_word = getattr(self.cfg, "wake_word", "jarvis")
                        aliases = list(set(getattr(self.cfg, "wake_aliases", [])) | {wake_word})
                        has_wake_word = self._wake_timestamp is not None or is_wake_word_detected(
                            text_lower, wake_word, aliases
                        )
                        if not has_wake_word:
                            print(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed but no wake word found in '{text_lower[:50]}...' "
                                f"(reasoning: {intent_judgment.reasoning})",
                                "voice"
                            )
                            # Don't accept - fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{intent_judgment.query}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(intent_judgment.query)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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
                                    print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                                    self._stop_thinking_tune()
                                    return
                                else:
                                    debug_log(
                                        f"echo in text (score={echo_score}) but judge extracted "
                                        f"non-echo query: \"{intent_judgment.query}\"", "voice"
                                    )

                        # The intent judge is explicitly designed to prune echo
                        # and extract the actual user query — always prefer its
                        # output when present. Falling back to raw heard text
                        # leaks partially-salvaged echo fragments into tool
                        # calls (e.g. "…amount now? okay, what is his best
                        # song?" reaching webSearch verbatim). If the judge
                        # returns an empty query (rare), fall back to raw text.
                        judge_query = (intent_judgment.query or "").strip()
                        hot_query = judge_query or text_lower
                        if judge_query and judge_query.lower() != text_lower:
                            debug_log(
                                f"using judge query over heard text: "
                                f"\"{judge_query}\" (heard: \"{text_lower[:80]}\")",
                                "voice",
                            )
                        debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{hot_query}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()

                        self.state_manager.start_collection(hot_query)

                        # Start thinking tune and show processing message
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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
                            print(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
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
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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
                                print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                                self._stop_thinking_tune()
                                return

                        debug_log(f"✅ Intent judge accepted (directed, high confidence, using actual text): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self.state_manager.start_collection(text_lower)
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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
                        last_tts_finish = self.echo_detector._last_tts_finish_time or 0.0
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
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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
                            print(f"  🧠 Intent override: accepting hot window speech (mixed echo+speech)", flush=True)
                            debug_log(
                                f"⚡ Overriding echo reasoning in hot window "
                                f"(echo_score={echo_score}, text longer than TTS): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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
                            print(f"  🧠 Intent override: accepting hot window speech", flush=True)
                            debug_log(
                                f"⚡ Overriding intent judge in hot window "
                                f"(echo_score={echo_score}, reasoning={intent_judgment.reasoning}): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self.state_manager.start_collection(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
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

        wake_detected = is_wake_word_detected(text_lower, wake_word, list(aliases), fuzzy_ratio)
        debug_log(f"wake word check: '{wake_word}' in '{text_lower}' → {wake_detected}", "voice")

        if wake_detected:
            # Cancel any pending hot window activation when new query starts
            self.state_manager.cancel_hot_window_activation()

            # Mark the current segment as processed to prevent re-extraction
            self._transcript_buffer.mark_segment_processed(text_lower)

            # Clear audio buffers to prevent concatenation issues
            self._clear_audio_buffers()

            query_fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
            self.state_manager.start_collection(query_fragment)

            # Start thinking tune and show processing message
            self._start_thinking_tune()
            try:
                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
            except Exception:
                pass
            return

        # Priority 5: Collection mode handling
        if self.state_manager.is_collecting():
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
                print(f"  ⏳ Heard during TTS (waiting for hot window): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
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

        # Language is dynamic (UA default, RU/EN/DE on explicit switch)
        # — passed as a SEPARATE system message so it doesn't bust the
        # cache on the main prompt. ~30 tokens, evals in <1s.
        lang_directive = self._language_directive(getattr(self, "_active_language", "uk"))

        # Build messages with rolling dialog history so the model has
        # context across turns AND across sessions (persistent memory
        # loaded at startup from ~/.config/jarvis/memory/dialog-YYYY-MM-DD.jsonl).
        # We pass the last 10 messages (5 pairs) — fits easily in
        # num_ctx=2560 along with system prompts + current query.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": lang_directive},
        ]
        # Use only the last 4 messages (2 turns) for context. On a 7b
        # CPU model, more history causes "positional anchoring": the
        # model latches onto the previous topic and ignores new
        # queries. 2 turns is enough to maintain coherent dialogue
        # without dragging stale context.
        messages.extend(self._dialog_history[-4:])
        messages.append({"role": "user", "content": query})

        try:
            t0 = _time.time()
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
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.4,
                        # 150 tokens for 7b chat. 7b @ Hetzner CCX23 CPU
                        # ≈ 8 tok/s → 150 tokens = ~18s worst-case warm.
                        # Most voice answers fit in 80-120 tokens; this
                        # leaves headroom without paying a full 25s tail.
                        "num_predict": 120,
                        "repeat_penalty": 1.25,
                        "repeat_last_n": 128,
                        "presence_penalty": 0.4,
                        "frequency_penalty": 0.3,
                        # 2560 ctx fits static prompt (~500t) + lang (~30t)
                        # + 10 history msgs (~1500t) + query (~50t) + reply
                        # (~200t) with margin. Smaller ctx = faster
                        # prompt-eval on CPU (linear in ctx size).
                        "num_ctx": 2560,
                        # 4 worker threads (Hetzner CCX23 = 4 vCPU).
                        "num_thread": 4,
                        # Larger batch = more tokens per prompt-eval pass
                        # = better CPU throughput.
                        "num_batch": 256,
                    },
                },
                # 120s hard limit — 7b cold-cache rebuild can hit 60-90s
                # on first call; warm subsequent calls return in 7-25s.
                timeout=120.0,
                stream=True,
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
                cleaned = self._strip_lazy_prefix(
                    self._sanitize_for_piper_uk(sent)
                ).strip()
                if not cleaned:
                    return
                if not (self.tts and self.tts.enabled):
                    return
                try:
                    # First sentence — track for echo detector.
                    # Subsequent sentences extend the same TTS turn,
                    # so we update the tracker as new text comes in.
                    if not sentences_spoken:
                        self.track_tts_start(cleaned)
                        debug_log(
                            f"stream sent #1: {cleaned[:60]}", "voice"
                        )
                    else:
                        debug_log(
                            f"stream sent #{len(sentences_spoken)+1}: "
                            f"{cleaned[:60]}",
                            "voice",
                        )
                    self.tts.speak(cleaned)
                    sentences_spoken.append(cleaned)
                except Exception as e:
                    debug_log(f"stream-speak failed: {e}", "voice")

            try:
                for raw in response.iter_lines(decode_unicode=False):
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
                            first_tok_t = _time.time() - t0
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

            # Flush any trailing fragment (no terminal punctuation)
            tail = buf.strip()
            if tail and len(tail) >= 2:
                _flush_sentence(tail)

            elapsed = _time.time() - t0
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
                lc = content.lower()
                if (
                    "не знаю напевно" in lc
                    or "потрібен пошук" in lc
                    or "потрібен веб-пошук" in lc
                ) and len(query) > 4:
                    debug_log(f"web-search fallback for: '{query[:60]}'", "voice")
                    print(f"  🌐 Шукаю в інтернеті: \"{query[:50]}\"", flush=True)
                    results = self._web_search(query, max_results=4)
                    if results:
                        ctx = "\n".join(
                            f"- {r['title']}: {r['snippet']}"
                            for r in results if r.get("snippet")
                        )
                        retry_msgs = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content":
                                f"{query}\n\n[Контекст з веб-пошуку — використай ці факти:]\n{ctx}\n\n"
                                f"Дай ЗМІСТОВНУ відповідь українською мовою на основі цього контексту."},
                        ]
                        try:
                            t1 = _time.time()
                            r2 = requests.post(
                                f"{base_url.rstrip('/')}/api/chat",
                                json={
                                    "model": model,
                                    "messages": retry_msgs,
                                    "stream": False,
                                    "keep_alive": "30m",
                                    "options": {
                                        "temperature": 0.4,
                                        "num_predict": 500,
                                        "repeat_penalty": 1.3,
                                        "repeat_last_n": 128,
                                        "presence_penalty": 0.5,
                                        "frequency_penalty": 0.4,
                                        "num_ctx": 4096,
                                    },
                                },
                                timeout=90.0,
                            )
                            if r2.status_code == 200:
                                d2 = r2.json()
                                m2 = d2.get("message", {}) if isinstance(d2, dict) else {}
                                c2 = (m2.get("content") or "").strip()
                                if c2 and len(c2) > 20:
                                    content = self._sanitize_for_piper_uk(c2)
                                    content = self._strip_lazy_prefix(content)
                                    debug_log(
                                        f"web-search retry ok in {_time.time()-t1:.1f}s",
                                        "voice",
                                    )
                        except Exception as e:
                            debug_log(f"web-search retry error: {e}", "voice")
                # Append to rolling dialog history AND persist to disk
                # so Jarvis remembers across daemon restarts.
                self._dialog_history.append({"role": "user", "content": query})
                self._dialog_history.append({"role": "assistant", "content": content})
                # Trim in-memory to last 16 messages (8 turns) to bound
                # prompt size. Disk has the full history.
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
        except requests.exceptions.Timeout:
            debug_log("voice direct-chat: timeout after 12s", "voice")
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
        replacements = [
            (r"\bJarvis\b", "Джарвіс"),
            (r"\bJARVIS\b", "Джарвіс"),
            (r"\bAI\b", "ШІ"),
            (r"\bOK\b", "добре"),
            (r"\bok\b", "добре"),
            (r"\bGPT\b", "Джі-Пі-Ті"),
            (r"\bLLM\b", "мовна модель"),
            (r"\bAPI\b", "АПІ"),
            (r"\bCEO\b", "генеральний директор"),
            (r"\bUI\b", "інтерфейс"),
            (r"\bUX\b", "юзабіліті"),
            (r"\bNexus\b", "Нексус"),
            (r"\bStudio\b", "Студіо"),
        ]
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
            r"добре", r"зрозуміло", r"зрозумів", r"гаразд",
            r"так(?:,\s*я)?", r"звичайно", r"звісно",
            r"окей", r"ок", r"авжеж", r"я можу",
            r"я знаю", r"я готовий", r"я тут",
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
        import re
        q = query.lower().strip()
        q = re.sub(r"[!?,.;:\"'`()\[\]{}«»…]+", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        # Drop any leading wake-word fragment if intent-judge missed it.
        for wake in ("джарвіс", "джарвис", "ярвіс", "ярвис", "jarvis",
                     "джарвес", "джарваз", "джавіс", "ярвес"):
            if q.startswith(wake + " "):
                q = q[len(wake) + 1:].strip()
                break

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
            return "Привіт, Даниле! Чим можу допомогти?"

        # Acknowledgements/thanks.
        thanks = {
            "дякую", "дяки", "спасибі", "вдячний",
            "спасибо", "благодарю",
            "thanks", "thank you", "thx",
            "danke", "vielen dank",
        }
        if q in thanks:
            return "Прошу. Звертайтесь."

        # Farewells.
        farewells = {
            "па", "пока", "бувай", "до побачення", "побачимось", "па-па",
            "bye", "goodbye", "see you", "cya",
            "tschüss", "auf wiedersehen",
        }
        if q in farewells:
            return "До зустрічі, Даниле."

        # Status check — "як справи?" / "як ти?" / "як ся маєш?"
        status = {
            "як справи", "як ти", "як ся маєш", "як ся маєте", "як настрій",
            "як дела", "как дела", "как ты",
            "how are you", "how's it going", "what's up",
        }
        if q in status:
            return "Все добре, працюю. А у вас?"

        # Single-word "test"/"перевірка"/"чуєш мене" — confirm liveness.
        liveness = {
            "тест", "перевірка", "тестування", "чуєш", "чуєш мене", "ти тут",
            "ти мене чуєш", "чути", "ти живий", "ти працюєш",
            "test", "testing", "are you there", "can you hear me",
            "проверка", "слышишь меня",
        }
        if q in liveness:
            return "Чую вас. Працюю."

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
            return "Гаразд."

        # Identity / who-are-you — common test phrases.
        identity = {
            "хто ти", "як тебе звати", "ти хто", "представся",
            "who are you", "what are you", "what's your name",
            "кто ты", "как тебя зовут",
        }
        if q in identity:
            return "Я — Джарвіс, ваш AI-асистент. Готовий допомогти."

        # Sorry / apology — symmetric ack.
        apologies = {
            "вибач", "вибачте", "пробач", "прости", "перепрошую",
            "sorry", "my bad", "excuse me",
            "извини", "извините",
        }
        if q in apologies:
            return "Нічого, працюємо далі."

        # Quick yes/no questions about Jarvis capability are misleading
        # to canned — leave to LLM.

        # Common short questions that don't really need LLM — answer
        # straight away.
        if q in {"котра година", "котра зараз година", "скільки часу", "який час"}:
            from datetime import datetime
            now = datetime.now()
            weekday = ["понеділок", "вівторок", "середа", "четвер",
                       "п'ятниця", "субота", "неділя"][now.weekday()]
            return f"Зараз {now.strftime('%H:%M')}, {weekday}."

        if q in {"яке сьогодні число", "яка дата", "який сьогодні день",
                 "який день тижня", "сьогодні який день"}:
            from datetime import datetime
            now = datetime.now()
            months = ["січня", "лютого", "березня", "квітня", "травня", "червня",
                      "липня", "серпня", "вересня", "жовтня", "листопада", "грудня"]
            weekday = ["понеділок", "вівторок", "середа", "четвер",
                       "п'ятниця", "субота", "неділя"][now.weekday()]
            return f"Сьогодні {weekday}, {now.day} {months[now.month-1]} {now.year} року."

        # Compliment / nice — friendly echo.
        compliments = {
            "ти класний", "молодець", "круто", "супер", "відмінно",
            "good job", "well done", "nice",
            "молодец", "класс",
        }
        if q in compliments:
            return "Дякую! Радий допомогти."

        return ""

    def _lang_name(self, code: str) -> str:
        """Human-readable Ukrainian name of a language code."""
        return {
            "uk": "українською",
            "ru": "російською",
            "en": "англійською",
            "de": "німецькою",
        }.get(code, code)

    def _language_directive(self, code: str) -> str:
        """Return the system-prompt clause that locks reply language."""
        directives = {
            "uk": (
                "МОВА ВІДПОВІДІ: Українська (кирилиця). Завжди українською, "
                "навіть якщо в історії є RU/EN/DE — не повторюй той вибір. "
                "Замість 'Jarvis' пиши 'Джарвіс'."
            ),
            "ru": (
                "ЯЗЫК ОТВЕТА: Русский. Пользователь явно попросил перейти "
                "на русский — отвечай только на русском, кириллицей."
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
        return directives.get(code, directives["uk"])

    def _run_self_upgrade_async(self, user_request: str) -> None:
        """Spawn Claude Code in a background thread to self-upgrade."""
        import threading as _t
        def _runner():
            try:
                from ..agent.self_upgrade import (
                    write_upgrade_brief, spawn_claude,
                    wait_for_completion_and_restart,
                )
                brief = write_upgrade_brief(user_request)
                proc = spawn_claude(brief)
                if proc is None:
                    self._speak_and_continue(
                        "Не знайшов Claude Code CLI. Встанови його через "
                        "npm install minus g at-anthropic-ai-claude-code."
                    )
                    return
                ok, summary = wait_for_completion_and_restart(proc)
                # After kickstart we'll be dead — speak BEFORE restart.
                if ok and "Перезапускаюсь" in summary:
                    self._speak_and_continue(summary)
                    # Daemon will be killed by kickstart; new instance loads.
                else:
                    self._speak_and_continue(summary)
            except Exception as e:
                debug_log(f"self-upgrade thread error: {e}", "voice")
                self._speak_and_continue(f"Помилка самооновлення: {e}")
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
        debug_log(f"dispatching query: '{query}'", "voice")

        # Language switch check: "говори російською" / "switch to English"
        # → stage with confirmation. Active language defaults to UA on
        # daemon start and after force_end_session.
        try:
            from .action_dispatcher import detect_language_switch
            lang_req = detect_language_switch(query)
            if lang_req is not None and not getattr(self, "_pending_lang_switch", None):
                lang_code, ack = lang_req
                current = getattr(self, "_active_language", "uk")
                if lang_code != current:
                    self._pending_lang_switch = lang_code
                    self._speak_and_continue(
                        f"Хочеш щоб я говорив {self._lang_name(lang_code)}? "
                        f"Підтверди — скажи 'виконуй'."
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
                    try:
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
                    self._speak_and_continue("Скасовано, залишаюсь на українській.")
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
                self._speak_and_continue(
                    "Запускаю самооновлення через Claude Code. "
                    "Це може зайняти 5-15 хвилин. Підтверди — скажи 'виконуй'."
                )
                return
            if getattr(self, "_pending_upgrade", None):
                from .action_dispatcher import is_confirmation as _is_conf, is_denial as _is_den
                if _is_conf(query):
                    req = self._pending_upgrade
                    self._pending_upgrade = None
                    self._speak_and_continue("Починаю самооновлення.")
                    self._run_self_upgrade_async(req)
                    return
                if _is_den(query):
                    self._pending_upgrade = None
                    self._speak_and_continue("Скасовано.")
                    return
                self._pending_upgrade = None
        except Exception as e:
            debug_log(f"upgrade-check error: {e}", "voice")

        # Confirmation flow: if there's a pending action and the user
        # is confirming/denying, run/cancel BEFORE going to LLM.
        from .action_dispatcher import (
            parse_action, is_confirmation, is_denial,
        )
        pending = getattr(self, "_pending_action", None)
        if pending is not None:
            if is_confirmation(query):
                debug_log(f"executing pending action: {pending.name}", "voice")
                ok, msg = pending.fn()
                self._pending_action = None
                self._speak_and_continue(msg if ok else f"Не вийшло. {msg}")
                return
            if is_denial(query):
                debug_log(f"cancelled pending action: {pending.name}", "voice")
                self._pending_action = None
                self._speak_and_continue("Скасовано.")
                return
            # Otherwise — user said something new (not confirm/deny).
            # Drop the pending action AND remove the "Зараз відкрию X.
            # Підтверди..." proposal from dialog history, otherwise the
            # LLM keeps anchoring the next reply to that abandoned
            # action and ignores the actual new query.
            self._pending_action = None
            try:
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
                # Store in dialog history so context stays coherent
                try:
                    self._dialog_history.append({"role": "user", "content": query})
                    self._dialog_history.append({"role": "assistant", "content": msg})
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
        action = parse_action(reply)
        if action is not None:
            if is_confirmation(query):
                # Inline confirm — user said "виконуй ..." right away.
                debug_log(f"inline-confirmed action: {action.name}", "voice")
                ok, msg = action.fn()
                reply = f"{action.description}. {msg}" if ok else f"Не вийшло: {msg}"
            else:
                # Stage for the next utterance's confirmation.
                self._pending_action = action
                if "підтверди" not in reply.lower():
                    reply = reply.rstrip(".!?") + ". Підтверди — скажи 'виконуй'."

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
                    self.echo_detector._tts_exact_duration = duration

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
        """Determine if audio frame contains speech."""
        if np is None:
            return True

        # Track energy for echo detection
        rms = float(np.sqrt(np.mean(np.square(frame))))
        self._recent_audio_energy.append(rms)

        if self._vad is None:
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))

        # Use WebRTC VAD
        try:
            pcm16 = np.clip(frame.flatten() * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return bool(self._vad.is_speech(pcm16, getattr(self, "_stream_samplerate", self._samplerate)))
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
                    # Marginal confidence - show in log viewer (not debug)
                    print(f"🔇 Low confidence ({confidence:.2f}): \"{seg.text.strip()[:50]}...\"", flush=True)
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
        """
        if not text:
            return False
        lower = text.lower()
        # Known idle-noise patterns. Each phrase observed multiple times in
        # production logs with `whisper_no_speech_threshold` ≥ 0.85.
        KNOWN_PATTERNS = (
            "дмитро павловський",
            "білян ліна керівськ",
            "білян ліпчук",
            "хіднось продавав",
            "thank you",            # English silence hallucination
            "thanks for watching",  # YouTube-style training-data leakage
            "subscribe to",
            "see you next",
        )
        for pat in KNOWN_PATTERNS:
            if pat in lower:
                return True
        return False

    def _check_query_timeout(self) -> None:
        """Check if there's a pending query that has timed out, and check hot window expiry."""
        if self.state_manager.check_collection_timeout():
            query = self.state_manager.clear_collection()
            if query.strip():
                self._dispatch_query(query)
            else:
                # Empty collection on timeout — user said just "Джарвіс"
                # with no follow-up question. Without this branch the daemon
                # went silent (no dispatch, no reply), leaving the HUD coin
                # stuck on LISTENING with no audible response. Speak a
                # short prompt so the user knows Jarvis is awake and waiting,
                # AND reset the face state so the next "Джарвіс" can fire.
                debug_log("collection timed out with empty query — speaking ack", "voice")
                self._stop_thinking_tune()
                if self.tts and self.tts.enabled:
                    self.tts.speak("Слухаю, Даниле.")
                # Reset HUD to IDLE so coin doesn't stay on LISTENING.
                try:
                    from desktop_app.face_widget import get_jarvis_state, JarvisState
                    get_jarvis_state().set_state(JarvisState.IDLE)
                except Exception:
                    pass

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
                            self.tts.interrupt()
                            # Clear echo-detector's last_tts so the next
                            # transcript isn't auto-rejected as echo.
                            try:
                                self.echo_detector._last_tts_text = ""
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
                        warm_lang = self._language_directive("uk")
                        warm_resp = _rq.post(
                            f"{base_url.rstrip('/')}/api/chat",
                            json={
                                "model": chat_model,
                                "messages": [
                                    {"role": "system", "content": VOICE_STATIC_SYSTEM_PROMPT},
                                    {"role": "system", "content": warm_lang},
                                    {"role": "user", "content": "[warmup] привіт"},
                                ],
                                "stream": False,
                                "keep_alive": "30m",
                                "options": {
                                    # Match _voice_direct_chat options
                                    # EXACTLY so warmup primes the same
                                    # cache slot. Any drift = cache miss
                                    # on first real call = 15-25s pause.
                                    "temperature": 0.4,
                                    "num_predict": 120,
                                    "repeat_penalty": 1.25,
                                    "repeat_last_n": 128,
                                    "presence_penalty": 0.4,
                                    "frequency_penalty": 0.3,
                                    "num_ctx": 2560,
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
            while not getattr(self.state_manager, "_should_stop", False):
                try:
                    if os.path.exists(ctrl_path):
                        with open(ctrl_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        ts = float(data.get("ts", 0))
                        action = str(data.get("action", "")).lower().strip()
                        if ts > last_ts and action:
                            last_ts = ts
                            debug_log(f"HUD control: action='{action}' ts={ts}", "voice")
                            if action in ("end_session", "stop_session", "session_end"):
                                if self.tts and self.tts.enabled and self.tts.is_speaking():
                                    self.tts.interrupt()
                                if hasattr(self, "_dialog_history"):
                                    self._dialog_history.clear()
                                self.state_manager.force_end_session()
                                print("  🛑 Session ended (HUD right-click)", flush=True)
                            elif action in ("interrupt_tts", "stop_speaking", "interrupt"):
                                # Just stop the current TTS, keep the
                                # hot window active so the user can
                                # immediately ask a follow-up.
                                if self.tts and self.tts.enabled and self.tts.is_speaking():
                                    self.tts.interrupt()
                                    print("  ⏸  TTS interrupted (HUD click)", flush=True)
                                # Clear echo-detector last_tts so the
                                # next utterance isn't auto-rejected.
                                try:
                                    self.echo_detector._last_tts_text = ""
                                except Exception:
                                    pass
                                self._set_face_state_listening()
                            elif action == "confirm":
                                # User confirmed a pending action — flag
                                # is read by the next dispatch turn.
                                self._pending_confirmation = "yes"
                                print("  ✅ Confirmed (HUD)", flush=True)
                            elif action == "deny":
                                self._pending_confirmation = "no"
                                print("  ❌ Denied (HUD)", flush=True)
                            elif action == "mute":
                                if self.tts and self.tts.enabled:
                                    self.tts.interrupt()
                                print("  🔇 TTS muted (HUD right-click)", flush=True)
                            # Delete to avoid re-firing on next poll.
                            try:
                                os.remove(ctrl_path)
                            except Exception:
                                pass
                except Exception as e:
                    debug_log(f"HUD control watcher error: {e}", "voice")
                time.sleep(0.5)

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
        self._llm_warmup_started_at = time.time()
        self._llm_warmup_threads = self._start_llm_warmup()

        # Start HUD control-file watcher — picks up "end_session" / "stop"
        # commands written by the HUD's right-click menu.
        self._start_hud_control_watcher()

        # Determine and initialise Whisper backend
        self._whisper_backend = self._determine_whisper_backend()
        model_name = getattr(self.cfg, "whisper_model", "small")

        # Validate large-v3-turbo support for faster-whisper backend
        if model_name == "large-v3-turbo" and self._whisper_backend != "mlx":
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

        if self._whisper_backend == "mlx":
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
        else:
            # faster-whisper backend
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
            deadline = getattr(self, "_llm_warmup_started_at", time.time()) + budget
            for t in warmup_threads:
                remaining = max(0.0, deadline - time.time())
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
                        else:
                            # Maintain pre-roll buffer
                            self._pre_roll.append(frame.copy())
                            while len(self._pre_roll) > pre_roll_max_frames:
                                try:
                                    self._pre_roll.popleft()
                                except Exception:
                                    break
                    else:
                        if is_voice:
                            self._utterance_frames.append(frame.copy())
                            self._silence_frames = 0
                        else:
                            self._silence_frames += 1
                            # Use shorter timeout during TTS for quick stop command detection
                            current_max_frames = tts_max_utt_frames if (self.tts and self.tts.is_speaking()) else normal_max_utt_frames
                            if self._silence_frames >= endpoint_silence_frames or len(self._utterance_frames) >= current_max_frames:
                                self._finalize_utterance()
                                self._pre_roll.clear()

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

    def _finalize_utterance(self) -> None:
        """Process completed utterance through speech recognition."""
        if np is None or not self._utterance_frames:
            self.is_speech_active = False
            self._silence_frames = 0
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
            if self._whisper_backend == "mlx":
                # MLX Whisper transcription
                with self.transcribe_lock:
                    # Forced language hint — Whisper auto-detect on short
                    # wake-word audio routinely guesses English (training
                    # imbalance) and hallucinates "Thank you" / "Charlie's"
                    # over real Ukrainian. Forcing 'uk' keeps it on the
                    # right phonetic map and still transcribes RU and most
                    # EN wake-word mishearings acceptably.
                    forced_lang = getattr(self.cfg, "whisper_language", None) or None
                    result = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo=self._mlx_model_repo,
                        language=forced_lang,
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
                                # Marginal confidence - show in log viewer (not debug)
                                print(f"🔇 Low confidence ({confidence:.2f}): \"{seg_text[:50]}...\"", flush=True)
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
                # where the attribute may be absent.
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
        separator = "" if self._first_utterance else f"\n{'─' * 50}"
        self._first_utterance = False
        print(f"{separator}\n📝 Heard: \"{text}\"", flush=True)

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
            tts_finish_time = self.echo_detector._last_tts_finish_time
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

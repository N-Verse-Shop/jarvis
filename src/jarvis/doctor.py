"""Jarvis health-check CLI — startup validation.

Inspired by `claw-code doctor`. Verifies that all required components
(audio, STT, TTS, LLM endpoint, IPC paths, action dispatcher) are
healthy BEFORE starting the daemon. Prints a clear go/no-go report
with actionable fix suggestions.

Run: `python -m jarvis.doctor`  or  `jarvis-isair doctor`

Exit codes:
  0 — all checks passed
  1 — one or more critical checks failed (daemon won't start)
  2 — warnings only (daemon will start but some features degraded)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable


# ─── reporting helpers ───────────────────────────────────────────────────

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _icon(status: str) -> str:
    return {
        "ok": f"{GREEN}✓{RESET}",
        "warn": f"{YELLOW}⚠{RESET}",
        "fail": f"{RED}✗{RESET}",
    }.get(status, "?")


def _print_check(name: str, status: str, detail: str = "", fix: str = "") -> None:
    print(f"  {_icon(status)}  {name:<40s}  {detail}")
    if fix and status != "ok":
        print(f"       {YELLOW}→ {fix}{RESET}")


# ─── individual checks ───────────────────────────────────────────────────

def check_python_version() -> tuple[str, str, str]:
    """Python ≥ 3.11 required for jarvis-isair."""
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return "ok", detail, ""
    return "fail", detail, "Install Python 3.11+ via Homebrew: brew install python@3.11"


def check_ollama_endpoint() -> tuple[str, str, str]:
    """Ollama server reachable at $JARVIS_OLLAMA_URL (default Hetzner tunnel)."""
    url = os.environ.get("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")
    try:
        t0 = time.time()
        with urllib.request.urlopen(f"{url}/api/version", timeout=3) as r:
            data = r.read().decode()
            dt = (time.time() - t0) * 1000
        return "ok", f"{url} ({dt:.0f}ms) — {data.strip()[:50]}", ""
    except Exception as e:
        return "fail", f"{url} — {str(e)[:60]}", \
            "Check tailscale + ssh tunnel: launchctl kickstart -k gui/$(id -u)/com.jarvis.ollama-tunnel"


def check_ollama_models() -> tuple[str, str, str]:
    """Required models present on Ollama server."""
    url = os.environ.get("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")
    # Audit round 14 fix: this hardcoded ``gemma2:9b`` drifted off the
    # actual default (``qwen3:8b`` for chat, ``qwen2.5:3b`` for intent
    # judge) and meant ``jarvis doctor`` would report "missing" for a
    # perfectly working install. Read from settings so the check
    # tracks whatever the user has configured.
    try:
        from .config import load_settings as _load_settings
        _cfg = _load_settings()
        required = [
            getattr(_cfg, "ollama_chat_model", "qwen3:8b"),
            getattr(_cfg, "intent_judge_model", "qwen2.5:3b")
                or getattr(_cfg, "ollama_chat_model", "qwen3:8b"),
        ]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        required = [m for m in required if m and not (m in seen or seen.add(m))]
    except Exception:
        # If config load fails (very early boot, malformed JSON),
        # fall back to the historic doctor list so the check still
        # tells the user *something*.
        required = ["qwen3:8b", "qwen2.5:3b"]
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as r:
            import json
            data = json.loads(r.read())
        present = {m["name"].split(":")[0] + ":" + m["name"].split(":")[1].split("-")[0]
                   for m in data.get("models", [])}
        present_norm = {m["name"] for m in data.get("models", [])}
        missing = [m for m in required if m not in present_norm
                   and not any(p.startswith(m) for p in present_norm)]
        if not missing:
            return "ok", f"all {len(required)} models present", ""
        return "fail", f"missing: {', '.join(missing)}", \
            f"On Hetzner: ollama pull {' '.join(missing)}"
    except Exception as e:
        return "warn", f"could not list models: {str(e)[:60]}", \
            "Ensure ollama endpoint check passes first"


def check_piper_voice() -> tuple[str, str, str]:
    """Piper TTS voice files present (uk_UA mykyta)."""
    candidates = [
        Path.home() / ".local/share/jarvis/models/piper/uk_UA-ukrainian_tts-medium.onnx",
        Path.home() / ".local/share/piper/voices/uk_UA-ukrainian_tts-medium.onnx",
        Path.home() / ".cache/piper/uk_UA-ukrainian_tts-medium.onnx",
        Path("/opt/jarvis/piper-voices/uk_UA-ukrainian_tts-medium.onnx"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1_000_000:
            cfg = p.with_suffix(".onnx.json")
            if cfg.exists():
                size_mb = p.stat().st_size / 1024 / 1024
                return "ok", f"{p.name} ({size_mb:.1f}MB)", ""
            return "warn", f"{p.name} found, config .json missing", \
                f"Download config: curl -o {cfg} https://huggingface.co/rhasspy/piper-voices/raw/main/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx.json"
    return "fail", "no uk_UA voice file found", \
        "Run: scripts/install-piper-voices.sh  (or fetch manually from HF rhasspy/piper-voices)"


def check_whisper_model() -> tuple[str, str, str]:
    """MLX Whisper model cache present."""
    cache_dirs = [
        Path.home() / ".cache/huggingface/hub",
        Path.home() / ".cache/whisper",
    ]
    for d in cache_dirs:
        if not d.exists():
            continue
        # Look for whisper-medium-mlx folders
        for sub in d.rglob("*whisper-medium*"):
            if sub.is_dir():
                return "ok", f"medium model at {sub}", ""
    return "warn", "whisper-medium cache not found", \
        "Will auto-download on first run (~1.5GB, takes 2-5min)"


def check_audio_devices() -> tuple[str, str, str]:
    """PortAudio enumerates at least 1 input device."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        devs = sd.query_devices()
        inputs = [d for d in devs if d.get("max_input_channels", 0) > 0]
        if not inputs:
            return "fail", "no input devices found", \
                "macOS: System Settings > Privacy & Security > Microphone — grant access to Terminal/Python"
        default = sd.default.device[0] if sd.default.device else None
        default_name = devs[default]["name"] if default is not None else "system default"
        return "ok", f"{len(inputs)} input(s), default: {default_name}", ""
    except ImportError:
        return "warn", "sounddevice not installed", "pip install sounddevice"
    except Exception as e:
        return "fail", f"PortAudio error: {str(e)[:60]}", \
            "brew install portaudio && pip install --force-reinstall sounddevice"


def check_ipc_paths() -> tuple[str, str, str]:
    """HUD ↔ daemon IPC directory writable."""
    ipc_dir = Path.home() / "Library/Application Support/jarvis"
    try:
        ipc_dir.mkdir(parents=True, exist_ok=True)
        test_file = ipc_dir / ".doctor-test"
        test_file.write_text("ok")
        test_file.unlink()
        return "ok", str(ipc_dir), ""
    except Exception as e:
        return "fail", f"{ipc_dir} — {e}", \
            f"chmod 755 {ipc_dir} or check disk permissions"


def check_action_dispatcher() -> tuple[str, str, str]:
    """parse_user_command + action patterns load cleanly."""
    try:
        from .listening.action_dispatcher import (
            parse_user_command, parse_action, ACTION_PATTERNS,
            USER_COMMAND_PATTERNS, APP_ALIASES,
        )
        # Sanity test one of each
        a1 = parse_user_command("відкрий Safari")
        a2 = parse_action("Зараз відкрию Safari")
        if a1 is None or a2 is None:
            return "fail", "patterns load but don't match", \
                "Run: python3 -m pytest tests/test_action_dispatcher.py"
        return "ok", f"{len(USER_COMMAND_PATTERNS)} user-cmd patterns, " \
                     f"{len(ACTION_PATTERNS)} LLM patterns, {len(APP_ALIASES)} app aliases", ""
    except Exception as e:
        return "fail", f"import error: {str(e)[:80]}", \
            "Check src/jarvis/listening/action_dispatcher.py for syntax errors"


def check_mac_permissions() -> tuple[str, str, str]:
    """macOS Automation + Accessibility permissions (needed for AppleScript)."""
    if sys.platform != "darwin":
        return "ok", "non-macOS, skipping", ""
    # Try a harmless osascript to verify Automation isn't blocked
    try:
        r = subprocess.run(
            ["osascript", "-e", "tell application \"System Events\" to get name of first process"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            return "ok", "AppleScript executes", ""
        if "not authorized" in (r.stderr or "").lower():
            return "warn", "Automation permission missing", \
                "System Settings > Privacy & Security > Automation — enable for Terminal/Python"
        return "warn", r.stderr.strip()[:80], \
            "Some PC-control actions may fail"
    except Exception as e:
        return "warn", f"check failed: {e}", ""


def check_disk_space() -> tuple[str, str, str]:
    """At least 2GB free for logs, model caches, persistent memory."""
    try:
        usage = shutil.disk_usage(Path.home())
        free_gb = usage.free / 1024**3
        if free_gb < 2:
            return "fail", f"only {free_gb:.1f}GB free", \
                "Free up disk — Whisper + Piper + caches need 2GB minimum"
        if free_gb < 10:
            return "warn", f"{free_gb:.1f}GB free", "Recommended: 10GB+ for model caches and logs"
        return "ok", f"{free_gb:.1f}GB free", ""
    except Exception as e:
        return "warn", f"check failed: {e}", ""


# ─── main ────────────────────────────────────────────────────────────────

CHECKS: list[tuple[str, Callable[[], tuple[str, str, str]]]] = [
    ("Python version", check_python_version),
    ("Ollama endpoint", check_ollama_endpoint),
    ("Ollama models (gemma2:9b, qwen2.5:3b)", check_ollama_models),
    ("Piper TTS voice (uk_UA)", check_piper_voice),
    ("Whisper STT model cache", check_whisper_model),
    ("Audio input devices", check_audio_devices),
    ("IPC directory writable", check_ipc_paths),
    ("Action dispatcher patterns", check_action_dispatcher),
    ("macOS Automation permission", check_mac_permissions),
    ("Free disk space", check_disk_space),
]


def main(argv: list[str] | None = None) -> int:
    """Run all health checks and return exit code."""
    print(f"\n{BOLD}🩺  Jarvis Doctor — health check{RESET}\n")

    fails = 0
    warns = 0
    for name, fn in CHECKS:
        try:
            status, detail, fix = fn()
        except Exception as e:
            status, detail, fix = "fail", f"check crashed: {e}", \
                "Open an issue with this traceback"
        _print_check(name, status, detail, fix)
        if status == "fail":
            fails += 1
        elif status == "warn":
            warns += 1

    print()
    if fails:
        print(f"{RED}{BOLD}✗  {fails} critical issue(s) — daemon will not start cleanly.{RESET}\n")
        return 1
    if warns:
        print(f"{YELLOW}{BOLD}⚠  {warns} warning(s) — daemon will start but some features degraded.{RESET}\n")
        return 2
    print(f"{GREEN}{BOLD}✓  All checks passed. Jarvis is ready.{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

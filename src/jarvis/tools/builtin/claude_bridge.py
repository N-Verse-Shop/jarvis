"""Claude Code bridge — Jarvis delegates coding tasks to local Claude CLI.

R35-S7. Користувач має Claude Max 5x підписку. Замість використання
дорогого Anthropic API через ключ, Jarvis викликає локальний ``claude``
CLI у headless режимі (``--print``). Це використовує SAME subscription
quota без extra cost.

Призначення:
  - "Створи лендінг для X"
  - "Напиши скрипт що робить Y"
  - "Виправ баг у файлі Z"
  - "Зроби рев'ю цього коду"
  - "Налаштуй проект Foo з deploy на Cloudflare"

Workflow:
  1. Jarvis отримує voice transcript (RU/UA/EN mixed).
  2. Tool router підбирає ``claudeCodeSpawn`` коли запит містить
     "створи / напиши / реалізуй / виправ / зроби / build / fix" + код-сигнали.
  3. Цей tool запускає ``claude --print --output-format json
     --dangerously-skip-permissions --add-dir <workdir> <prompt>``
     у subprocess.
  4. Stdout повертається назад. Jarvis читає коротке резюме голосом,
     повна відповідь лишається в logs.

Безпека:
  - workdir МАЄ бути всередині дозволених коренів (``allowed_roots``):
    ``~/Projects``, ``~/Desktop/Nexus_Studio_Digital_Innovation``,
    ``/tmp/jarvis-claude``. Будь-який інший шлях відхиляється.
  - ``--dangerously-skip-permissions`` дозволено ТІЛЬКИ всередині цих
    коренів; якщо workdir НЕ в списку — користувача повідомляють, що
    треба підтвердити вручну.
  - ``timeout`` — 600s default, max 3600s. Жодне subprocess не може
    висіти більше за це.
  - Жодних shell escapes — args передаються списком до ``subprocess``.
  - Жодних credentials в логах — stderr фільтрується перед записом.

Контракт відповіді (voice-friendly):
  - Success: коротке RU речення "Готово. <2-3 reasons>. Дивись лог: <path>."
  - Failure: RU речення з причиною + path до stderr.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Hard cap on prompt length — voice transcripts rarely exceed 500 chars,
# but a malicious / hallucinated payload could try to OOM Claude CLI.
_MAX_PROMPT_LEN = 4000

# Default timeout (s). 10 min = comfortable for "create a landing
# page" but bounded so a stuck claude doesn't pin the dialogue loop.
_DEFAULT_TIMEOUT_S = 600.0
_MAX_TIMEOUT_S = 3600.0  # 1 hour absolute ceiling

# Allowed work-dir roots. Anything outside these refused outright.
_ALLOWED_ROOTS = [
    Path.home() / "Projects",
    Path.home() / "Desktop" / "Nexus_Studio_Digital_Innovation",
    Path("/tmp/jarvis-claude"),
]

# Default work-dir for ad-hoc projects (auto-named with timestamp).
_DEFAULT_WORK_ROOT = Path("/tmp/jarvis-claude")

# Log dir for full Claude stdout / stderr captures. Keeps voice replies
# concise while preserving the full output for later review.
_LOG_DIR = Path.home() / ".local" / "share" / "jarvis" / "claude-bridge"

# Where `claude` binary lives. PATH lookup at call time, but pre-cache
# a known good location so a tool-call doesn't fail due to non-login
# shell PATH absence.
_CLAUDE_BIN_CANDIDATES = [
    Path.home() / ".npm-global" / "bin" / "claude",
    Path("/opt/homebrew/bin/claude"),
    Path("/usr/local/bin/claude"),
]


def _claude_cfg(key: str, default: str) -> str:
    """Read a claude-spawn config key from config.json, best-effort.

    R35-S30 — lets the user pin the model / reasoning effort used for
    complex tasks without touching code. Never raises: a broken config
    falls back to ``default`` (quality-first).
    """
    try:
        from ...config import load_config
        val = (load_config() or {}).get(key)
        return str(val) if val else default
    except Exception:
        return default


def _complex_free_fallback(prompt: str) -> Optional[str]:
    """R35-S30: strongest FREE model when the Claude CLI is unavailable.

    When ``claude`` is missing or its weekly Opus limit is hit, complex
    tasks would otherwise hard-fail. Instead we answer with the strongest
    free cloud model we can reach — Cerebras ``gpt-oss-120b`` (120B), then
    the full Groq→Cerebras→NIM ladder. Text-only (not agentic file edits),
    but it keeps Jarvis a genius for hard reasoning questions. Returns the
    answer text, or ``None`` if every provider is unreachable.
    """
    sys_prompt = (
        "You are an elite senior software engineer and analyst. Think "
        "carefully and give a thorough, correct, actionable answer."
    )
    try:
        from ...llm_providers import (
            PROVIDERS,
            call_provider_streaming,
            call_tier_ladder,
            _get_key,
        )
        model = _claude_cfg("complex_free_model", "gpt-oss-120b")
        cerebras = next((p for p in PROVIDERS if p.name == "cerebras"), None)
        if cerebras is not None and _get_key(cerebras) is not None:
            r = call_provider_streaming(
                provider=cerebras,
                chat_model=model,
                system_prompt=sys_prompt,
                user_content=prompt,
                timeout_sec=60.0,
                max_tokens=1500,
            )
            if r:
                return r
        reply, _who = call_tier_ladder(
            system_prompt=sys_prompt,
            user_content=prompt,
            per_provider_timeout_sec=30.0,
            max_tokens=1500,
        )
        return reply
    except Exception:
        return None


def _find_claude_binary() -> Optional[Path]:
    """Resolve ``claude`` CLI path. Returns None if not installed."""
    # Try PATH first.
    path = shutil.which("claude")
    if path:
        return Path(path)
    # Fallback to known install locations.
    for candidate in _CLAUDE_BIN_CANDIDATES:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _is_path_in_allowed_root(path: Path) -> bool:
    """Path must resolve to a real location within an allowed root.

    ``Path.resolve()`` defangs ``..`` traversal AND symlink-escape
    attempts that would otherwise let a hostile transcript point the
    tool at ``~/.ssh`` or similar.
    """
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    for root in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _slug(text: str) -> str:
    """Make a filesystem-friendly slug from free-form text."""
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:48] if s else "task"


def _short_summary(stdout: str, max_chars: int = 320) -> str:
    """Extract a voice-friendly summary from Claude's full output.

    Claude's --output-format json wraps response in a structure;
    --output-format text returns prose. Try JSON first, fall back to
    last meaningful prose block. Trim aggressively for TTS.
    """
    s = stdout.strip()
    if not s:
        return "Готово, але Claude нічого не повернув. Дивись лог."
    # Try JSON path.
    try:
        obj = json.loads(s)
        # Common Claude --print --output-format json shape:
        # {"type":"result","subtype":"success","result":"...","is_error":false,"session_id":"...","duration_ms":...}
        if isinstance(obj, dict) and "result" in obj:
            result_text = str(obj.get("result", "")).strip()
            if result_text:
                # Drop common preambles; take last meaningful paragraph.
                paragraphs = [p.strip() for p in result_text.split("\n\n") if p.strip()]
                if paragraphs:
                    summary = paragraphs[-1]
                else:
                    summary = result_text
                if len(summary) > max_chars:
                    summary = summary[: max_chars - 1].rstrip() + "…"
                return summary
    except (json.JSONDecodeError, ValueError):
        pass
    # Plain text fallback — last paragraph.
    paragraphs = [p.strip() for p in s.split("\n\n") if p.strip()]
    summary = paragraphs[-1] if paragraphs else s
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def _scrub_secrets(text: str) -> str:
    """Best-effort redaction of common credential shapes before logging.

    Belt-and-braces. The Jarvis architecture says credentials never
    cross this boundary, but if a hostile / hallucinated transcript
    asks Claude to echo an .env file, we don't want it in our log.
    """
    # Long base64-ish tokens (40+ chars).
    text = re.sub(r"[A-Za-z0-9+/_=-]{40,}", "<REDACTED>", text)
    # api_key= / password= / token= / Bearer XXX
    text = re.sub(r"(?i)(api[_-]?key|password|secret|token|bearer)\s*[:=]\s*\S+", r"\1=<REDACTED>", text)
    return text


def _write_log(stdout: str, stderr: str, prompt: str, workdir: Path, exit_code: int) -> Path:
    """Write full session log under ~/.local/share/jarvis/claude-bridge/."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(prompt[:60])
    log_path = _LOG_DIR / f"{ts}_{slug}.log"
    try:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"# Claude Code Bridge log\n")
            f.write(f"# timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# workdir:   {workdir}\n")
            f.write(f"# exit_code: {exit_code}\n")
            f.write(f"# prompt:\n{prompt}\n\n")
            f.write("# === stdout ===\n")
            f.write(_scrub_secrets(stdout))
            f.write("\n\n# === stderr ===\n")
            f.write(_scrub_secrets(stderr))
    except OSError as exc:
        debug_log(f"claude_bridge: failed to write log {log_path}: {exc}")
    return log_path


class ClaudeCodeSpawnTool(Tool):
    """Run a one-shot Claude Code task in the background using user's subscription."""

    @property
    def name(self) -> str:
        return "claudeCodeSpawn"

    @property
    def description(self) -> str:
        # R35-S15: description doubles as keyword-router scoring corpus AND
        # quality-tier signal. Triggers for BOTH coding tasks AND general
        # "I want a thoughtful, high-quality answer" queries — both route
        # to Claude Sonnet 4.6 via the local CLI (Max 5x subscription).
        return (
            # Coding triggers (R35-S7 baseline)
            "Создать сайт, написать код, реализовать программу, исправить баг, "
            "написать скрипт, сделать ревью кода, разработать приложение, "
            "построить лендинг, сгенерировать компонент, создать проект, "
            "написати програму, виправити код, створити сайт, побудувати, "
            "розробити, реалізувати, code review, build, fix bug, write code, "
            "generate component, create project, landing page, app, website. "
            # Quality / depth triggers (R35-S15 — explicit escalation)
            "Деталь, детально, ретельно, глибоко, професійно, якісно, "
            "ґрунтовно, подумай добре, проаналізуй уважно, складна задача, "
            "обґрунтуй, поясни детально, опиши повністю, дай якісну відповідь, "
            "через клода, claude, deep, detail, thorough, professional, "
            "carefully, analyse, complex task, в деталях, глубоко, "
            "обоснуй, объясни подробно, через клод, сложная задача. "
            # Strategic / planning triggers
            "Архітектура, стратегія, план, дорожня карта, рекомендація, "
            "архитектура, стратегия, план, дорожная карта, рекомендация. "
            # Mechanism note
            "Делегирует ЛЮБОЙ запрос, требующий качественного ответа, "
            "в локальный Claude Code CLI (Claude Sonnet 4.6 через "
            "подписку Claude Max — НЕ Anthropic API, без доп. расходов). "
            "Используй для сложных задач: код, архитектура, анализ, "
            "стратегические решения, длинные ответы."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Що саме Claude Code повинен зробити. "
                        "Природна мова. Один абзац. "
                        "Приклади: «Створи односторінковий лендінг для IBONS», "
                        "«Виправ помилку TypeError у app.py», "
                        "«Зроби рев'ю PR #42»."
                    ),
                },
                "workdir": {
                    "type": "string",
                    "description": (
                        "Робоча директорія для Claude. За замовчуванням "
                        "/tmp/jarvis-claude/<slug>. ОБМЕЖЕНО: ~/Projects, "
                        "~/Desktop/Nexus_Studio_Digital_Innovation, /tmp/jarvis-claude."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Максимум секунд (default 600, cap 3600).",
                    "minimum": 30,
                    "maximum": 3600,
                },
                "effort": {
                    "type": "string",
                    "description": "Рівень зусиль Claude: low, medium, high, xhigh, max.",
                    "enum": ["low", "medium", "high", "xhigh", "max"],
                },
                "agent": {
                    "type": "string",
                    "description": (
                        "Optional Claude subagent. Наявні: code-reviewer, "
                        "debugger, devops-engineer, expo-react-native-expert, "
                        "search-specialist, security-auditor."
                    ),
                },
                "structured_output": {
                    "type": "boolean",
                    "description": "Якщо true — попросити Claude повернути JSON (для post-processing).",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        }

    def run(
        self,
        args: Optional[Dict[str, Any]],
        context: ToolContext,
    ) -> ToolExecutionResult:
        args = args or {}
        prompt = str(args.get("prompt", "")).strip()

        # ---- Validate inputs -------------------------------------------------
        if not prompt:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Не задано prompt — нічого делегувати в Claude Code.",
            )
        if len(prompt) > _MAX_PROMPT_LEN:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=f"Prompt задовгий: {len(prompt)} > {_MAX_PROMPT_LEN}.",
            )

        # Resolve binary.
        claude_bin = _find_claude_binary()
        if claude_bin is None:
            # R35-S30: Claude CLI missing → don't hard-fail. Answer with the
            # strongest FREE cloud model (gpt-oss-120b / ladder) so complex
            # reasoning still works (text-only, not agentic file edits).
            fb = _complex_free_fallback(prompt)
            if fb:
                return ToolExecutionResult(
                    success=True,
                    reply_text=fb,
                    error_message=None,
                )
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=(
                    "Claude CLI не знайдено. Встанови: "
                    "npm install -g @anthropic-ai/claude-code@latest"
                ),
            )

        # Resolve workdir.
        workdir_arg = args.get("workdir")
        if workdir_arg:
            workdir = Path(str(workdir_arg)).expanduser()
            if not _is_path_in_allowed_root(workdir):
                roots = ", ".join(str(r) for r in _ALLOWED_ROOTS)
                return ToolExecutionResult(
                    success=False,
                    reply_text=None,
                    error_message=f"Workdir {workdir} поза дозволеними коренями: {roots}",
                )
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = _slug(prompt[:48])
            workdir = _DEFAULT_WORK_ROOT / f"{ts}_{slug}"
        workdir.mkdir(parents=True, exist_ok=True)

        timeout_s = float(args.get("timeout_s") or _DEFAULT_TIMEOUT_S)
        timeout_s = min(max(timeout_s, 30.0), _MAX_TIMEOUT_S)

        # ---- Build the claude command ---------------------------------------
        cmd: List[str] = [
            str(claude_bin),
            "--print",
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
            "--add-dir",
            str(workdir),
        ]
        # R35-S30: complex tasks must run on the STRONGEST Claude model for
        # max quality. The Max subscription covers Opus (no API charge), and
        # Claude only ever runs with internet anyway, so there's no reason to
        # use a weaker default. Config-driven (claude_spawn_model, default
        # "opus"); tool-arg ``model`` overrides per call.
        _claude_model = args.get("model") or _claude_cfg("claude_spawn_model", "opus")
        if _claude_model:
            cmd += ["--model", str(_claude_model)]
        # Default reasoning effort to "high" for quality (was: only when the
        # caller passed it). Caller / config can still dial it down.
        effort = args.get("effort") or _claude_cfg("claude_effort", "high")
        if effort in {"low", "medium", "high", "xhigh", "max"}:
            cmd += ["--effort", str(effort)]
        agent = args.get("agent")
        if agent and re.match(r"^[a-z0-9-]+$", str(agent)):
            cmd += ["--agent", str(agent)]
        if args.get("structured_output"):
            # Hint Claude to return structured JSON inside the result.
            prompt = prompt + "\n\nReturn the result as a single JSON object."

        # Prompt goes last as positional arg.
        cmd.append(prompt)

        # ---- Execute ---------------------------------------------------------
        # cwd=workdir so Claude has natural project context.
        # env: keep parent's OAuth / config (~/.claude/) untouched; only
        # set CLAUDE_CODE_BARE=0 to allow hooks/skills (default).
        env = os.environ.copy()
        # Suppress verbose hook output that bloats voice replies.
        env.setdefault("CLAUDE_CODE_SUPPRESS_HOOK_LOGS", "1")

        context.user_print(
            f"🤖 Делегую в Claude Code (subscription, workdir={workdir.name})…"
        )
        debug_log(f"claude_bridge: {' '.join(shlex.quote(c) for c in cmd[:6])}…  workdir={workdir}")
        started = time.monotonic()
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            log_path = _write_log(
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                prompt=prompt,
                workdir=workdir,
                exit_code=-1,
            )
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=(
                    f"Claude не встиг за {elapsed:.0f}s (timeout={timeout_s:.0f}s). "
                    f"Лог: {log_path}"
                ),
            )
        except (OSError, ValueError) as exc:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=f"Не вдалося запустити claude: {exc}",
            )

        elapsed = time.monotonic() - started

        # ---- Process output --------------------------------------------------
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        log_path = _write_log(
            stdout=stdout,
            stderr=stderr,
            prompt=prompt,
            workdir=workdir,
            exit_code=completed.returncode,
        )

        if completed.returncode != 0:
            short_err = (stderr.splitlines() or ["(no stderr)"])[-1][:200]
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=(
                    f"Claude exited {completed.returncode}: {short_err}. "
                    f"Лог: {log_path}"
                ),
            )

        summary = _short_summary(stdout)
        voice_reply = (
            f"Готово за {elapsed:.0f} секунд. {summary} "
            f"Робоча папка: {workdir.name}."
        )

        return ToolExecutionResult(
            success=True,
            reply_text=voice_reply,
        )

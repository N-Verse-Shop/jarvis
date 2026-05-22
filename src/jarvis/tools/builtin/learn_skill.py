"""Self-learning skill — when Jarvis doesn't know how to do X, learn it.

R35-S9. The user's mandate: "якщо щось не буде знати як виконати то він
повинен виконувати пошук як йому це навчитися та находити вихід ситуації
та навчатися щоб достигнути результату ідеального професійного рівня".

Workflow:
  1. User asks Jarvis to do X (e.g. "deploy to Cloudflare Pages").
  2. If Jarvis can't find a matching tool/skill, this tool fires.
  3. learnSkill delegates to the local ``claude`` CLI (R35-S7 bridge):
     "Research how to do <topic>. Write a 200-300 word how-to as a
      markdown skill that future-Jarvis can read and follow. Include
      exact commands, env vars, gotchas. Save to <vault>/skills/<slug>.md."
  4. Future invocations on the same topic load the saved skill and
     follow the recipe — no relearning needed.

Why not embed inside ``claudeCodeSpawn``?
  Separating ``learnSkill`` lets the keyword router recognise *learning*
  intent ("як це зробити", "навчись", "вивчи", "розберись") as a
  distinct trigger from *coding* intent ("створи", "напиши"). The router
  picks the cheaper, faster tool when the user is asking a how-to vs
  ordering a build.

Storage:
  - Vault path: ``~/Documents/Nexus-Brain/04-KNOWLEDGE/skills/<slug>.md``
  - The skills directory is gitignored only for sensitive subfolders;
    learnt skills go to the main 04-KNOWLEDGE/ which IS versioned —
    so every learnt skill gets a free `git commit` next time you push.
  - Filename uses slug of topic to deduplicate (re-learning overwrites).
  - Each file starts with frontmatter:
      ```
      ---
      topic: <verbatim user topic>
      learned_at: ISO timestamp
      sources: [url1, url2, ...]
      ---
      ```

Voice-friendly reply contract:
  - Success: "Навчився <topic>. Збережено: <filename>."
  - Failure: "Не зміг вивчити <topic>. Причина: <reason>."
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Where learned skills land. Kept in 04-KNOWLEDGE so the vault `git push`
# carries them forward across machines automatically.
_SKILLS_DIR = (
    Path.home() / "Documents" / "Nexus-Brain" / "04-KNOWLEDGE" / "skills-learned"
)

# Default learning effort. Higher = better research but slower.
_DEFAULT_EFFORT = "medium"

# Cap on raw topic length (we slug it for filename, plus pass through
# verbatim to claude). 400 chars covers "deploy a Hydrogen storefront
# to Cloudflare Pages with custom domain via wrangler".
_MAX_TOPIC_LEN = 400


def _slug(text: str) -> str:
    """Filesystem-friendly slug. Cyrillic-aware via class allowlist."""
    s = text.lower().strip()
    # Replace whitespace + punctuation with single hyphen.
    s = re.sub(r"[^\wа-яё]+", "-", s, flags=re.IGNORECASE)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:64] if s else "topic"


def _existing_skill_path(topic: str) -> Optional[Path]:
    """Return existing learnt skill file if it already exists."""
    slug = _slug(topic)
    candidate = _SKILLS_DIR / f"{slug}.md"
    return candidate if candidate.exists() else None


class LearnSkillTool(Tool):
    """When Jarvis hits a knowledge gap, learn the topic and save a skill."""

    @property
    def name(self) -> str:
        return "learnSkill"

    @property
    def description(self) -> str:
        return (
            "Самостоятельно изучить новую тему, инструмент, технологию когда "
            "Jarvis не знает как сделать. Делает веб-поиск, читает источники, "
            "сохраняет markdown-памятку в vault. "
            "Тригеры: «навчись», «вивчи», «дізнайся як», «розберись», «дослідь», "
            "«як це зробити», «не знаю як», «розкажи як», «навчити», «learn how», "
            "«research how to», «explain how», «teach yourself», «figure out». "
            "Сохраняет результат в ~/Documents/Nexus-Brain/04-KNOWLEDGE/skills-learned/."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "Тема для вивчення. Природна мова. Один абзац максимум."
                        "Приклади: «deploy a Hydrogen site to Cloudflare Pages», "
                        "«як зробити Telegram bot з n8n», "
                        "«як налаштувати Tailscale ACL».",
                    ),
                },
                "force_relearn": {
                    "type": "boolean",
                    "description": (
                        "Якщо true — перезаписати існуючий навичковий файл. "
                        "За замовчуванням false: якщо вже вивчили — повертаємо "
                        "збережене."
                    ),
                },
                "effort": {
                    "type": "string",
                    "description": "Рівень зусиль Claude під час дослідження.",
                    "enum": ["low", "medium", "high", "xhigh", "max"],
                },
            },
            "required": ["topic"],
            "additionalProperties": False,
        }

    def run(
        self,
        args: Optional[Dict[str, Any]],
        context: ToolContext,
    ) -> ToolExecutionResult:
        args = args or {}
        topic = str(args.get("topic", "")).strip()

        if not topic:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message="Тема відсутня.",
            )
        if len(topic) > _MAX_TOPIC_LEN:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=f"Тема задовга: {len(topic)} > {_MAX_TOPIC_LEN}",
            )

        force = bool(args.get("force_relearn", False))
        effort = args.get("effort") or _DEFAULT_EFFORT

        _SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        # Cached skill present? Reuse unless force.
        cached = _existing_skill_path(topic)
        if cached and not force:
            try:
                cached_text = cached.read_text(encoding="utf-8")[:600]
            except OSError:
                cached_text = ""
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Вже знаю цю тему — навичка збережена у {cached.name}. "
                    f"Попередній витяг: {cached_text[:240]}…"
                ),
            )

        # Delegate the actual research+write to claudeCodeSpawn.
        # We construct a prompt that produces a structured md file.
        slug = _slug(topic)
        target_path = _SKILLS_DIR / f"{slug}.md"
        now_iso = datetime.now().isoformat(timespec="seconds")

        claude_prompt = (
            f"Research the topic: «{topic}».\n\n"
            f"1. Use WebSearch + WebFetch tools to find authoritative sources "
            f"(prefer official docs, recent posts; avoid SEO spam).\n"
            f"2. Synthesize a concise, actionable how-to skill in Ukrainian "
            f"or Russian, matching the language of the topic.\n"
            f"3. Required structure (in Markdown):\n"
            f"   ---\n"
            f"   topic: {topic}\n"
            f"   learned_at: {now_iso}\n"
            f"   sources: [url1, url2, ...]\n"
            f"   ---\n"
            f"   # <Title in user's language>\n"
            f"   ## Коли використовувати\n"
            f"   1-2 sentences when this applies.\n"
            f"   ## Кроки\n"
            f"   Numbered list. Exact commands in code fences. ENV vars in <ALL_CAPS>.\n"
            f"   ## Підводні камені\n"
            f"   3-5 bullets: gotchas, common errors, version-specific quirks.\n"
            f"   ## Перевірка\n"
            f"   1-2 commands that confirm success.\n\n"
            f"4. Length: 200–350 words total. Voice-friendly: every sentence ≤ 18 words.\n"
            f"5. Write the result to: {target_path}\n"
            f"6. Print a 2-sentence summary at the end.\n\n"
            f"Constraints:\n"
            f"- Use Bash and Write tools as needed; do NOT ask for confirmation.\n"
            f"- If a source contradicts another, mark the conflict with > ⚡ CONFLICT:.\n"
            f"- Never invent commands; if you're unsure of a flag, omit it and note "
            f"  «verify in docs» in Підводні камені."
        )

        # Import the bridge tool here (avoid circular at module load).
        from .claude_bridge import ClaudeCodeSpawnTool

        bridge = ClaudeCodeSpawnTool()
        context.user_print(f"📚 Вивчаю тему «{topic[:60]}…»")
        started = time.monotonic()

        bridge_result = bridge.run(
            {
                "prompt": claude_prompt,
                # Allow Claude to write to the skills dir.
                "workdir": str(_SKILLS_DIR.parent),
                "timeout_s": 600,
                "effort": effort,
                "agent": "search-specialist",
            },
            context,
        )

        elapsed = time.monotonic() - started

        if not bridge_result.success:
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=(
                    f"Не зміг вивчити: {bridge_result.error_message or 'unknown error'}"
                ),
            )

        # Verify the file landed.
        if not target_path.exists():
            # Maybe Claude saved to a slightly different filename; look for
            # any new .md in skills-learned/ within the last 60 seconds.
            recent = [
                p
                for p in _SKILLS_DIR.glob("*.md")
                if time.time() - p.stat().st_mtime < 120
            ]
            if recent:
                target_path = recent[0]
            else:
                return ToolExecutionResult(
                    success=False,
                    reply_text=None,
                    error_message=(
                        f"Claude завершив, але skill-файл не з'явився у "
                        f"{_SKILLS_DIR}. Можливо змінено шлях."
                    ),
                )

        try:
            preview = target_path.read_text(encoding="utf-8").splitlines()
            # Skip frontmatter; show first 2 prose lines.
            in_body = False
            prose = []
            for line in preview:
                if line.strip() == "---":
                    in_body = not in_body if not prose else True
                    continue
                if in_body and line.strip() and not line.startswith("#"):
                    prose.append(line.strip())
                if len(prose) >= 2:
                    break
            sample = " ".join(prose)[:240]
        except OSError:
            sample = ""

        return ToolExecutionResult(
            success=True,
            reply_text=(
                f"Навчився «{topic[:50]}» за {elapsed:.0f} секунд. "
                f"Збережено: {target_path.name}. "
                f"Витяг: {sample}…" if sample else f"Збережено: {target_path.name}."
            ),
        )

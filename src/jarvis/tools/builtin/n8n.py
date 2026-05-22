"""n8n voice tool — manage workflows on the user's self-hosted n8n.

R35-S1. Single tool with an ``op`` argument so the LLM doesn't have to
juggle 7+ separate tool names. Supported ops:

  * ``list``                     — list workflows (optional active filter)
  * ``show``                     — get one workflow's summary
  * ``executions``               — recent executions (last N)
  * ``trigger``                  — fire a webhook (production by default)
  * ``activate`` / ``deactivate`` — flip the active flag
  * ``create_from_template``     — instantiate a stored JSON template
  * ``delete``                   — irreversible; requires ``confirm=true``

Auth + base URL are handled by :class:`jarvis.integrations.N8NClient`
which reads ``JARVIS_N8N_API_KEY`` / ``JARVIS_N8N_BASE_URL`` from the
environment (set in ``~/.config/jarvis/.env`` with ``chmod 0600``).

Replies are RU-only by default — the persona post-S48 is Russian; the
tool surface text matches so TTS doesn't accent-mangle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...debug import debug_log
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult


# Status → short RU label used in spoken replies.
_STATUS_RU = {
    "success": "успешно",
    "error": "ошибка",
    "running": "выполняется",
    "waiting": "в ожидании",
    "canceled": "отменено",
}


def _fmt_workflow_line(wf, index: int = 0) -> str:
    """Render one workflow line for spoken / printed output."""
    flag = "▶" if wf.active else "⏸"
    tags = ", ".join(wf.tags) if wf.tags else ""
    tag_suffix = f"  [{tags}]" if tags else ""
    prefix = f"{index}. " if index else ""
    return f"{prefix}{flag} {wf.name}{tag_suffix}"


def _fmt_execution_line(ex, wf_name: Optional[str] = None) -> str:
    """Render one execution line."""
    label = _STATUS_RU.get(ex.status, ex.status)
    started = (ex.started_at or "")[:19].replace("T", " ")
    name_part = f" — {wf_name}" if wf_name else ""
    return f"  • {started}{name_part}: {label}"


def _slug(s: str) -> str:
    """Lower-case alphanumeric slug for matching template filenames."""
    return "".join(c for c in (s or "").lower() if c.isalnum() or c == "_")


def _list_template_dir() -> Path:
    """Where workflow JSON templates live."""
    from ...integrations import n8n_client as _nc  # noqa: F401  (import side-effect ok)
    here = Path(__file__).resolve().parent.parent.parent  # …/jarvis/
    return here / "integrations" / "n8n_templates"


def _load_template(name: str) -> Optional[Dict[str, Any]]:
    """Find + parse a JSON template by name/slug. Returns ``None`` on miss."""
    target = _slug(name)
    if not target:
        return None
    tdir = _list_template_dir()
    if not tdir.exists():
        return None
    for p in tdir.iterdir():
        if p.suffix.lower() != ".json":
            continue
        if _slug(p.stem) == target or target in _slug(p.stem):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                debug_log(f"n8n template parse failed: {p.name}: {exc}", "tools")
                return None
    return None


def _list_available_templates() -> List[str]:
    """Names of templates currently shipped with Jarvis."""
    tdir = _list_template_dir()
    if not tdir.exists():
        return []
    return sorted(p.stem for p in tdir.iterdir() if p.suffix.lower() == ".json")


def _substitute(obj: Any, params: Dict[str, Any]) -> Any:
    """Recursively replace ``{{key}}`` placeholders in strings.

    Used to personalise a template before push (e.g. ``{{slack_channel}}``).
    """
    if isinstance(obj, str):
        out = obj
        for k, v in params.items():
            placeholder = "{{" + k + "}}"
            if placeholder in out:
                out = out.replace(placeholder, str(v))
        return out
    if isinstance(obj, list):
        return [_substitute(x, params) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute(v, params) for k, v in obj.items()}
    return obj


class N8NTool(Tool):
    """Voice-side tool wrapping the n8n REST client."""

    @property
    def name(self) -> str:
        return "n8nAutomation"

    @property
    def description(self) -> str:
        return (
            "Управляй автоматизациями (workflow) на n8n: список текущих, "
            "просмотр истории запусков, ручной запуск webhook, пауза/возобновление, "
            "создание новой автоматизации из шаблона, удаление (с подтверждением). "
            "Используй когда пользователь говорит: 'какие автоматизации', "
            "'запусти автоматизацию ...', 'настрой ...', 'останови автоматизацию ...', "
            "'покажи логи автоматизации ...', 'удали автоматизацию ...'. "
            "Не используй для одноразовых задач — для них есть отдельные tools."
        )

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": [
                        "list",
                        "show",
                        "executions",
                        "trigger",
                        "activate",
                        "deactivate",
                        "create_from_template",
                        "list_templates",
                        "delete",
                    ],
                    "description": "Operation to perform.",
                },
                "name_or_id": {
                    "type": "string",
                    "description": "Workflow name (case-insensitive) or ID. "
                                   "Required for show/trigger/activate/deactivate/delete.",
                },
                "webhook_path": {
                    "type": "string",
                    "description": "For op=trigger: the webhook path configured in "
                                   "the workflow's Webhook node (e.g. 'jarvis/morning-briefing').",
                },
                "payload": {
                    "type": "object",
                    "description": "For op=trigger: JSON body to send to the webhook.",
                    "additionalProperties": True,
                },
                "template": {
                    "type": "string",
                    "description": "For op=create_from_template: which template to instantiate. "
                                   "Use op=list_templates to see what's available.",
                },
                "params": {
                    "type": "object",
                    "description": "For op=create_from_template: {{placeholder}} substitutions "
                                   "(e.g. slack_channel, telegram_chat_id, github_repo).",
                    "additionalProperties": True,
                },
                "active_only": {
                    "type": "boolean",
                    "description": "For op=list: only return active workflows.",
                },
                "limit": {
                    "type": "integer",
                    "description": "For op=executions: how many to return (default 10, max 50).",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "MUST be true for op=delete. Other ops ignore it.",
                },
            },
            "required": ["op"],
        }

    # ─── dispatch ────────────────────────────────────────────────────

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        args = args or {}
        op = (args.get("op") or "").strip().lower()

        try:
            from ...integrations import get_n8n_client, N8NAuthError, N8NNotConfiguredError, N8NError
        except Exception as exc:  # import-time failure (shouldn't happen)
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Не могу загрузить n8n клиент: {exc}",
            )

        # list_templates is purely local (reads JSON files from disk) —
        # no API key required, so handle it before the configured-check.
        if op == "list_templates":
            return self._op_list_templates()

        client = get_n8n_client(getattr(context, "cfg", None))

        # Fail-fast when key isn't set — the LLM should see a clear
        # message it can read aloud instead of a stack trace.
        if not client.is_configured():
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "n8n ещё не настроен. Сгенерируй API ключ в "
                    "n8n → Settings → API, потом добавь "
                    "JARVIS_N8N_API_KEY=<ключ> в ~/.config/jarvis/.env "
                    "и перезапусти меня."
                ),
            )

        try:
            if op == "list":
                return self._op_list(client, args)
            if op == "show":
                return self._op_show(client, args)
            if op == "executions":
                return self._op_executions(client, args)
            if op == "trigger":
                return self._op_trigger(client, args)
            if op == "activate":
                return self._op_activate(client, args, activate=True)
            if op == "deactivate":
                return self._op_activate(client, args, activate=False)
            if op == "list_templates":
                return self._op_list_templates()
            if op == "create_from_template":
                return self._op_create_from_template(client, args)
            if op == "delete":
                return self._op_delete(client, args)

            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Неизвестная операция: {op!r}. "
                              "Доступно: list, show, executions, trigger, "
                              "activate, deactivate, create_from_template, "
                              "list_templates, delete.",
            )
        except N8NAuthError as exc:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "n8n отверг API ключ. Похоже он истёк или был отозван. "
                    "Сгенерируй новый в n8n → Settings → API."
                ),
            )
        except N8NNotConfiguredError as exc:
            return ToolExecutionResult(
                success=True, reply_text=str(exc),
            )
        except N8NError as exc:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Ошибка n8n: {exc}",
            )

    # ─── ops ─────────────────────────────────────────────────────────

    def _op_list(self, client, args: Dict[str, Any]) -> ToolExecutionResult:
        active = args.get("active_only")
        active_flag = bool(active) if active is not None else None
        workflows = client.list_workflows(active=active_flag, limit=100)
        if not workflows:
            return ToolExecutionResult(
                success=True,
                reply_text="У тебя пока нет автоматизаций на n8n."
                           if active_flag is None
                           else "Активных автоматизаций нет.",
            )

        # Sort by active first, then by name.
        workflows.sort(key=lambda w: (not w.active, w.name.lower()))

        # Cap visible names so the LLM doesn't spew 100 lines.
        head = workflows[:20]
        lines = [_fmt_workflow_line(wf, i + 1) for i, wf in enumerate(head)]
        active_count = sum(1 for w in workflows if w.active)
        summary = (
            f"Найдено {len(workflows)} автоматизаций "
            f"({active_count} активных):\n" + "\n".join(lines)
        )
        if len(workflows) > 20:
            summary += f"\n… и ещё {len(workflows) - 20}."
        return ToolExecutionResult(success=True, reply_text=summary)

    def _op_show(self, client, args: Dict[str, Any]) -> ToolExecutionResult:
        ref = (args.get("name_or_id") or "").strip()
        if not ref:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="op=show требует name_or_id",
            )
        # Try as ID first (cheap single call), fall back to name match.
        wf = None
        try:
            wf = client.get_workflow(ref)
        except N8NError if False else Exception:  # narrow at runtime
            try:
                wf = client.find_workflow_by_name(ref)
            except Exception:
                wf = None
        if wf is None:
            return ToolExecutionResult(
                success=True,
                reply_text=f"Не нашёл автоматизацию '{ref}'.",
            )
        state = "активна" if wf.active else "на паузе"
        tag_str = (", теги: " + ", ".join(wf.tags)) if wf.tags else ""
        nodes = wf.raw.get("nodes") or []
        node_types = sorted({n.get("type", "?").rsplit(".", 1)[-1] for n in nodes})
        nodes_str = (", шаги: " + ", ".join(node_types)) if node_types else ""
        return ToolExecutionResult(
            success=True,
            reply_text=f"{wf.name} — {state}{tag_str}{nodes_str}.",
        )

    def _op_executions(self, client, args: Dict[str, Any]) -> ToolExecutionResult:
        ref = (args.get("name_or_id") or "").strip()
        limit_raw = args.get("limit") or 10
        try:
            limit = max(1, min(50, int(limit_raw)))
        except (TypeError, ValueError):
            limit = 10

        wf_id: Optional[str] = None
        wf_name: Optional[str] = None
        if ref:
            wf = client.find_workflow_by_name(ref)
            if wf is None:
                # Maybe an ID was passed.
                try:
                    wf = client.get_workflow(ref)
                except Exception:
                    wf = None
            if wf is not None:
                wf_id = wf.id
                wf_name = wf.name

        executions = client.list_executions(workflow_id=wf_id, limit=limit)
        if not executions:
            return ToolExecutionResult(
                success=True,
                reply_text=(f"У '{wf_name}' нет недавних запусков." if wf_name
                            else "Недавних запусков нет."),
            )
        lines = [_fmt_execution_line(ex, wf_name if wf_id else None)
                 for ex in executions]
        header = (f"Последние запуски '{wf_name}':" if wf_name
                  else f"Последние {len(executions)} запусков:")
        return ToolExecutionResult(
            success=True, reply_text=header + "\n" + "\n".join(lines),
        )

    def _op_trigger(self, client, args: Dict[str, Any]) -> ToolExecutionResult:
        path = (args.get("webhook_path") or "").strip()
        payload = args.get("payload") or {}

        # If no explicit path, try to look up by name → webhook node path.
        if not path:
            ref = (args.get("name_or_id") or "").strip()
            if not ref:
                return ToolExecutionResult(
                    success=False, reply_text=None,
                    error_message="op=trigger требует webhook_path или name_or_id",
                )
            wf = client.find_workflow_by_name(ref)
            if wf is None:
                return ToolExecutionResult(
                    success=True,
                    reply_text=f"Не нашёл автоматизацию '{ref}' для запуска.",
                )
            # Search the workflow for a Webhook node and pull its path.
            for node in (wf.raw.get("nodes") or []):
                ntype = (node.get("type") or "").lower()
                if ntype.endswith("webhook"):
                    params = node.get("parameters") or {}
                    p = params.get("path") or params.get("webhookPath") or ""
                    if p:
                        path = p.lstrip("/")
                        break
            if not path:
                return ToolExecutionResult(
                    success=True,
                    reply_text=(
                        f"У '{wf.name}' нет webhook-триггера — её можно "
                        "запустить только по расписанию или вручную из n8n UI."
                    ),
                )

        result = client.trigger_webhook(path=path, payload=payload)
        ok_text = "Запустил."
        # Try to surface a friendly summary if the workflow returns a message.
        if isinstance(result, dict):
            for k in ("message", "reply", "summary"):
                if isinstance(result.get(k), str) and result[k].strip():
                    ok_text = result[k].strip()[:280]
                    break
        return ToolExecutionResult(success=True, reply_text=ok_text)

    def _op_activate(self, client, args: Dict[str, Any], activate: bool) -> ToolExecutionResult:
        ref = (args.get("name_or_id") or "").strip()
        if not ref:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=("op=activate" if activate else "op=deactivate") + " требует name_or_id",
            )
        wf = client.find_workflow_by_name(ref)
        if wf is None:
            try:
                wf = client.get_workflow(ref)
            except Exception:
                wf = None
        if wf is None:
            return ToolExecutionResult(
                success=True, reply_text=f"Не нашёл автоматизацию '{ref}'.",
            )
        if activate:
            client.activate_workflow(wf.id)
            return ToolExecutionResult(
                success=True, reply_text=f"Включил '{wf.name}'.",
            )
        client.deactivate_workflow(wf.id)
        return ToolExecutionResult(
            success=True, reply_text=f"Поставил '{wf.name}' на паузу.",
        )

    def _op_list_templates(self) -> ToolExecutionResult:
        names = _list_available_templates()
        if not names:
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    "Каталог шаблонов пуст. Сейчас в Jarvis нет встроенных n8n-шаблонов — "
                    "ты можешь либо импортировать workflow вручную, либо подождать "
                    "следующего обновления Jarvis."
                ),
            )
        return ToolExecutionResult(
            success=True,
            reply_text="Доступные шаблоны:\n" + "\n".join(f"  • {n}" for n in names),
        )

    def _op_create_from_template(self, client, args: Dict[str, Any]) -> ToolExecutionResult:
        tmpl_name = (args.get("template") or "").strip()
        if not tmpl_name:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="op=create_from_template требует template",
            )
        tmpl = _load_template(tmpl_name)
        if tmpl is None:
            avail = _list_available_templates()
            avail_str = (", ".join(avail) if avail else "пусто")
            return ToolExecutionResult(
                success=True,
                reply_text=f"Не нашёл шаблон '{tmpl_name}'. Есть: {avail_str}.",
            )
        params = args.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        # Substitute {{...}} placeholders.
        definition = _substitute(tmpl, params)
        # Detect un-substituted placeholders — fail loudly before push.
        leftover = _find_placeholders(definition)
        if leftover:
            missing = ", ".join(sorted(leftover))
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Чтобы создать '{tmpl_name}' мне нужны параметры: {missing}. "
                    "Сообщи их и я повторю."
                ),
            )
        try:
            wf = client.create_workflow(definition)
        except Exception as exc:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"n8n отказал при создании: {exc}",
            )
        # Auto-activate.
        try:
            client.activate_workflow(wf.id)
            active_note = " Включил."
        except Exception:
            active_note = ""
        return ToolExecutionResult(
            success=True,
            reply_text=f"Создал автоматизацию '{wf.name}'.{active_note}",
        )

    def _op_delete(self, client, args: Dict[str, Any]) -> ToolExecutionResult:
        ref = (args.get("name_or_id") or "").strip()
        if not ref:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message="op=delete требует name_or_id",
            )
        if not args.get("confirm"):
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Чтобы удалить '{ref}' окончательно — повтори с "
                    "подтверждением (confirm=true). Это необратимо."
                ),
            )
        wf = client.find_workflow_by_name(ref)
        if wf is None:
            try:
                wf = client.get_workflow(ref)
            except Exception:
                wf = None
        if wf is None:
            return ToolExecutionResult(
                success=True, reply_text=f"Не нашёл автоматизацию '{ref}'.",
            )
        client.delete_workflow(wf.id)
        return ToolExecutionResult(
            success=True, reply_text=f"Удалил '{wf.name}'.",
        )


def _find_placeholders(obj: Any) -> set:
    """Return any remaining ``{{...}}`` markers in a substituted blob."""
    import re
    found = set()

    def walk(x):
        if isinstance(x, str):
            for m in re.finditer(r"\{\{([a-zA-Z0-9_]+)\}\}", x):
                found.add(m.group(1))
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    walk(obj)
    return found

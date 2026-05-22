"""Unit tests for the n8nAutomation voice-tool dispatcher (R35-S3 F6).

Covers all 9 ops + template-validation parity (C-F5):
  * list_templates is local-only (no API key required)
  * other ops fail gracefully when not configured
  * argument validation (delete needs confirm, show needs name_or_id, …)
  * create_from_template auto-activate gate (C-F4)
  * the ``_op_show`` ID-then-name fallback (C-F3 sanity)
  * each shipped template parses + its ``required_params`` matches its
    actual ``{{placeholder}}`` strings — typos at ship time fail loudly.

No live n8n server is touched. The client is monkey-patched with a stub.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from jarvis.integrations.n8n_client import (
    Workflow,
    Execution,
    N8NError,
    N8NAuthError,
)
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.n8n import (
    N8NTool,
    _substitute,
    _find_placeholders,
    _list_template_dir,
    _list_available_templates,
    _load_template,
)


# ─── stubs ────────────────────────────────────────────────────────────


class _StubClient:
    """In-memory n8n client double for the dispatcher tests."""

    def __init__(self, *, configured: bool = True) -> None:
        self._configured = configured
        self.workflows: List[Workflow] = []
        self.calls: List[tuple] = []  # (method, args) trace
        self.fail_on_create: bool = False

    def is_configured(self) -> bool:
        return self._configured

    def list_workflows(self, active=None, tags=None, limit=100, max_pages=50):
        self.calls.append(("list_workflows", active, limit))
        if active is None:
            return list(self.workflows)
        return [w for w in self.workflows if bool(w.active) == bool(active)]

    def get_workflow(self, wf_id: str) -> Workflow:
        self.calls.append(("get_workflow", wf_id))
        for w in self.workflows:
            if w.id == wf_id:
                return w
        raise N8NError(f"not found: {wf_id}", status=404)

    def find_workflow_by_name(self, name: str) -> Optional[Workflow]:
        self.calls.append(("find_workflow_by_name", name))
        for w in self.workflows:
            if w.name.lower() == name.lower():
                return w
        return None

    def create_workflow(self, definition: Dict[str, Any]) -> Workflow:
        self.calls.append(("create_workflow", definition.get("name")))
        if self.fail_on_create:
            raise N8NError("denied", status=400)
        wf = Workflow(
            id="new-1", name=str(definition.get("name", "x")),
            active=False, tags=[], raw=definition,
        )
        self.workflows.append(wf)
        return wf

    def activate_workflow(self, wf_id: str) -> Workflow:
        self.calls.append(("activate_workflow", wf_id))
        for w in self.workflows:
            if w.id == wf_id:
                # mutate via re-assignment, since Workflow is a dataclass
                w_new = Workflow(
                    id=w.id, name=w.name, active=True, tags=w.tags, raw=w.raw
                )
                idx = self.workflows.index(w)
                self.workflows[idx] = w_new
                return w_new
        raise N8NError("not found", status=404)

    def deactivate_workflow(self, wf_id: str) -> Workflow:
        self.calls.append(("deactivate_workflow", wf_id))
        for w in self.workflows:
            if w.id == wf_id:
                w_new = Workflow(
                    id=w.id, name=w.name, active=False, tags=w.tags, raw=w.raw
                )
                idx = self.workflows.index(w)
                self.workflows[idx] = w_new
                return w_new
        raise N8NError("not found", status=404)

    def delete_workflow(self, wf_id: str, *, force: bool = False) -> bool:
        self.calls.append(("delete_workflow", wf_id, force))
        # mirror the real client's safety gate
        if not force:
            raise N8NError("force=True required")
        for w in list(self.workflows):
            if w.id == wf_id:
                self.workflows.remove(w)
                return True
        return False

    def list_executions(self, workflow_id=None, limit=10):
        self.calls.append(("list_executions", workflow_id, limit))
        return []

    def trigger_webhook(self, path, payload=None, test_mode=False):
        self.calls.append(("trigger_webhook", path, test_mode))
        return {"ok": True, "message": "ran"}


def _make_tool_with_stub(stub: _StubClient) -> N8NTool:
    """Patch the get_n8n_client lookup so the dispatcher uses the stub."""
    tool = N8NTool()
    return tool


def _ctx() -> ToolContext:
    """Minimal ToolContext stub the dispatcher accepts."""
    ctx = MagicMock(spec=ToolContext)
    ctx.cfg = MagicMock()
    return ctx


# ─── list_templates: no API key required ──────────────────────────────


def test_list_templates_works_without_api_key():
    """list_templates is a local file read — must work before n8n config."""
    tool = N8NTool()
    # Force "not configured" path: get_n8n_client returns a stub whose
    # is_configured() is False. But list_templates should still succeed
    # because the dispatcher routes BEFORE the configured-check.
    stub = _StubClient(configured=False)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "list_templates"}, _ctx())
    assert result.success is True
    # Must mention at least one shipped template by stem name.
    assert "jarvis_" in (result.reply_text or "").lower() \
        or "шаблон" in (result.reply_text or "").lower()


# ─── op-dispatch: not configured path ─────────────────────────────────


def test_list_returns_friendly_when_not_configured():
    tool = N8NTool()
    stub = _StubClient(configured=False)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "list"}, _ctx())
    assert result.success is True
    assert "не настроен" in (result.reply_text or "").lower() or \
           "api ключ" in (result.reply_text or "").lower()


def test_unknown_op_returns_error():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "nonsense"}, _ctx())
    assert result.success is False
    assert "list" in (result.error_message or "").lower()


# ─── op=list ───────────────────────────────────────────────────────────


def test_op_list_empty_workflows():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "list"}, _ctx())
    assert result.success is True
    assert "пока нет" in (result.reply_text or "").lower() or \
           "найдено 0" in (result.reply_text or "").lower()


def test_op_list_with_workflows():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [
        Workflow(id="1", name="Morning", active=True, tags=["daily"], raw={}),
        Workflow(id="2", name="Telegram", active=False, tags=[], raw={}),
    ]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "list"}, _ctx())
    assert result.success is True
    text = result.reply_text or ""
    assert "Morning" in text
    assert "Telegram" in text
    assert "Найдено 2" in text


def test_op_list_active_only_filter():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [
        Workflow(id="1", name="A", active=True, tags=[], raw={}),
        Workflow(id="2", name="B", active=False, tags=[], raw={}),
    ]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "list", "active_only": True}, _ctx())
    assert result.success is True
    assert "A" in (result.reply_text or "")
    # B is inactive — should be filtered out by stub.list_workflows(active=True)
    assert "B" not in (result.reply_text or "")


# ─── op=show ───────────────────────────────────────────────────────────


def test_op_show_requires_name_or_id():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "show"}, _ctx())
    assert result.success is False
    assert "name_or_id" in (result.error_message or "")


def test_op_show_id_first_then_name_fallback():
    """C-F3: get_workflow() raises → fall back to find_workflow_by_name."""
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [Workflow(id="abc123", name="Morning", active=True, tags=[], raw={"nodes": []})]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "show", "name_or_id": "Morning"}, _ctx())
    assert result.success is True
    assert "Morning" in (result.reply_text or "")
    # Must have tried get_workflow first (it raises because id != name),
    # then find_workflow_by_name.
    methods = [c[0] for c in stub.calls]
    assert "get_workflow" in methods
    assert "find_workflow_by_name" in methods


def test_op_show_not_found():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "show", "name_or_id": "Nope"}, _ctx())
    assert result.success is True
    assert "не нашёл" in (result.reply_text or "").lower()


# ─── op=trigger ────────────────────────────────────────────────────────


def test_op_trigger_requires_path_or_name():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "trigger"}, _ctx())
    assert result.success is False
    assert "webhook_path" in (result.error_message or "")


def test_op_trigger_explicit_path():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run(
            {"op": "trigger", "webhook_path": "jarvis/test", "payload": {"a": 1}},
            _ctx(),
        )
    assert result.success is True
    # Stub returns {"ok": True, "message": "ran"} → friendly summary
    assert "ran" in (result.reply_text or "").lower() or \
           "запустил" in (result.reply_text or "").lower()
    assert ("trigger_webhook", "jarvis/test", False) in stub.calls


# ─── op=activate / deactivate ──────────────────────────────────────────


def test_op_activate_by_name():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [Workflow(id="x1", name="Briefing", active=False, tags=[], raw={})]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "activate", "name_or_id": "Briefing"}, _ctx())
    assert result.success is True
    assert "включил" in (result.reply_text or "").lower()
    methods = [c[0] for c in stub.calls]
    assert "activate_workflow" in methods


def test_op_deactivate_by_name():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [Workflow(id="x1", name="Briefing", active=True, tags=[], raw={})]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "deactivate", "name_or_id": "Briefing"}, _ctx())
    assert result.success is True
    assert "паузу" in (result.reply_text or "").lower() or \
           "пауз" in (result.reply_text or "").lower()


# ─── op=delete ─────────────────────────────────────────────────────────


def test_op_delete_requires_confirm():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [Workflow(id="x1", name="X", active=False, tags=[], raw={})]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "delete", "name_or_id": "X"}, _ctx())
    assert result.success is True
    assert "confirm" in (result.reply_text or "").lower() or \
           "подтверд" in (result.reply_text or "").lower()
    # Must NOT have actually deleted.
    assert any(c[0] == "delete_workflow" for c in stub.calls) is False
    assert len(stub.workflows) == 1


def test_op_delete_with_confirm():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    stub.workflows = [Workflow(id="x1", name="X", active=False, tags=[], raw={})]
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run(
            {"op": "delete", "name_or_id": "X", "confirm": True}, _ctx()
        )
    assert result.success is True
    assert "удалил" in (result.reply_text or "").lower()
    assert len(stub.workflows) == 0
    # R35-S3 P3-32: verify the tool passed force=True to the client
    delete_calls = [c for c in stub.calls if c[0] == "delete_workflow"]
    assert delete_calls
    assert delete_calls[0][2] is True  # force kwarg


def test_client_delete_workflow_refuses_without_force():
    """P3-32: direct client call without force=True raises."""
    from jarvis.integrations.n8n_client import N8NClient
    c = N8NClient(api_key="eyJabc1234567890defGHIjklMNOpqrSTUv")
    with pytest.raises(N8NError) as exc:
        c.delete_workflow("xyz")
    assert "force" in str(exc.value).lower()


# ─── op=create_from_template ───────────────────────────────────────────


def test_op_create_from_template_unknown_template():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run(
            {"op": "create_from_template", "template": "does_not_exist_xyz"},
            _ctx(),
        )
    assert result.success is True
    assert "не нашёл" in (result.reply_text or "").lower()


def test_op_create_from_template_requires_template_name():
    tool = N8NTool()
    stub = _StubClient(configured=True)
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run({"op": "create_from_template"}, _ctx())
    assert result.success is False
    assert "template" in (result.error_message or "")


def test_op_create_from_template_missing_placeholders():
    """If template has un-substituted {{x}}, we ask for the params."""
    tool = N8NTool()
    stub = _StubClient(configured=True)
    templates = _list_available_templates()
    if not templates:
        pytest.skip("no shipped templates to test")
    with patch("jarvis.integrations.get_n8n_client", return_value=stub):
        # Pass NO params — every template has at least one {{placeholder}}.
        result = tool.run(
            {"op": "create_from_template", "template": templates[0]},
            _ctx(),
        )
    assert result.success is True
    # Should mention needing params.
    text = (result.reply_text or "").lower()
    assert "параметр" in text or "нужны" in text or "сообщи" in text


def test_op_create_from_template_auto_activate_default_true():
    """C-F4: default behaviour activates the new workflow."""
    tool = N8NTool()
    stub = _StubClient(configured=True)
    # Use a minimal handcrafted template path via _load_template patching
    fake_template = {"name": "Test", "nodes": [], "connections": {}}
    with patch("jarvis.tools.builtin.n8n._load_template", return_value=fake_template), \
         patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run(
            {"op": "create_from_template", "template": "anything"},
            _ctx(),
        )
    assert result.success is True
    assert "включил" in (result.reply_text or "").lower()
    methods = [c[0] for c in stub.calls]
    assert "activate_workflow" in methods


def test_op_create_from_template_auto_activate_false_keeps_paused():
    """C-F4: explicit auto_activate=false leaves the workflow paused."""
    tool = N8NTool()
    stub = _StubClient(configured=True)
    fake_template = {"name": "Test2", "nodes": [], "connections": {}}
    with patch("jarvis.tools.builtin.n8n._load_template", return_value=fake_template), \
         patch("jarvis.integrations.get_n8n_client", return_value=stub):
        result = tool.run(
            {
                "op": "create_from_template",
                "template": "anything",
                "auto_activate": False,
            },
            _ctx(),
        )
    assert result.success is True
    text = (result.reply_text or "").lower()
    assert "пауз" in text or "проверить" in text
    methods = [c[0] for c in stub.calls]
    # No activate_workflow call when auto_activate=False
    assert "activate_workflow" not in methods


# ─── Helper-function tests ─────────────────────────────────────────────


def test_substitute_replaces_placeholders():
    out = _substitute(
        {"url": "https://{{host}}/api/{{path}}"},
        {"host": "example.com", "path": "v1"},
    )
    assert out == {"url": "https://example.com/api/v1"}


def test_substitute_recursive_through_lists_and_dicts():
    obj = {
        "outer": [
            {"inner": "{{x}}"},
            "{{y}}",
        ]
    }
    out = _substitute(obj, {"x": "A", "y": "B"})
    assert out == {"outer": [{"inner": "A"}, "B"]}


def test_find_placeholders_collects_unsubstituted():
    obj = {"a": "hello {{name}}", "b": ["{{other}}", "ok"]}
    found = _find_placeholders(obj)
    assert found == {"name", "other"}


def test_find_placeholders_returns_empty_when_clean():
    assert _find_placeholders({"a": "hello world"}) == set()


# ─── Template-validation parity (C-F5) ─────────────────────────────────


_TDIR = _list_template_dir()


def _shipped_template_files() -> List[Path]:
    if not _TDIR.exists():
        return []
    return sorted(p for p in _TDIR.iterdir() if p.suffix.lower() == ".json")


@pytest.mark.parametrize("tpath", _shipped_template_files(), ids=lambda p: p.stem)
def test_shipped_template_parses_and_has_metadata(tpath: Path):
    """Each shipped template must be valid JSON with the metadata block."""
    raw = tpath.read_text(encoding="utf-8")
    data = json.loads(raw)  # raises on syntax error
    assert isinstance(data, dict)
    assert data.get("name"), f"{tpath.name} missing 'name' field"
    assert isinstance(data.get("nodes"), list), f"{tpath.name} missing nodes[]"
    meta = data.get("_jarvis_template_metadata") or {}
    assert isinstance(meta.get("required_params"), list), \
        f"{tpath.name} missing _jarvis_template_metadata.required_params"


@pytest.mark.parametrize("tpath", _shipped_template_files(), ids=lambda p: p.stem)
def test_shipped_template_required_params_match_placeholders(tpath: Path):
    """C-F5: the declared required_params MUST match the actual {{x}} strings.

    Without this test, a typo in either side (added param but forgot to
    declare it, or declared a param that doesn't appear in the body)
    only surfaces at runtime as a cryptic 'missing params' or 'unused
    parameter' error.
    """
    raw = tpath.read_text(encoding="utf-8")
    declared = set(json.loads(raw).get("_jarvis_template_metadata", {}).get("required_params", []))
    actual = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", raw))
    # Strict bidirectional check.
    missing_in_body = declared - actual
    unused_in_body = actual - declared
    assert not missing_in_body, (
        f"{tpath.name}: declared params {missing_in_body!r} not used in body"
    )
    assert not unused_in_body, (
        f"{tpath.name}: body uses {unused_in_body!r} but not declared"
    )

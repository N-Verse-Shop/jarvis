"""Tests for jarvis.capabilities — env-var gate logic."""

from __future__ import annotations

import os

import pytest

from jarvis import capabilities as cap


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Remove any pre-set JARVIS_ENABLE_*/DISABLE_* between tests."""
    for k in list(os.environ.keys()):
        if k.startswith(("JARVIS_ENABLE_", "JARVIS_DISABLE_")):
            monkeypatch.delenv(k, raising=False)
    yield


class TestIsGateOpen:
    def test_open_by_default(self):
        assert cap.is_gate_open("open_apps") is True
        assert cap.is_gate_open("clipboard") is True
        assert cap.is_gate_open("shell_ops") is True

    def test_contacts_closed_by_default(self):
        assert cap.is_gate_open("contacts") is False

    def test_unknown_gate_closed(self):
        assert cap.is_gate_open("nonexistent_gate") is False

    def test_empty_name_closed(self):
        assert cap.is_gate_open("") is False

    def test_enable_env_opens_default_closed_gate(self, monkeypatch):
        monkeypatch.setenv("JARVIS_ENABLE_CONTACTS", "true")
        assert cap.is_gate_open("contacts") is True

    def test_disable_env_kills_open_gate(self, monkeypatch):
        monkeypatch.setenv("JARVIS_DISABLE_SHELL_OPS", "true")
        assert cap.is_gate_open("shell_ops") is False

    def test_disable_beats_enable(self, monkeypatch):
        monkeypatch.setenv("JARVIS_ENABLE_SHELL_OPS", "true")
        monkeypatch.setenv("JARVIS_DISABLE_SHELL_OPS", "true")
        assert cap.is_gate_open("shell_ops") is False

    @pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_truthy_env_values(self, monkeypatch, truthy):
        monkeypatch.setenv("JARVIS_ENABLE_CONTACTS", truthy)
        assert cap.is_gate_open("contacts") is True

    @pytest.mark.parametrize("falsy", ["0", "false", "no", "off", ""])
    def test_falsy_env_values_keep_default(self, monkeypatch, falsy):
        # Open default + falsy enable env = stays at default (open)
        monkeypatch.setenv("JARVIS_ENABLE_SHELL_OPS", falsy)
        assert cap.is_gate_open("shell_ops") is True


class TestIsToolAllowed:
    def test_open_apps_tools(self):
        assert cap.is_tool_allowed("focus_app") is True
        assert cap.is_tool_allowed("open_url") is True

    def test_shell_ops_tools(self):
        assert cap.is_tool_allowed("run_shortcut") is True
        assert cap.is_tool_allowed("type_text") is True
        assert cap.is_tool_allowed("click_at") is True

    def test_unknown_tool_defaults_open(self):
        # Forward-compat: a tool not in the registry isn't blocked.
        assert cap.is_tool_allowed("some_future_tool") is True

    def test_disable_shell_ops_blocks_run_shortcut(self, monkeypatch):
        monkeypatch.setenv("JARVIS_DISABLE_SHELL_OPS", "true")
        assert cap.is_tool_allowed("run_shortcut") is False
        assert cap.is_tool_allowed("type_text") is False
        # Unrelated gate still open
        assert cap.is_tool_allowed("focus_app") is True

    def test_skill_loaders_inert(self):
        assert cap.is_tool_allowed("list_skills") is True
        assert cap.is_tool_allowed("load_skill") is True
        assert cap.is_tool_allowed("load_skill_reference") is True


class TestAllowedTools:
    def test_filter_drops_closed_gates(self, monkeypatch):
        monkeypatch.setenv("JARVIS_DISABLE_MESSAGING", "true")
        names = ["focus_app", "send_message", "open_url"]
        assert cap.allowed_tools(names) == ["focus_app", "open_url"]


class TestGateSummary:
    def test_summary_includes_all_gates(self):
        summary = cap.gate_summary()
        # No leading-underscore gates ("_open") in the public view
        assert all(not g.startswith("_") for g in summary)
        # Sanity: known gates listed
        assert "shell_ops" in summary
        assert "contacts" in summary

    def test_summary_reflects_env(self, monkeypatch):
        monkeypatch.setenv("JARVIS_DISABLE_CLIPBOARD", "true")
        s = cap.gate_summary()
        assert s["clipboard"]["active"] is False
        assert s["clipboard"]["default"] is True

    def test_summary_tools_grouped(self):
        s = cap.gate_summary()
        # shell_ops should list run_shortcut + type_text + click_at + key + list_shortcuts
        assert "run_shortcut" in s["shell_ops"]["tools"]
        assert "type_text" in s["shell_ops"]["tools"]


class TestGateForTool:
    def test_known(self):
        assert cap.gate_for_tool("run_shortcut") == "shell_ops"
        assert cap.gate_for_tool("send_message") == "messaging"

    def test_unknown(self):
        assert cap.gate_for_tool("some_new_tool") == ""

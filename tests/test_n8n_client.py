"""Unit tests for n8n REST client (R35-S1).

Mock requests; no live n8n needed. Covers:
  * placeholder-key detection
  * scrub helper redacts the key from log strings
  * 401 → N8NAuthError, 4xx/5xx → N8NError
  * list_workflows envelope parsing (n8n wraps in {"data": [...]})
  * webhook trigger path encoding
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from jarvis.integrations.n8n_client import (
    N8NClient,
    N8NError,
    N8NAuthError,
    N8NNotConfiguredError,
    Workflow,
    Execution,
    _looks_like_placeholder,
    _scrub,
)


# ─── helpers ──────────────────────────────────────────────────────────


class _MockResp:
    def __init__(self, status_code=200, json_body=None, text_body=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text_body or (json.dumps(json_body) if json_body else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


# ─── placeholder detection ───────────────────────────────────────────


def test_placeholder_detection_empty():
    assert _looks_like_placeholder("") is True
    assert _looks_like_placeholder("   ") is True


def test_placeholder_detection_short():
    assert _looks_like_placeholder("abc") is True  # < 8 chars


def test_placeholder_detection_keywords():
    assert _looks_like_placeholder("YOUR_API_KEY") is True
    assert _looks_like_placeholder("placeholder123") is True
    assert _looks_like_placeholder("xxx_xxx_xxx") is True


def test_placeholder_detection_real_key():
    # JWT-shaped real n8n key
    fake = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJ1IiwiaWQiOiJ4IiwiaWF0Ijox"
    )
    assert _looks_like_placeholder(fake) is False


# ─── scrub helper ────────────────────────────────────────────────────


def test_scrub_replaces_full_key():
    key = "secret_key_abcdefghij1234567890"
    text = f"Error: bad request with key={key}"
    out = _scrub(text, key)
    assert key not in out
    assert "***" in out


def test_scrub_replaces_partial_key():
    key = "secret_key_abcdefghij1234567890"
    text = f"Error: prefix only={key[:12]}…"
    out = _scrub(text, key)
    assert key[:12] not in out


def test_scrub_handles_exception():
    out = _scrub(RuntimeError("boom secret_key_abc1234567890"), "secret_key_abc1234567890")
    assert "secret_key" not in out


def test_scrub_no_key_passthrough():
    assert _scrub("hello", "") == "hello"


# ─── client configuration ─────────────────────────────────────────────


def test_client_not_configured_when_no_key():
    c = N8NClient(api_key="")
    assert c.is_configured() is False


def test_client_not_configured_when_placeholder():
    c = N8NClient(api_key="YOUR_KEY_HERE")
    assert c.is_configured() is False


def test_client_configured_with_real_key():
    # JWT-shaped key — must avoid substrings the placeholder detector
    # treats as "looks fake" ("xxx", "placeholder", "your_key", ...).
    c = N8NClient(api_key="eyJabc1234567890defGHIjklMNOpqrSTUv")
    assert c.is_configured() is True


def test_request_raises_when_not_configured():
    c = N8NClient(api_key="")
    with pytest.raises(N8NNotConfiguredError):
        c._request("GET", "/workflows")


# ─── REST behaviour (mocked) ─────────────────────────────────────────


def _client():
    return N8NClient(api_key="eyJabc1234567890defGHIjklMNOpqrSTUv", base_url="https://n8n.example.com")


@patch.object(N8NClient, "_request")
def test_list_workflows_unwraps_envelope(mock_req):
    mock_req.return_value = {
        "data": [
            {"id": "1", "name": "Morning", "active": True, "tags": [{"name": "daily"}]},
            {"id": "2", "name": "Telegram", "active": False, "tags": []},
        ]
    }
    wfs = _client().list_workflows()
    assert len(wfs) == 2
    assert wfs[0].name == "Morning"
    assert wfs[0].active is True
    assert "daily" in wfs[0].tags
    assert wfs[1].active is False


@patch.object(N8NClient, "_request")
def test_list_workflows_accepts_bare_list(mock_req):
    """Some n8n versions return a list directly, not envelope."""
    mock_req.return_value = [
        {"id": "1", "name": "Morning", "active": True}
    ]
    wfs = _client().list_workflows()
    assert len(wfs) == 1
    assert wfs[0].id == "1"


@patch.object(N8NClient, "_request")
def test_find_workflow_by_name_exact_match(mock_req):
    mock_req.return_value = {"data": [
        {"id": "1", "name": "Morning Briefing", "active": True},
        {"id": "2", "name": "Telegram Relay", "active": False},
    ]}
    c = _client()
    wf = c.find_workflow_by_name("Morning Briefing")
    assert wf is not None
    assert wf.id == "1"


@patch.object(N8NClient, "_request")
def test_find_workflow_by_name_returns_none_on_miss(mock_req):
    mock_req.return_value = {"data": [{"id": "1", "name": "Other", "active": True}]}
    c = _client()
    assert c.find_workflow_by_name("Nonexistent") is None


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_auth_error_raises_n8n_auth(mock_request):
    mock_request.return_value = _MockResp(status_code=401, text_body="unauthorized")
    c = _client()
    with pytest.raises(N8NAuthError):
        c.list_workflows()


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_server_error_raises_n8n_error(mock_request):
    mock_request.return_value = _MockResp(status_code=500, text_body="oops")
    c = _client()
    with pytest.raises(N8NError) as exc_info:
        c.list_workflows()
    assert exc_info.value.status == 500


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_404_raises_n8n_error(mock_request):
    mock_request.return_value = _MockResp(status_code=404, text_body="not found")
    c = _client()
    with pytest.raises(N8NError) as exc_info:
        c.get_workflow("missing")
    assert exc_info.value.status == 404


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_create_workflow_requires_name(mock_request):
    c = _client()
    with pytest.raises(N8NError):
        c.create_workflow({})


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_create_workflow_sends_post(mock_request):
    mock_request.return_value = _MockResp(
        status_code=201, json_body={"id": "x1", "name": "Test", "active": False}
    )
    c = _client()
    wf = c.create_workflow({"name": "Test", "nodes": []})
    assert wf.id == "x1"
    # Verify POST was used
    args, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert "/workflows" in kwargs["url"]


# ─── webhook ──────────────────────────────────────────────────────────


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_trigger_webhook_path_normalised(mock_request):
    mock_request.return_value = _MockResp(status_code=200, json_body={"ok": True})
    c = _client()
    c.trigger_webhook("/jarvis/test", payload={"a": 1})
    args, kwargs = mock_request.call_args
    # Leading slash should be stripped before joining.
    assert kwargs["url"] == "https://n8n.example.com/webhook/jarvis/test"


@patch("jarvis.integrations.n8n_client.requests.Session.request")
def test_trigger_webhook_test_mode(mock_request):
    mock_request.return_value = _MockResp(status_code=200, json_body={})
    c = _client()
    c.trigger_webhook("jarvis/test", test_mode=True)
    args, kwargs = mock_request.call_args
    assert "/webhook-test/" in kwargs["url"]


# ─── model parsing ───────────────────────────────────────────────────


def test_workflow_from_api_handles_string_tags():
    w = Workflow.from_api({"id": "1", "name": "X", "active": True, "tags": ["a", "b"]})
    assert w.tags == ["a", "b"]


def test_workflow_from_api_handles_dict_tags():
    w = Workflow.from_api({"id": "1", "name": "X", "active": True,
                           "tags": [{"id": 1, "name": "daily"}]})
    assert w.tags == ["daily"]


def test_execution_from_api_normalises_status_from_finished_flag():
    # Older n8n versions used "finished" boolean instead of "status".
    ex = Execution.from_api({"id": "e1", "workflowId": "w1", "finished": True,
                             "stoppedAt": "2026-05-22T10:00:00Z"})
    assert ex.status in ("success", "error")


def test_execution_from_api_uses_explicit_status_when_present():
    ex = Execution.from_api({"id": "e1", "workflowId": "w1", "status": "running"})
    assert ex.status == "running"

"""Unit tests for the dashboard /api/voice/inject endpoint (R35-S3 P2-18).

This is the write endpoint used by n8n templates (morning_briefing,
notion_to_memory) to push text into the voice pipeline. Failure modes
must be enumerated because a 404 / 500 here means the n8n workflow
runs but Jarvis stays silent — silent failure mode.

We exercise the handler directly without a live aiohttp server because
the route is auth-gated by middleware (covered separately).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.dashboard.server import DashboardServer


# ─── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def server():
    s = DashboardServer(token="testtoken")
    return s


def _make_request(body):
    """Build a fake aiohttp Request whose .json() returns ``body``."""
    req = MagicMock()
    if isinstance(body, str):
        async def _json():
            raise ValueError("invalid JSON")
        req.json = _json
    else:
        async def _json():
            return body
        req.json = _json
    return req


def _resp_body(resp) -> dict:
    """Extract dict body from aiohttp.web.Response (json_response)."""
    return json.loads(resp.body.decode("utf-8"))


# ─── /api/voice/inject ─────────────────────────────────────────────────


def test_voice_inject_rejects_invalid_json(server, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    req = _make_request("invalid")
    resp = asyncio.run(server._h_voice_inject(req))
    assert resp.status == 400
    assert "invalid JSON" in _resp_body(resp)["error"]


def test_voice_inject_rejects_empty_text(server, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    resp = asyncio.run(server._h_voice_inject(_make_request({"text": ""})))
    assert resp.status == 400
    assert "empty" in _resp_body(resp)["error"]


def test_voice_inject_rejects_whitespace_only(server, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    resp = asyncio.run(server._h_voice_inject(_make_request({"text": "   "})))
    assert resp.status == 400


def test_voice_inject_rejects_oversize(server, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    huge = "x" * 2500
    resp = asyncio.run(server._h_voice_inject(_make_request({"text": huge})))
    assert resp.status == 400
    assert "too long" in _resp_body(resp)["error"]


def test_voice_inject_writes_flag_file(server, tmp_path, monkeypatch):
    """Happy path: writes inject_transcription.flag, returns 200."""
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    payload = {"text": "Доброе утро, погода в Берлине +12°C"}
    resp = asyncio.run(server._h_voice_inject(_make_request(payload)))
    assert resp.status == 200
    body = _resp_body(resp)
    assert body["ok"] is True
    flag_path = Path(body["flag"])
    assert flag_path.exists()
    assert flag_path.read_text(encoding="utf-8") == payload["text"]


def test_voice_inject_preserves_unicode(server, tmp_path, monkeypatch):
    """RU/UA characters must round-trip without corruption (template payloads)."""
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    text = "Привіт, Джарвіс! Завтра нарада о 10:00 на тему R36."
    resp = asyncio.run(server._h_voice_inject(_make_request({"text": text})))
    assert resp.status == 200
    flag = Path(_resp_body(resp)["flag"])
    assert flag.read_text(encoding="utf-8") == text


def test_voice_inject_overwrites_previous_flag(server, tmp_path, monkeypatch):
    """Consecutive injects replace, not append — the voice loop consumes once."""
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    asyncio.run(server._h_voice_inject(_make_request({"text": "first"})))
    resp = asyncio.run(server._h_voice_inject(_make_request({"text": "second"})))
    assert resp.status == 200
    flag = Path(_resp_body(resp)["flag"])
    assert flag.read_text(encoding="utf-8") == "second"


def test_voice_inject_trims_surrounding_whitespace(server, tmp_path, monkeypatch):
    """The handler strips() so leading/trailing whitespace is gone from the flag."""
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    resp = asyncio.run(server._h_voice_inject(_make_request({"text": "  hello  "})))
    assert resp.status == 200
    flag = Path(_resp_body(resp)["flag"])
    assert flag.read_text(encoding="utf-8") == "hello"


def test_voice_inject_handles_missing_text_field(server, tmp_path, monkeypatch):
    """text is required — missing it is treated as empty."""
    monkeypatch.setattr(
        "jarvis.dashboard.server._dashboard_data_dir",
        lambda: tmp_path,
    )
    resp = asyncio.run(server._h_voice_inject(_make_request({})))
    assert resp.status == 400

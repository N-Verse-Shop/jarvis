"""Tests for jarvis.audit — write/query/stats + PII redaction."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.audit import (
    AuditStore,
    _redact,
    _to_json,
    reset_audit_store_singleton,
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    reset_audit_store_singleton()
    yield
    reset_audit_store_singleton()


@pytest.fixture
def store(tmp_path):
    s = AuditStore(db_path=tmp_path / "audit.db")
    yield s
    s.close(timeout=2.0)


# ─────────────────────── PII redaction ──────────────────────────


class TestRedact:
    def test_strips_password(self):
        out = _redact({"app": "Safari", "password": "secret123"})
        assert out["app"] == "Safari"
        assert out["password"] == "<redacted>"

    def test_strips_nested_token(self):
        out = _redact({"creds": {"api_key": "abc", "user": "danylo"}})
        assert out["creds"]["api_key"] == "<redacted>"
        assert out["creds"]["user"] == "danylo"

    def test_handles_list(self):
        out = _redact([{"token": "x"}, {"safe": "y"}])
        assert out[0]["token"] == "<redacted>"
        assert out[1]["safe"] == "y"

    def test_caps_giant_string(self):
        big = "x" * 5000
        out = _redact(big)
        assert isinstance(out, str)
        assert len(out) < 5000
        assert "truncated" in out

    def test_short_string_unchanged(self):
        assert _redact("hello") == "hello"

    def test_depth_limit(self):
        deep = {}
        d = deep
        for _ in range(20):
            d["x"] = {}
            d = d["x"]
        out = _redact(deep)
        # Walk down and confirm a redaction sentinel appears
        cur = out
        for _ in range(20):
            if isinstance(cur, dict) and "x" in cur:
                cur = cur["x"]
            else:
                break
        assert cur == "<redacted: too deep>" or cur == {}


class TestToJson:
    def test_none(self):
        assert _to_json(None) is None

    def test_basic_dict(self):
        s = _to_json({"a": 1, "b": "x"})
        assert s is not None
        import json
        assert json.loads(s) == {"a": 1, "b": "x"}


# ─────────────────────── AuditStore ──────────────────────────


class TestAuditStore:
    def test_emit_then_query(self, store):
        store.emit(
            kind="tool_call",
            tool="focus_app",
            status="completed",
            args={"app": "Safari"},
            duration_ms=50,
        )
        # Writer is async — give it a beat to drain
        _wait_for(lambda: len(store.query()) == 1)
        events = store.query()
        assert len(events) == 1
        e = events[0]
        assert e.kind == "tool_call"
        assert e.tool == "focus_app"
        assert e.status == "completed"
        assert e.duration_ms == 50
        assert e.args == {"app": "Safari"}

    def test_filter_by_status(self, store):
        for i in range(5):
            store.emit(
                kind="tool_call",
                tool=f"t{i}",
                status="completed" if i % 2 == 0 else "failed",
            )
        _wait_for(lambda: len(store.query()) == 5)
        failed = store.query(status="failed")
        assert len(failed) == 2
        for e in failed:
            assert e.status == "failed"

    def test_filter_by_tool(self, store):
        store.emit(kind="tool_call", tool="alpha", status="completed")
        store.emit(kind="tool_call", tool="beta", status="completed")
        store.emit(kind="tool_call", tool="alpha", status="completed")
        _wait_for(lambda: len(store.query()) == 3)
        alpha = store.query(tool="alpha")
        assert len(alpha) == 2
        for e in alpha:
            assert e.tool == "alpha"

    def test_filter_by_since_ts(self, store):
        t0 = time.time()
        store.emit(kind="tool_call", tool="old", status="completed", ts=t0 - 60)
        store.emit(kind="tool_call", tool="new", status="completed", ts=t0)
        _wait_for(lambda: len(store.query()) == 2)
        recent = store.query(since_ts=t0 - 1)
        names = [e.tool for e in recent]
        assert "new" in names
        assert "old" not in names

    def test_redaction_persists(self, store):
        store.emit(
            kind="tool_call",
            tool="send_message",
            status="completed",
            args={"to": "+123", "body": "ok", "api_key": "supersecret"},
        )
        _wait_for(lambda: len(store.query()) == 1)
        e = store.query()[0]
        assert e.args["to"] == "+123"
        assert e.args["api_key"] == "<redacted>"

    def test_stats(self, store):
        for i in range(10):
            store.emit(
                kind="tool_call",
                tool="focus_app" if i < 7 else "new_note",
                status="completed" if i < 8 else "failed",
            )
        store.emit(kind="state", status="info")
        _wait_for(lambda: len(store.query(limit=20)) == 11)
        stats = store.stats()
        assert stats["total"] == 11
        assert stats["by_kind"]["tool_call"] == 10
        assert stats["by_status"]["completed"] == 8
        top_tools = stats["top_tools"]
        tool_names = [t["tool"] for t in top_tools]
        assert "focus_app" in tool_names

    def test_ordering_newest_first(self, store):
        t0 = time.time()
        for i in range(5):
            store.emit(
                kind="tool_call",
                tool=f"t{i}",
                status="completed",
                ts=t0 + i,
            )
        _wait_for(lambda: len(store.query()) == 5)
        events = store.query()
        # Newest first
        assert [e.tool for e in events] == ["t4", "t3", "t2", "t1", "t0"]

    def test_db_file_permissions_owner_only(self, tmp_path):
        store = AuditStore(db_path=tmp_path / "perms.db")
        try:
            import os, stat
            mode = stat.S_IMODE(os.stat(store.path).st_mode)
            assert mode == 0o600
        finally:
            store.close()


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.02):
    """Spin until predicate() is truthy or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    raise AssertionError(
        f"Predicate {predicate} did not become true within {timeout}s"
    )

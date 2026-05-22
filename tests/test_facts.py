"""Tests for jarvis.memory.facts — atomic facts with decay + FTS5.

We test the public contracts:
  - add → query → score ranking
  - decay actually decays
  - idempotent upsert (same key+value bumps timestamp, not row count)
  - FTS5 fuzzy search across Cyrillic + Latin
  - prune removes low-score rows
  - render_facts_for_prompt produces compact block
  - touch updates last_used and hits
  - tombstone (soft delete) hides from queries
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from jarvis.memory.facts import (
    DEFAULT_HALF_LIFE_DAYS,
    Fact,
    FactStore,
    render_facts_for_prompt,
)


@pytest.fixture()
def store(tmp_path: Path) -> FactStore:
    s = FactStore(tmp_path / "facts.db")
    yield s
    s.close()


class TestBasicCRUD:
    def test_add_and_get_roundtrips(self, store: FactStore):
        fid = store.add(
            "user.preference",
            "drinks black coffee, no sugar",
            source="voice",
            confidence=0.9,
        )
        assert fid > 0
        fact = store.get(fid)
        assert fact is not None
        assert fact.key == "user.preference"
        assert fact.value == "drinks black coffee, no sugar"
        assert fact.source == "voice"
        assert fact.confidence == pytest.approx(0.9)
        assert fact.hits == 0

    def test_add_rejects_empty(self, store: FactStore):
        with pytest.raises(ValueError):
            store.add("", "v")
        with pytest.raises(ValueError):
            store.add("k", "")
        with pytest.raises(ValueError):
            store.add("   ", "  ")

    def test_add_clamps_confidence(self, store: FactStore):
        fid_hi = store.add("k", "v1", confidence=99.0)
        fid_lo = store.add("k", "v2", confidence=-1.0)
        assert store.get(fid_hi).confidence == 1.0
        assert store.get(fid_lo).confidence == 0.0

    def test_idempotent_upsert(self, store: FactStore):
        fid_a = store.add("user.location", "Berlin")
        time.sleep(0.01)
        fid_b = store.add("user.location", "Berlin")
        assert fid_a == fid_b, "same key+value should not create a second row"
        assert len(store.all()) == 1

    def test_upsert_bumps_confidence(self, store: FactStore):
        fid = store.add("k", "v", confidence=0.4)
        store.add("k", "v", confidence=0.4)
        store.add("k", "v", confidence=0.4)
        fact = store.get(fid)
        # Each upsert moves halfway to 1.0:
        # 0.4 → 0.7 → 0.85 → 0.925
        assert fact.confidence > 0.8

    def test_soft_delete(self, store: FactStore):
        fid = store.add("k", "v")
        assert store.delete(fid) is True
        assert store.get(fid) is None
        assert len(store.all()) == 0
        # Idempotent
        assert store.delete(fid) is False
        # Tombstoned row still visible with include_tombstones
        assert len(store.all(include_tombstones=True)) == 1

    def test_forget_key(self, store: FactStore):
        store.add("user.likes", "coffee")
        store.add("user.likes", "tea")
        store.add("user.location", "Berlin")
        n = store.forget_key("user.likes")
        assert n == 2
        assert len(store.all()) == 1
        assert store.all()[0].key == "user.location"


class TestRetrieval:
    def test_by_key_prefix(self, store: FactStore):
        store.add("user.likes", "coffee")
        store.add("user.location", "Berlin")
        store.add("system.theme", "dark")
        user_facts = store.by_key("user.")
        assert {f.key for f in user_facts} == {"user.likes", "user.location"}

    def test_fts_finds_substring_match(self, store: FactStore):
        store.add("user.likes", "drinks black coffee in the morning")
        store.add("user.location", "Berlin, Germany")
        store.add("system.theme", "dark mode preferred")
        results = store.search("coffee")
        assert len(results) >= 1
        assert any("coffee" in f.value for f in results)

    def test_fts_finds_cyrillic(self, store: FactStore):
        store.add("user.lang", "розмовляє українською")
        store.add("user.lang", "speaks English")
        results = store.search("українською")
        assert len(results) == 1
        assert "українською" in results[0].value

    def test_fts_returns_empty_for_no_match(self, store: FactStore):
        store.add("k", "alpha bravo charlie")
        results = store.search("xyzzy")
        assert results == []

    def test_search_empty_query_returns_all(self, store: FactStore):
        store.add("user.a", "one")
        store.add("user.b", "two")
        results = store.search("")
        assert len(results) == 2

    def test_search_with_key_prefix_filters(self, store: FactStore):
        store.add("user.likes", "coffee")
        store.add("system.theme", "coffee-themed")
        results = store.search("coffee", key_prefix="user.")
        assert len(results) == 1
        assert results[0].key == "user.likes"

    def test_search_survives_fts_syntax_error(self, store: FactStore):
        store.add("k", "hello world")
        # FTS5 chokes on bare operators; we should fall back.
        results = store.search('AND OR NOT "')
        assert isinstance(results, list)


class TestDecayScoring:
    def test_fresh_fact_scores_near_confidence(self):
        now = time.time()
        f = Fact(
            id=1,
            key="k",
            value="v",
            source="s",
            confidence=0.8,
            ts_utc=now,
            last_used=now,
            hits=0,
        )
        assert f.score(now=now) == pytest.approx(0.8, abs=0.01)

    def test_old_fact_decays(self):
        now = time.time()
        old_ts = now - 60 * 86400  # 60 days ago, half-life=30d
        f = Fact(
            id=1, key="k", value="v", source="s",
            confidence=1.0, ts_utc=old_ts, last_used=old_ts, hits=0,
        )
        # After 2 half-lives, score should be ~0.25
        s = f.score(now=now, half_life_days=DEFAULT_HALF_LIFE_DAYS)
        assert 0.20 < s < 0.30

    def test_hits_boost_score(self):
        now = time.time()
        f_cold = Fact(1, "k", "v", "s", 1.0, now, now, 0)
        f_warm = Fact(2, "k", "v", "s", 1.0, now, now, 50)
        assert f_warm.score(now=now) > f_cold.score(now=now)

    def test_search_orders_by_decayed_score(self, store: FactStore):
        # Two matching rows, one ancient. The fresh one should win
        # despite identical content.
        old_id = store.add("user.coffee", "coffee fact alpha")
        store._conn.execute(
            "UPDATE facts SET ts_utc=?, last_used=? WHERE id=?",
            (time.time() - 200 * 86400, time.time() - 200 * 86400, old_id),
        )
        store.add("user.coffee", "coffee fact bravo")
        results = store.search("coffee")
        assert len(results) >= 2
        assert results[0].value == "coffee fact bravo"


class TestTouch:
    def test_touch_updates_last_used_and_hits(self, store: FactStore):
        fid = store.add("k", "v")
        before = store.get(fid)
        time.sleep(0.01)
        store.touch(fid)
        after = store.get(fid)
        assert after.hits == before.hits + 1
        assert after.last_used > before.last_used

    def test_touch_many_batches(self, store: FactStore):
        ids = [store.add(f"k{i}", f"v{i}") for i in range(5)]
        store.touch_many(ids)
        for fid in ids:
            assert store.get(fid).hits == 1

    def test_touch_ignores_tombstone(self, store: FactStore):
        fid = store.add("k", "v")
        store.delete(fid)
        store.touch(fid)  # should not raise
        # get returns None for tombstoned
        assert store.get(fid) is None


class TestPrune:
    def test_prune_removes_low_score_rows(self, store: FactStore):
        # Three rows: one ancient + low conf, one fresh, one with hits.
        ancient = store.add("k1", "vanish", confidence=0.3)
        store._conn.execute(
            "UPDATE facts SET ts_utc=?, last_used=? WHERE id=?",
            (time.time() - 500 * 86400, time.time() - 500 * 86400, ancient),
        )
        store.add("k2", "stays fresh")
        n = store.prune()
        assert n >= 1
        keys_remaining = {f.key for f in store.all()}
        assert "k1" not in keys_remaining
        assert "k2" in keys_remaining

    def test_prune_respects_hard_age(self, store: FactStore):
        # High confidence but ancient (1 year+) → still pruned.
        fid = store.add("k", "ancient but loved", confidence=1.0)
        store._conn.execute(
            "UPDATE facts SET ts_utc=?, last_used=?, hits=? WHERE id=?",
            (time.time() - 400 * 86400, time.time() - 400 * 86400, 100, fid),
        )
        n = store.prune(hard_age_days=365.0)
        assert n == 1

    def test_prune_is_idempotent(self, store: FactStore):
        store.add("k", "v")
        store.prune()
        # Second call should not double-tombstone.
        n2 = store.prune()
        assert n2 == 0

    def test_vacuum_reclaims_space(self, store: FactStore):
        for i in range(20):
            fid = store.add(f"k{i}", f"v{i}")
            store.delete(fid)
        store.vacuum()
        assert len(store.all(include_tombstones=True)) == 0


class TestStats:
    def test_stats_reports_counts(self, store: FactStore):
        store.add("user.a", "x")
        store.add("user.a", "y")
        store.add("user.b", "z")
        fid = store.add("system.c", "w")
        store.delete(fid)
        s = store.stats()
        assert s["active"] == 3
        assert s["tombstoned"] == 1
        keys = {k["key"] for k in s["top_keys"]}
        assert "user.a" in keys


class TestPromptRender:
    def test_render_empty_for_no_facts(self, store: FactStore):
        assert render_facts_for_prompt(store) == ""

    def test_render_includes_top_facts(self, store: FactStore):
        store.add("user.coffee", "drinks black coffee")
        store.add("user.lang", "speaks Ukrainian")
        block = render_facts_for_prompt(store)
        # R34-S53.1 Phase 6c: was "Known about you:" — header migrated
        # to RU ("Что я знаю о тебе:") in S52 Phase 3 when the persona
        # went RU-only.
        assert block.startswith("Что я знаю о тебе:")
        assert "black coffee" in block
        assert "Ukrainian" in block

    def test_render_respects_max_chars(self, store: FactStore):
        for i in range(20):
            store.add(f"user.x{i}", "x" * 100)
        block = render_facts_for_prompt(store, max_chars=300)
        assert len(block) < 400  # header + a few lines

    def test_render_touches_returned_facts(self, store: FactStore):
        fid = store.add("user.coffee", "drinks coffee")
        block = render_facts_for_prompt(store)
        assert "coffee" in block
        assert store.get(fid).hits == 1


class TestRedaction:
    def test_redacts_api_keys_in_values(self, store: FactStore):
        store.add("user.note", "my key is sk-proj-aaaaaaaaaaaaaaaaaaaaa")
        all_facts = store.all()
        assert "<redacted>" in all_facts[0].value
        assert "sk-proj" not in all_facts[0].value

    def test_redacts_bearer_tokens(self, store: FactStore):
        store.add("user.auth", "Bearer abcdef1234567890")
        assert "<redacted>" in store.all()[0].value

    def test_sensitive_key_redacts_whole_value(self, store: FactStore):
        # When the KEY says it's a secret, the entire value gets
        # redacted even if the value itself doesn't match a regex.
        store.add("user.api_key", "abc123xyz")
        store.add("user.password", "qwerty")
        store.add("user.ssh_key", "anything at all")
        store.add("user.credentials.aws", "literal-text")
        for f in store.all():
            assert f.value == "<redacted>", (
                f"sensitive key '{f.key}' should redact value, got {f.value!r}"
            )

    def test_normal_key_keeps_normal_value(self, store: FactStore):
        store.add("user.likes", "coffee in the morning")
        assert store.all()[0].value == "coffee in the morning"

    def test_redact_handles_empty_safely(self, store: FactStore):
        from jarvis.memory.facts import _redact
        assert _redact("") == ""
        assert _redact("safe text") == "safe text"
        assert _redact("safe", key="user.password") == "<redacted>"


class TestThreadSafety:
    def test_concurrent_adds_do_not_corrupt(self, store: FactStore):
        import threading

        errors = []

        def worker(i: int):
            try:
                for j in range(20):
                    store.add(f"user.{i}", f"val-{i}-{j}")
            except Exception as exc:  # pragma: no cover — should not fire
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors
        assert len(store.all()) == 4 * 20

"""Tests for jarvis.skills — frontmatter parsing + L1/L2/L3 model.

Adapted from KAOS's ``tests/test_skills.py`` style but trimmed to
Jarvis's tighter API (no publish/import-from-URL flow).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.skills.store import (
    Skill,
    SkillStore,
    _parse_frontmatter,
    _parse_list_field,
    reset_skill_store_singleton,
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """Each test gets a fresh global state — no cross-test leakage."""
    reset_skill_store_singleton()
    yield
    reset_skill_store_singleton()


# ─────────────────────── frontmatter parser ──────────────────────────


class TestParseFrontmatter:
    def test_no_frontmatter_returns_empty_and_body_unchanged(self):
        text = "# Just a heading\n\nSome body."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_plain_key_values(self):
        text = "---\nname: foo\ndescription: bar\n---\nbody\n"
        meta, body = _parse_frontmatter(text)
        assert meta == {"name": "foo", "description": "bar"}
        assert body == "body"

    def test_multiline_folded_scalar(self):
        text = (
            "---\n"
            "description: >\n"
            "  This is a long\n"
            "  multi-line value.\n"
            "---\n"
            "body"
        )
        meta, _ = _parse_frontmatter(text)
        assert meta["description"] == "This is a long multi-line value."

    def test_inline_list(self):
        text = "---\ntags: [a, b, c]\n---\nbody"
        meta, _ = _parse_frontmatter(text)
        # _parse_frontmatter keeps the raw bracket form; _parse_list_field
        # is what actually splits it.
        assert "[a, b, c]" in meta["tags"]
        assert _parse_list_field(meta["tags"]) == ["a", "b", "c"]


# ─────────────────────────── _parse_list_field ───────────────────────


class TestParseListField:
    def test_bracketed(self):
        assert _parse_list_field("[a, b, c]") == ["a", "b", "c"]

    def test_bare_csv(self):
        assert _parse_list_field("a, b, c") == ["a", "b", "c"]

    def test_strips_quotes(self):
        assert _parse_list_field("['a', \"b\"]") == ["a", "b"]

    def test_empty(self):
        assert _parse_list_field("") == []
        assert _parse_list_field("[]") == []


# ─────────────────────────── SkillStore ──────────────────────────────


def _seed_skill(
    root: Path,
    name: str,
    description: str = "Test skill.",
    body: str = "Body text.",
    extra_meta: dict | None = None,
    references: dict | None = None,
) -> None:
    """Write a minimal SKILL.md tree under ``root/<name>/``."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    meta_lines = [f"name: {name}", f"description: {description}"]
    for k, v in (extra_meta or {}).items():
        meta_lines.append(f"{k}: {v}")
    frontmatter = "---\n" + "\n".join(meta_lines) + "\n---\n"
    (skill_dir / "SKILL.md").write_text(frontmatter + body, encoding="utf-8")
    if references:
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(exist_ok=True)
        for ref_name, ref_body in references.items():
            (refs_dir / f"{ref_name}.md").write_text(ref_body, encoding="utf-8")


class TestSkillStore:
    def test_empty_dir_returns_no_skills(self, tmp_path):
        store = SkillStore(skills_dir=tmp_path)
        assert store.list_skills() == []
        assert store.l1_catalog() == []
        assert store.catalog_block() == ""

    def test_missing_dir_doesnt_crash(self, tmp_path):
        store = SkillStore(skills_dir=tmp_path / "nonexistent")
        assert store.list_skills() == []
        # No exception raised — silent skip is the contract.

    def test_loads_minimal_skill(self, tmp_path):
        _seed_skill(tmp_path, "alpha")
        store = SkillStore(skills_dir=tmp_path)
        skills = store.list_skills()
        assert len(skills) == 1
        assert skills[0].name == "alpha"
        assert skills[0].description == "Test skill."
        assert skills[0].content == "Body text."

    def test_l1_catalog_line_format(self, tmp_path):
        _seed_skill(tmp_path, "foo", description="Does foo things.")
        store = SkillStore(skills_dir=tmp_path)
        lines = store.l1_catalog()
        assert lines == ["- foo — Does foo things."]

    def test_catalog_block_contains_lines(self, tmp_path):
        _seed_skill(tmp_path, "alpha", description="Alpha task.")
        _seed_skill(tmp_path, "beta", description="Beta task.")
        store = SkillStore(skills_dir=tmp_path)
        block = store.catalog_block()
        assert "alpha — Alpha task." in block
        assert "beta — Beta task." in block
        assert "load_skill(name)" in block  # invitation phrasing

    def test_draft_status_is_excluded(self, tmp_path):
        _seed_skill(tmp_path, "active-one")
        _seed_skill(
            tmp_path, "draft-one", extra_meta={"status": "draft"}
        )
        store = SkillStore(skills_dir=tmp_path)
        names = [s.name for s in store.list_skills()]
        assert "active-one" in names
        assert "draft-one" not in names

    def test_references_loaded(self, tmp_path):
        _seed_skill(
            tmp_path,
            "with-refs",
            references={"EXAMPLE": "# Example body", "SETUP": "# Setup body"},
        )
        store = SkillStore(skills_dir=tmp_path)
        skill = store.get_skill("with-refs")
        assert skill is not None
        assert sorted(skill.references.keys()) == ["EXAMPLE", "SETUP"]

    def test_load_reference_returns_content(self, tmp_path):
        _seed_skill(
            tmp_path,
            "doc",
            references={"GUIDE": "# Guide\n\nHello world"},
        )
        store = SkillStore(skills_dir=tmp_path)
        text = store.load_reference("doc", "GUIDE")
        assert text is not None
        assert "Hello world" in text

    def test_load_reference_unknown_returns_none(self, tmp_path):
        _seed_skill(tmp_path, "x")
        store = SkillStore(skills_dir=tmp_path)
        assert store.load_reference("x", "missing") is None
        assert store.load_reference("missing-skill", "anything") is None

    def test_symlink_traversal_rejected(self, tmp_path):
        """A malicious skill can't read files outside skills_dir via
        a symlink in its references directory."""
        import os
        # Put the skills dir in a CHILD of tmp_path and the sensitive
        # file as a SIBLING — that way the symlink genuinely escapes
        # the sandbox.
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        sensitive = tmp_path / "outside.txt"
        sensitive.write_text("SECRET TOKEN")

        # Create a skill whose references/ contains a symlink to that file.
        _seed_skill(skills_dir, "evil")
        refs_dir = skills_dir / "evil" / "references"
        refs_dir.mkdir(parents=True, exist_ok=True)
        link = refs_dir / "STEAL.md"
        os.symlink(str(sensitive), str(link))

        store = SkillStore(skills_dir=skills_dir)
        skill = store.get_skill("evil")
        assert skill is not None
        assert "STEAL" in skill.references
        # The actual read must be blocked because the target escapes
        # the sandbox root.
        result = store.load_reference("evil", "STEAL")
        assert result is None, (
            f"path-traversal load should return None, got {result!r}"
        )

    def test_in_sandbox_symlink_still_allowed(self, tmp_path):
        """A symlink pointing INSIDE skills_dir should still work
        (some users legitimately symlink shared docs between skills)."""
        import os
        _seed_skill(tmp_path, "host", references={"REAL": "# real body"})
        _seed_skill(tmp_path, "client")
        # Create a symlink in client's refs that points to host's real ref.
        client_refs = tmp_path / "client" / "references"
        client_refs.mkdir(parents=True, exist_ok=True)
        target = tmp_path / "host" / "references" / "REAL.md"
        link = client_refs / "SHARED.md"
        os.symlink(str(target), str(link))

        store = SkillStore(skills_dir=tmp_path)
        result = store.load_reference("client", "SHARED")
        assert result is not None
        assert "real body" in result

    def test_large_reference_is_capped(self, tmp_path):
        big = "x" * (300 * 1024)
        _seed_skill(tmp_path, "big", references={"HUGE": big})
        store = SkillStore(skills_dir=tmp_path)
        text = store.load_reference("big", "HUGE")
        assert text is not None
        # 256 KB cap + truncation marker
        assert len(text) <= 256 * 1024 + 200
        assert "truncated" in text

    def test_get_skill_unknown_returns_none(self, tmp_path):
        _seed_skill(tmp_path, "x")
        store = SkillStore(skills_dir=tmp_path)
        assert store.get_skill("unknown") is None

    def test_broken_skill_does_not_break_others(self, tmp_path):
        """A SKILL.md with totally invalid frontmatter shouldn't prevent
        loading sibling skills."""
        _seed_skill(tmp_path, "good")
        # Hand-craft a deliberately corrupt SKILL.md
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text(
            "---\nname: bad\nthis is not valid yaml at all: [\n",
            encoding="utf-8",
        )
        store = SkillStore(skills_dir=tmp_path)
        names = [s.name for s in store.list_skills()]
        # 'good' must load even if 'bad' parses incompletely.
        assert "good" in names

    def test_reload_picks_up_new_skill(self, tmp_path):
        _seed_skill(tmp_path, "first")
        store = SkillStore(skills_dir=tmp_path)
        assert len(store.list_skills()) == 1
        _seed_skill(tmp_path, "second")
        store.reload()
        names = [s.name for s in store.list_skills()]
        assert sorted(names) == ["first", "second"]

    def test_skill_l1_line_strips_pipes(self):
        """Pipes in description must not leak into the markdown table."""
        skill = Skill(
            name="x",
            description="Does a | b | c",
            content="",
            path=Path("/dev/null"),
        )
        assert "|" not in skill.l1_line

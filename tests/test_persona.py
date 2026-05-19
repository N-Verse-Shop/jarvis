"""Tests for jarvis.persona — markdown parsing + render_prompt_block."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.persona import (
    Persona,
    PersonaStore,
    _parse_bullets,
    _parse_frontmatter,
    _split_h2_sections,
    reset_persona_singleton,
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    reset_persona_singleton()
    yield
    reset_persona_singleton()


class TestFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("Just markdown.")
        assert meta == {}
        assert body == "Just markdown."

    def test_strips_inline_comments(self):
        text = "---\nlang: uk  # default\n---\n# Body"
        meta, _ = _parse_frontmatter(text)
        assert meta == {"lang": "uk"}


class TestH2Split:
    def test_empty(self):
        assert _split_h2_sections("") == {}

    def test_two_sections(self):
        text = "## Voice Rules\n- one\n\n## Boundaries\n- two\n"
        result = _split_h2_sections(text)
        assert sorted(result.keys()) == ["boundaries", "voice rules"]
        assert "- one" in result["voice rules"]
        assert "- two" in result["boundaries"]


class TestBullets:
    def test_dash_bullets(self):
        assert _parse_bullets("- one\n- two\n- three\n") == ["one", "two", "three"]

    def test_star_bullets(self):
        assert _parse_bullets("* one\n* two\n") == ["one", "two"]

    def test_continuation_joins(self):
        text = "- first line\n  continued here\n- second\n"
        result = _parse_bullets(text)
        # First bullet should have continuation merged in
        assert result[0] == "first line continued here"
        assert result[1] == "second"

    def test_ignores_blank_and_non_bullet_text(self):
        text = "Some intro.\n\n- only this counts\n\nMore prose.\n"
        assert _parse_bullets(text) == ["only this counts"]


class TestPersonaStore:
    def test_seeds_default_when_missing(self, tmp_path):
        target = tmp_path / "subdir" / "persona.md"
        store = PersonaStore(path=target)
        persona = store.get()
        assert target.exists()
        assert persona.name == "Jarvis"
        assert persona.owner == "Danylo Molyanko"
        assert persona.identity  # populated from default body
        # Voice rules deliberately empty in the default seed —
        # those live in pipecat_loop's RU/UK base prompt to avoid
        # duplication (R33-S1 design choice).
        assert persona.voice_rules == []
        assert len(persona.boundaries) >= 3
        assert len(persona.shortcuts) >= 3

    def test_reads_user_file(self, tmp_path):
        target = tmp_path / "persona.md"
        target.write_text(
            """\
---
name: Custom
owner: Test User
language_preference: ru
---

# Custom Persona

## Identity

You are Custom — for testing only.

## Voice Rules

- always reply in haiku
- never use semicolons

## Boundaries

- decline anything political

## Shortcuts

- "TU" — Test User
""",
            encoding="utf-8",
        )
        store = PersonaStore(path=target)
        persona = store.get()
        assert persona.name == "Custom"
        assert persona.owner == "Test User"
        assert persona.language_preference == "ru"
        assert "Custom — for testing only" in persona.identity
        assert "always reply in haiku" in persona.voice_rules
        assert "decline anything political" in persona.boundaries
        assert any("Test User" in s for s in persona.shortcuts)

    def test_render_prompt_block_packs_dense(self, tmp_path):
        target = tmp_path / "persona.md"
        target.write_text(
            """\
---
name: J
owner: X
---

## Identity

Be concise.

## Voice Rules

- short replies
- no emojis

## Boundaries

- no secrets

## Shortcuts

- "X" — owner
""",
            encoding="utf-8",
        )
        store = PersonaStore(path=target)
        block = store.get().render_prompt_block()
        # All sections appear; bullets numbered inline for density
        assert "Be concise" in block
        assert "(1) short replies" in block
        assert "(2) no emojis" in block
        assert "[1] no secrets" in block

    def test_to_dict_round_trip(self, tmp_path):
        target = tmp_path / "persona.md"
        store = PersonaStore(path=target)
        persona = store.get()
        as_dict = persona.to_dict()
        assert as_dict["name"] == "Jarvis"
        assert isinstance(as_dict["voice_rules"], list)
        assert isinstance(as_dict["boundaries"], list)
        assert as_dict["path"] == str(target)

    def test_reload_picks_up_edits(self, tmp_path):
        target = tmp_path / "persona.md"
        store = PersonaStore(path=target)
        first = store.get()
        assert first.name == "Jarvis"  # default
        # Overwrite the file
        target.write_text(
            "---\nname: Renamed\nowner: X\n---\n## Identity\nNew identity.\n",
            encoding="utf-8",
        )
        second = store.reload()
        assert second.name == "Renamed"
        assert "New identity" in second.identity

    def test_missing_sections_are_empty(self, tmp_path):
        target = tmp_path / "persona.md"
        target.write_text(
            "---\nname: Minimal\n---\n## Identity\nJust identity.\n",
            encoding="utf-8",
        )
        store = PersonaStore(path=target)
        p = store.get()
        assert p.voice_rules == []
        assert p.boundaries == []
        assert p.shortcuts == []


class TestPersonaPromptBlock:
    def test_empty_persona_renders_empty(self):
        p = Persona()
        assert p.render_prompt_block() == ""

    def test_only_identity_renders(self):
        p = Persona(identity="Be helpful.")
        assert p.render_prompt_block() == "Be helpful."

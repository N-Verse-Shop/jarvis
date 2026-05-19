"""Skills — workspace-local procedures loaded on demand.

Inspired by KAOS (kronos-agent-os v0.1.0) ``kronos/skills``. Same
progressive-disclosure model adapted for Jarvis's voice-first stack:

    L1 — name + one-line description, always in the voice system prompt
    L2 — full SKILL.md body, loaded via ``load_skill(name)`` tool call
    L3 — supporting reference files, loaded via
         ``load_skill_reference(name, ref)`` tool call

The voice prompt stays tiny (≤130 tokens for the Jarvis persona +
~50 tokens for the L1 catalog at 5-10 skills) — small qwen3:8b
context, fast cold-cache evals. Heavy procedures live in SKILL.md
files the LLM pulls only when the task matches.

Skill layout::

    ~/.config/jarvis/skills/
    └── research-brief/
        ├── SKILL.md            # L2 protocol
        └── references/
            └── EXAMPLE.md      # L3 reference

Public API::

    from jarvis.skills import get_skill_store
    store = get_skill_store()
    catalog = store.l1_catalog()         # "name — description" lines
    skill = store.get_skill("research-brief")
    refs = store.list_references("research-brief")
    text = store.load_reference("research-brief", "EXAMPLE")

Tool-layer integration lives in
``jarvis.listening.pipecat_loop._register_skill_tools`` so the LLM
sees ``list_skills`` / ``load_skill`` / ``load_skill_reference``
function-calls. The L1 catalog string from ``catalog_block()`` is
appended to the voice system prompt at pipeline build time.
"""

from __future__ import annotations

from .store import (
    Skill,
    SkillStore,
    get_skill_store,
    reset_skill_store_singleton,
)

__all__ = [
    "Skill",
    "SkillStore",
    "get_skill_store",
    "reset_skill_store_singleton",
]

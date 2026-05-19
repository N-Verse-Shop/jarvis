"""Skill store — scans, parses, and serves SKILL.md files.

Ported from KAOS (kronos-agent-os) ``kronos/skills/store.py``. The
KAOS implementation targets a langchain agent loop; ours is voice-
first, so the public API is trimmed:

* No publish/import-from-URL flow (Jarvis skills are user-edited
  local files; we don't pull from a registry).
* No "review_required" + remote-source metadata (same reason).
* No shared workspace overlay (single user, single workspace).

What we keep from KAOS:

* YAML frontmatter parser with multi-line scalars + inline list
  syntax (``[a, b, c]``). Tolerant of hand-edited markdown.
* L1 catalog vs L2 body vs L3 references model.
* Lazy directory scan + in-memory index; cheap to call repeatedly.
* Singleton accessor so multiple voice turns reuse one parse.

Storage layout::

    ~/.config/jarvis/skills/
    └── <skill_name>/
        ├── SKILL.md
        └── references/
            └── *.md
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.skills.store")


# Default location — user-writable, separate from the daemon's data
# dir so a `rm -rf ~/Library/Application Support/jarvis` doesn't
# nuke hand-authored skills.
def _default_skills_dir() -> Path:
    env = os.environ.get("JARVIS_SKILLS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "jarvis" / "skills"


# ─────────────────────── frontmatter parser ──────────────────────────


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse ``---`` YAML frontmatter from markdown.

    Supports:

    * ``key: value`` (plain)
    * ``key: >`` / ``key: |`` followed by indented continuation lines
      (YAML folded / literal scalars)
    * ``key: [a, b, c]`` (inline list)

    Returns ``(metadata, body)``. Unknown YAML constructs (anchors,
    refs, mappings) are best-effort — we keep the raw string value.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    meta_text, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}

    lines = meta_text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if value in (">", "|", ">-", "|-"):
                # Multi-line scalar — collect indented continuation
                parts = []
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                    parts.append(lines[i].strip())
                    i += 1
                meta[key] = " ".join(parts)
                continue
            meta[key] = value
        i += 1

    return meta, body.strip()


def _parse_list_field(raw: str) -> list[str]:
    """Parse an inline-list YAML value like ``[a, b, c]`` or ``a, b, c``."""
    if not raw:
        return []
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    return [t.strip().strip("\"'") for t in stripped.split(",") if t.strip()]


# ─────────────────────────── data model ──────────────────────────────


@dataclass
class Skill:
    """A loaded skill definition.

    All fields except ``name``/``description``/``content`` are optional;
    a minimal SKILL.md need only have ``name`` and ``description`` in
    the frontmatter (or just a top-level ``# heading`` as a fallback
    description source).
    """

    name: str
    description: str
    content: str            # SKILL.md body, frontmatter stripped
    path: Path
    references: dict[str, Path] = field(default_factory=dict)
    status: str = "active"  # "active" | "draft"
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    risk: str = "low"       # "low" | "medium" | "high" — gates rendering
    locale: str = ""        # ISO-639-1; "" = applies to any language
    loaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def l1_line(self) -> str:
        """Single-line summary for the system-prompt L1 catalog."""
        # Keep ASCII pipes out so the markdown table renders cleanly.
        desc = self.description.replace("|", "/").strip()
        return f"- {self.name} — {desc}"


# ─────────────────────────── store ──────────────────────────────


class SkillStore:
    """Loads skills from a directory and serves them by name.

    Construction is cheap (just stat the dir); a full scan happens
    on first call to :meth:`list_skills` / :meth:`l1_catalog` /
    :meth:`get_skill`. Subsequent calls are O(1). Use
    :meth:`reload` to pick up edits to SKILL.md on disk without
    restarting the daemon.
    """

    def __init__(self, skills_dir: Optional[Path] = None) -> None:
        self._dir = (skills_dir or _default_skills_dir()).expanduser()
        self._skills: dict[str, Skill] = {}
        self._loaded = False
        self._lock = threading.RLock()

    # ---------------------------------------------------- scanning --
    def reload(self) -> None:
        """Force re-scan of the skills directory."""
        with self._lock:
            self._skills.clear()
            self._loaded = False
            self._load_all()

    def _load_all(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._dir.is_dir():
                log.info("Skills dir not found, skipping load: %s", self._dir)
                return
            count = 0
            for skill_dir in sorted(self._dir.iterdir()):
                try:
                    if not skill_dir.is_dir():
                        continue
                    skill_file = skill_dir / "SKILL.md"
                    if not skill_file.is_file():
                        continue
                    self._load_one(skill_dir, skill_file)
                    count += 1
                except Exception as exc:
                    # One broken skill must not break the whole store
                    log.warning(
                        "Failed to load skill %s: %s",
                        skill_dir.name,
                        exc,
                    )
            log.info(
                "Loaded %d skills from %s: %s",
                count,
                self._dir,
                list(self._skills.keys()),
            )

    def _load_one(self, skill_dir: Path, skill_file: Path) -> None:
        raw = skill_file.read_text(encoding="utf-8").strip()
        meta, body = _parse_frontmatter(raw)

        name = meta.get("name", skill_dir.name).strip()
        description = meta.get("description", "").strip()
        if not description:
            # Fallback: first paragraph of the body, capped at 200
            # chars so the L1 catalog stays readable.
            first_para = body.split("\n\n")[0] if body else ""
            description = first_para[:200].strip()
            # Strip a leading markdown ``# heading`` if that was the
            # first "paragraph" — a heading alone isn't a description.
            if description.startswith("#"):
                description = description.lstrip("#").strip()

        # References: scan SUB-DIRECTORY ``references/`` for .md files.
        refs: dict[str, Path] = {}
        refs_dir = skill_dir / "references"
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.iterdir()):
                if ref_file.is_file() and ref_file.suffix.lower() == ".md":
                    refs[ref_file.stem] = ref_file

        skill = Skill(
            name=name,
            description=description,
            content=body,
            path=skill_file,
            references=refs,
            status=meta.get("status", "active").strip().lower(),
            version=meta.get("version", "1.0.0").strip(),
            author=meta.get("author", "").strip(),
            tags=_parse_list_field(meta.get("tags", "")),
            tools=_parse_list_field(meta.get("tools", "")),
            risk=meta.get("risk", "low").strip().lower(),
            locale=meta.get("locale", "").strip().lower(),
        )
        self._skills[name] = skill

    # ---------------------------------------------------- public API --
    def list_skills(self) -> list[Skill]:
        self._load_all()
        with self._lock:
            return [s for s in self._skills.values() if s.status == "active"]

    def get_skill(self, name: str) -> Optional[Skill]:
        self._load_all()
        with self._lock:
            return self._skills.get(name)

    def l1_catalog(self, *, active_locale: str = "") -> list[str]:
        """Return one ``- name — description`` line per active skill.

        ``locale`` in a SKILL.md is a CONTENT-language hint (the
        SKILL.md body is written in that language), not a gate. We
        include every active skill regardless of active_locale —
        the LLM can read a Ukrainian SKILL.md even when the user
        spoke Russian, and the protocol itself is language-neutral.
        ``active_locale`` is reserved for future strict-mode use.
        """
        del active_locale  # currently unused; reserved
        return [s.l1_line for s in self.list_skills()]

    def catalog_block(
        self, *, active_locale: str = "", title: str = "Available skills"
    ) -> str:
        """Return a ready-to-paste system-prompt block.

        Empty string if there are no skills, so callers can
        unconditionally concatenate to a base prompt.
        """
        lines = self.l1_catalog(active_locale=active_locale)
        if not lines:
            return ""
        body = "\n".join(lines)
        return f"\n\n{title} (call `load_skill(name)` for the full protocol):\n{body}"

    def list_references(self, skill_name: str) -> list[str]:
        skill = self.get_skill(skill_name)
        if skill is None:
            return []
        return sorted(skill.references.keys())

    def load_reference(
        self, skill_name: str, ref_name: str
    ) -> Optional[str]:
        skill = self.get_skill(skill_name)
        if skill is None:
            return None
        ref_path = skill.references.get(ref_name)
        if ref_path is None:
            return None
        try:
            return ref_path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning(
                "Failed to read reference %s/%s: %s",
                skill_name,
                ref_name,
                exc,
            )
            return None

    # Diagnostic helpers
    @property
    def skills_dir(self) -> Path:
        return self._dir


# ─────────────────────────── singleton ──────────────────────────────


_INSTANCE: Optional[SkillStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_skill_store(skills_dir: Optional[Path] = None) -> SkillStore:
    """Process-wide singleton.

    Pass ``skills_dir`` to override the location (mostly useful for
    tests). Subsequent calls ignore the argument and return the
    same instance — use :func:`reset_skill_store_singleton` for
    tests that need a clean state.
    """
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = SkillStore(skills_dir=skills_dir)
    return _INSTANCE


def reset_skill_store_singleton() -> None:
    """Drop the singleton so the next ``get_skill_store`` rebuilds.

    Used by tests; not part of the runtime hot path.
    """
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None

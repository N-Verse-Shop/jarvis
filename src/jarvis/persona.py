"""Persona — central voice / behavioural definition for Jarvis.

Inspired by KAOS ``kronos/persona.py``. The persona file is a small
markdown document the user (Danylo) can edit to change Jarvis's
voice, tone, mannerisms, and boundaries without touching code.

File location:

    ~/.config/jarvis/persona.md

If the file doesn't exist on first read, we seed it with a minimal
default (so the user can `open` it and start editing).

The file is parsed once at daemon boot and cached. Hot-reload is
exposed via :meth:`PersonaStore.reload` for callers that want to
pick up edits without restarting the daemon (the dashboard backend
uses this so a save in the persona editor takes effect immediately).

The persona surface is:

* ``identity`` — one paragraph: who Jarvis is, who they serve
* ``voice_rules`` — bullet list: brevity, tone, language rules
* ``boundaries`` — bullet list: hard guardrails (no harm, no secrets)
* ``shortcuts`` — bullet list: user-specific catchphrases / nicknames

The voice system prompt (``pipecat_loop._system_prompt_for``) reads
:meth:`render_prompt_block` to inject the persona content into the
RU/UK voice prompts.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.persona")


# Default seed file content — kept compact (~50 tokens of content)
# so the cold-cache prompt eval on qwen3:8b stays fast. The voice
# format rules (brevity, no-filler) live in the RU/UK base prompt
# in pipecat_loop._VOICE_SYSTEM_PROMPT_* — no duplication here.
# Persona = identity + shortcuts + boundaries that the model
# couldn't otherwise infer from the slim base.
_DEFAULT_PERSONA_MD = """\
---
name: Jarvis
owner: Danylo Molyanko
language_preference: uk  # 'uk', 'ru', or 'auto'
---

# Jarvis Persona

## Identity

You are Jarvis — personal voice assistant for Danylo Molyanko (CEO, Nexus Studio, Rehburg-Loccum, Germany). Concise, decisive, respect the user's time.

## Boundaries

- Never log secrets, tokens, passwords, or .env contents.
- Refuse to send money, initiate transfers, or place orders on user's behalf.
- Confirm before destructive actions (delete, mass-update, send-to-many).
- Don't speculate on health/legal/financial advice — defer to a professional.

## Shortcuts

- Owner's name "Данило" — conjugate: Данило / Данила / Данилу / Даниле. Never "Нило" or "Даня".
- "Nexus" = Nexus Studio (the agency).
- "IBONS" = active Shopify Hydrogen project.
- "Нексус-мобайл" = Nexus Mobile (React Native + Expo).
"""


def _default_persona_path() -> Path:
    env = os.environ.get("JARVIS_PERSONA_PATH")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "jarvis" / "persona.md"


# ─────────────────────── frontmatter parser ──────────────────────────


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Minimal YAML frontmatter parser (delimited by ``---``).

    Identical contract to ``jarvis.skills.store._parse_frontmatter`` —
    duplicated here so persona has no skills dependency.
    """
    import re as _re

    match = _re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, _re.DOTALL)
    if not match:
        return {}, text
    meta_text, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in meta_text.strip().splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            # Strip inline `# comment` from value
            value = value.split("#", 1)[0]
            meta[key.strip()] = value.strip()
    return meta, body.strip()


# ─────────────────────── dataclass ──────────────────────────


@dataclass
class Persona:
    """In-memory representation of persona.md."""

    name: str = "Jarvis"
    owner: str = ""
    language_preference: str = "auto"  # 'uk' | 'ru' | 'auto'
    identity: str = ""
    voice_rules: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    shortcuts: list[str] = field(default_factory=list)
    raw_body: str = ""  # full markdown body (without frontmatter)
    path: Optional[Path] = None

    def render_prompt_block(self, *, lang: str = "uk") -> str:
        """Return a compact system-prompt block (~150-250 tokens).

        Used by ``pipecat_loop._system_prompt_for`` to seed the
        voice LLM with the persona. Keep the output dense — every
        token costs on every voice turn.

        ``lang`` is the active voice language; we drop language
        rules that don't apply (e.g. an English-only rule wouldn't
        show in a UK turn). Currently no rules are language-gated,
        but the API leaves room.
        """
        del lang  # reserved for future language-gated rules

        parts: list[str] = []
        if self.identity:
            parts.append(self.identity.strip())
        if self.voice_rules:
            rules_inline = " ".join(
                f"({i + 1}) {r.strip()}" for i, r in enumerate(self.voice_rules)
            )
            parts.append(f"Rules: {rules_inline}")
        if self.boundaries:
            bd_inline = " ".join(
                f"[{i + 1}] {b.strip()}" for i, b in enumerate(self.boundaries)
            )
            parts.append(f"Boundaries: {bd_inline}")
        if self.shortcuts:
            sh_inline = " ".join(s.strip() for s in self.shortcuts)
            parts.append(f"Shortcuts: {sh_inline}")
        return "\n".join(parts).strip()

    def to_dict(self) -> dict:
        """JSON-serialisable view for the dashboard API."""
        return {
            "name": self.name,
            "owner": self.owner,
            "language_preference": self.language_preference,
            "identity": self.identity,
            "voice_rules": list(self.voice_rules),
            "boundaries": list(self.boundaries),
            "shortcuts": list(self.shortcuts),
            "path": str(self.path) if self.path else None,
        }


# ─────────────────────── markdown section parser ──────────────────────────


def _split_h2_sections(body: str) -> dict[str, str]:
    """Split a markdown body by ``## H2`` headings.

    Returns ``{lower_heading: section_body}``. The body string
    spans from after the heading line to the next ``##`` (or EOF),
    with surrounding whitespace stripped.
    """
    sections: dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip().lower()
            current_lines = []
        else:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _parse_bullets(section_body: str) -> list[str]:
    """Extract ``- bullet`` lines from a markdown section body.

    Tolerant of ``*`` / ``-`` bullets and indented continuations.
    Returns each top-level bullet as a single line; an indented
    continuation line is merged in only when the previous line was
    a bullet or part of one. A BLANK LINE flushes the current bullet
    so trailing prose like

        - foo

        More prose afterwards.

    leaves ``foo`` as the only captured bullet, not ``foo More prose``.
    """
    bullets: list[str] = []
    current: list[str] = []

    def _flush():
        if current:
            bullets.append(" ".join(current).strip())
            current.clear()

    for line in section_body.splitlines():
        stripped = line.lstrip()
        if not stripped:
            # Blank line — end the current bullet's continuation.
            _flush()
            continue
        if line.startswith(("- ", "* ")) or line.startswith(("\t- ", "  - ")):
            _flush()
            text = stripped[2:] if stripped.startswith(("- ", "* ")) else stripped
            current.append(text.strip())
        elif current and (line.startswith((" ", "\t"))):
            # Indented continuation only — non-indented prose is a
            # paragraph break, not part of the bullet.
            current.append(stripped.strip())
        else:
            # Non-indented prose after a bullet => not a continuation.
            _flush()
    _flush()
    return [b for b in bullets if b]


# ─────────────────────── store ──────────────────────────


class PersonaStore:
    """Loads + caches the persona.md file.

    Thread-safe. Auto-seeds the file with :data:`_DEFAULT_PERSONA_MD`
    if the path doesn't exist (first-boot UX — the user can `open`
    the file and start editing).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = (path or _default_persona_path()).expanduser()
        self._persona: Optional[Persona] = None
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_seeded(self) -> None:
        if self._path.exists():
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(_DEFAULT_PERSONA_MD, encoding="utf-8")
            log.info("Seeded default persona at %s", self._path)
        except OSError as exc:
            log.warning("Failed to seed persona at %s: %s", self._path, exc)

    def reload(self) -> Persona:
        with self._lock:
            self._persona = None
            return self.get()

    def get(self) -> Persona:
        with self._lock:
            if self._persona is not None:
                return self._persona
            self._ensure_seeded()
            try:
                text = self._path.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning(
                    "Persona file unreadable (%s); using built-in defaults",
                    exc,
                )
                text = _DEFAULT_PERSONA_MD

            meta, body = _parse_frontmatter(text)
            sections = _split_h2_sections(body)

            # Identity is the first prose paragraph under ## Identity,
            # not bullets. Strip blank lines and treat the body of the
            # section as one paragraph collapsed to single spaces.
            identity = sections.get("identity", "").strip()

            persona = Persona(
                name=meta.get("name", "Jarvis"),
                owner=meta.get("owner", ""),
                language_preference=meta.get("language_preference", "auto").strip().lower(),
                identity=identity,
                voice_rules=_parse_bullets(sections.get("voice rules", "")),
                boundaries=_parse_bullets(sections.get("boundaries", "")),
                shortcuts=_parse_bullets(sections.get("shortcuts", "")),
                raw_body=body,
                path=self._path,
            )
            self._persona = persona
            return persona


# ─────────────────────── singleton ──────────────────────────


_INSTANCE: Optional[PersonaStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_persona_store(path: Optional[Path] = None) -> PersonaStore:
    """Process-wide singleton accessor."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = PersonaStore(path=path)
    return _INSTANCE


def reset_persona_singleton() -> None:
    """Drop the singleton — used by tests + dashboard hot-reload."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None

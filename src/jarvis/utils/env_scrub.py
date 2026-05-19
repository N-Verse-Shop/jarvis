"""Environment-variable scrubber for subprocess spawn paths.

Centralises the logic that strips secret-shaped env vars before
handing an environment to a child process. Used by:

  * ``agent/self_upgrade.py`` — when spawning the ``claude`` CLI for
    self-upgrade. Claude authenticates via ``~/.claude``, so it does
    not need ``OPENAI_API_KEY`` / ``JARVIS_*_TOKEN`` etc.
  * ``tools/external/mcp_client.py`` — when spawning stdio MCP server
    subprocesses (every Node/Python MCP installed via npm/pip is a
    third-party process and should not inherit the full secret env of
    the daemon).

Originally defined locally in ``self_upgrade.py`` in audit round 19;
audit round 20 extracts it to a shared utility because the MCP-spawn
path had the same leak surface.

Design notes
------------
The matcher is a suffix/prefix denylist rather than a strict allowlist
because the legitimate set of env vars children may need is large and
varies by tool (LANG, LC_*, TERM, USER, SHELL, HOME, PATH, PYTHONPATH,
NODE_PATH, NPM_CONFIG_*, ELECTRON_*, DISPLAY, WAYLAND_DISPLAY, ...).
The denylist captures the dangerous-by-shape names — anything ending
in _TOKEN / _KEY / _SECRET / _PASSWORD / _PASSPHRASE / _PRIVATE_KEY,
plus known vendor prefixes (OPENAI_, ANTHROPIC_, AWS_, GCP_, AZURE_,
GITHUB_, GITLAB_, NOTION_, SLACK_, HF_, HUGGINGFACE_, etc.).

When a child process legitimately needs a secret (e.g. a Notion MCP
server needs ``NOTION_TOKEN``), the user explicitly supplies it via
the per-server ``env`` config dict — which is merged on top of the
scrubbed parent env AFTER scrubbing. So the user keeps control of
which secrets each MCP can see, while accidental inheritance of every
secret in the daemon's process env is prevented.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Mapping, Optional


# Audit round 20 — extracted from self_upgrade.py so multiple
# subprocess-spawn sites share one definition. Adding a pattern here
# automatically protects every spawn site (self_upgrade, mcp_client,
# any future agent that shells out).
_SECRET_ENV_PATTERNS = (
    # Generic suffix shapes.
    re.compile(r".*_TOKEN$"),
    re.compile(r".*_API_KEY$"),
    re.compile(r".*_SECRET(?:_KEY)?$"),
    re.compile(r".*_PASSWORD$"),
    re.compile(r".*_PASSPHRASE$"),
    re.compile(r".*_PRIVATE_KEY$"),
    re.compile(r".*_BEARER$"),
    re.compile(r".*_OAUTH.*$"),
    re.compile(r".*_ACCESS_TOKEN$"),
    re.compile(r".*_REFRESH_TOKEN$"),
    re.compile(r".*_SESSION_TOKEN$"),
    re.compile(r".*_BOT_TOKEN$"),
    # Jarvis-internal token names that historically don't follow the
    # _TOKEN convention.
    re.compile(r"^JARVIS_.*_TOKEN$"),
    re.compile(r"^JARVIS_.*_KEY$"),
    # Vendor-prefix names that may not match the generic shape.
    re.compile(
        r"^(?:OPENAI|ANTHROPIC|GROQ|MISTRAL|XAI|DEEPSEEK|GEMINI|GOOGLE|COHERE|REPLICATE)_API_KEY$"
    ),
    re.compile(
        r"^(?:GITHUB|GITLAB|BITBUCKET|NOTION|LINEAR|SLACK|DISCORD|TELEGRAM)_TOKEN$"
    ),
    re.compile(r"^(?:HF|HUGGINGFACE)_TOKEN$"),
    re.compile(r"^(?:AWS|GCP|AZURE)_.*$"),
    re.compile(r"^SUPABASE_(?:KEY|TOKEN|URL)$"),
    re.compile(r"^STRIPE_(?:SECRET|PUBLISHABLE|RESTRICTED)_KEY$"),
    re.compile(r"^TWILIO_(?:AUTH_TOKEN|API_KEY|API_SECRET)$"),
    re.compile(r"^SENDGRID_API_KEY$"),
    re.compile(r"^RESEND_API_KEY$"),
    re.compile(r"^MAPBOX_(?:ACCESS_TOKEN|SECRET_TOKEN)$"),
    re.compile(r"^BRAVE_(?:SEARCH_)?API_KEY$"),
    re.compile(r"^TAVILY_API_KEY$"),
    re.compile(r"^PERPLEXITY_API_KEY$"),
)


def is_secret_env_name(name: str) -> bool:
    """Return True if ``name`` matches any known secret-env pattern.

    Caller code uses this to scrub env dicts before passing them to
    a subprocess. The match is anchored (`^...$`) at the pattern level
    so a substring match like "JARVIS_OPENAI_API_KEY_BACKUP" still
    fires the suffix check (`_KEY$`) — good — but a name like
    "TOKEN_DESCRIPTION" does NOT match `_TOKEN$` — also good.
    """
    return any(p.match(name) for p in _SECRET_ENV_PATTERNS)


def scrub_secret_env(
    source: Optional[Mapping[str, str]] = None,
    *,
    extra_keep: Optional[Iterable[str]] = None,
) -> dict:
    """Return a dict copy of ``source`` (default ``os.environ``) with
    every secret-shaped env var removed.

    Args:
        source: Mapping to scrub. Defaults to a fresh dict-copy of
            ``os.environ`` so the caller can't accidentally mutate
            the live process env.
        extra_keep: Optional iterable of names to forcibly KEEP even
            if they match a secret pattern. Use sparingly — only when
            the caller is certain a specific named variable is
            required and is not actually a credential. (Currently
            unused; reserved for the cases where the user names a
            non-secret variable with a misleading shape.)

    Returns:
        A new dict with all matching keys removed. Iteration order is
        the same as the source mapping for predictability.
    """
    src = dict(os.environ) if source is None else dict(source)
    keep = set(extra_keep or ())
    scrubbed: dict = {}
    for k, v in src.items():
        if k in keep:
            scrubbed[k] = v
            continue
        if is_secret_env_name(k):
            continue
        scrubbed[k] = v
    return scrubbed

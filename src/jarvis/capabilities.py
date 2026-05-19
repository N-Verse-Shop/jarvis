"""Capability gates — env-var allow-list for risky tools.

Inspired by KAOS ``ENABLE_DYNAMIC_TOOLS=false`` / ``ENABLE_SERVER_OPS=false``
pattern in ``docs/MCP.md``. The default posture is conservative: tools
that can spend money, send messages, mutate filesystem outside the
workspace, or run shell commands are GATED behind an explicit
``JARVIS_ENABLE_*=true`` env var.

Why gates instead of "just remove the tool":

* The tool's CODE PATH still ships — useful for power users who flip
  the env var, but a default install can't accidentally fire them.
* The gate is one place to audit: read this file, you know which
  ops are dangerous AND how to turn them on.
* The gate decision is logged via :func:`audit` so a flipped env
  var is visible in the dashboard, not hidden in shell rc files.

Gates are intentionally NAMED for the action, not the tool. A future
``execute_shortcut`` that ran arbitrary AppleScript would inherit the
same ``shell_ops`` gate as ``run_shortcut`` (KAOS shortcuts run in a
sandboxed shortcut runner; ours don't — same risk class).

Gate semantics:

* ``JARVIS_ENABLE_<NAME>=true`` (or ``1``, ``yes``, ``on``) → open
* anything else, including unset → closed
* Closed gates filter the tool out of the LLM's function schema +
  refuse the call from the fast path. The user sees a polite
  "this capability is disabled" rather than a silent no-op.

Defined gates:

================ ===================================================
``shell_ops``    Run arbitrary AppleScript / Shortcuts that can
                 invoke shell, ssh, modify system state outside
                 the workspace. (run_shortcut, type_text, click_at,
                 key)
``messaging``    Send messages on the user's behalf via Messages.app.
                 (send_message)
``volume``       Change system volume / mute state.
                 (set_volume, set_mute)
``clipboard``    Replace clipboard contents.
                 (clipboard_set, clipboard_get)
``calendar``     Read user calendar events.
                 (query_calendar)
``contacts``     Read / use contacts data (reserved for future).
================ ===================================================

The first four ship CLOSED by default. ``calendar`` ships OPEN
because Jarvis already reads it for daily-recap. ``contacts``
is reserved (no current tool depends on it).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

log = logging.getLogger("jarvis.capabilities")


# Tool → gate name. A tool can only have ONE gate (the strictest).
# When adding a tool that could fit multiple, pick the riskiest.
_TOOL_GATES: dict[str, str] = {
    # mac_control ops (the ones we register in pipecat_loop)
    "focus_app": "open_apps",
    "open_url": "open_apps",
    "new_note": "personal_data",
    "new_reminder": "personal_data",
    "query_calendar": "calendar",
    "send_message": "messaging",
    "run_shortcut": "shell_ops",
    "list_shortcuts": "shell_ops",
    "system_info": "system_read",
    "set_volume": "volume",
    "set_mute": "volume",
    "clipboard_set": "clipboard",
    "clipboard_get": "clipboard",
    # input-synthesis (registered as part of mac_control but high-risk)
    "type_text": "shell_ops",
    "click_at": "shell_ops",
    "key": "shell_ops",
    # skill loaders are inert metadata reads — no gate
    "list_skills": "_open",
    "load_skill": "_open",
    "load_skill_reference": "_open",
}


# Default state of each gate. Jarvis is a personal power-user agent —
# the defaults are OPEN so the user gets full functionality out of
# the box. The gates exist as KILL SWITCHES: a security-conscious
# user (or a borrowed laptop) can flip ``JARVIS_DISABLE_SHELL_OPS=true``
# and lose only that capability. ``contacts`` is the one default-closed
# gate because no current tool depends on it AND opening it would
# leak PII without a user-visible request first.
_DEFAULT_GATES: dict[str, bool] = {
    "_open": True,             # always-on pseudo-gate for inert ops
    "open_apps": True,         # focus_app, open_url
    "personal_data": True,     # notes / reminders
    "calendar": True,          # read-only
    "system_read": True,       # system_info
    "volume": True,            # set_volume, set_mute
    "clipboard": True,         # clipboard_get / clipboard_set
    "messaging": True,         # send_message (user-driven)
    "shell_ops": True,         # run_shortcut + input synthesis
    "contacts": False,         # CLOSED — PII; no current tool depends
}


_TRUTHY = ("1", "true", "yes", "on")


def is_gate_open(gate_name: str) -> bool:
    """Return whether the given capability gate is open.

    Lookup priority:
      1. ``JARVIS_DISABLE_<NAME>=true`` overrides to CLOSED (kill switch).
      2. ``JARVIS_ENABLE_<NAME>=true`` overrides to OPEN.
      3. Fallback to :data:`_DEFAULT_GATES`.

    Unknown gate names default to CLOSED — we treat "I don't know what
    this is" as "don't run it".
    """
    if not gate_name:
        return False
    upper = gate_name.upper().strip()
    # 1. Disable override (kill switch) — beats everything else.
    if os.environ.get(f"JARVIS_DISABLE_{upper}", "").lower() in _TRUTHY:
        return False
    # 2. Enable override.
    if os.environ.get(f"JARVIS_ENABLE_{upper}", "").lower() in _TRUTHY:
        return True
    # 3. Built-in default.
    return _DEFAULT_GATES.get(gate_name, False)


def is_tool_allowed(tool_name: str) -> bool:
    """Whether a registered tool may be invoked right now."""
    gate = _TOOL_GATES.get(tool_name)
    if gate is None:
        # Unknown tools default to OPEN — third-party additions that
        # don't register a gate are still callable. They CAN add an
        # entry to _TOOL_GATES to opt into gating.
        return True
    return is_gate_open(gate)


def gate_for_tool(tool_name: str) -> str:
    """Return the gate name this tool falls under, or '' if ungated."""
    return _TOOL_GATES.get(tool_name, "")


def allowed_tools(tool_names: Iterable[str]) -> list[str]:
    """Filter a list of tool names by their gate state.

    Useful at LLM-schema registration time so a closed gate's tool
    never even appears in the function-call menu.
    """
    return [t for t in tool_names if is_tool_allowed(t)]


def gate_summary() -> dict:
    """Snapshot of every gate's state — for dashboard + audit."""
    seen: set[str] = set()
    out: dict[str, dict] = {}
    for gate in list(_DEFAULT_GATES.keys()) + list(set(_TOOL_GATES.values())):
        if gate in seen or gate.startswith("_"):
            continue
        seen.add(gate)
        out[gate] = {
            "default": _DEFAULT_GATES.get(gate, False),
            "active": is_gate_open(gate),
            "env_enable": f"JARVIS_ENABLE_{gate.upper()}",
            "env_disable": f"JARVIS_DISABLE_{gate.upper()}",
            "tools": sorted(
                t for t, g in _TOOL_GATES.items() if g == gate
            ),
        }
    return out


def log_gate_state() -> None:
    """One-shot log at startup so the user sees what's open/closed."""
    summary = gate_summary()
    open_ones = sorted(g for g, info in summary.items() if info["active"])
    closed = sorted(g for g, info in summary.items() if not info["active"])
    log.info(
        "Capability gates — OPEN: %s | CLOSED: %s",
        ", ".join(open_ones) or "(none)",
        ", ".join(closed) or "(none)",
    )

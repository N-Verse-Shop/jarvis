"""Dashboard — local-only HTTP control room for Jarvis.

Inspired by KAOS ``dashboard/`` + ``docs/DASHBOARD.md``. Trimmed to
Jarvis's voice-first scope: instead of multi-agent swarm visualisers,
we expose just the surfaces a personal voice agent needs (memory,
audit, skills, persona, live events, capability gates).

Listens on ``127.0.0.1:8789`` by default, bind address overridable
via the ``JARVIS_DASHBOARD_HOST`` env var (NEVER bind to ``0.0.0.0``
on a multi-user machine — the dashboard exposes the user's audit
DB, conversations, and the action surface). Auth is a bearer token
either:

* ``JARVIS_DASHBOARD_TOKEN`` env var, OR
* generated at first boot and written to
  ``~/Library/Application Support/jarvis/dashboard-token``
  (0o600). The user reads the file once + pastes into the SPA.

The server runs on its own asyncio loop in a background thread so
it doesn't block the voice pipeline.
"""

from __future__ import annotations

from .server import DashboardServer, get_dashboard_token, start_dashboard

__all__ = ["DashboardServer", "get_dashboard_token", "start_dashboard"]

"""Local-only dashboard HTTP server.

aiohttp-based — picked over FastAPI because pipecat already depends on
aiohttp; reusing avoids a fresh ~5MB dependency tree at daemon boot.

Routing surface (all JSON):

  GET  /api/health           — daemon liveness + uptime + state
  GET  /api/state            — current state.json contents
  GET  /api/persona          — parsed persona
  POST /api/persona/reload   — hot-reload persona.md
  GET  /api/skills           — list of skills (L1 catalog)
  GET  /api/skills/{name}    — full SKILL.md + references list
  GET  /api/skills/{name}/refs/{ref} — load one L3 reference
  POST /api/skills/reload    — rescan ~/.config/jarvis/skills
  GET  /api/audit            — query audit events (?limit, ?kind, ?tool, ?since)
  GET  /api/audit/stats      — aggregate counts
  GET  /api/capabilities     — gate state snapshot
  GET  /api/facts            — list / FTS search persistent facts
  POST /api/facts            — add new fact {key, value, source?, confidence?}
  DEL  /api/facts/{id}       — soft-delete (tombstone) a fact
  POST /api/facts/prune      — run decay-based prune now
  GET  /api/facts/stats      — counts + top keys
  GET  /api/events           — tail of events.jsonl (N most recent lines)
  WS   /ws/events            — live event stream (line-delimited JSON)

Threading model:

The server runs on its own asyncio loop in a daemon thread spun
up by :func:`start_dashboard`. The daemon's main thread (and the
Pipecat thread) remain unaffected. All blocking work (skill scans,
SQLite queries) happens in worker threads via ``loop.run_in_executor``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("jarvis.dashboard")


# ────────────────────── paths + tokens ─────────────────────────


def _dashboard_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/jarvis"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".local/share")
    return base / "jarvis"


_TOKEN_PATH = _dashboard_data_dir() / "dashboard-token"


def get_dashboard_token() -> str:
    """Return the active dashboard auth token.

    Priority:
      1. ``JARVIS_DASHBOARD_TOKEN`` env var (always wins)
      2. ``~/Library/Application Support/jarvis/dashboard-token``
      3. New random 32-byte hex written to (2), mode 0o600

    The token never leaves the local filesystem. Even ``ps`` users
    on the same machine can't read it unless they're the file owner.
    """
    env = os.environ.get("JARVIS_DASHBOARD_TOKEN", "").strip()
    if env:
        return env
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _TOKEN_PATH.exists():
        try:
            return _TOKEN_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    new = secrets.token_hex(32)
    try:
        _TOKEN_PATH.write_text(new, encoding="utf-8")
        try:
            _TOKEN_PATH.chmod(0o600)
        except OSError:
            pass
    except OSError as exc:
        log.warning("Could not persist dashboard token: %s", exc)
    return new


# ────────────────────── server ─────────────────────────


class DashboardServer:
    """aiohttp app wrapper exposing Jarvis's control surface.

    Construction is cheap — the actual HTTP socket binds when
    :meth:`start` is called.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        token: Optional[str] = None,
    ) -> None:
        self._host = host or os.environ.get("JARVIS_DASHBOARD_HOST", "127.0.0.1")
        try:
            self._port = int(
                port if port is not None
                else os.environ.get("JARVIS_DASHBOARD_PORT", "8789")
            )
        except (TypeError, ValueError):
            self._port = 8789
        self._token = (token or get_dashboard_token()).strip()
        self._started_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner = None
        self._site = None
        self._ws_clients: set = set()

    # ----- public lifecycle ---------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        ready = threading.Event()
        err: dict = {}

        def _runner_thread() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                loop.run_until_complete(self._serve_forever(ready))
            except Exception as exc:
                err["exc"] = exc
                ready.set()
                log.error("dashboard thread crashed: %s", exc, exc_info=True)

        self._thread = threading.Thread(
            target=_runner_thread, name="JarvisDashboard", daemon=True
        )
        self._thread.start()
        if not ready.wait(timeout=5.0):
            log.warning("dashboard slow to bind (>5s) — continuing anyway")
        if err:
            log.error("dashboard start error: %s", err["exc"])
        elif self._started_at:
            log.info(
                "Dashboard ready on http://%s:%d — token at %s",
                self._host,
                self._port,
                _TOKEN_PATH,
            )

    async def _serve_forever(self, ready: threading.Event) -> None:
        from aiohttp import web

        app = web.Application(middlewares=[self._auth_mw, self._cors_mw])
        self._register_routes(app)
        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        self._site = site
        self._started_at = time.time()
        ready.set()
        # Keep the loop alive
        while True:
            await asyncio.sleep(3600)

    # ----- middlewares --------------------------------------------
    @staticmethod
    def _bearer(req) -> Optional[str]:
        # Header first
        header = req.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        # Query fallback (handy for WebSocket connects from the browser)
        return req.query.get("token", "").strip() or None

    @staticmethod
    def _safe_compare(a: str, b: str) -> bool:
        if not a or not b or len(a) != len(b):
            return False
        out = 0
        for x, y in zip(a, b):
            out |= ord(x) ^ ord(y)
        return out == 0

    # Paths that must be reachable WITHOUT a token — the SPA shell
    # has to load before the user types one. Everything under /api
    # (except /api/health) still requires auth.
    _PUBLIC_PREFIXES = ("/dashboard", "/")

    @classmethod
    def _is_public_path(cls, path: str) -> bool:
        if path == "/api/health":
            return True
        if path == "/":
            return True
        if path.startswith("/dashboard"):
            return True
        return False

    async def _auth_mw(self, app, handler):
        async def _mw(req):
            if self._is_public_path(req.path):
                return await handler(req)
            provided = self._bearer(req)
            if not provided or not self._safe_compare(provided, self._token):
                from aiohttp import web as _w
                return _w.json_response(
                    {"error": "unauthorized"}, status=401
                )
            return await handler(req)
        return _mw

    async def _cors_mw(self, app, handler):
        async def _mw(req):
            from aiohttp import web as _w
            if req.method == "OPTIONS":
                resp = _w.Response(status=204)
            else:
                resp = await handler(req)
            # CORS for the Electron app + a local dev server.
            origin = req.headers.get("Origin", "")
            if origin.startswith(("http://127.0.0.1", "http://localhost", "file://")):
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,PUT,PATCH,OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Authorization,Content-Type"
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["Referrer-Policy"] = "no-referrer"
            return resp
        return _mw

    # ----- routes -------------------------------------------------
    def _register_routes(self, app) -> None:
        from aiohttp import web

        # ── Static SPA ─────────────────────────────────────
        # Mounted at /dashboard/ so the SPA loads its own
        # styles + JS from there. Root / redirects to the
        # SPA index.
        static_dir = Path(__file__).parent / "static"
        if static_dir.is_dir():
            app.router.add_get("/", self._h_root_redirect)
            app.router.add_get("/dashboard", self._h_dashboard_index)
            app.router.add_get("/dashboard/", self._h_dashboard_index)
            app.router.add_static(
                "/dashboard/", static_dir, name="dashboard-static"
            )

        app.router.add_get("/api/health", self._h_health)
        app.router.add_get("/api/state", self._h_state)
        app.router.add_get("/api/persona", self._h_persona)
        app.router.add_post("/api/persona/reload", self._h_persona_reload)
        app.router.add_get("/api/persona/raw", self._h_persona_raw)
        app.router.add_get("/api/skills", self._h_skills)
        app.router.add_get("/api/skills/{name}", self._h_skill_one)
        app.router.add_get(
            "/api/skills/{name}/refs/{ref}", self._h_skill_ref
        )
        app.router.add_post("/api/skills/reload", self._h_skills_reload)
        # R34-S23: skill + persona CRUD via dashboard
        app.router.add_post("/api/skills", self._h_skill_create)
        app.router.add_put("/api/skills/{name}", self._h_skill_update)
        app.router.add_delete("/api/skills/{name}", self._h_skill_delete)
        app.router.add_put("/api/persona", self._h_persona_update)
        app.router.add_get("/api/audit", self._h_audit)
        app.router.add_get("/api/audit/stats", self._h_audit_stats)
        app.router.add_get("/api/capabilities", self._h_capabilities)
        app.router.add_get("/api/facts", self._h_facts)
        app.router.add_post("/api/facts", self._h_facts_add)
        app.router.add_delete("/api/facts/{fact_id}", self._h_facts_delete)
        app.router.add_post("/api/facts/prune", self._h_facts_prune)
        app.router.add_get("/api/facts/stats", self._h_facts_stats)
        app.router.add_get("/api/events", self._h_events_tail)
        app.router.add_get("/ws/events", self._h_ws_events)
        # R34-S18: settings panel — read + update + restart-daemon.
        app.router.add_get("/api/settings", self._h_settings_get)
        app.router.add_patch("/api/settings", self._h_settings_patch)
        app.router.add_post("/api/restart", self._h_restart)
        # OPTIONS fallback for every route (CORS preflight)
        app.router.add_route("OPTIONS", "/{tail:.*}", self._h_options)

    @staticmethod
    async def _h_options(req):
        from aiohttp import web
        return web.Response(status=204)

    @staticmethod
    async def _h_root_redirect(req):
        from aiohttp import web
        raise web.HTTPFound("/dashboard/")

    @staticmethod
    async def _h_dashboard_index(req):
        from aiohttp import web
        index_path = Path(__file__).parent / "static" / "index.html"
        if not index_path.is_file():
            return web.Response(status=404, text="dashboard not built")
        return web.Response(
            body=index_path.read_bytes(),
            content_type="text/html",
            charset="utf-8",
        )

    # ----- handlers -----------------------------------------------
    async def _h_health(self, req):
        from aiohttp import web
        return web.json_response(
            {
                "ok": True,
                "started_at": self._started_at,
                "uptime_s": (
                    time.time() - self._started_at
                    if self._started_at else 0
                ),
                "pid": os.getpid(),
                "version": "R34",
                "brand": "Nexus Studio",
            }
        )

    async def _h_state(self, req):
        from aiohttp import web
        state_path = _dashboard_data_dir() / "state.json"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {"state": "UNKNOWN", "ts": 0.0, "level": 0.0}
        return web.json_response(data)

    async def _h_persona(self, req):
        from aiohttp import web
        from ..persona import get_persona_store
        p = await asyncio.to_thread(lambda: get_persona_store().get())
        return web.json_response(p.to_dict())

    async def _h_persona_reload(self, req):
        from aiohttp import web
        from ..persona import get_persona_store
        p = await asyncio.to_thread(lambda: get_persona_store().reload())
        return web.json_response(
            {"ok": True, "persona": p.to_dict()}
        )

    async def _h_persona_raw(self, req):
        """GET /api/persona/raw — raw persona.md text for the editor.

        Read-only; the parsed view at /api/persona keeps the chat-side
        consumers happy, but the dashboard editor needs full fidelity
        so it can preserve our `## Style` section + comments that
        the parser doesn't surface."""
        from aiohttp import web
        from ..persona import get_persona_store
        store = get_persona_store()
        path = store.path
        if not path.exists():
            return web.json_response({"ok": False, "error": "not_found"}, status=404)
        try:
            content = await asyncio.to_thread(
                lambda: path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        return web.json_response({"ok": True, "path": str(path), "content": content})

    async def _h_skills(self, req):
        from aiohttp import web
        from ..skills import get_skill_store
        skills = await asyncio.to_thread(
            lambda: get_skill_store().list_skills()
        )
        payload = [
            {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "tags": s.tags,
                "tools": s.tools,
                "risk": s.risk,
                "locale": s.locale,
                "references": sorted(s.references.keys()),
                "path": str(s.path),
            }
            for s in skills
        ]
        return web.json_response({"skills": payload})

    async def _h_skill_one(self, req):
        from aiohttp import web
        from ..skills import get_skill_store
        name = req.match_info["name"]
        s = await asyncio.to_thread(
            lambda: get_skill_store().get_skill(name)
        )
        if s is None:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(
            {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "tags": s.tags,
                "tools": s.tools,
                "risk": s.risk,
                "locale": s.locale,
                "references": sorted(s.references.keys()),
                "path": str(s.path),
                "content": s.content,
            }
        )

    async def _h_skill_ref(self, req):
        from aiohttp import web
        from ..skills import get_skill_store
        name = req.match_info["name"]
        ref = req.match_info["ref"]
        store = get_skill_store()
        text = await asyncio.to_thread(
            lambda: store.load_reference(name, ref)
        )
        if text is None:
            return web.json_response({"error": "not_found"}, status=404)
        return web.json_response(
            {"name": name, "reference": ref, "content": text}
        )

    async def _h_skills_reload(self, req):
        from aiohttp import web
        from ..skills import get_skill_store
        await asyncio.to_thread(get_skill_store().reload)
        return web.json_response({"ok": True})

    # ────────────────────────────────────────────────────────────
    # R34-S23: skill CRUD via dashboard
    # ────────────────────────────────────────────────────────────
    _SKILL_NAME_RE = __import__("re").compile(r"^[a-z][a-z0-9-]{1,38}[a-z0-9]$")
    _SKILL_BODY_MAX = 200_000      # 200 KB cap
    _PERSONA_BODY_MAX = 200_000    # same for persona.md

    @classmethod
    def _validate_skill_name(cls, name: str) -> str:
        """Lowercase kebab-case 3-40 chars. Rejects path traversal."""
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        n = name.strip().lower()
        if not cls._SKILL_NAME_RE.fullmatch(n):
            raise ValueError(
                "name must be lowercase kebab-case (a-z, 0-9, '-'), "
                "3..40 chars, no edge dashes"
            )
        # belt-and-braces: forbid anything that looks like a path segment
        if "/" in n or ".." in n or n.startswith("."):
            raise ValueError("invalid characters in name")
        return n

    def _skill_dir(self, name: str) -> "Path":
        """Resolve a skill directory under the user skills root, with a
        post-resolution sandbox check to defend against traversal even
        if the validator above is bypassed in some future refactor."""
        from ..skills import get_skill_store
        root = get_skill_store().skills_dir.resolve()
        target = (root / name).resolve()
        # Must be a direct child of `root`. .parents[0] catches that.
        if target.parent != root:
            raise ValueError("skill path escapes the skills directory")
        return target

    @staticmethod
    def _compose_skill_md(meta: dict, body: str) -> str:
        """Render frontmatter + body into a SKILL.md string."""
        lines = ["---"]
        # Ordered keys for stable, human-readable output.
        ordered = [
            "name", "description", "status", "version", "author",
            "upstream", "license", "tags", "tools", "risk", "locale",
        ]
        seen = set()
        for k in ordered:
            if k in meta and meta[k] not in (None, "", []):
                v = meta[k]
                if isinstance(v, list):
                    lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
                else:
                    lines.append(f"{k}: {v}")
                seen.add(k)
        for k, v in meta.items():
            if k in seen or v in (None, "", []):
                continue
            if isinstance(v, list):
                lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
            else:
                lines.append(f"{k}: {v}")
        lines.append("---")
        lines.append("")
        lines.append(body.rstrip() + "\n")
        return "\n".join(lines)

    async def _h_skill_create(self, req):
        """POST /api/skills — create a new skill folder + SKILL.md."""
        from aiohttp import web
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON body"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"ok": False, "error": "body must be an object"}, status=400
            )
        try:
            name = self._validate_skill_name(body.get("name", ""))
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        content = body.get("content", "")
        description = (body.get("description") or "").strip()
        if not description:
            return web.json_response(
                {"ok": False, "error": "description is required"}, status=400
            )
        if not isinstance(content, str) or len(content) > self._SKILL_BODY_MAX:
            return web.json_response(
                {"ok": False, "error": f"content must be ≤{self._SKILL_BODY_MAX} bytes"},
                status=400,
            )

        try:
            skill_dir = self._skill_dir(name)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        if skill_dir.exists():
            return web.json_response(
                {"ok": False, "error": f"skill '{name}' already exists"},
                status=409,
            )

        meta = {
            "name": name,
            "description": description,
            "status": (body.get("status") or "active").strip().lower(),
            "version": (body.get("version") or "1.0.0").strip(),
            "author": (body.get("author") or "").strip(),
            "tags": body.get("tags") or [],
            "tools": body.get("tools") or [],
            "risk": (body.get("risk") or "low").strip().lower(),
            "locale": (body.get("locale") or "ru").strip().lower(),
        }
        md = self._compose_skill_md(meta, content)

        def _write():
            skill_dir.mkdir(parents=True, exist_ok=False)
            (skill_dir / "SKILL.md").write_text(md, encoding="utf-8")
            try:
                (skill_dir / "SKILL.md").chmod(0o644)
            except OSError:
                pass

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            log.warning("skill create failed: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        # Hot-reload the store so the new skill appears immediately.
        from ..skills import get_skill_store
        await asyncio.to_thread(get_skill_store().reload)
        return web.json_response({"ok": True, "name": name})

    async def _h_skill_update(self, req):
        """PUT /api/skills/{name} — overwrite SKILL.md."""
        from aiohttp import web
        try:
            name = self._validate_skill_name(req.match_info["name"])
            skill_dir = self._skill_dir(name)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        if not skill_dir.is_dir():
            return web.json_response({"ok": False, "error": "not_found"}, status=404)
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON body"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"ok": False, "error": "body must be an object"}, status=400
            )
        content = body.get("content", "")
        description = (body.get("description") or "").strip()
        if not description:
            return web.json_response(
                {"ok": False, "error": "description is required"}, status=400
            )
        if not isinstance(content, str) or len(content) > self._SKILL_BODY_MAX:
            return web.json_response(
                {"ok": False, "error": f"content must be ≤{self._SKILL_BODY_MAX} bytes"},
                status=400,
            )

        # Merge with existing metadata so callers don't have to send everything.
        from ..skills import get_skill_store
        existing = get_skill_store().get_skill(name)
        if existing:
            meta = {
                "name": name,
                "description": description,
                "status": (body.get("status") or existing.status or "active").strip().lower(),
                "version": (body.get("version") or existing.version or "1.0.0").strip(),
                "author": (body.get("author") or existing.author or "").strip(),
                "tags": body.get("tags") if body.get("tags") is not None else existing.tags,
                "tools": body.get("tools") if body.get("tools") is not None else existing.tools,
                "risk": (body.get("risk") or existing.risk or "low").strip().lower(),
                "locale": (body.get("locale") or existing.locale or "ru").strip().lower(),
            }
        else:
            meta = {
                "name": name,
                "description": description,
                "status": (body.get("status") or "active").strip().lower(),
                "version": (body.get("version") or "1.0.0").strip(),
                "author": (body.get("author") or "").strip(),
                "tags": body.get("tags") or [],
                "tools": body.get("tools") or [],
                "risk": (body.get("risk") or "low").strip().lower(),
                "locale": (body.get("locale") or "ru").strip().lower(),
            }
        md = self._compose_skill_md(meta, content)

        target = skill_dir / "SKILL.md"
        tmp = target.with_suffix(".md.tmp")

        def _write():
            tmp.write_text(md, encoding="utf-8")
            try:
                tmp.chmod(0o644)
            except OSError:
                pass
            os.replace(tmp, target)

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            log.warning("skill update failed: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        await asyncio.to_thread(get_skill_store().reload)
        return web.json_response({"ok": True, "name": name})

    async def _h_skill_delete(self, req):
        """DELETE /api/skills/{name} — remove the skill directory.

        Two-phase: actual move to a `__trash__/` subdir under the skills
        root, NOT rm -rf. Lets the user recover by hand if they regret
        it, and keeps us inside the user's own data dir at all times."""
        from aiohttp import web
        try:
            name = self._validate_skill_name(req.match_info["name"])
            skill_dir = self._skill_dir(name)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        if not skill_dir.is_dir():
            return web.json_response({"ok": False, "error": "not_found"}, status=404)

        from ..skills import get_skill_store
        root = get_skill_store().skills_dir
        trash = root / "__trash__"

        def _move():
            trash.mkdir(parents=True, exist_ok=True)
            import datetime as _dt
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = trash / f"{name}.{stamp}"
            os.rename(skill_dir, dest)
            return str(dest)

        try:
            trashed = await asyncio.to_thread(_move)
        except Exception as exc:
            log.warning("skill delete failed: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        await asyncio.to_thread(get_skill_store().reload)
        return web.json_response({"ok": True, "name": name, "moved_to": trashed})

    async def _h_persona_update(self, req):
        """PUT /api/persona — overwrite persona.md raw text."""
        from aiohttp import web
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON body"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"ok": False, "error": "body must be an object"}, status=400
            )
        content = body.get("content", "")
        if not isinstance(content, str):
            return web.json_response(
                {"ok": False, "error": "content must be a string"}, status=400
            )
        if not content.strip():
            return web.json_response(
                {"ok": False, "error": "content cannot be empty"}, status=400
            )
        if len(content) > self._PERSONA_BODY_MAX:
            return web.json_response(
                {"ok": False, "error": f"content must be ≤{self._PERSONA_BODY_MAX} bytes"},
                status=400,
            )
        # Defence in depth: persona.md MUST start with YAML frontmatter
        # so the loader doesn't crash on malformed input. The very
        # first chars need to be `---\n` or `---\r\n`.
        if not content.lstrip().startswith("---"):
            return web.json_response(
                {"ok": False, "error": "persona.md must start with YAML frontmatter (---)"},
                status=400,
            )

        from ..persona import get_persona_store
        store = get_persona_store()
        target = store.path
        tmp = target.with_suffix(".md.tmp")

        def _write():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(content, encoding="utf-8")
            try:
                tmp.chmod(0o600)
            except OSError:
                pass
            os.replace(tmp, target)

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            log.warning("persona update failed: %s", exc)
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

        await asyncio.to_thread(store.reload)
        return web.json_response({"ok": True, "path": str(target)})

    async def _h_audit(self, req):
        from aiohttp import web
        from ..audit import get_audit_store
        q = req.query
        try:
            limit = int(q.get("limit", "100"))
        except ValueError:
            limit = 100
        kind = q.get("kind") or None
        tool = q.get("tool") or None
        status = q.get("status") or None
        try:
            since = float(q.get("since")) if q.get("since") else None
        except (TypeError, ValueError):
            since = None
        store = get_audit_store()
        events = await asyncio.to_thread(
            lambda: store.query(
                kind=kind,
                tool=tool,
                status=status,
                since_ts=since,
                limit=min(max(1, limit), 1000),
            )
        )
        return web.json_response(
            {"events": [e.to_dict() for e in events]}
        )

    async def _h_audit_stats(self, req):
        from aiohttp import web
        from ..audit import get_audit_store
        store = get_audit_store()
        try:
            since = (
                float(req.query.get("since"))
                if req.query.get("since") else None
            )
        except (TypeError, ValueError):
            since = None
        stats = await asyncio.to_thread(
            lambda: store.stats(since_ts=since)
        )
        return web.json_response(stats)

    async def _h_capabilities(self, req):
        from aiohttp import web
        from ..capabilities import gate_summary
        return web.json_response({"gates": gate_summary()})

    async def _h_facts(self, req):
        """List or search persistent facts.

        Query params:
          q          — FTS5 search query (optional)
          key_prefix — filter by key prefix (e.g. "user.")
          limit      — max rows (default 50, cap 500)
        """
        from aiohttp import web
        from ..memory.facts import get_facts_store
        q = req.query.get("q", "").strip()
        prefix = req.query.get("key_prefix", "").strip() or None
        try:
            limit = int(req.query.get("limit", "50"))
        except ValueError:
            limit = 50
        limit = min(max(1, limit), 500)
        store = get_facts_store()
        if q:
            facts = await asyncio.to_thread(
                lambda: store.search(q, limit=limit, key_prefix=prefix)
            )
        else:
            facts = await asyncio.to_thread(
                lambda: store.by_key(prefix or "", limit=limit)
            )
        import time as _time
        now = _time.time()
        return web.json_response({
            "facts": [
                {
                    "id": f.id,
                    "key": f.key,
                    "value": f.value,
                    "source": f.source,
                    "confidence": f.confidence,
                    "ts_utc": f.ts_utc,
                    "last_used": f.last_used,
                    "hits": f.hits,
                    "score": f.score(now=now),
                }
                for f in facts
            ]
        })

    async def _h_facts_add(self, req):
        """Add a new fact. Body: {key, value, source?, confidence?}."""
        from aiohttp import web
        from ..memory.facts import get_facts_store
        try:
            body = await req.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        key = (body.get("key") or "").strip()
        value = (body.get("value") or "").strip()
        if not key or not value:
            return web.json_response(
                {"error": "key and value required"}, status=400
            )
        source = body.get("source") or "dashboard"
        try:
            confidence = float(body.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        store = get_facts_store()
        try:
            fid = await asyncio.to_thread(
                lambda: store.add(
                    key, value, source=source, confidence=confidence,
                )
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response({"id": fid, "ok": True})

    async def _h_facts_delete(self, req):
        from aiohttp import web
        from ..memory.facts import get_facts_store
        try:
            fid = int(req.match_info["fact_id"])
        except (KeyError, ValueError):
            return web.json_response({"error": "bad id"}, status=400)
        store = get_facts_store()
        removed = await asyncio.to_thread(lambda: store.delete(fid))
        return web.json_response({"removed": bool(removed)})

    async def _h_facts_prune(self, req):
        from aiohttp import web
        from ..memory.facts import get_facts_store
        store = get_facts_store()
        n = await asyncio.to_thread(lambda: store.prune())
        return web.json_response({"pruned": n})

    async def _h_facts_stats(self, req):
        from aiohttp import web
        from ..memory.facts import get_facts_store
        store = get_facts_store()
        stats = await asyncio.to_thread(lambda: store.stats())
        return web.json_response(stats)

    async def _h_events_tail(self, req):
        from aiohttp import web
        try:
            n = int(req.query.get("n", "100"))
        except ValueError:
            n = 100
        n = min(max(1, n), 2000)
        events_path = _dashboard_data_dir() / "events.jsonl"
        lines: list[dict] = []
        if events_path.exists():
            try:
                # Tail in a thread — file can be ~8MB.
                lines = await asyncio.to_thread(self._tail_jsonl, events_path, n)
            except Exception as exc:
                log.warning("events tail failed: %s", exc)
        return web.json_response({"events": lines})

    @staticmethod
    def _tail_jsonl(path: Path, n: int) -> list[dict]:
        """Return last ``n`` JSONL-parsed records from ``path``.

        Simple O(file size) read because events.jsonl caps at 8 MB
        before rotation — chunk-based reverse-read isn't worth the
        complexity here.
        """
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        out: list[dict] = []
        for line in raw.splitlines()[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # ── Settings panel (R34-S18) ─────────────────────────────────
    # Whitelist of config.json keys the dashboard is allowed to read.
    # Everything ELSE (DB paths, MCP creds, internal toggles) stays
    # opaque even to a local-network attacker who has the token.
    _SETTINGS_READ_KEYS = (
        "piper_voice",
        "vad_aggressiveness",
        "endpoint_silence_ms",
        "max_utterance_ms",
        "voice_min_energy",
        "wake_word",
        "wake_fuzzy_ratio",
        "hot_window_seconds",
        "active_language",
        "whisper_language",
        "whisper_model",
        "voice_device",
        "ollama_chat_model",
        "ollama_chat_temperature",
        "ollama_chat_num_predict",
        "voice_engine",
        "tts_enabled",
        "silero_vad_threshold",
    )
    # Subset of the above that the dashboard is allowed to WRITE.
    # We deliberately keep this tighter than the read set — changing
    # paths / engine choices mid-session breaks things in ways that
    # need a daemon restart anyway.
    _SETTINGS_WRITE_KEYS = (
        "piper_voice",
        "vad_aggressiveness",
        "endpoint_silence_ms",
        "max_utterance_ms",
        "voice_min_energy",
        "wake_word",
        "wake_fuzzy_ratio",
        "hot_window_seconds",
        "active_language",
        "whisper_language",
        "whisper_model",
        "voice_device",
        "ollama_chat_model",
        "silero_vad_threshold",
    )
    # Per-key validators — return the coerced value or raise ValueError.
    @staticmethod
    def _validate_setting(key: str, value):  # noqa: C901 — intentional schema
        import re as _re
        s = value
        if key == "piper_voice":
            s = str(s).strip()
            if not _re.fullmatch(r"[a-zA-Z]{2,3}_[A-Z]{2}-[a-zA-Z0-9_]+-[a-z_]+", s):
                raise ValueError(
                    "piper_voice must look like 'ru_RU-ruslan-medium'"
                )
            return s
        if key == "vad_aggressiveness":
            v = int(s)
            if not 0 <= v <= 3:
                raise ValueError("vad_aggressiveness must be 0..3")
            return v
        if key in ("endpoint_silence_ms", "max_utterance_ms"):
            v = int(s)
            if not 100 <= v <= 30000:
                raise ValueError(f"{key} must be 100..30000 ms")
            return v
        if key in ("voice_min_energy", "wake_fuzzy_ratio", "silero_vad_threshold"):
            v = float(s)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{key} must be 0.0..1.0")
            return v
        if key == "hot_window_seconds":
            v = float(s)
            if not 0.5 <= v <= 600.0:
                raise ValueError("hot_window_seconds must be 0.5..600")
            return v
        if key in ("wake_word", "voice_device", "whisper_model"):
            s = str(s).strip()
            if not s or len(s) > 200:
                raise ValueError(f"{key} must be 1..200 chars")
            return s
        if key in ("active_language", "whisper_language"):
            s = str(s).strip().lower()
            if s not in ("ru", "uk", "en", "de"):
                raise ValueError(f"{key} must be ru | uk | en | de")
            return s
        if key == "ollama_chat_model":
            s = str(s).strip()
            # Ollama tags look like 'qwen3:8b' or 'library/llama3.2:1b'.
            # Slashes are legal (registry prefix) but '..' is a clear
            # path-traversal attempt and must never reach disk-adjacent
            # code paths. Same for leading slash.
            if not _re.fullmatch(r"[a-zA-Z0-9._:\-/]+", s):
                raise ValueError("ollama_chat_model has invalid chars")
            if ".." in s or s.startswith("/"):
                raise ValueError("ollama_chat_model has path-traversal pattern")
            if len(s) > 120:
                raise ValueError("ollama_chat_model too long")
            return s
        raise ValueError(f"{key} is not a writable setting")

    async def _h_settings_get(self, req):
        from aiohttp import web
        try:
            from ..config import load_config
            raw = await asyncio.to_thread(load_config) or {}
        except Exception as exc:
            log.warning("settings load failed: %s", exc)
            raw = {}
        out = {k: raw.get(k) for k in self._SETTINGS_READ_KEYS}
        # Piper voice catalog — let the UI populate a dropdown without
        # the user typing a magic string.
        out["_piper_voice_catalog"] = [
            {"id": "uk_UA-mykyta-high", "label": "Микита (UA, чоловічий, high)"},
            {"id": "uk_UA-oleksa-high", "label": "Олекса (UA, чоловічий, high)"},
            {"id": "uk_UA-ukrainian_tts-medium", "label": "Lada (UA, жіночий, medium)"},
            {"id": "uk_UA-lada-x_low", "label": "Lada (UA, жіночий, x_low)"},
            {"id": "uk_UA-tetiana-high", "label": "Тетяна (UA, жіночий, high)"},
            {"id": "ru_RU-ruslan-medium", "label": "Руслан (RU, чоловічий, medium)"},
            {"id": "ru_RU-dmitri-medium", "label": "Дмитро (RU, чоловічий, medium)"},
            {"id": "ru_RU-denis-medium", "label": "Денис (RU, чоловічий, medium)"},
            {"id": "ru_RU-irina-medium", "label": "Ірина (RU, жіночий, medium)"},
        ]
        out["_writable_keys"] = list(self._SETTINGS_WRITE_KEYS)
        return web.json_response(out)

    async def _h_settings_patch(self, req):
        from aiohttp import web
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON body"}, status=400
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"ok": False, "error": "body must be a JSON object"},
                status=400,
            )
        # Validate every key first; reject the WHOLE batch on any
        # failure so the file is never left in a half-written state.
        validated: dict = {}
        for k, v in body.items():
            if k not in self._SETTINGS_WRITE_KEYS:
                return web.json_response(
                    {"ok": False, "error": f"{k!r} is not writable"},
                    status=400,
                )
            try:
                validated[k] = self._validate_setting(k, v)
            except (ValueError, TypeError) as exc:
                return web.json_response(
                    {"ok": False, "error": f"{k}: {exc}"},
                    status=400,
                )

        # Merge into config.json atomically.
        def _write_config():
            from ..config import default_config_path
            import os as _os
            path = default_config_path()
            try:
                raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except json.JSONDecodeError:
                raw = {}
            raw.update(validated)
            tmp = path.with_suffix(path.suffix + f".tmp-{_os.getpid()}")
            tmp.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _os.replace(tmp, path)
            try:
                _os.chmod(path, 0o600)
            except OSError:
                pass

        try:
            await asyncio.to_thread(_write_config)
        except Exception as exc:
            log.warning("settings patch failed: %s", exc)
            return web.json_response(
                {"ok": False, "error": f"write failed: {exc}"}, status=500
            )

        return web.json_response(
            {"ok": True, "updated": validated, "restart_required": True}
        )

    async def _h_restart(self, req):
        """Trigger a daemon restart via launchctl (macOS launchd).

        R34-S20: was racing the SIGTERM against the HTTP response
        flush — browsers saw `TypeError: Failed to fetch` because the
        socket closed mid-response. Two fixes layered together:

        1. Explicit Connection: close + Content-Length so the browser
           treats one full read as the complete response (no chunked
           transfer that can be truncated).
        2. Bumped the pre-kill sleep from 0.3s → 1.2s. aiohttp writes
           into kernel buffers then yields; the kernel still needs a
           round-trip to actually ship them to localhost before
           launchctl unloads us.

        We also delegate the kill to a subprocess that survives our
        own SIGTERM — `nohup launchctl ... &` — so the load step still
        runs even after the unload kills us.
        """
        from aiohttp import web
        import subprocess as _sp

        plist = Path.home() / "Library/LaunchAgents/com.jarvis.assistant.plist"
        plist_path = str(plist) if plist.exists() else None

        async def _do_restart():
            # Long enough for the response above to fully flush, even
            # under load. 1.2s is well under the user-visible feedback
            # window (we show a toast immediately).
            await asyncio.sleep(1.2)
            if plist_path:
                # Spawn a detached shell that survives our own death:
                # unload → small wait → load. We never await it; we're
                # about to be killed by the first unload anyway.
                _sp.Popen(
                    [
                        "/bin/sh", "-c",
                        f'launchctl unload "{plist_path}" >/dev/null 2>&1; '
                        f'sleep 2; '
                        f'launchctl load "{plist_path}" >/dev/null 2>&1',
                    ],
                    start_new_session=True,
                    stdin=_sp.DEVNULL,
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                )
            else:
                # No plist → fall back to graceful exit; user's
                # supervisor/shell can re-spawn.
                import os as _os, signal as _sig
                _os.kill(_os.getpid(), _sig.SIGTERM)

        asyncio.create_task(_do_restart())
        # Force HTTP/1.1 Connection: close so the browser treats the
        # response as complete on socket close. Pin Content-Length to
        # avoid any chunked-transfer ambiguity.
        body = json.dumps({"ok": True, "restarting": True}).encode("utf-8")
        return web.Response(
            body=body,
            content_type="application/json",
            headers={
                "Connection": "close",
                "Cache-Control": "no-store",
                "Content-Length": str(len(body)),
            },
        )

    async def _h_ws_events(self, req):
        from aiohttp import web, WSMsgType
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self._ws_clients.add(ws)
        events_path = _dashboard_data_dir() / "events.jsonl"
        try:
            # Stream tail of file + then poll for appends.
            last_size = events_path.stat().st_size if events_path.exists() else 0
            # Send last 50 events first so the dashboard has context.
            seed = self._tail_jsonl(events_path, 50) if events_path.exists() else []
            for ev in seed:
                await ws.send_json(ev)
            # Poll for new content
            while not ws.closed:
                await asyncio.sleep(0.25)
                if not events_path.exists():
                    continue
                try:
                    size = events_path.stat().st_size
                except OSError:
                    continue
                if size < last_size:
                    # Rotation happened → reset; consumer will get
                    # next batch as it lands.
                    last_size = 0
                    continue
                if size > last_size:
                    new_bytes = b""
                    try:
                        with open(events_path, "rb") as fh:
                            fh.seek(last_size)
                            new_bytes = fh.read(size - last_size)
                    except OSError:
                        new_bytes = b""
                    last_size = size
                    for raw in new_bytes.decode("utf-8", errors="replace").splitlines():
                        if not raw.strip():
                            continue
                        try:
                            ev = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if ws.closed:
                            break
                        try:
                            await ws.send_json(ev)
                        except ConnectionResetError:
                            break
        finally:
            self._ws_clients.discard(ws)
            try:
                await ws.close()
            except Exception:
                pass
        return ws


# ────────────────────── module-level start helper ─────────────────────────


_SERVER: Optional[DashboardServer] = None
_SERVER_LOCK = threading.Lock()


def start_dashboard(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
) -> DashboardServer:
    """Start the dashboard on its own thread (idempotent).

    Returns the server instance so callers can stash it for tests.
    """
    global _SERVER
    if _SERVER is not None:
        return _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = DashboardServer(host=host, port=port, token=token)
            _SERVER.start()
    return _SERVER

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
        # R34-S58.3 B1.2: serialise chat requests. Ollama can only
        # process one chat at a time on a given model, and pipelining
        # multiple chats over the same daemon process used to wedge
        # both — the second request waited on Ollama while still
        # holding the aiohttp task queue, blocking unrelated endpoints
        # (state, transcript, audit). One semaphore = one in-flight
        # chat; subsequent requests get an immediate 429 instead of
        # piling up. Lazily initialised inside ``_h_chat`` because the
        # asyncio loop doesn't exist at __init__ time.
        self._chat_semaphore: Optional[asyncio.Semaphore] = None

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

        # R34-S58.3 B1.3: framework-level cap on request body size.
        # 256 KB is well above what any legitimate dashboard endpoint
        # accepts (chat caps at 8 KB, most CRUD endpoints under 4 KB)
        # but small enough that an attacker can't queue a multi-GB
        # body in RAM before our per-endpoint size checks fire. aiohttp
        # rejects with 413 BEFORE invoking ``json.loads`` / ``.text()``.
        app = web.Application(
            middlewares=[self._auth_mw, self._cors_mw],
            client_max_size=262144,
        )
        self._register_routes(app)
        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner
        # R34-S24: JARVIS_DASHBOARD_HOST can be a comma-separated list
        # of bind targets so the same dashboard answers both the local
        # 127.0.0.1 client AND the Tailscale-side reverse proxy without
        # going through 0.0.0.0 (which would also accept LAN traffic).
        # Example: "127.0.0.1,100.90.14.99"
        hosts = [h.strip() for h in str(self._host).split(",") if h.strip()]
        if not hosts:
            hosts = ["127.0.0.1"]
        sites = []
        for h in hosts:
            site = web.TCPSite(runner, h, self._port)
            try:
                await site.start()
                sites.append(site)
                log.info("Dashboard bound on %s:%d", h, self._port)
            except OSError as exc:
                log.warning(
                    "Dashboard could not bind %s:%d — %s. Continuing on remaining hosts.",
                    h, self._port, exc,
                )
        if not sites:
            raise RuntimeError(
                f"Dashboard failed to bind any of {hosts!r} on port {self._port}"
            )
        # _site kept for back-compat (status endpoint may read it).
        self._site = sites[0]
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
        # R34-S55.1 Phase 8b (P2 auth crypto hygiene): replaced the
        # hand-rolled byte-compare with stdlib ``hmac.compare_digest``.
        # The previous implementation:
        #   1) Returned early on ``len(a) != len(b)`` — leaking token
        #      length via response timing (microseconds, but observable
        #      on a LAN with enough samples).
        #   2) Iterated via ``zip(a, b)`` + ``ord()`` per character —
        #      pure Python, slower than the C path in ``hmac``, and
        #      the loop body wasn't reviewed by anyone outside this
        #      file. ``hmac.compare_digest`` is the canonical primitive,
        #      audited, constant-time independent of input contents.
        # ``compare_digest`` handles None/empty by accepting str|bytes;
        # we coerce ``None`` → ``""`` so empty inputs return False
        # via the length mismatch rather than raising TypeError.
        import hmac
        return hmac.compare_digest(a or "", b or "")

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
        # R34-S24: public deploy surfaces — robots.txt and favicon
        # need to respond without auth so crawlers (which we block)
        # and browser-chrome can probe without 401 spam.
        if path in ("/robots.txt", "/favicon.ico"):
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

    # R34-S24: extra origins for the public deployment behind
    # Cloudflare Tunnel + Hetzner Caddy. Comma-separated env var so
    # ops can add more without a code change.
    @staticmethod
    def _public_origin_allowlist() -> tuple[str, ...]:
        default = "https://jarvis.nexus-studio-innovation.com"
        raw = os.environ.get("JARVIS_DASHBOARD_ORIGINS", default)
        return tuple(o.strip() for o in raw.split(",") if o.strip())

    async def _cors_mw(self, app, handler):
        async def _mw(req):
            from aiohttp import web as _w
            if req.method == "OPTIONS":
                resp = _w.Response(status=204)
            else:
                resp = await handler(req)
            # CORS allowlist:
            #   • local dev (127.0.0.1 / localhost / file://)
            #   • the public CF-tunnelled origin(s) from env
            #
            # R34-S55.1 Phase 8b (P2 DNS-rebinding hardening): the prior
            # ``startswith("http://127.0.0.1")`` matched
            # ``http://127.0.0.1.evil.com`` — a DNS-rebinding attack
            # vector where an attacker-controlled DNS responder returns
            # the loopback address for a subdomain. The browser sends
            # ``Origin: http://127.0.0.1.evil.com`` (the original
            # request URL), which our prefix check used to accept.
            # Tightening: only accept the bare host OR the host
            # followed by a port colon — never a host-suffix.
            origin = req.headers.get("Origin", "")
            ok = (
                origin == "http://127.0.0.1"
                or origin == "http://localhost"
                or origin.startswith("http://127.0.0.1:")
                or origin.startswith("http://localhost:")
                or origin.startswith("file://")
            )
            if not ok and origin:
                ok = origin in self._public_origin_allowlist()
            if ok:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,PUT,PATCH,OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = (
                "Authorization,Content-Type,"
                # Cloudflare Access headers (so the browser keeps them
                # on retried preflights).
                "Cf-Access-Jwt-Assertion,Cf-Access-Authenticated-User-Email"
            )
            # Hardening headers — same for local AND public routes.
            resp.headers["X-Content-Type-Options"] = "nosniff"
            resp.headers["Referrer-Policy"] = "no-referrer"
            # noindex everywhere — R34-S24 user req: «без налаштування
            # сео, щоб пошукові системи його не шукали».
            resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
            # Frame deny so the panel can't be embedded in another site.
            resp.headers["X-Frame-Options"] = "DENY"
            # Tight CSP for the SPA — only loads its own static assets,
            # talks to its own origin for XHR/WS, plus Cloudflare's
            # Access JS if it's served alongside (we don't load any
            # external JS but allow self-only just in case).
            if req.path.startswith("/dashboard") or req.path == "/":
                resp.headers["Content-Security-Policy"] = (
                    "default-src 'self'; "
                    "img-src 'self' data:; "
                    "style-src 'self' 'unsafe-inline'; "
                    "script-src 'self'; "
                    "connect-src 'self' ws: wss:; "
                    "font-src 'self' data:; "
                    "frame-ancestors 'none'; "
                    "base-uri 'self'; "
                    "form-action 'self'"
                )
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
        # R34-S24: robots.txt blocking all crawlers. User: «без
        # налаштування сео, щоб пошукові системи його не шукали».
        app.router.add_get("/robots.txt", self._h_robots_txt)
        # Don't expose a favicon.ico crawlable trail.
        app.router.add_get("/favicon.ico", self._h_favicon_204)

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
        # R34-S29: text chat fallback for when voice path is wedged.
        # POST {"text": "...", "tts": false} → Ollama → reply text.
        # POST /api/interrupt → signal voice thread to stop current TTS.
        app.router.add_post("/api/chat", self._h_chat)
        app.router.add_post("/api/interrupt", self._h_interrupt)
        # R34-S30: test endpoint — push a TranscriptionFrame into the
        # voice pipeline without going through the microphone.
        app.router.add_post("/api/voice/inject", self._h_voice_inject)
        app.router.add_get("/api/brain", self._h_brain)
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
            # R35-S33: never cache the HTML shell so the browser always
            # revalidates and picks up new ?v= asset versions. Without
            # this the browser cached index.html → stale app.js → users
            # never saw new dashboard features/fixes until a hard reload.
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    @staticmethod
    async def _h_robots_txt(req):
        """Block every crawler — this UI is private."""
        from aiohttp import web
        body = "User-agent: *\nDisallow: /\n"
        return web.Response(text=body, content_type="text/plain")

    @staticmethod
    async def _h_favicon_204(req):
        # 204 No Content keeps the browser quiet without leaking
        # a discoverable static asset URL.
        from aiohttp import web
        return web.Response(status=204)

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
        # R35-S31: full model/privacy/behaviour control surface.
        "ollama_base_url",
        "ollama_chat_keep_alive",
        "ollama_chat_num_ctx",
        "tts_rate",
        "nvidia_nim_chat_model",
        "mistral_chat_model",
        "privacy_llm",
        "claude_spawn_model",
        "claude_effort",
        "complex_free_model",
        "voice_confirm_actions",
        "privacy_keywords",
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
        # R35-S31: model/privacy/behaviour control surface.
        "ollama_base_url",
        "ollama_chat_keep_alive",
        "ollama_chat_num_ctx",
        "ollama_chat_num_predict",
        "tts_rate",
        "nvidia_nim_chat_model",
        "mistral_chat_model",
        "privacy_llm",
        "claude_spawn_model",
        "claude_effort",
        "complex_free_model",
        "voice_confirm_actions",
        "privacy_keywords",
    )
    # R35-S31: schema drives the dashboard's auto-rendered controls.
    # Each entry: key, label (RU/UA UI), type (text|int|enum|bool),
    # group, and (for enum) options / (for int) min,max.
    _SETTINGS_SCHEMA = (
        {"key": "ollama_chat_num_predict", "label": "Ollama: макс токенів відповіді", "type": "int", "group": "Моделі", "min": 10, "max": 1000},
        {"key": "ollama_chat_num_ctx", "label": "Ollama: контекст num_ctx", "type": "int", "group": "Моделі", "min": 256, "max": 8192},
        {"key": "ollama_base_url", "label": "Ollama URL (приватний/офлайн бекенд)", "type": "text", "group": "Моделі"},
        {"key": "ollama_chat_keep_alive", "label": "Ollama keep_alive (24h, 5m…)", "type": "text", "group": "Моделі"},
        {"key": "nvidia_nim_chat_model", "label": "NVIDIA NIM модель", "type": "text", "group": "Моделі"},
        {"key": "mistral_chat_model", "label": "Mistral модель (EU, приватне)", "type": "text", "group": "Приватність"},
        {"key": "privacy_llm", "label": "Бекенд приватних запитів", "type": "enum", "group": "Приватність", "options": ["auto", "mistral", "ollama"]},
        {"key": "privacy_keywords", "label": "Приватні ключові слова (по одному на рядок)", "type": "list", "group": "Приватність"},
        {"key": "claude_spawn_model", "label": "Claude модель (складні задачі)", "type": "enum", "group": "Claude / Геній", "options": ["opus", "sonnet", "haiku"]},
        {"key": "claude_effort", "label": "Claude reasoning effort", "type": "enum", "group": "Claude / Геній", "options": ["low", "medium", "high", "xhigh", "max"]},
        {"key": "complex_free_model", "label": "Безкоштовна модель fallback", "type": "text", "group": "Claude / Геній"},
        {"key": "voice_confirm_actions", "label": "Питати підтвердження дій", "type": "bool", "group": "Поведінка"},
        {"key": "tts_rate", "label": "Швидкість голосу (TTS rate)", "type": "int", "group": "Голос", "min": 80, "max": 400},
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
        # ── R35-S31: model / privacy / behaviour settings ─────────────
        if key in ("nvidia_nim_chat_model", "mistral_chat_model", "complex_free_model"):
            s = str(s).strip()
            if not _re.fullmatch(r"[a-zA-Z0-9._:\-/]+", s):
                raise ValueError(f"{key} has invalid chars")
            if ".." in s or s.startswith("/") or len(s) > 120:
                raise ValueError(f"{key} invalid (traversal / too long)")
            return s
        if key == "claude_spawn_model":
            s = str(s).strip()
            if not _re.fullmatch(r"[a-zA-Z0-9._\-]+", s) or len(s) > 60:
                raise ValueError("claude_spawn_model invalid (e.g. opus|sonnet|haiku)")
            return s
        if key == "claude_effort":
            s = str(s).strip().lower()
            if s not in ("low", "medium", "high", "xhigh", "max"):
                raise ValueError("claude_effort must be low|medium|high|xhigh|max")
            return s
        if key == "privacy_llm":
            s = str(s).strip().lower()
            if s not in ("auto", "mistral", "ollama"):
                raise ValueError("privacy_llm must be auto|mistral|ollama")
            return s
        if key == "ollama_base_url":
            s = str(s).strip()
            if not _re.fullmatch(r"https?://[\w.\-]+(:\d+)?(/[\w./\-]*)?", s) or len(s) > 200:
                raise ValueError("ollama_base_url must be a http(s) URL")
            return s
        if key == "ollama_chat_keep_alive":
            s = str(s).strip()
            if not _re.fullmatch(r"-?\d+[smhd]?", s) or len(s) > 12:
                raise ValueError("ollama_chat_keep_alive like '24h', '5m', '0', '-1'")
            return s
        if key == "ollama_chat_num_ctx":
            v = int(s)
            if not 256 <= v <= 8192:
                raise ValueError("ollama_chat_num_ctx must be 256..8192")
            return v
        if key == "ollama_chat_num_predict":
            v = int(s)
            if not 10 <= v <= 1000:
                raise ValueError("ollama_chat_num_predict must be 10..1000")
            return v
        if key == "tts_rate":
            v = int(s)
            if not 80 <= v <= 400:
                raise ValueError("tts_rate must be 80..400")
            return v
        if key == "voice_confirm_actions":
            if isinstance(s, bool):
                return s
            sv = str(s).strip().lower()
            if sv in ("true", "1", "yes", "on"):
                return True
            if sv in ("false", "0", "no", "off"):
                return False
            raise ValueError("voice_confirm_actions must be boolean")
        if key == "privacy_keywords":
            # Accept a list, or newline/comma-separated text from the UI.
            if isinstance(s, (list, tuple)):
                items = [str(x).strip() for x in s]
            elif isinstance(s, str):
                items = [x.strip() for x in s.replace(",", "\n").splitlines()]
            else:
                raise ValueError("privacy_keywords must be a list or text")
            items = [x for x in items if x]
            if len(items) > 300:
                raise ValueError("too many privacy_keywords (max 300)")
            for x in items:
                if len(x) > 80:
                    raise ValueError("a privacy keyword is too long (max 80 chars)")
            # Dedup case-insensitively, preserve order.
            seen, deduped = set(), []
            for x in items:
                if x.lower() not in seen:
                    seen.add(x.lower())
                    deduped.append(x)
            return deduped
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
        # R34-S52 H: gender labels were UA "чоловічий"/"жіночий" —
        # for a RU-speaking user the dropdown UI should also be RU.
        # Proper-noun voice names (Микита, Lada, Руслан, etc.) stay.
        out["_piper_voice_catalog"] = [
            {"id": "uk_UA-mykyta-high", "label": "Микита (UA, мужской, high)"},
            {"id": "uk_UA-oleksa-high", "label": "Олекса (UA, мужской, high)"},
            {"id": "uk_UA-ukrainian_tts-medium", "label": "Lada (UA, женский, medium)"},
            {"id": "uk_UA-lada-x_low", "label": "Lada (UA, женский, x_low)"},
            {"id": "uk_UA-tetiana-high", "label": "Тетяна (UA, женский, high)"},
            {"id": "ru_RU-ruslan-medium", "label": "Руслан (RU, мужской, medium)"},
            {"id": "ru_RU-dmitri-medium", "label": "Дмитрий (RU, мужской, medium)"},
            {"id": "ru_RU-denis-medium", "label": "Денис (RU, мужской, medium)"},
            {"id": "ru_RU-irina-medium", "label": "Ирина (RU, женский, medium)"},
        ]
        out["_writable_keys"] = list(self._SETTINGS_WRITE_KEYS)
        # R35-S31: schema for the auto-rendered "advanced" controls.
        out["_schema"] = [dict(s) for s in self._SETTINGS_SCHEMA]
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
                # R34-S52 C-5: was previously
                #   _sp.Popen(["/bin/sh", "-c", f'launchctl unload "{plist_path}" ...'])
                # which interpolates ``plist_path`` into a shell string.
                # The path was gated by ``plist.exists()`` above so the
                # known instance was safe, but the construction is the
                # kind of pattern a future refactor accidentally
                # neuters. Switched to a detached Python orchestrator
                # invoked via argv — ``plist`` arrives as ``sys.argv[1]``
                # and is passed to launchctl as an argv element, never
                # interpreted by a shell. Eliminates the whole class.
                import sys as _sys
                _orchestrator = (
                    "import subprocess, time, sys;"
                    "plist = sys.argv[1];"
                    "subprocess.run("
                    "    ['launchctl', 'unload', plist],"
                    "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
                    ");"
                    "time.sleep(2);"
                    "subprocess.run("
                    "    ['launchctl', 'load', plist],"
                    "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
                    ")"
                )
                _sp.Popen(
                    [_sys.executable, "-c", _orchestrator, plist_path],
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

        # R34-S55.1 Phase 8c (P3 hygiene): retain the task reference.
        # ``asyncio.create_task`` returns a Task whose only reference is
        # held by the event loop's _scheduled queue. PEP-3156 / GH-bpo-37658
        # documents that the loop can drop the task before it runs if no
        # one keeps a strong reference. In practice the restart happens
        # in tens of ms so the risk is low, but stashing on a class-level
        # set ensures durability.
        self._restart_task = asyncio.create_task(_do_restart())
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

    # ── R34-S29: text-chat fallback ───────────────────────────────────
    # When the voice path is wedged (mic perm, wake mishearing, Continuity
    # rerouting…) the user still needs a way to drive Jarvis. This endpoint
    # takes plain text, runs it through Ollama directly with the persona
    # system prompt, and returns the reply. No Pipecat involvement — the
    # text-chat path stays alive even when the voice loop is on the floor.
    async def _h_chat(self, req):
        from aiohttp import web
        # R34-S58.3 B1.2: enforce one-in-flight chat per server. Lazy
        # init because asyncio.Semaphore needs a running loop, which
        # only exists inside the dashboard thread (not __init__).
        if self._chat_semaphore is None:
            self._chat_semaphore = asyncio.Semaphore(1)
        if self._chat_semaphore.locked():
            # Another chat is already running — refuse fast rather
            # than pile up. 429 with Retry-After lets the dashboard
            # UI back off without surfacing a confusing 5xx.
            return web.json_response(
                {"error": "chat busy, try again shortly"},
                status=429,
                headers={"Retry-After": "2"},
            )
        async with self._chat_semaphore:
            return await self._h_chat_inner(req)

    async def _h_chat_inner(self, req):
        from aiohttp import web
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON"}, status=400
            )
        text = str(body.get("text", "")).strip()
        if not text:
            return web.json_response(
                {"ok": False, "error": "empty text"}, status=400
            )
        if len(text) > 8000:
            return web.json_response(
                {"ok": False, "error": "text too long (max 8000 chars)"},
                status=400,
            )
        # R35-S3 P2-11: honour the voice-pipeline standby flag for the
        # dashboard text path too. Without this, a user who pressed the
        # HUD Close button (= voice silenced) could still drive the
        # daemon by typing — bypassing the verbal-resume gate. Match
        # the gate behaviour in pipecat_loop.py:4064/2496 so the
        # dashboard and voice are consistent. Allow opt-out via a body
        # field for tests / power users who need to debug while paused.
        try:
            from ..listening.pipecat_loop import _is_standby
            if _is_standby() and not bool(body.get("force_when_standby")):
                return web.json_response(
                    {
                        "ok": False,
                        "error": (
                            "Daemon is in STANDBY (voice closed). Say the "
                            "wake word to resume, or pass force_when_standby=true."
                        ),
                    },
                    status=503,
                )
        except Exception:
            # If standby state can't be queried (cold start, import
            # failure) we let the request through rather than block.
            pass
        # R35-S3 F2: route through run_reply_engine so the dashboard
        # textbox gets the SAME tool affordance as the voice path. The
        # legacy implementation POSTed straight to Ollama /api/chat
        # without any `tools=` parameter — meaning n8nAutomation, web
        # search, mac control, memory, etc. were ALL invisible to the
        # LLM. Live evidence (Phase A audit): user typed "какие у меня
        # сейчас автоматизации" and got a confident fabrication about
        # "CI/CD на GitLab, мониторинг через Prometheus и Grafana"
        # because the n8n tool was never offered. This block falls back
        # to the raw Ollama call if the engine can't be set up (e.g.
        # daemon globals not initialised yet during cold start).
        try:
            from ..config import load_settings
            from ..memory.db import Database
            from .. import daemon as _daemon_mod
            from ..reply.engine import run_reply_engine
            _cfg = load_settings()
            _db = Database(_cfg.db_path, _cfg.sqlite_vss_path)
            # R35-S3 FIX (smoke-test follow-up): ``python -m jarvis.daemon``
            # creates BOTH __main__ and jarvis.daemon module objects.
            # main() only sets the global in __main__'s copy, so direct
            # getattr on the imported module returns None. The helper
            # walks both locations and returns the live DialogueMemory.
            try:
                _dm = _daemon_mod.get_dialogue_memory()
            except Exception:
                _dm = getattr(_daemon_mod, "_global_dialogue_memory", None)
            if _dm is not None:
                # Run the engine on a worker thread — it's a synchronous
                # function that holds the GIL for LLM streaming and would
                # block the dashboard event loop if awaited directly.
                loop = asyncio.get_running_loop()
                def _run_engine_sync() -> str:
                    return run_reply_engine(
                        _db, _cfg, None,  # tts=None — text-only response
                        text, _dm,
                        language=None,
                        abort_event=None,
                    ) or ""
                reply = await loop.run_in_executor(None, _run_engine_sync)
                # Best-effort: emit a stream event for audit/HUD parity.
                try:
                    from ..ipc import get_stream
                    get_stream().emit(
                        "dashboard_chat_reply",
                        text=text[:200],
                        reply=reply[:200],
                        path="engine",
                    )
                except Exception:
                    pass
                return web.json_response({
                    "ok": True,
                    "reply": reply,
                    "model": getattr(_cfg, "ollama_chat_model", "qwen3:8b"),
                    "path": "engine",
                })
        except Exception as exc:
            log.warning(
                "chat: engine path failed, falling back to raw Ollama: %r", exc
            )
        # ── fallback: raw Ollama (no tools, no memory) ─────────────────
        # Kept as a safety net so the textbox at least returns SOMETHING
        # if the engine path errors (cold-start race, db lock, etc.).
        cfg = None
        base_url = "http://127.0.0.1:11434"
        # qwen2.5:3b — chosen because it has no reasoning mode (the
        # OpenAI /v1 shim dumps qwen3's output into `reasoning` and
        # leaves `content` empty, so Pipecat hears nothing).
        model = "qwen2.5:3b"
        num_predict = 220
        # R34-S54.1 Phase 7c: default 4096 → 8192 to mirror
        # llm.py:call_llm_direct (S53.1 I-1) and call_llm_streaming
        # (S52 Phase 5). Aligned defaults keep Ollama's KV-cache
        # prefix reuse intact across voice / dashboard / direct paths.
        # ``cfg.ollama_chat_num_ctx`` overrides this in practice; this
        # literal is the unconfigured-cfg fallback path.
        num_ctx = 8192
        try:
            from ..config import load_settings
            cfg = load_settings()
            base_url = str(getattr(cfg, "ollama_base_url", base_url))
            model = str(getattr(cfg, "ollama_chat_model", model))
            num_predict = int(getattr(cfg, "ollama_chat_num_predict", num_predict))
            num_ctx = int(getattr(cfg, "ollama_chat_num_ctx", num_ctx))
        except Exception as exc:
            log.warning("chat: settings load failed: %r", exc)
        # Build the same system prompt the voice pipeline assembles.
        try:
            from ..system_prompt import build_system_prompt
            system_prompt = build_system_prompt("Jarvis")
        except Exception as exc:
            log.warning("chat: system prompt build failed: %r", exc)
            # R34-S53.1 (Phase 6a): UA fallback prompt → RU. The
            # daemon's persona is RU-only (R34-S48/S51); if
            # ``build_system_prompt`` failed and we landed here, the
            # fallback used to leak UA tokens into qwen3:8b's context
            # — which then echoed UA back through TTS. Now matches the
            # persona's primary tone.
            system_prompt = (
                "Ты — Jarvis, личный голосовой ассистент. "
                "Отвечай кратко, по сути, на русском."
            )
        # R34-S30: switched from /v1/chat/completions (OpenAI shim) to
        # native /api/chat. The OpenAI shim IGNORES the `think: false`
        # flag — confirmed empirically: a qwen3:8b call over /v1 still
        # routes every token into `message.reasoning` regardless of
        # whether `think:false` is at top level or under `options`.
        # The native /api/chat endpoint honours `think:false` properly
        # and returns `{"message": {"content": "..."}}`.
        import aiohttp as _ah
        url = base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        try:
            timeout = _ah.ClientTimeout(total=180)
            async with _ah.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        return web.json_response(
                            {
                                "ok": False,
                                "error": f"ollama {resp.status}: {err_text[:300]}",
                            },
                            status=502,
                        )
                    data = await resp.json()
        except asyncio.TimeoutError:
            return web.json_response(
                {"ok": False, "error": "Ollama timed out (180s)"}, status=504
            )
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": f"Ollama unreachable: {exc!r}"},
                status=502,
            )
        # R34-S30: native /api/chat returns {"message": {"content": "..."}}
        # — NOT the OpenAI-shim `choices[0].message.content` shape.
        reply = ""
        try:
            reply = str(data.get("message", {}).get("content", "") or "")
        except Exception:
            pass
        # Best-effort: emit a stream event so audit picks up the
        # text-chat exchange alongside voice turns.
        #
        # R34-S55.1 Phase 8b (P2 privacy): scrub user_text + reply_text
        # through ``scrub_secrets`` BEFORE the emit. The event lands in
        # events.jsonl AND fans out to every connected WebSocket client
        # via ``_h_ws_events`` — including connections still alive from
        # a previously-authenticated browser tab. The 500-char window
        # is wide enough to carry pasted API keys / OAuth tokens /
        # passwords / one-time codes. Voice turns route through
        # ``_safe_user_text`` (which calls the same scrubber), so the
        # dashboard chat path should match.
        try:
            from ..ipc import get_stream
            from ..utils.redact import scrub_secrets as _scrub
            try:
                safe_user = _scrub(text or "")[:500]
            except Exception:
                safe_user = (text or "")[:500]
            try:
                safe_reply = _scrub(reply or "")[:500]
            except Exception:
                safe_reply = (reply or "")[:500]
            get_stream().emit(
                "chat_text",
                source="dashboard",
                user_text=safe_user,
                reply_text=safe_reply,
                model=model,
            )
        except Exception:
            pass
        return web.json_response(
            {"ok": True, "reply": reply, "model": model}
        )

    # ── R34-S29: interrupt current Jarvis turn (stop TTS/LLM) ────────
    # The voice thread polls ``~/Library/Application Support/jarvis/
    # interrupt.flag`` once per pipeline tick. When the flag exists the
    # voice thread pushes an InterruptionFrame into its PipelineTask and
    # deletes the flag. Dashboard just needs to touch the file.
    async def _h_interrupt(self, req):
        from aiohttp import web
        try:
            base = _dashboard_data_dir()
            flag = base / "interrupt.flag"
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text(str(time.time()), encoding="utf-8")
            # Also emit an event so the dashboard sees the action.
            try:
                from ..ipc import get_stream
                get_stream().emit(
                    "interrupt_request",
                    source="dashboard",
                )
            except Exception:
                pass
            return web.json_response({"ok": True, "flag": str(flag)})
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": repr(exc)}, status=500
            )

    # ── R34-S30: voice inject (test endpoint) ─────────────────────
    # Writes a flag file that the voice pipeline's interrupt poller
    # picks up and converts to a TranscriptionFrame. Used by smoke
    # tests and the dashboard's voice-test button so we can verify
    # the full LLM+TTS path without needing the user to speak.
    async def _h_voice_inject(self, req):
        from aiohttp import web
        try:
            body = await req.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "invalid JSON"}, status=400
            )
        text = str(body.get("text", "")).strip()
        if not text:
            return web.json_response(
                {"ok": False, "error": "empty text"}, status=400
            )
        if len(text) > 2000:
            return web.json_response(
                {"ok": False, "error": "text too long"}, status=400
            )
        try:
            base = _dashboard_data_dir()
            base.mkdir(parents=True, exist_ok=True)
            flag = base / "inject_transcription.flag"
            flag.write_text(text, encoding="utf-8")
            try:
                from ..ipc import get_stream
                get_stream().emit(
                    "voice_inject",
                    source="dashboard",
                    text=text[:200],
                )
            except Exception:
                pass
            return web.json_response({"ok": True, "flag": str(flag)})
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": repr(exc)}, status=500
            )

    # ── R34-S27: brain-view data (Three.js dashboard tab) ────────────
    # Reads cheaply from existing sources — no new computation in the
    # voice loop. The endpoint compiles "regions" from skill categories
    # + memory facts + recent audit events, each "neuron" maps to one
    # underlying record. Firing rate = events in the last 5 minutes.
    # Cached at 5s TTL to keep cost negligible during fast UI polling.
    async def _h_brain(self, req):
        from aiohttp import web
        # Tiny TTL cache so a 1 Hz UI poll doesn't hammer the audit DB.
        cache = getattr(self, "_brain_cache", None)
        now = time.time()
        if cache and now - cache.get("ts", 0) < 5.0:
            return web.json_response(cache["data"])
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self._brain_snapshot
            )
        except Exception as exc:
            log.warning("brain snapshot failed: %r", exc)
            data = {"ok": False, "error": repr(exc), "regions": []}
        self._brain_cache = {"ts": now, "data": data}
        return web.json_response(data)

    def _brain_snapshot(self) -> dict:
        """Build the brain-view payload. Runs in a thread to keep the
        event loop free during DB reads."""
        out = {"ok": True, "regions": [], "totals": {}}
        # ── Region 1: Skills, grouped by ``risk`` tier ────────────────
        try:
            from ..skills import get_skill_store
            skills = get_skill_store().list_skills() or []
        except Exception:
            skills = []
        skill_groups: dict = {}
        for sk in skills:
            tier = str(getattr(sk, "risk", None) or "info").lower()
            skill_groups.setdefault(tier, []).append(sk)
        for tier, group in sorted(skill_groups.items()):
            neurons = []
            for sk in group:
                name = getattr(sk, "name", None) or "?"
                neurons.append({
                    "id": f"skill:{name}",
                    "label": str(name),
                    "weight": 1.0,
                })
            out["regions"].append({
                "id": f"skills:{tier}",
                "label": f"Skills · {tier}",
                "kind": "skills",
                "tier": tier,
                "neurons": neurons,
                "firing_rate": 0.0,
            })
        # ── Region 2: Memory facts, top-N by recency ─────────────────
        try:
            from ..memory.facts import get_facts_store
            store = get_facts_store()
            facts = store.by_key("", limit=80) if store else []
        except Exception:
            facts = []
        if facts:
            neurons = []
            now_ts = time.time()
            for f in facts:
                key = getattr(f, "key", "?")
                try:
                    score = float(f.score(now=now_ts))
                except Exception:
                    score = 1.0
                neurons.append({
                    "id": f"fact:{key}",
                    "label": str(key)[:48],
                    "weight": score,
                })
            out["regions"].append({
                "id": "memory:facts",
                "label": "Memory · Facts",
                "kind": "memory",
                "tier": "info",
                "neurons": neurons,
                "firing_rate": 0.0,
            })
        # ── Region 3: Recent audit events (firing rate) ──────────────
        try:
            from ..audit import get_audit_store
            since_ts = time.time() - 300  # last 5 minutes
            recent = get_audit_store().query(
                since_ts=since_ts, limit=500
            ) or []
        except Exception:
            recent = []
        by_tool: dict = {}
        for ev in recent:
            tool = getattr(ev, "tool", None)
            if not tool:
                continue
            by_tool[tool] = by_tool.get(tool, 0) + 1
        if by_tool:
            neurons = [
                {
                    "id": f"tool:{name}",
                    "label": name,
                    "weight": float(count),
                }
                for name, count in sorted(
                    by_tool.items(), key=lambda kv: -kv[1]
                )[:20]
            ]
            total = sum(by_tool.values())
            out["regions"].append({
                "id": "audit:tools",
                "label": "Live · Tools (5min)",
                "kind": "audit",
                "tier": "live",
                "neurons": neurons,
                # Firing rate normalized 0..1 over the last 5 min.
                "firing_rate": min(1.0, total / 60.0),
            })
        out["totals"] = {
            "neurons": sum(len(r["neurons"]) for r in out["regions"]),
            "regions": len(out["regions"]),
        }
        return out

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

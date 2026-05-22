"""n8n REST API client.

Talks to a self-hosted n8n instance (default: the user's
``n8n.nexus-studio-innovation.com``). Authentication via
``X-N8N-API-KEY`` header — the key is read from the
``JARVIS_N8N_API_KEY`` env var (NEVER hardcoded, NEVER logged).

R35-S1: voice tool layer can:
  * list_workflows / get_workflow
  * create_workflow / update_workflow / delete_workflow
  * activate / deactivate
  * list_executions / get_execution
  * trigger webhook by path (separate from REST — runs the workflow)

Error model:
  * N8NError — generic API error (network, 4xx/5xx, JSON parse)
  * N8NAuthError — subclass thrown specifically on 401/403 so callers
    can surface a "configure API key" prompt to the user

Security:
  * All request/response paths run the API key through a scrub layer
    before any debug_log() — exception messages get sanitised too.
  * The constructor refuses to send requests when the key looks like
    a placeholder ("YOUR_KEY", "...", empty) — fails-fast.
"""

from __future__ import annotations

import os
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests

from ..debug import debug_log


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://n8n.nexus-studio-innovation.com"
"""Default n8n base URL — overridable via env JARVIS_N8N_BASE_URL or
``Settings.n8n_base_url``."""

API_PREFIX = "/api/v1"
"""All REST endpoints sit under this prefix in n8n versions ≥0.218."""

# Reasonable defaults — overridable per-call.
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 30.0

_PLACEHOLDER_PATTERNS = (
    "your_key", "your-key", "your_api_key", "your-api-key",
    "replace_me", "placeholder", "xxx", "...",
)


# ──────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────

class N8NError(Exception):
    """Generic n8n API failure. ``status`` is the HTTP code if available."""

    def __init__(self, message: str, status: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body


class N8NAuthError(N8NError):
    """401/403 from the n8n API.

    Callers should surface a user-facing "configure API key in
    ~/.config/jarvis/.env" message rather than retry."""


class N8NNotConfiguredError(N8NError):
    """The API key env var is missing or looks like a placeholder.

    Distinct from ``N8NAuthError`` (server rejected a real key) so
    callers can show a different message ("set the key" vs "key
    invalid").
    """


# ──────────────────────────────────────────────────────────────────────
# Models — light dataclasses so callers don't have to remember the JSON
# shape. Only the most commonly accessed fields are typed; the original
# dict is preserved on ``.raw`` for advanced use.
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Workflow:
    """Slim view of an n8n workflow."""
    id: str
    name: str
    active: bool
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "Workflow":
        tags = []
        for t in (d.get("tags") or []):
            if isinstance(t, dict):
                tags.append(t.get("name") or str(t.get("id", "")))
            elif isinstance(t, str):
                tags.append(t)
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            active=bool(d.get("active", False)),
            tags=tags,
            created_at=d.get("createdAt"),
            updated_at=d.get("updatedAt"),
            raw=d,
        )


@dataclass
class Execution:
    """Slim view of a workflow execution (run)."""
    id: str
    workflow_id: str
    status: str  # "success" | "error" | "running" | "waiting" | "canceled"
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, d: Dict[str, Any]) -> "Execution":
        # n8n status field varies: "finished + success/error" boolean OR
        # "status" string in newer versions. Normalise to a single token.
        status = d.get("status")
        if status is None:
            if d.get("finished") is True:
                status = "error" if d.get("stoppedAt") and (d.get("data") or {}).get("resultData", {}).get("error") else "success"
            else:
                status = "running"
        return cls(
            id=str(d.get("id", "")),
            workflow_id=str(d.get("workflowId", "")),
            status=str(status),
            started_at=d.get("startedAt"),
            stopped_at=d.get("stoppedAt"),
            raw=d,
        )


# ──────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────

def _looks_like_placeholder(value: str) -> bool:
    """True when the value is empty or matches a known placeholder pattern."""
    if not value:
        return True
    low = value.strip().lower()
    if len(low) < 8:  # real n8n keys are long JWT-style strings
        return True
    return any(p in low for p in _PLACEHOLDER_PATTERNS)


def _scrub(text: Any, api_key: Optional[str]) -> str:
    """Strip the API key out of any logged string.

    Defensive: callers can pass exceptions/dicts/anything — we
    str()-ify first.
    """
    s = str(text)
    if api_key and len(api_key) > 4:
        s = s.replace(api_key, "***")
        # Also redact lookalikes (URL-encoded, partial prefixes 12+ chars).
        if len(api_key) >= 12:
            s = s.replace(api_key[:12], "***")
    return s


class N8NClient:
    """Synchronous n8n REST API client.

    Thread-safe — uses a single ``requests.Session`` guarded by a
    re-entrant lock. Connections pool naturally between calls.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
    ):
        # Base URL precedence: explicit arg → env → default.
        self.base_url = (
            base_url
            or os.environ.get("JARVIS_N8N_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")

        # Key precedence: explicit arg → env. NEVER persisted on disk
        # by the client itself; the user maintains ~/.config/jarvis/.env.
        self.api_key = (api_key or os.environ.get("JARVIS_N8N_API_KEY") or "").strip()

        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        self._session = requests.Session()
        self._lock = threading.RLock()

    # ─── lifecycle ───────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """True when we have a non-placeholder API key.

        Voice tools should check this BEFORE the user triggers an
        action so we can produce a friendly "set up your key first"
        prompt instead of a 401.
        """
        return not _looks_like_placeholder(self.api_key)

    def close(self) -> None:
        with self._lock:
            try:
                self._session.close()
            except Exception:
                pass

    # ─── low-level request helper ────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        expect_json: bool = True,
        extra_timeout: Optional[float] = None,
    ) -> Any:
        """Perform a single REST call. Internal — public methods wrap this.

        Raises ``N8NNotConfiguredError`` when no usable key, ``N8NAuthError``
        on 401/403, generic ``N8NError`` for everything else.
        """
        if not self.is_configured():
            raise N8NNotConfiguredError(
                "JARVIS_N8N_API_KEY is not set or looks like a placeholder. "
                "Set it in ~/.config/jarvis/.env and restart the daemon."
            )

        # Normalise path — accept "/workflows" or "workflows".
        if not path.startswith("/"):
            path = "/" + path
        full_url = f"{self.base_url}{API_PREFIX}{path}"

        headers = {
            "X-N8N-API-KEY": self.api_key,
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        timeout = (
            self.connect_timeout,
            (extra_timeout if extra_timeout is not None else self.read_timeout),
        )

        # Single retry on connection-level failures (stale Tailscale NAT, etc.)
        # — matches the LLM-side retry policy. Read-timeouts are NOT retried
        # because they reflect the caller's budget.
        last_exc: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                with self._lock:
                    resp = self._session.request(
                        method=method,
                        url=full_url,
                        params=params,
                        json=json_body,
                        headers=headers,
                        timeout=timeout,
                    )
                break  # success path
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ConnectTimeout,
            ) as exc:
                last_exc = exc
                if attempt == 1:
                    time.sleep(0.5)  # short backoff
                    continue
                # Sanitise BEFORE raising so the message doesn't leak the key.
                raise N8NError(
                    f"n8n network error after retries: {_scrub(exc, self.api_key)}"
                ) from None
            except requests.exceptions.ReadTimeout as exc:
                raise N8NError(
                    f"n8n read timeout (>{timeout[1]}s): {_scrub(exc, self.api_key)}"
                ) from None
            except requests.exceptions.RequestException as exc:
                raise N8NError(
                    f"n8n request failed: {_scrub(exc, self.api_key)}"
                ) from None

        # Auth errors → distinct exception so callers can react.
        if resp.status_code in (401, 403):
            raise N8NAuthError(
                f"n8n auth rejected ({resp.status_code}). "
                "Check JARVIS_N8N_API_KEY is valid and not revoked.",
                status=resp.status_code,
            )

        if resp.status_code == 404:
            raise N8NError(f"n8n resource not found: {method} {path}", status=404)

        if not (200 <= resp.status_code < 300):
            # 4xx/5xx — scrub the body before logging it.
            body_excerpt = ""
            try:
                body_excerpt = _scrub(resp.text[:500], self.api_key)
            except Exception:
                pass
            raise N8NError(
                f"n8n API {resp.status_code} on {method} {path}: {body_excerpt}",
                status=resp.status_code,
                body=body_excerpt,
            )

        if not expect_json:
            return resp.text

        try:
            return resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise N8NError(
                f"n8n returned non-JSON body on {method} {path}: "
                f"{_scrub(exc, self.api_key)}"
            ) from None

    # ─── workflows ──────────────────────────────────────────────────

    def list_workflows(
        self,
        active: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Workflow]:
        """List workflows. Optionally filter by active/tags."""
        params: Dict[str, Any] = {"limit": int(limit)}
        if active is not None:
            params["active"] = "true" if active else "false"
        if tags:
            params["tags"] = ",".join(tags)
        data = self._request("GET", "/workflows", params=params)
        items = data.get("data") if isinstance(data, dict) else data
        return [Workflow.from_api(w) for w in (items or [])]

    def get_workflow(self, workflow_id: str) -> Workflow:
        """Fetch a single workflow by ID."""
        data = self._request("GET", f"/workflows/{workflow_id}")
        return Workflow.from_api(data)

    def create_workflow(self, definition: Dict[str, Any]) -> Workflow:
        """Create a workflow from a JSON definition.

        ``definition`` is the full workflow object (name, nodes,
        connections, settings, ...). Template helpers can produce
        this shape from a stored .json file + parameter substitution.
        """
        if not isinstance(definition, dict) or not definition.get("name"):
            raise N8NError("workflow definition must be a dict with a 'name' field")
        data = self._request("POST", "/workflows", json_body=definition)
        return Workflow.from_api(data)

    def update_workflow(self, workflow_id: str, definition: Dict[str, Any]) -> Workflow:
        """Replace a workflow's definition. ``definition`` must be a complete object."""
        data = self._request("PUT", f"/workflows/{workflow_id}", json_body=definition)
        return Workflow.from_api(data)

    def delete_workflow(self, workflow_id: str) -> bool:
        """Hard-delete a workflow. Irreversible — voice flows MUST gate
        this with a user confirmation."""
        self._request("DELETE", f"/workflows/{workflow_id}", expect_json=False)
        return True

    def activate_workflow(self, workflow_id: str) -> Workflow:
        data = self._request("POST", f"/workflows/{workflow_id}/activate")
        return Workflow.from_api(data)

    def deactivate_workflow(self, workflow_id: str) -> Workflow:
        data = self._request("POST", f"/workflows/{workflow_id}/deactivate")
        return Workflow.from_api(data)

    # ─── executions ─────────────────────────────────────────────────

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Execution]:
        params: Dict[str, Any] = {"limit": int(limit)}
        if workflow_id:
            params["workflowId"] = workflow_id
        if status:
            params["status"] = status
        data = self._request("GET", "/executions", params=params)
        items = data.get("data") if isinstance(data, dict) else data
        return [Execution.from_api(e) for e in (items or [])]

    def get_execution(self, execution_id: str) -> Execution:
        data = self._request("GET", f"/executions/{execution_id}")
        return Execution.from_api(data)

    # ─── webhook trigger ────────────────────────────────────────────
    # Note: webhook URLs sit OUTSIDE the /api/v1 prefix. They use a
    # path the workflow author chose. This helper still uses the
    # configured base_url + auth header — n8n ignores X-N8N-API-KEY on
    # webhook routes (webhooks have their own auth — Basic / Header /
    # JWT — configured per-trigger). We send the header anyway because
    # it doesn't hurt and may be needed if the webhook is auth=header.

    def trigger_webhook(
        self,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
        method: str = "POST",
    ) -> Dict[str, Any]:
        """Hit a workflow's webhook trigger.

        ``path`` is the webhook path configured in the workflow's
        Webhook node (e.g. ``"jarvis/morning-briefing"``). Set
        ``test_mode=True`` to use the n8n test URL prefix (only fires
        when the editor is open + the workflow is in "Listen for
        test event" mode).
        """
        if not path:
            raise N8NError("webhook path is required")
        prefix = "/webhook-test/" if test_mode else "/webhook/"
        path = path.lstrip("/")
        full = f"{self.base_url}{prefix}{path}"

        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["X-N8N-API-KEY"] = self.api_key
        if payload is not None:
            headers["Content-Type"] = "application/json"

        try:
            with self._lock:
                resp = self._session.request(
                    method=method,
                    url=full,
                    json=payload,
                    headers=headers,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
        except requests.exceptions.RequestException as exc:
            raise N8NError(
                f"webhook trigger failed: {_scrub(exc, self.api_key)}"
            ) from None

        if not (200 <= resp.status_code < 300):
            raise N8NError(
                f"webhook returned HTTP {resp.status_code}: "
                f"{_scrub(resp.text[:300], self.api_key)}",
                status=resp.status_code,
            )

        try:
            return resp.json() if resp.text else {}
        except (ValueError, json.JSONDecodeError):
            return {"raw": resp.text}

    # ─── convenience ────────────────────────────────────────────────

    def find_workflow_by_name(self, name: str) -> Optional[Workflow]:
        """Case-insensitive name match. Returns ``None`` when not found.

        Voice flows often refer to a workflow by spoken name; this
        helper resolves that to an ID without a separate list+filter
        dance at the caller.
        """
        target = (name or "").strip().lower()
        if not target:
            return None
        for wf in self.list_workflows():
            if wf.name.strip().lower() == target:
                return wf
        # Fuzzy fallback — substring match.
        candidates = [wf for wf in self.list_workflows() if target in wf.name.lower()]
        if len(candidates) == 1:
            return candidates[0]
        return None


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton accessor
# ──────────────────────────────────────────────────────────────────────

_GLOBAL_CLIENT_LOCK = threading.Lock()
_GLOBAL_CLIENT: Optional[N8NClient] = None


def get_n8n_client(cfg: Optional[Any] = None) -> N8NClient:
    """Return the process-wide N8NClient singleton.

    ``cfg`` is an optional Settings-like object — when provided we
    honour ``cfg.n8n_base_url`` and ``cfg.n8n_api_key`` (the latter
    only useful for tests; production reads from env).
    """
    global _GLOBAL_CLIENT
    with _GLOBAL_CLIENT_LOCK:
        if _GLOBAL_CLIENT is None:
            base = None
            key = None
            if cfg is not None:
                base = getattr(cfg, "n8n_base_url", None) or None
                key = getattr(cfg, "n8n_api_key", None) or None
            _GLOBAL_CLIENT = N8NClient(base_url=base, api_key=key)
        return _GLOBAL_CLIENT

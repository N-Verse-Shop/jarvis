"""Direct LLM interaction utilities without extra features like temporal context."""

from __future__ import annotations
import ipaddress
import threading
import time
from typing import Optional, Any, Dict, List, Generator, Callable
from urllib.parse import urlparse
import requests
import json

from .debug import debug_log


class ToolsNotSupportedError(Exception):
    """Raised when the model returns HTTP 400 because native tool calling is not supported."""
    pass


# Audit round 15 fix F4: SSRF guard for ``ollama_base_url``.
#
# The config-loaded ``ollama_base_url`` is used directly in
# ``requests.post(f"{base_url}/api/chat", ...)``. A tampered config (or
# a future self-upgrade malfunction that writes config) could point it
# at ``http://169.254.169.254/...`` (cloud metadata), ``file:///...``,
# or an attacker-controlled host on the LAN — and every system prompt
# + user transcript + memory digest would be POSTed there in clear.
#
# Policy:
#   * scheme MUST be http or https
#   * host MUST be one of:
#       - loopback (127.0.0.0/8, ::1)
#       - RFC1918 private (10/8, 172.16/12, 192.168/16)
#       - RFC4193 unique-local IPv6 (fc00::/7)
#       - link-local IPv4 EXCEPT 169.254.169.254 (cloud metadata)
#       - the Tailscale CGNAT range 100.64.0.0/10
#       - a hostname in the explicit allow-list below
#
# Cache validated URLs so we don't pay DNS on every call.
_VALIDATED_URLS: set[str] = set()
_HOSTNAME_ALLOW_LIST = frozenset({
    "localhost", "ollama", "ollama.local",
})


class _BaseUrlRejected(Exception):
    """Raised by ``_validate_base_url`` when the URL fails the SSRF guard."""
    pass


def _validate_base_url(base_url: str) -> None:
    """Reject ``base_url`` values that point at non-private targets.

    Idempotent + cached so the per-call cost is one set-lookup on the
    hot path.
    """
    if not base_url:
        raise _BaseUrlRejected("empty base_url")
    if base_url in _VALIDATED_URLS:
        return
    parsed = urlparse(base_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise _BaseUrlRejected(f"unsupported scheme {scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise _BaseUrlRejected("missing hostname")
    # Hostname allow-list short-circuit (avoids DNS round-trip).
    if host in _HOSTNAME_ALLOW_LIST:
        _VALIDATED_URLS.add(base_url)
        return
    # Reject the AWS/GCP/Azure metadata IPs explicitly even if they
    # would otherwise be link-local.
    if host in ("169.254.169.254", "fd00:ec2::254"):
        raise _BaseUrlRejected(f"refusing to send LLM traffic to cloud metadata {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname (non-IP) — best-effort resolve and re-check.
        # R34-S56.1 Phase 9b (P2): the bare ``socket.gethostbyname``
        # call can block ~30s on Tailscale NXDOMAIN because mDNS +
        # search-domain probing has no upper bound on macOS. Wrap it
        # in a thread + 2s deadline; the SSRF guard fails closed if we
        # can't resolve in time (cheap, refuses on doubt vs. allows by
        # default).
        import socket as _socket
        import concurrent.futures as _cf
        def _resolve_blocking() -> str:
            return _socket.gethostbyname(host)
        try:
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(_resolve_blocking)
                resolved = _fut.result(timeout=2.0)
            ip = ipaddress.ip_address(resolved)
        except _cf.TimeoutError as exc:
            raise _BaseUrlRejected(f"DNS resolve timed out for {host} after 2s")
        except (OSError, ValueError) as exc:
            raise _BaseUrlRejected(f"could not resolve {host}: {exc}")
    # Tailscale CGNAT 100.64.0.0/10 is officially "Carrier-Grade NAT"
    # so ``is_private`` returns False; whitelist explicitly.
    if ip in ipaddress.ip_network("100.64.0.0/10"):
        _VALIDATED_URLS.add(base_url)
        return
    if ip.is_loopback or ip.is_private or (
        isinstance(ip, ipaddress.IPv6Address) and ip in ipaddress.ip_network("fc00::/7")
    ):
        _VALIDATED_URLS.add(base_url)
        return
    raise _BaseUrlRejected(
        f"refusing to send LLM traffic to public/unknown host {host} ({ip})"
    )


# Transient HTTP statuses that warrant a small retry. 502/503/504 are
# typical when Ollama is reloading a model or briefly OOM; 408 is the
# server-side analogue of ``Timeout``. Audit round 15 fix F5.
_RETRYABLE_STATUS = {408, 502, 503, 504}
_MAX_RETRIES = 3

# R34-S56.1 Phase 9b (P2): Tailscale CGNAT mappings can go stale —
# urllib3's default keep-alive keeps the dead socket cached in the
# connection pool, so the *next* request inherits a dead TCP path and
# stalls for the full ``timeout_sec`` before the kernel notices. Force
# the server to tear the socket down after each response so every call
# gets a fresh ARP/route resolution. This is the same fix applied in
# intent_judge.py + listener.py — keep the headers identical so the
# behaviour is uniform across modules.
_OLLAMA_HEADERS = {"Connection": "close"}


def call_llm_direct(base_url: str, chat_model: str, system_prompt: str, user_content: str, timeout_sec: float = 10.0, thinking: bool = False, num_ctx: int = 8192, temperature: Optional[float] = None) -> Optional[str]:
    """Direct LLM call without temporal context, location, or other ask_coach features.

    ``num_ctx`` controls Ollama's context window for this call. Default 8192
    matches ``call_llm_streaming``'s default (R34-S52 Phase 5 raised that).
    Keeping the two in sync is what lets Ollama reuse the KV-cache prefix
    across direct and streaming calls; a 4096 ↔ 8192 mismatch silently
    invalidated the prefix on every alternating turn (cold-eval +15 s).
    Callers that assemble richer prompts (planner with dialogue + memory +
    tool catalogue) can still pass an even larger value.

    ``temperature`` is forwarded to Ollama when set. Pass ``0.0`` for
    classification / extraction calls where determinism beats creativity —
    Ollama defaults to ~0.8 otherwise, which can flake small models on
    rule-following tasks (e.g. the knowledge extractor's banned-form list).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    options: Dict[str, Any] = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": thinking,
    }
    
    # Audit round 15 fix F4: refuse to send LLM traffic to non-private
    # destinations. If the operator has explicitly pointed ollama at a
    # public IP they need to add it to the allow-list — silent SSRF
    # is worse than a noisy refusal at start-up.
    try:
        _validate_base_url(base_url)
    except _BaseUrlRejected as exc:
        debug_log(f"call_llm_direct: refused base_url — {exc}", "llm")
        return None

    # Audit round 15 fix F5: retry on transient 5xx (Ollama briefly
    # reloading a model or under memory pressure). Up to 3 attempts
    # with bounded exponential backoff (0.5s, 1s). Do NOT retry on
    # 4xx — the existing ``ToolsNotSupportedError`` path needs the
    # first error verbatim, and ``HTTPError`` from a 400 means the
    # request was malformed, not transient.
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=timeout_sec,
                headers=_OLLAMA_HEADERS,
            ) as resp:
                if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                    sleep_for = 0.5 * (2 ** (attempt - 1))
                    debug_log(
                        f"call_llm_direct: HTTP {resp.status_code} attempt {attempt}/{_MAX_RETRIES} — retrying in {sleep_for:.1f}s",
                        "llm",
                    )
                    time.sleep(sleep_for)
                    continue
                resp.raise_for_status()
                data = resp.json()

            if isinstance(data, dict):
                content = extract_text_from_response(data)
                if isinstance(content, str) and content.strip():
                    return content
                debug_log(f"call_llm_direct: empty content from response keys={list(data.keys())}", "llm")
            return None
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            debug_log(f"call_llm_direct: timeout after {timeout_sec}s (attempt {attempt}/{_MAX_RETRIES})", "llm")
            # Don't retry on timeout — the timeout itself is the
            # backoff; the caller chose ``timeout_sec``.
            return None
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                sleep_for = 0.5 * (2 ** (attempt - 1))
                debug_log(
                    f"call_llm_direct: connection error attempt {attempt}/{_MAX_RETRIES} — retrying in {sleep_for:.1f}s ({exc})",
                    "llm",
                )
                time.sleep(sleep_for)
                continue
            break
        except Exception as e:
            debug_log(f"call_llm_direct: request failed — {e}", "llm")
            return None

    debug_log(f"call_llm_direct: gave up after {_MAX_RETRIES} attempts (last: {last_exc})", "llm")
    return None


def call_llm_streaming(
    base_url: str,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 30.0,
    thinking: bool = False,
    abort_event: Optional["threading.Event"] = None,
) -> Optional[str]:
    """
    Streaming LLM call that invokes on_token callback for each token received.

    Args:
        base_url: Ollama base URL
        chat_model: Model name
        system_prompt: System prompt
        user_content: User message
        on_token: Callback invoked with each token as it arrives
        timeout_sec: Request timeout
        thinking: Enable thinking/reasoning mode
        abort_event: Optional ``threading.Event`` checked on every
            iter_lines tick. When set, the loop short-circuits and the
            function returns ``None``. This is how the HUD stop button
            and the voice "стоп" command interrupt the LLM mid-stream
            — without it, Ollama keeps generating tokens (and TTS keeps
            queuing them) for many seconds after the user cancelled.
            Audit round 20 — directly fixes user-reported "кнопки
            завершення не працюють".

    Returns:
        Complete response text, or None on error / abort.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # R34-S52 M: num_ctx was 4096 here but ``chat_with_messages`` uses
    # 8192. The diary summariser is the heaviest consumer of this
    # streaming path — a full day's chunks easily exceed 4096 tokens
    # with the deflection-strip + attribution rules included, so
    # Ollama silently truncated the system prompt server-side and the
    # summariser's instruction list got cut, producing diary entries
    # that contained the very "the assistant said" leaks the rules
    # were designed to prevent. Match the rest of the LLM surface.
    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": 8192},
        "think": thinking,
    }

    # Use ``with`` so the streaming response (and the underlying TCP
    # connection) is released even if iter_lines exits early via an
    # exception or the caller stops consuming. Without this an aborted
    # stream pinned the connection until GC, which could happen many
    # turns later under sustained reply load.
    # Audit round 18 fix: SSRF parity for the streaming path. The
    # round-15 guard was only on ``call_llm_direct``; streaming was
    # left wide open with the same vector (an attacker-supplied
    # base_url pointing at 169.254.169.254 / 127.0.0.1 / link-local
    # would have leaked Ollama-bound payloads to arbitrary internal
    # services).
    try:
        _validate_base_url(base_url)
    except _BaseUrlRejected as e:
        try:
            from .debug import debug_log
            debug_log(f"call_llm_streaming: SSRF guard tripped — {e}", "llm")
        except Exception:
            pass
        return None
    try:
        with requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_sec,
            stream=True,
            headers=_OLLAMA_HEADERS,
        ) as resp:
            resp.raise_for_status()

            full_response = []
            # Audit round 18 fix: detect truncated streams. Previously
            # if Ollama dropped the TCP stream mid-token (network blip,
            # model OOM, server kill) ``iter_lines`` simply exited and
            # we returned the partial text as if it were complete — the
            # caller had no way to distinguish a finished response from
            # a torn one. Ollama's protocol marks the final chunk with
            # ``"done": true``; we now require it before treating the
            # accumulated text as a valid result.
            saw_done = False
            aborted = False
            for line in resp.iter_lines():
                # Audit round 20 fix: check abort flag on every line —
                # this is the only mechanism by which the HUD stop
                # button / voice "стоп" / wake-interrupt can break a
                # streaming reply. Without this, Ollama keeps
                # producing tokens for the full reply duration and the
                # daemon keeps appending them to the TTS queue even
                # though the user explicitly cancelled.
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    try:
                        from .debug import debug_log
                        debug_log(
                            f"call_llm_streaming: aborted by event after "
                            f"{len(full_response)} chunks",
                            "llm",
                        )
                    except Exception:
                        pass
                    break
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and isinstance(data["message"], dict):
                            content = data["message"].get("content", "")
                            if content:
                                full_response.append(content)
                                if on_token:
                                    on_token(content)
                        if data.get("done") is True:
                            saw_done = True
                    except json.JSONDecodeError:
                        continue
            if aborted:
                # Caller asked us to stop. Returning None signals "no
                # usable result" so the caller doesn't act on the
                # partial text — mirrors the behaviour for timeout /
                # torn stream.
                return None

            if not saw_done:
                try:
                    from .debug import debug_log
                    debug_log(
                        "call_llm_streaming: stream ended without 'done: true' "
                        f"after {len(full_response)} chunks — treating as failed",
                        "llm",
                    )
                except Exception:
                    pass
                return None

            result = "".join(full_response)
            return result if result.strip() else None

    except requests.exceptions.Timeout:
        # Audit round 8 fix I4: previously silent — diary updates hit
        # this on every cold-model rebuild and the "no chunks pending"
        # / "Ollama 503" / "timeout" / "connection refused" failures
        # all returned None indistinguishably. Now logged so
        # `~/Library/Logs/jarvis-assistant.err.log` shows what failed.
        try:
            from .debug import debug_log
            debug_log("call_llm_streaming: timeout", "llm")
        except Exception:
            pass
        return None
    except Exception as e:
        try:
            from .debug import debug_log
            debug_log(f"call_llm_streaming: request failed — {type(e).__name__}: {e}", "llm")
        except Exception:
            pass
        return None


def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
    """Extract text from LLM response - supports multiple response formats."""
    # Preferred: Ollama chat non-stream format
    if "message" in data and isinstance(data["message"], dict):
        content = data["message"].get("content")
        if isinstance(content, str):
            return content
    
    # Fallback: OpenAI-style format
    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if isinstance(content, str):
                    return content
            elif "text" in choice:
                content = choice["text"]
                if isinstance(content, str):
                    return content
    
    # Another fallback: direct "content" field
    if "content" in data:
        content = data["content"]
        if isinstance(content, str):
            return content
    
    return None


def chat_with_messages(
    base_url: str,
    chat_model: str,
    messages: List[Dict[str, str]],
    timeout_sec: float = 30.0,
    extra_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking: bool = False,
    abort_event: Optional[threading.Event] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send an arbitrary messages array to the LLM and return the raw response JSON.
    Caller is responsible for interpreting assistant content (including JSON/tool calls).

    Args:
        base_url: Ollama base URL
        chat_model: Model name
        messages: Conversation messages
        timeout_sec: Request timeout
        extra_options: Additional model options
        tools: Optional list of tools in OpenAI-compatible JSON schema format for native tool calling
        thinking: Enable thinking/reasoning mode
        abort_event: Optional ``threading.Event`` that, when set, causes
            the function to abort cleanly. Checked BEFORE the initial
            request (so a click-stop between turns wins immediately),
            and again at each retry boundary (so a stop during transient
            5xx retry doesn't have to wait the full backoff window).
            This is a best-effort fast-fail — once ``requests.post`` is
            mid-flight, only the ``timeout_sec`` deadline can interrupt
            it. Audit round 20.

    Returns the parsed JSON response dict on success, or None on error/timeout/abort.
    """
    # Main agentic chat uses 8192 so the system prompt (tool list + protocol
    # guidance + memory context) doesn't overflow and force ollama to truncate
    # — which previously dropped the tool schema on smaller models like
    # gemma4:e2b, tipping them into their pre-trained tool_code scaffolding.
    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 8192},
        "think": thinking,
    }
    if extra_options and isinstance(extra_options, dict):
        # Merge shallowly into options
        payload["options"].update(extra_options)

    # Add tools for native tool calling support (Ollama 0.4+)
    if tools and isinstance(tools, list) and len(tools) > 0:
        payload["tools"] = tools

    # Audit round 18 fix: SSRF parity. ``call_llm_direct`` was guarded
    # in round 15 but ``chat_with_messages`` — the MAIN agentic-loop
    # entry — was left wide open. Any operator misconfiguration or
    # attacker-supplied base_url could send the live conversation
    # (memory enrichment, user query, tool outputs) to an arbitrary
    # internal target. Same validator, same fail-closed contract.
    try:
        _validate_base_url(base_url)
    except _BaseUrlRejected as exc:
        try:
            from .debug import debug_log
            debug_log(f"chat_with_messages: refused base_url — {exc}", "llm")
        except Exception:
            pass
        return None

    # Audit round 18 fix: retry on transient 5xx matches the round-15
    # behaviour of ``call_llm_direct``. Ollama briefly returning 503
    # mid-conversation (model reload, OOM recovery) used to hard-fail
    # the user's turn with nothing recoverable. Bounded backoff
    # (0.5s, 1s) keeps total wall time under the parent timeout.
    import time as _time
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        # Audit round 20 abort hook: check BEFORE every attempt so a
        # stop signal that arrives during the backoff window wins
        # immediately instead of being absorbed by the full retry
        # deadline. Fast-fail returns None (the universal "no result"
        # signal the caller already handles).
        if abort_event is not None and abort_event.is_set():
            try:
                from .debug import debug_log
                debug_log(
                    f"chat_with_messages: aborted by event before attempt {attempt + 1}",
                    "llm",
                )
            except Exception:
                pass
            return None
        try:
            with requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=timeout_sec,
                headers=_OLLAMA_HEADERS,
            ) as resp:
                if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                    try:
                        from .debug import debug_log
                        debug_log(
                            f"chat_with_messages: transient {resp.status_code} — "
                            f"retry {attempt + 1}/{_MAX_RETRIES - 1}",
                            "llm",
                        )
                    except Exception:
                        pass
                    _time.sleep(0.5 * (2 ** attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
            if isinstance(data, dict):
                return data
            return None
        except requests.exceptions.Timeout:
            print("  ⏱️ LLM request timed out", flush=True)
            return None
        except requests.exceptions.ConnectionError as e:
            # Connection errors are often transient (Ollama restart);
            # apply the same retry envelope.
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                _time.sleep(0.5 * (2 ** attempt))
                continue
            # Audit round 18 fix: scrub URL userinfo from the printed
            # message so a misconfigured ``http://user:pass@host``
            # base_url cannot leak credentials to stdout / log file.
            print(f"  ❌ LLM connection error: {_sanitise_request_error(e)}", flush=True)
            return None
        except requests.exceptions.HTTPError as e:
            # Raise a specific error when the model rejects the tools parameter (HTTP 400).
            # This lets the caller fall back to text-based tool calling automatically.
            if e.response is not None and e.response.status_code == 400 and tools:
                raise ToolsNotSupportedError(
                    f"Model {chat_model!r} returned HTTP 400 — native tools API not supported"
                )
            print(f"  ❌ LLM HTTP error: {_sanitise_request_error(e)}", flush=True)
            return None
        except Exception as e:
            print(f"  ❌ LLM error: {type(e).__name__}", flush=True)
            return None

    # All retries exhausted with a transient response — drop into the
    # silent-None contract the caller expects.
    if last_exc is not None:
        try:
            from .debug import debug_log
            debug_log(
                f"chat_with_messages: exhausted retries — {type(last_exc).__name__}",
                "llm",
            )
        except Exception:
            pass
    return None


def _sanitise_request_error(e: Exception) -> str:
    """Strip URL userinfo + bearer-token-shaped fragments from a requests error.

    Audit round 18 fix: ``str(requests.exceptions.HTTPError)`` includes
    the failing URL verbatim. A base_url like ``http://user:pass@host``
    (common when the operator wires up a private proxy) would leak the
    embedded credentials into stdout — which the desktop_app pipes
    into the user-visible log file. Strip the userinfo segment and
    common bearer-token shapes before printing.
    """
    msg = str(e)
    try:
        import re as _re
        # url userinfo: scheme://user:pass@host -> scheme://[REDACTED]@host
        msg = _re.sub(
            r"(https?://)[^/\s@:]+:[^/\s@]+@",
            r"\1[REDACTED]@",
            msg,
        )
        # bearer tokens
        msg = _re.sub(r"(Bearer\s+)\S+", r"\1[REDACTED]", msg, flags=_re.IGNORECASE)
    except Exception:
        pass
    return msg

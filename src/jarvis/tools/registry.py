from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import sys
import re
import requests
import threading
from pathlib import Path
import os

from .builtin.screenshot import ScreenshotTool
from .builtin.web_search import WebSearchTool
from .builtin.local_files import LocalFilesTool
from .builtin.fetch_web_page import FetchWebPageTool
from .builtin.nutrition.log_meal import LogMealTool
from .builtin.nutrition.fetch_meals import FetchMealsTool
from .builtin.nutrition.delete_meal import DeleteMealTool
from .builtin.refresh_mcp_tools import RefreshMCPToolsTool
from .builtin.weather import WeatherTool
from .builtin.stop import StopTool
from .builtin.tool_search import ToolSearchTool
from .builtin.mac_control import MacControlTool
from .builtin.n8n import N8NTool
from .types import ToolExecutionResult
from ..config import Settings
from .external.mcp_client import MCPClient
from ..debug import debug_log


# Registry of all builtin tools.
# Audit round 20 P3 — ``macControl`` added so the agent has first-class
# window-mgmt + native-app automation (Notes / Reminders / Finder /
# focus/list apps / open URL with strict scheme allowlist).
BUILTIN_TOOLS = {
    "screenshot": ScreenshotTool(),
    "webSearch": WebSearchTool(),
    "localFiles": LocalFilesTool(),
    "fetchWebPage": FetchWebPageTool(),
    "logMeal": LogMealTool(),
    "fetchMeals": FetchMealsTool(),
    "deleteMeal": DeleteMealTool(),
    "refreshMCPTools": RefreshMCPToolsTool(),
    "getWeather": WeatherTool(),
    "stop": StopTool(),
    "toolSearchTool": ToolSearchTool(),
    "macControl": MacControlTool(),
    # R35-S1: n8n self-hosted workflow management — gated behind
    # JARVIS_N8N_API_KEY env var (tool returns a friendly "configure
    # me" reply when unset, so registration is always safe).
    "n8nAutomation": N8NTool(),
}

# Global MCP tools cache
_mcp_tools_cache: Dict[str, "ToolSpec"] = {}
_mcp_tools_cache_lock = threading.Lock()
_mcp_config_cache: Dict[str, Any] = {}


def initialize_mcp_tools(mcps_config: Dict[str, Any], verbose: bool = True) -> Tuple[Dict[str, "ToolSpec"], Dict[str, str]]:
    """
    Initialize MCP tools cache at startup.

    Args:
        mcps_config: MCP server configuration
        verbose: Whether to print status messages

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    global _mcp_tools_cache, _mcp_config_cache

    with _mcp_tools_cache_lock:
        _mcp_config_cache = mcps_config or {}
        _mcp_tools_cache, errors = discover_mcp_tools(mcps_config)

        if verbose and _mcp_tools_cache:
            debug_log(f"MCP tools cache initialized with {len(_mcp_tools_cache)} tools", "mcp")

    # Audit round 21 (F11): MCP cache changed — bust the
    # description / schema caches so the next call recomputes with
    # the new tool set. Outside ``_mcp_tools_cache_lock`` so the
    # description cache's own lock isn't nested under it.
    _bump_tools_generation()
    return _mcp_tools_cache.copy(), errors


def get_cached_mcp_tools() -> Dict[str, "ToolSpec"]:
    """Get cached MCP tools without rediscovering."""
    with _mcp_tools_cache_lock:
        return _mcp_tools_cache.copy()


def refresh_mcp_tools(verbose: bool = True) -> Tuple[Dict[str, "ToolSpec"], Dict[str, str]]:
    """
    Refresh MCP tools cache by rediscovering all tools.

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    global _mcp_tools_cache

    with _mcp_tools_cache_lock:
        if not _mcp_config_cache:
            debug_log("No MCP config cached, skipping refresh", "mcp")
            return {}, {}

        if verbose:
            print("🔄 Refreshing MCP tools...", flush=True)

        _mcp_tools_cache, errors = discover_mcp_tools(_mcp_config_cache)

        if verbose:
            print(f"  ✅ Found {len(_mcp_tools_cache)} MCP tools", flush=True)

        debug_log(f"MCP tools cache refreshed with {len(_mcp_tools_cache)} tools", "mcp")
    # Audit round 21 (F11): same generation-bump as
    # ``initialize_mcp_tools``. Outside the lock.
    _bump_tools_generation()
    return _mcp_tools_cache.copy(), errors


def is_mcp_cache_initialized() -> bool:
    """Check if MCP tools cache has been initialized."""
    with _mcp_tools_cache_lock:
        return len(_mcp_config_cache) > 0 or len(_mcp_tools_cache) > 0



# ToolSpec for MCP compatibility
@dataclass(frozen=True)
class ToolSpec:
    name: str  # canonical tool identifier (camelCase)
    description: str  # Human-readable description (matches MCP format)
    inputSchema: Optional[Dict[str, Any]] = None  # JSON Schema for arguments (matches MCP format)


def discover_mcp_tools(mcps_config: Dict[str, Any]) -> Tuple[Dict[str, ToolSpec], Dict[str, str]]:
    """Discover all tools from configured MCP servers and create ToolSpec entries for them.

    Returns:
        Tuple of (discovered_tools, errors) where errors maps server name to error message.
    """
    if not mcps_config:
        return {}, {}

    try:
        client = MCPClient(mcps_config)
        discovered_tools = {}
        errors: Dict[str, str] = {}

        for server_name in mcps_config.keys():
            try:
                tools = client.list_tools(server_name)
                for tool_info in tools:
                    tool_name = tool_info.get("name")
                    if not tool_name:
                        continue

                    # Create a unique tool name: server__toolname
                    full_tool_name = f"{server_name}__{tool_name}"

                    # Create a ToolSpec for this MCP tool
                    description = tool_info.get("description", f"Tool from {server_name} MCP server")
                    input_schema = tool_info.get("inputSchema", {"type": "object", "properties": {}, "required": []})
                    discovered_tools[full_tool_name] = ToolSpec(
                        name=full_tool_name,
                        description=description,
                        inputSchema=input_schema
                    )

            except BaseException as e:
                # ExceptionGroups (from anyio TaskGroup) wrap the real cause;
                # extract the first sub-exception for a useful error message.
                cause = e
                if hasattr(e, "exceptions") and e.exceptions:
                    cause = e.exceptions[0]
                debug_log(f"Failed to discover tools from MCP server '{server_name}': {cause}", "mcp")
                errors[server_name] = str(cause)
                continue

        return discovered_tools, errors

    except Exception as e:
        debug_log(f"Failed to discover MCP tools: {e}", "mcp")
        return {}, {"_global": str(e)}


# Audit round 21 fix (F11) — cache the tool-description and tool-
# JSON-schema outputs across replies. The two functions are called
# at least twice per voice turn (engine init + every tool_search
# widening). Without a cache, every turn paid the cost of:
#   • iterating BUILTIN_TOOLS (~12 tools, each a property access on
#     name/description/inputSchema — descriptors that may allocate).
#   • iterating mcp_tools (often 20-30 tools at ~200 char descs).
#   • building ~6 KB of strings.
# The output is a pure function of (sorted-allowed-tool-set,
# mcp_tools_generation). Cache the result keyed on that. Bust the
# cache when ``initialize_mcp_tools``/``refresh_mcp_tools`` updates
# ``_mcp_tools_cache`` — see ``_bump_tools_generation`` below.
#
# R34-S55.1 Phase 8a (P1 leak): bounded LRU. Before this cap, every
# unique ``tuple(sorted(allowed_tools))`` produced a permanent dict
# entry — and ``selection.py`` routes a per-query tool subset that
# varies with the user's voice content, so a long-running daemon
# accumulated tens of thousands of subset combinations × ~5-20 KB
# of cached JSON-schema strings. ``_bump_tools_generation`` only
# clears on MCP / builtin-set mutation, which can be days apart.
# Net: multi-100 MB drift over a year of normal voice traffic.
#
# 256 entries is generous (covers the typical 50-80 distinct subsets
# the selector actually emits) and uses OrderedDict's move-to-end
# semantics for true LRU eviction.
_TOOLS_RESULT_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_TOOLS_RESULT_CACHE_LOCK = threading.Lock()
_TOOLS_CACHE_GENERATION: int = 0
_TOOLS_RESULT_CACHE_MAX = 256


def _bump_tools_generation() -> None:
    """Invalidate the description / schema caches.

    Call from any path that mutates the available tool set
    (``initialize_mcp_tools``, ``refresh_mcp_tools``, builtin
    registration changes). Cheap — flips a counter; the cache lookup
    keys on that counter so the next call recomputes."""
    global _TOOLS_CACHE_GENERATION
    with _TOOLS_RESULT_CACHE_LOCK:
        _TOOLS_CACHE_GENERATION += 1
        _TOOLS_RESULT_CACHE.clear()


def _tools_cache_key(
    allowed_tools: Optional[List[str]],
    mcp_tools: Optional[Dict[str, "ToolSpec"]],
    kind: str,
) -> Tuple[Any, ...]:
    """Build a hashable cache key from the inputs.

    ``allowed_tools=None`` is treated as "all built-ins + all
    mcp_tools" — collapse to a sentinel so repeated calls with
    ``None`` share the same key.
    """
    if allowed_tools is None:
        allowed_key: Any = None
    else:
        allowed_key = tuple(sorted(allowed_tools))
    mcp_key = id(mcp_tools) if mcp_tools else 0
    return (kind, allowed_key, mcp_key, _TOOLS_CACHE_GENERATION)


def generate_tools_json_schema(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> List[Dict[str, Any]]:
    """
    Generate tools in OpenAI-compatible JSON schema format for native tool calling.

    This format is supported by Ollama for models with native tool calling support
    (Llama 3.1+, Llama 3.2, Qwen 3, Mistral, etc.).

    Returns a list of tool definitions in this format:
    [
        {
            "type": "function",
            "function": {
                "name": "toolName",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    ]

    Audit round 21 (F11): memoised keyed on (allowed_tools-set,
    mcp_tools-id, generation). Result is deep-copied on hit so a
    caller mutating the returned list doesn't poison the cache.
    """
    cache_key = _tools_cache_key(allowed_tools, mcp_tools, "schema")
    with _TOOLS_RESULT_CACHE_LOCK:
        cached = _TOOLS_RESULT_CACHE.get(cache_key)
        if cached is not None:
            # LRU bookkeeping — move the touched entry to the end so
            # the rarely-used ones drop off first when we hit the cap.
            _TOOLS_RESULT_CACHE.move_to_end(cache_key)
    if cached is not None:
        # Deep-ish copy — the function dict structure is small;
        # ``json.loads(json.dumps(...))`` would be cleanest but is
        # 5x slower than a list comprehension of dict copies on
        # this shape.
        return [dict(t, function=dict(t["function"])) for t in cached]
    names = list(allowed_tools or list(BUILTIN_TOOLS.keys()))
    tools: List[Dict[str, Any]] = []

    # Add built-in tools
    for tool_name in names:
        tool = BUILTIN_TOOLS.get(tool_name)
        if not tool:
            continue

        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema or {"type": "object", "properties": {}, "required": []},
            }
        }
        tools.append(tool_def)

    # Add discovered MCP tools
    if mcp_tools:
        for tool_name, spec in mcp_tools.items():
            if tool_name in names:  # Only include if allowed
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.inputSchema or {"type": "object", "properties": {}, "required": []},
                    }
                }
                tools.append(tool_def)

    with _TOOLS_RESULT_CACHE_LOCK:
        _TOOLS_RESULT_CACHE[cache_key] = tools
        _TOOLS_RESULT_CACHE.move_to_end(cache_key)
        # LRU eviction — drop the oldest entry once we exceed the cap.
        while len(_TOOLS_RESULT_CACHE) > _TOOLS_RESULT_CACHE_MAX:
            _TOOLS_RESULT_CACHE.popitem(last=False)
    # Caller may mutate the returned list — give them a copy.
    return [dict(t, function=dict(t["function"])) for t in tools]


def generate_tools_description(allowed_tools: Optional[List[str]] = None, mcp_tools: Optional[Dict[str, ToolSpec]] = None) -> str:
    """Produce a compact tool help string for the system prompt using OpenAI standard format.

    Audit round 21 (F11): memoised — same key shape as
    ``generate_tools_json_schema``. Save ~5-20 ms per turn.
    """
    cache_key = _tools_cache_key(allowed_tools, mcp_tools, "desc")
    with _TOOLS_RESULT_CACHE_LOCK:
        cached = _TOOLS_RESULT_CACHE.get(cache_key)
        if cached is not None:
            _TOOLS_RESULT_CACHE.move_to_end(cache_key)
    if cached is not None:
        return cached
    names = list(allowed_tools or list(BUILTIN_TOOLS.keys()))
    lines: List[str] = []
    lines.append("Tool-use protocol: Use the tool_calls field in your response:")
    lines.append('tool_calls: [{"id": "call_<id>", "type": "function", "function": {"name": "<toolName>", "arguments": "<json_string>"}}]')
    lines.append("\nAvailable tools and when to use them:")

    # Add built-in tools
    for tool_name in names:
        tool = BUILTIN_TOOLS.get(tool_name)
        if not tool:
            continue
        lines.append(f"\n{tool.name}: {tool.description}")
        if tool.inputSchema:
            # Extract a simple parameter summary from the JSON schema
            props = tool.inputSchema.get("properties", {})
            required = tool.inputSchema.get("required", [])
            param_descriptions = []
            for prop_name, prop_def in props.items():
                prop_type = prop_def.get("type", "any")
                is_required = prop_name in required
                req_marker = " (required)" if is_required else ""
                param_descriptions.append(f"{prop_name}: {prop_type}{req_marker}")
            if param_descriptions:
                lines.append(f"Input: {', '.join(param_descriptions)}")

    # Add discovered MCP tools
    if mcp_tools:
        for tool_name, spec in mcp_tools.items():
            if tool_name in names:  # Only include if allowed
                lines.append(f"\n{spec.name}: {spec.description}")
                if spec.inputSchema:
                    # Extract a simple parameter summary from the JSON schema
                    props = spec.inputSchema.get("properties", {})
                    required = spec.inputSchema.get("required", [])
                    param_descriptions = []
                    for prop_name, prop_def in props.items():
                        prop_type = prop_def.get("type", "any")
                        is_required = prop_name in required
                        req_marker = " (required)" if is_required else ""
                        param_descriptions.append(f"{prop_name}: {prop_type}{req_marker}")
                    if param_descriptions:
                        lines.append(f"Input: {', '.join(param_descriptions)}")

    result = "\n".join(lines)
    with _TOOLS_RESULT_CACHE_LOCK:
        _TOOLS_RESULT_CACHE[cache_key] = result
        _TOOLS_RESULT_CACHE.move_to_end(cache_key)
        while len(_TOOLS_RESULT_CACHE) > _TOOLS_RESULT_CACHE_MAX:
            _TOOLS_RESULT_CACHE.popitem(last=False)
    return result

# Audit round 11 fix M1: previously a duplicate ``_normalize_time_range``
# lived here AND in nutrition/fetch_meals.py. Only the nutrition copy
# was used; the registry copy was dead code that future maintainers
# would have copy-pasted out of sync. Removed; if any other tool needs
# the helper, import it from ``nutrition/fetch_meals.py``.


def run_tool_with_retries(
    db,
    cfg: Settings,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]],
    system_prompt: str,
    original_prompt: str,
    redacted_text: str,
    max_retries: int = 1,
    language: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None,
) -> ToolExecutionResult:
    """Dispatch a tool call. Audit round 11 fix C1: enforce the active
    allow-list AT THE REGISTRY LAYER, not just at the engine call sites.

    The engine guards both dispatch sites today, but the registry is the
    real security boundary — any future caller (a CLI test harness, the
    currently-unwired evaluator, an experimental agent path) would
    silently get full-catalog access. ``allowed_tools=None`` preserves
    the prior behaviour (no enforcement) for callers that haven't yet
    been updated; engine.py passes the live allow-list explicitly.
    """
    # Normalize tool name to canonical camelCase
    raw_name = (tool_name or "").strip()
    name = raw_name

    # Allow-list enforcement happens before any dispatch path so MCP
    # tools, builtins, and the unknown-tool tail share one gate.
    if allowed_tools is not None and name not in allowed_tools:
        debug_log(f"registry: rejected non-allowed tool: {name!r} (allow-list size={len(allowed_tools)})", "tools")
        return ToolExecutionResult(
            success=False,
            reply_text=None,
            error_message=f"Tool '{name}' is not in the active allow-list.",
        )

    # Audit round 16 fix: builtin tools ALWAYS win over MCP-discovered
    # tools, regardless of name shape. Previously a malicious MCP
    # server could register a tool called ``screenshot__steal`` or
    # even reuse a builtin name with a double-underscore suffix
    # (``web__Search``); the ``"__" in raw_name`` branch fired before
    # the builtin check ran, so the MCP got the dispatch.
    if name in BUILTIN_TOOLS:
        # Fall through to the BUILTIN dispatch below.
        pass
    elif "__" in raw_name:
        # Check if tool name is a discovered MCP tool (server__toolname format)
        server_name, mcp_tool_name = raw_name.split("__", 1)
        mcps_config = getattr(cfg, "mcps", {})
        # Extra guard (defence in depth): even after the builtin
        # check above, refuse to dispatch via MCP if the server_name
        # half collides with a builtin name. This catches the
        # degenerate case where the engine's name-normalisation drops
        # the suffix before lookup.
        if server_name in BUILTIN_TOOLS:
            debug_log(
                f"registry: refusing MCP dispatch — server_name {server_name!r} "
                f"shadows a builtin tool",
                "tools",
            )
            return ToolExecutionResult(
                success=False,
                reply_text=None,
                error_message=f"Refused: MCP server name '{server_name}' collides with a builtin tool.",
            )
        if mcps_config and server_name in mcps_config:
            try:
                if MCPClient is None:
                    return ToolExecutionResult(success=False, reply_text=None, error_message="MCP client not available. Install 'mcp' package.")

                client = MCPClient(mcps_config)
                result = client.invoke_tool(server_name=server_name, tool_name=mcp_tool_name, arguments=tool_args or {})
                is_error = bool(result.get("isError", False))
                text = result.get("text") or None
                return ToolExecutionResult(success=(not is_error), reply_text=text, error_message=(text if is_error else None))
            except Exception as e:
                return ToolExecutionResult(success=False, reply_text=None, error_message=f"MCP tool '{raw_name}' error: {e}")

    # Friendly user print helper (non-debug only)
    def _user_print(message: str) -> None:
        # 4-space indent: tool messages happen INSIDE an agentic-loop
        # turn. The turn header (`  🔁 Turn N/M`) sits at 2 spaces, so
        # per-tool activity nests one level deeper for visual hierarchy.
        if not getattr(cfg, "voice_debug", False):
            try:
                print(f"    {message}")
            except Exception:
                pass

    # Check builtin tools first
    if name in BUILTIN_TOOLS:
        tool = BUILTIN_TOOLS[name]
        return tool.execute(
            db=db,
            cfg=cfg,
            tool_args=tool_args,
            system_prompt=system_prompt,
            original_prompt=original_prompt,
            redacted_text=redacted_text,
            max_retries=max_retries,
            user_print=_user_print,
            language=language,
        )

    # Unknown tool
    debug_log(f"unknown tool requested: {tool_name}", "tools")
    return ToolExecutionResult(success=False, reply_text=None, error_message=f"Unknown tool: {tool_name}")



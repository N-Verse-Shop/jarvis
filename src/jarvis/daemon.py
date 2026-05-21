"""
Jarvis Voice Assistant Daemon

Main orchestrator that coordinates listening, reply generation, and output.
"""

from __future__ import annotations
import sys
import os
import time
import signal
import threading

# Fix OpenBLAS threading crash in bundled apps (must be before numpy imports)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Fix Windows console encoding for Unicode/emoji characters
# Skip in bundled mode (frozen) - encoding is handled by desktop_app.py
if sys.platform == 'win32' and not getattr(sys, 'frozen', False):
    try:
        import io
        # Only wrap if stdout has a proper binary buffer (not a custom writer)
        if hasattr(sys.stdout, 'buffer') and hasattr(sys.stdout.buffer, 'write'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and hasattr(sys.stderr.buffer, 'write'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from typing import Optional
from faster_whisper import WhisperModel

from .config import load_settings
from .memory.db import Database
from .memory.conversation import DialogueMemory, update_diary_from_dialogue_memory
from .output.tts import create_tts_engine
from .tools.registry import initialize_mcp_tools
from .debug import debug_log
from .listening.listener import VoiceListener
from .utils.location import get_location_context, is_location_available

# Global instances for coordination between modules
_global_dialogue_memory: Optional[DialogueMemory] = None
_global_stop_requested: bool = False
_warm_profile_graph_listener = None  # registered callback, kept for shutdown unregister
_global_tts_engine = None  # TTS engine reference for face animation polling
_global_dictation_engine = None  # Dictation engine reference for history UI

# Shutdown timeout for diary update (shorter than normal to allow reasonable quit time)
# Desktop app's stop_daemon() should wait at least this long + buffer
SHUTDOWN_DIARY_TIMEOUT_SEC = 45.0

# Callbacks for desktop app to receive diary update progress
# Set by desktop app before calling request_stop()
_diary_update_callbacks: dict = {
    "on_token": None,  # Callable[[str], None] - called for each LLM token
    "on_status": None,  # Callable[[str], None] - called for status updates
    "on_chunks": None,  # Callable[[List[str]], None] - called with pending chunks
    "on_complete": None,  # Callable[[bool], None] - called when done (success/fail)
}


def request_stop() -> None:
    """Request the daemon to stop gracefully. Used by desktop app for QThread shutdown."""
    global _global_stop_requested
    _global_stop_requested = True


def set_diary_update_callbacks(
    on_token=None,
    on_status=None,
    on_chunks=None,
    on_complete=None,
) -> None:
    """
    Set callbacks for diary update progress during shutdown.

    These are used by the desktop app to show a live diary update dialog.

    Args:
        on_token: Called with each LLM token as it's generated
        on_status: Called with status messages
        on_chunks: Called with the list of pending conversation chunks
        on_complete: Called when diary update completes (bool = success)
    """
    global _diary_update_callbacks
    _diary_update_callbacks["on_token"] = on_token
    _diary_update_callbacks["on_status"] = on_status
    _diary_update_callbacks["on_chunks"] = on_chunks
    _diary_update_callbacks["on_complete"] = on_complete


def get_pending_diary_chunks() -> list:
    """Get pending conversation chunks from dialogue memory (for UI display only).

    Uses ``get_pending_chunks()`` which discards the atomic snapshot timestamp.
    Do not use the result of this function to drive diary saves — the actual
    save path goes through ``update_diary_from_dialogue_memory``, which calls
    ``get_pending_chunks_with_snapshot()`` internally.
    """
    global _global_dialogue_memory
    if _global_dialogue_memory is None:
        return []
    return _global_dialogue_memory.get_pending_chunks()


# Diary IPC protocol prefix - desktop app intercepts lines starting with this
DIARY_IPC_PREFIX = "__DIARY__:"


def _emit_diary_event(event_type: str, data) -> None:
    """
    Emit a diary update event to stdout for IPC with desktop app.

    Used in subprocess mode where callbacks aren't available.
    Desktop app intercepts these lines and forwards to diary dialog.

    Args:
        event_type: One of "chunks", "token", "status", "complete"
        data: Event payload (list for chunks, str for token/status, bool for complete)
    """
    import json
    try:
        event = {"type": event_type, "data": data}
        line = f"{DIARY_IPC_PREFIX}{json.dumps(event)}"
        print(line, flush=True)
        # Debug: also print to stderr so we can verify it's being called
        if event_type != "token":  # Don't spam for tokens
            debug_log(f"IPC event emitted: {event_type}", "diary_ipc")
    except Exception as e:
        debug_log(f"IPC emit error: {e}", "diary_ipc")


def is_stop_requested() -> bool:
    """Check if a stop has been requested."""
    return _global_stop_requested


def get_tts_engine():
    """Get the global TTS engine for speaking state polling (used by face widget)."""
    return _global_tts_engine


def get_dictation_engine():
    """Get the global dictation engine (used by desktop app for history window)."""
    return _global_dictation_engine


# Audit round 16 fix: dictation state updates fire on every hotkey
# press and used to swallow ALL exceptions silently. The dominant
# failure mode is ``ImportError`` in headless mode (expected — no
# Qt installed in CI / SSH session) which we keep silent; anything
# else (e.g. ``set_state`` raised, the singleton was torn down)
# is a real bug. We dedupe via ``_FACE_WIDGET_WARN_SEEN`` so a
# transient failure logs once instead of every dictation tick.
_FACE_WIDGET_WARN_SEEN: set[str] = set()


def _safe_set_face_state(state_name: str) -> None:
    """Best-effort UI state hint for the face widget.

    ``state_name`` is the ``JarvisState`` member name ("DICTATING",
    "DICTATION_PROCESSING", "IDLE"). Failures are logged ONCE per
    ``(state, error_class)`` pair so the log doesn't fill up on
    a recurring exception.
    """
    try:
        from desktop_app.face_widget import JarvisState, get_jarvis_state
        get_jarvis_state().set_state(getattr(JarvisState, state_name))
    except ImportError:
        # Headless / unbundled mode — no widget to update. Silent
        # by design; this fires whenever the desktop app isn't
        # available and is not a bug.
        return
    except Exception as e:
        sig = f"{state_name}:{type(e).__name__}"
        if sig not in _FACE_WIDGET_WARN_SEEN:
            _FACE_WIDGET_WARN_SEEN.add(sig)
            debug_log(
                f"face widget set_state({state_name}) failed once: "
                f"{type(e).__name__}: {e}",
                "ui",
            )


def _install_signal_handlers() -> None:
    """Ensure signals like Ctrl+Break trigger clean shutdown."""
    def _raise_keyboard_interrupt(_signum, _frame):
        raise KeyboardInterrupt()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _raise_keyboard_interrupt)
            except (ValueError, OSError) as e:
                # ValueError: signal only installable on main thread
                # (the only legitimate swallow). OSError: signal unsupported
                # on this OS (Windows SIGTERM is partial). Anything else
                # is a real bug that used to vanish silently — audit
                # round 16 surfaces it as a debug breadcrumb.
                debug_log(
                    f"signal handler install failed for {sig_name}: "
                    f"{type(e).__name__}: {e}",
                    "daemon",
                )
            except Exception as e:
                debug_log(
                    f"unexpected error installing signal {sig_name}: "
                    f"{type(e).__name__}: {e}",
                    "daemon",
                )


def _check_and_update_diary(
    db: Database, cfg, verbose: bool = False, force: bool = False, timeout_sec: Optional[float] = None,
    use_callbacks: bool = False, use_ipc: bool = False
) -> None:
    """Check if diary should be updated and perform batch update if needed.

    Args:
        timeout_sec: Optional override for LLM timeout. If None, uses cfg.llm_chat_timeout_sec.
                    During shutdown, a shorter timeout is used to allow graceful quit.
        use_callbacks: If True, uses the global diary update callbacks for UI updates.
        use_ipc: If True, emits diary events to stdout for IPC with desktop app (subprocess mode).
    """
    global _global_dialogue_memory, _diary_update_callbacks

    # R34-S49 OPT — silence the no-op "check" line. This runs every
    # ~30s in the daemon's diary-poll loop and accounted for ~20% of
    # all debug-log volume in the err log. Keep it for verbose=True
    # (manual / forced diary updates surface diagnostics).
    if verbose or force:
        debug_log(f"diary update check: force={force}, verbose={verbose}", "memory")

    # Helper to safely call callbacks and/or emit IPC events
    def _notify(event_type: str, data):
        # Map event types to callback names
        callback_map = {"chunks": "on_chunks", "status": "on_status", "token": "on_token", "complete": "on_complete"}
        callback_name = callback_map.get(event_type)

        # Call callback if set (bundled mode)
        if use_callbacks and callback_name and _diary_update_callbacks.get(callback_name):
            try:
                _diary_update_callbacks[callback_name](data)
            except Exception as _cb_err:
                # Audit round 16: surface the callback failure. A buggy
                # UI callback used to silently break diary UI updates
                # with no log trail; debug breadcrumb lets us bisect.
                debug_log(
                    f"diary callback {callback_name!r} raised: "
                    f"{type(_cb_err).__name__}: {_cb_err}",
                    "memory",
                )

        # Emit IPC event (subprocess mode)
        if use_ipc:
            _emit_diary_event(event_type, data)

    if _global_dialogue_memory is None:
        debug_log("diary update skipped: dialogue_memory is None", "memory")
        _notify("complete", False)
        return

    try:
        should_update = force or _global_dialogue_memory.should_update_diary()
        # R34-S49 OPT — only log the no-op case when verbose. The
        # interesting branch (should_update=True) is logged via the
        # "found N pending chunks" line below.
        if verbose or should_update or force:
            debug_log(f"diary update: should_update={should_update}, force={force}", "memory")

        if should_update:
            # Display-only: get a snapshot of pending chunks to notify the UI.
            # The atomic snapshot for the actual save is captured inside
            # update_diary_from_dialogue_memory via get_pending_chunks_with_snapshot().
            pending_chunks = _global_dialogue_memory.get_pending_chunks()
            debug_log(f"diary update: found {len(pending_chunks)} pending chunks", "memory")

            if not pending_chunks:
                debug_log("diary update skipped: no pending chunks", "memory")
                _notify("complete", False)
                return

            # Notify about chunks and status
            _notify("chunks", pending_chunks)
            _notify("status", "Writing diary entry...")

            if verbose:
                try:
                    print("📝 Updating your diary. Please wait… (don't press Ctrl+C again)", file=sys.stderr, flush=True)
                except Exception:
                    pass

            source_app = "stdin" if cfg.use_stdin else "voice"
            effective_timeout = timeout_sec if timeout_sec is not None else cfg.llm_chat_timeout_sec

            # Create token handler that notifies via callback and/or IPC
            # For IPC mode, batch tokens to avoid overwhelming the receiver
            token_buffer = []
            last_flush_time = [time.time()]  # Use list for closure mutability
            TOKEN_FLUSH_INTERVAL = 0.1  # Flush every 100ms

            def on_token_handler(token: str):
                if use_callbacks:
                    # Callbacks can handle individual tokens (same process)
                    _notify("token", token)
                elif use_ipc:
                    # IPC mode: batch tokens to reduce event frequency
                    token_buffer.append(token)
                    now = time.time()
                    if now - last_flush_time[0] >= TOKEN_FLUSH_INTERVAL:
                        if token_buffer:
                            _emit_diary_event("token", "".join(token_buffer))
                            token_buffer.clear()
                        last_flush_time[0] = now

            # Only use token handler if we have callbacks or IPC enabled
            on_token = on_token_handler if (use_callbacks or use_ipc) else None

            # Graph best-child picker is a one-digit classification — reuse the
            # tool-router model chain so placement runs on a small model instead
            # of paging in the big chat model for every fact.
            from .reply.engine import resolve_tool_router_model
            graph_picker_model = resolve_tool_router_model(cfg)

            summary_id = update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=_global_dialogue_memory,
                ollama_base_url=cfg.ollama_base_url,
                ollama_chat_model=cfg.ollama_chat_model,
                ollama_embed_model=cfg.ollama_embed_model,
                source_app=source_app,
                voice_debug=cfg.voice_debug,
                timeout_sec=effective_timeout,
                force=force,
                on_token=on_token,
                thinking=getattr(cfg, 'llm_thinking_enabled', False),
                graph_picker_model=graph_picker_model,
            )

            # Flush any remaining tokens in IPC mode
            if use_ipc and token_buffer:
                _emit_diary_event("token", "".join(token_buffer))
                token_buffer.clear()

            if summary_id:
                debug_log(f"diary updated from dialogue memory: id={summary_id}", "memory")
                _notify("complete", True)
            else:
                debug_log("diary update from dialogue memory failed", "memory")
                _notify("complete", False)

            if verbose:
                try:
                    if summary_id:
                        print("✅ Diary update finished.", file=sys.stderr, flush=True)
                    else:
                        print("⚠️ Diary update failed. Shutting down anyway.", file=sys.stderr, flush=True)
                except Exception:
                    pass
        else:
            # No update needed
            _notify("complete", False)
    except Exception as e:
        debug_log(f"diary update check error: {e}", "memory")
        _notify("complete", False)


def main() -> None:
    """Main daemon entry point."""
    global _global_dialogue_memory, _global_stop_requested, _global_tts_engine, _global_dictation_engine
    global _warm_profile_graph_listener

    # Reset stop flag at start (in case of restart)
    _global_stop_requested = False

    _install_signal_handlers()

    cfg = load_settings()
    db = Database(cfg.db_path, cfg.sqlite_vss_path)

    debug_log("daemon started", "jarvis")
    print("✓ Daemon started", flush=True)
    print(f"🧠 Using chat model: {cfg.ollama_chat_model}", flush=True)
    print(f"🎤 Using whisper model: {cfg.whisper_model}", flush=True)

    # Typed event: daemon startup. Single source of truth for HUD +
    # observers to discover model/version without polling state.json.
    try:
        from .ipc import get_stream
        from . import get_version
        version, channel = get_version()
        get_stream().emit(
            "daemon_startup",
            version=version,
            channel=channel,
            pid=os.getpid(),
            chat_model=cfg.ollama_chat_model,
            judge_model=getattr(cfg, "intent_judge_model", ""),
            whisper_model=cfg.whisper_model,
        )
    except Exception as _e:
        debug_log(f"failed to emit daemon_startup: {_e}", "jarvis")

    # MCP preflight: discover and cache external MCP tools
    mcps = getattr(cfg, "mcps", {}) or {}
    if mcps:
        print(f"📡 Discovering MCP tools from {len(mcps)} server(s)...", flush=True)
        try:
            mcp_tools, mcp_errors = initialize_mcp_tools(mcps, verbose=False)

            # Group tools by server for display
            tools_by_server: dict = {}
            for tool_name in mcp_tools.keys():
                if "__" in tool_name:
                    server_name = tool_name.split("__")[0]
                    if server_name not in tools_by_server:
                        tools_by_server[server_name] = []
                    tools_by_server[server_name].append(tool_name)

            for server_name in mcps.keys():
                count = len(tools_by_server.get(server_name, []))
                if count > 0:
                    print(f"  ✅ {server_name}: {count} tools available", flush=True)
                elif server_name in mcp_errors:
                    print(f"  ❌ {server_name}: {mcp_errors[server_name]}", flush=True)
                else:
                    print(f"  ⚠️ {server_name}: no tools discovered", flush=True)

            debug_log(f"MCP tools cached: {len(mcp_tools)} total", "mcp")
        except Exception as e:
            debug_log(f"MCP discovery failed: {e}", "mcp")
            print(f"  ⚠️ MCP discovery failed: {e}", flush=True)
    else:
        print("📡 No MCP servers configured", flush=True)

    # Initialize dialogue memory with timeout
    print("💾 Initializing dialogue memory...", flush=True)
    _global_dialogue_memory = DialogueMemory(
        inactivity_timeout=cfg.dialogue_memory_timeout,
        max_interactions=20
    )
    print("✓ Dialogue memory initialized", flush=True)

    # Wire the conversation-scoped warm-profile cache to graph mutations.
    # When the User or Directives branch is mutated mid-conversation, the
    # cached warm profile is dropped so the next reply rebuilds it from
    # the current graph state. World-branch writes (typical webSearch
    # extractions) do not touch warm profile, so they are ignored.
    try:
        from .memory.graph import (
            BRANCH_DIRECTIVES,
            BRANCH_USER,
            register_graph_mutation_listener,
        )

        _wp_relevant_branches = {BRANCH_USER, BRANCH_DIRECTIVES}

        # Read the DialogueMemory ref through the module global at fire
        # time, not via closure capture, so a future singleton swap (tests
        # or hot-reload) routes invalidation to the live instance instead
        # of the freed one.
        def _invalidate_wp_on_graph_mutation(*, action, node_id, branch):
            del action, node_id  # Only the branch matters for warm-profile filtering.
            if branch not in _wp_relevant_branches:
                return
            dm = _global_dialogue_memory
            if dm is None:
                return
            try:
                dm.invalidate_warm_profile()
                debug_log(
                    f"warm profile invalidated by {branch} graph mutation",
                    "memory",
                )
            except Exception as exc:
                debug_log(
                    f"warm profile invalidation failed (non-fatal): {exc}",
                    "memory",
                )

        # If a previous run left a listener registered (re-entry without
        # full process restart), drop it before installing the new one so
        # the registry never accumulates stale closures.
        if _warm_profile_graph_listener is not None:
            try:
                from .memory.graph import unregister_graph_mutation_listener
                unregister_graph_mutation_listener(_warm_profile_graph_listener)
            except Exception:
                pass
        register_graph_mutation_listener(_invalidate_wp_on_graph_mutation)
        _warm_profile_graph_listener = _invalidate_wp_on_graph_mutation
    except Exception as exc:
        debug_log(
            f"warm profile mutation listener wiring failed (non-fatal): {exc}",
            "memory",
        )

    # Knowledge graph: wipe + re-seed if the on-disk shape predates the
    # User/Directives/World taxonomy. Non-destructive to the diary —
    # users can re-import via the memory viewer.
    try:
        from .memory.graph import GraphMemoryStore
        _graph_store_boot = GraphMemoryStore(cfg.db_path)
        if _graph_store_boot.migrate_legacy_shape():
            print("🧹 Wiped legacy knowledge graph; re-seeded User / Directives / World branches", flush=True)
            print("   📥 Open the memory viewer and use 'Import from Diary' to repopulate.", flush=True)
        _graph_store_boot.close()
    except Exception as e:
        debug_log(f"graph legacy-shape migration failed (non-fatal): {e}", "memory")

    # Check location detection status
    if cfg.location_enabled:
        location_context = get_location_context(
            config_ip=cfg.location_ip_address,
            auto_detect=cfg.location_auto_detect,
            resolve_cgnat_public_ip=cfg.location_cgnat_resolve_public_ip,
            location_cache_minutes=cfg.location_cache_minutes,
        )
        if location_context == "Location: Unknown":
            print("📍 Location detection not available", flush=True)
            if not is_location_available():
                print("     GeoLite2 database not found. Download from:", flush=True)
                print("     https://www.maxmind.com/en/geolite2/signup", flush=True)
            else:
                print("     Could not detect public IP address.", flush=True)
                print("     Configure 'location_ip_address' in config.json", flush=True)
                print("     or run the setup wizard to configure location.", flush=True)
        else:
            print(f"📍 {location_context}", flush=True)
    else:
        print("📍 Location services disabled", flush=True)

    # Initialize TTS
    print(f"🔊 Initializing TTS engine ({cfg.tts_engine})...", flush=True)
    tts = create_tts_engine(
        engine=cfg.tts_engine,
        enabled=cfg.tts_enabled,
        voice=cfg.tts_voice,
        rate=cfg.tts_rate,
        # Chatterbox parameters
        device=cfg.tts_chatterbox_device,
        audio_prompt_path=cfg.tts_chatterbox_audio_prompt,
        exaggeration=cfg.tts_chatterbox_exaggeration,
        cfg_weight=cfg.tts_chatterbox_cfg_weight,
        # Piper parameters
        piper_model_path=cfg.tts_piper_model_path,
        piper_speaker=cfg.tts_piper_speaker,
        piper_length_scale=cfg.tts_piper_length_scale,
        piper_noise_scale=cfg.tts_piper_noise_scale,
        piper_noise_w=cfg.tts_piper_noise_w,
        piper_sentence_silence=cfg.tts_piper_sentence_silence,
        # CRITICAL: pass through the per-language voice map. Without this,
        # SystemTTS falls back to its hardcoded defaults (`piper:mykyta` for UA)
        # which булькає on Cyrillic. With this, the daemon uses whatever the
        # user configured in `tts_system_voice_map` (e.g. Lesya UA / Milena RU).
        system_voice_map=cfg.tts_system_voice_map,
    )
    _global_tts_engine = tts  # Expose for face widget speaking animation
    if tts.enabled:
        tts.start()
        print("✓ TTS engine started", flush=True)
    else:
        print("  TTS disabled", flush=True)

    # Initialize voice listening (only if dependencies available)
    # R31: ``voice_engine`` config flag toggles between the legacy
    # hand-rolled listener and the Pipecat-based loop. Default is
    # "legacy" until the Pipecat path passes end-to-end integration
    # in Stage 6.
    voice_engine = str(getattr(cfg, "voice_engine", "legacy")).lower()
    print(
        f"🎤 Initializing voice listener (engine={voice_engine}, "
        "this may take a moment to load the model)...",
        flush=True,
    )
    voice_thread = None
    if voice_engine == "pipecat":
        try:
            from .listening.pipecat_loop import PipecatVoiceThread
            voice_thread = PipecatVoiceThread(
                db, cfg, tts, _global_dialogue_memory
            )
            print(
                "✓ Pipecat voice thread constructed — starting...",
                flush=True,
            )
        except Exception as exc:
            print(
                f"⚠ Pipecat init failed ({exc!r}); falling back to "
                "legacy listener.",
                flush=True,
            )
            voice_thread = None
    if voice_thread is None:
        voice_thread = VoiceListener(db, cfg, tts, _global_dialogue_memory)
    voice_thread.start()
    print(
        "✓ Voice listener thread started (loading model in background)",
        flush=True,
    )

    # R33-S5: optional local-only dashboard control room. Enabled by
    # default (gated env var) so the user gets the dashboard on
    # first daemon restart after this commit. Set
    # ``JARVIS_DASHBOARD_DISABLE=true`` to turn it off entirely.
    if os.environ.get("JARVIS_DASHBOARD_DISABLE", "").lower() not in (
        "1", "true", "yes", "on"
    ):
        try:
            from .dashboard import start_dashboard, get_dashboard_token
            dashboard = start_dashboard()
            token = get_dashboard_token()
            host = os.environ.get("JARVIS_DASHBOARD_HOST", "127.0.0.1")
            port = int(os.environ.get("JARVIS_DASHBOARD_PORT", "8789"))
            print(
                f"✓ Dashboard ready on http://{host}:{port}",
                flush=True,
            )
            print(
                "    token (first 12 chars): "
                f"{token[:12]}…  — full token at "
                f"~/Library/Application\\ Support/jarvis/dashboard-token",
                flush=True,
            )
        except Exception as _dash_exc:
            print(
                f"⚠ Dashboard init skipped: {_dash_exc!r}",
                flush=True,
            )

    # R33-S2: nightly facts-prune. Decay-based pruning is cheap (~ms for
    # ~10k rows) so we just run it once every 6 h in a daemon thread.
    # Disable with JARVIS_FACTS_DISABLE=true.
    if os.environ.get("JARVIS_FACTS_DISABLE", "").lower() not in (
        "1", "true", "yes", "on"
    ):
        def _facts_prune_loop():
            import time as _t
            from .memory.facts import get_facts_store
            _t.sleep(60)  # let startup finish before first sweep
            while True:
                try:
                    store = get_facts_store()
                    n = store.prune()
                    if n:
                        print(f"facts: pruned {n} stale row(s)", flush=True)
                except Exception as exc:
                    print(f"facts prune failed: {exc!r}", flush=True)
                _t.sleep(6 * 3600)  # every 6 hours

        threading.Thread(
            target=_facts_prune_loop,
            name="facts-prune",
            daemon=True,
        ).start()

    # R34-S10: pipeline heartbeat. Emits a lightweight `heartbeat`
    # event every 60s with current state.json snapshot + uptime. Gives
    # the HUD a live "last seen" timestamp so users can tell the
    # difference between "Jarvis is idle but alive" and "Jarvis is
    # frozen / stalled / fell off CoreAudio". Disable with
    # JARVIS_HEARTBEAT_DISABLE=true.
    if os.environ.get("JARVIS_HEARTBEAT_DISABLE", "").lower() not in (
        "1", "true", "yes", "on"
    ):
        _hb_started_at = time.time()

        def _heartbeat_loop():
            import json as _json
            import time as _t
            from pathlib import Path as _P
            state_path = _P.home() / "Library/Application Support/jarvis/state.json"
            from .ipc import get_stream as _gs
            stream = _gs()
            _t.sleep(10)  # let warmup land before the first beat
            while True:
                try:
                    snap = {}
                    if state_path.exists():
                        try:
                            snap = _json.loads(state_path.read_text(encoding="utf-8"))
                        except (_json.JSONDecodeError, OSError):
                            snap = {}
                    stream.emit(
                        "heartbeat",
                        state=str(snap.get("state", "UNKNOWN")),
                        level=float(snap.get("level", 0.0) or 0.0),
                        uptime_s=round(_t.time() - _hb_started_at, 1),
                        pid=os.getpid(),
                    )
                except Exception as exc:
                    # Never let observability kill the loop.
                    debug_log(f"heartbeat emit failed: {exc!r}", "daemon")
                _t.sleep(60)

        threading.Thread(
            target=_heartbeat_loop,
            name="heartbeat",
            daemon=True,
        ).start()

    # R34-S17: mic-RMS probe. PyAudio on macOS allows multiple readers
    # on the same input device — opening a second stream lets us sample
    # the real signal independently of Pipecat's audio callback. Emits
    # a `mic_probe` event every 30 s carrying the peak RMS over the
    # last 5 seconds. The user (and the dashboard) can finally tell:
    #   * RMS == 0          → daemon doesn't actually have mic permission
    #                         (or PortAudio routed to a silent device)
    #   * RMS > 0, no VAD   → Silero confidence threshold still too high
    #   * RMS > 0, VAD fires→ pipeline is reaching STT; wake-word filter
    #                         is the next suspect
    # Disable with JARVIS_MIC_PROBE_DISABLE=true (e.g. for unit tests
    # or single-mic devices that can't tolerate parallel readers).
    #
    # R34-S26 LESSON LEARNED: on macOS 26+ this probe HARMS the daemon
    # it tries to diagnose. The "PyAudio on macOS allows multiple
    # readers" assumption is FALSE for AUHAL exclusive-mode streams
    # when Continuity Camera/iPhone is in the device list — the second
    # open re-arbitrates AUHAL and silently kills Pipecat's input
    # callback. Symptom: pipecat_audio_rms events fire for ~17 s then
    # stop, wake-word never triggers, daemon "не викликається" even
    # though it's running. We now ship with JARVIS_MIC_PROBE_DISABLE=true
    # in the launchd plist. JarvisAudioProbeProcessor inside the Pipecat
    # pipeline emits pipecat_audio_rms from the SAME stream that VAD
    # sees — that's both safer and a more accurate diagnostic.
    if os.environ.get("JARVIS_MIC_PROBE_DISABLE", "").lower() not in (
        "1", "true", "yes", "on"
    ):
        def _mic_probe_loop():
            import time as _t
            try:
                import pyaudio  # noqa: F401
                import numpy as np  # noqa: F401
            except ImportError:
                debug_log("mic-probe: pyaudio/numpy missing", "daemon")
                return
            from .ipc import get_stream as _gs
            from .listening.pipecat_loop import _resolve_input_device_index
            stream = _gs()
            # Wait for the main pipeline to finish warmup so we don't
            # contend on PortAudio's setup phase.
            _t.sleep(20)
            # Resolve the same device the daemon's voice pipeline uses
            # (voice_device substring → numeric idx). If we can't, fall
            # back to PortAudio default — better than silently nothing.
            device_idx = _resolve_input_device_index(
                getattr(cfg, "input_device_index", None),
                getattr(cfg, "voice_device", None),
            )
            while True:
                try:
                    import pyaudio as _pa
                    import numpy as _np
                    pa = _pa.PyAudio()
                    try:
                        stm = pa.open(
                            format=_pa.paInt16,
                            channels=1,
                            rate=16000,
                            input=True,
                            frames_per_buffer=1600,
                            input_device_index=device_idx,
                        )
                        # Sample for 5 s, take peak RMS so a single
                        # syllable is enough to register.
                        peak = 0.0
                        n_frames = 0
                        for _ in range(50):  # 50 × 100 ms = 5 s
                            try:
                                data = stm.read(1600, exception_on_overflow=False)
                            except Exception:
                                break
                            arr = _np.frombuffer(data, dtype=_np.int16)
                            arr_f = arr.astype(_np.float32) / 32768.0
                            rms = float(_np.sqrt(_np.mean(arr_f * arr_f)))
                            if rms > peak:
                                peak = rms
                            n_frames += 1
                        stm.close()
                    finally:
                        pa.terminate()
                    # Resolve device name for the event.
                    try:
                        pa2 = _pa.PyAudio()
                        name = pa2.get_device_info_by_index(
                            device_idx if device_idx is not None
                            else pa2.get_default_input_device_info()["index"]
                        )["name"]
                        pa2.terminate()
                    except Exception:
                        name = "unknown"
                    stream.emit(
                        "mic_probe",
                        device_index=device_idx,
                        device_name=name,
                        peak_rms=round(peak, 5),
                        sampled_frames=n_frames,
                    )
                except Exception as exc:
                    debug_log(f"mic-probe failed: {exc!r}", "daemon")
                _t.sleep(30)

        threading.Thread(
            target=_mic_probe_loop,
            name="mic-probe",
            daemon=True,
        ).start()

    # Initialize dictation engine (hold-to-dictate)
    dictation = None
    if bool(getattr(cfg, "dictation_enabled", True)):
        try:
            from .dictation.dictation_engine import DictationEngine as _DE  # noqa: F811

            def _on_dictation_start():
                voice_thread._dictation_active = True
                _safe_set_face_state("DICTATING")
                debug_log("dictation started — listener paused", "dictation")

            def _on_dictation_processing_start():
                _safe_set_face_state("DICTATION_PROCESSING")
                debug_log("dictation processing started — transcribing captured audio", "dictation")

            def _on_dictation_end():
                voice_thread._dictation_active = False
                _safe_set_face_state("IDLE")
                debug_log("dictation ended — listener resumed", "dictation")

            dictation = _DE(
                whisper_model_ref=lambda: voice_thread.model,
                whisper_backend_ref=lambda: voice_thread._whisper_backend,
                mlx_repo_ref=lambda: voice_thread._mlx_model_repo,
                hotkey=cfg.dictation_hotkey,
                sample_rate=int(getattr(cfg, "sample_rate", 16000)),
                on_dictation_start=_on_dictation_start,
                on_dictation_processing_start=_on_dictation_processing_start,
                on_dictation_end=_on_dictation_end,
                transcribe_lock=voice_thread.transcribe_lock,
                voice_device=getattr(cfg, "voice_device", None),
                filler_removal=getattr(cfg, "dictation_filler_removal", False),
                custom_dictionary=getattr(cfg, "dictation_custom_dictionary", []),
                ollama_base_url=getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434"),
                ollama_model=cfg.ollama_chat_model,
                thinking=getattr(cfg, "dictation_thinking_enabled", False),
            )
            dictation.start()
            _global_dictation_engine = dictation
            if dictation._started:
                from jarvis.dictation.dictation_engine import format_hotkey_display
                hotkey_display = format_hotkey_display(cfg.dictation_hotkey)
                print(f"🎙️ Dictation enabled (hold {hotkey_display} to dictate)", flush=True)
        except Exception as e:
            debug_log(f"dictation engine init failed: {e}", "dictation")
            print(f"  ⚠ Dictation not available: {e}", flush=True)
    else:
        print("🎙️ Dictation disabled", flush=True)

    # Periodic diary update checking
    last_diary_check = time.time()
    diary_check_interval = 60.0

    # Start stdin monitor thread for Windows shutdown signal
    # On Windows, CTRL_BREAK_EVENT doesn't work reliably with CREATE_NO_WINDOW
    # So we also check for stdin being closed as a shutdown signal
    def stdin_monitor():
        global _global_stop_requested
        try:
            # Read until explicit SHUTDOWN command OR EOF. We do NOT
            # treat EOF alone as shutdown when running under launchd:
            # launchd attaches /dev/null to stdin → first readline()
            # returns "" immediately → daemon would self-kill at boot.
            # See I1-regression in round 8 audit.
            while True:
                line = sys.stdin.readline()
                if not line:  # EOF — thread exits silently, no stop signal
                    break
                line = line.strip()
                if line == "SHUTDOWN":
                    debug_log("SHUTDOWN command received, requesting stop", "jarvis")
                    _global_stop_requested = True
                    break
        except Exception:
            pass  # stdin might not be available

    # Audit round 8 fix I1: previously gated to win32 only. The
    # Electron HUD + desktop_app launcher both spawn the daemon as a
    # subprocess on macOS too, and send "SHUTDOWN\n" to stdin for a
    # clean exit. Without the monitor running on darwin, that
    # SHUTDOWN was ignored — daemon could only be killed via SIGTERM,
    # which raced with the blocking diary loop (see I2).
    #
    # I1-regression guard: only enable the monitor when stdin is a
    # PIPE (parent process writing to us). Under launchd stdin is
    # /dev/null (character device), under a terminal it's a tty —
    # in both cases there's no useful SHUTDOWN traffic to wait for.
    def _stdin_is_pipe() -> bool:
        try:
            import stat
            return stat.S_ISFIFO(os.fstat(0).st_mode)
        except Exception:
            return False
    if not getattr(sys, 'frozen', False) and _stdin_is_pipe():
        stdin_thread = threading.Thread(target=stdin_monitor, daemon=True)
        stdin_thread.start()
        debug_log("stdin SHUTDOWN monitor enabled (pipe detected)", "jarvis")

    try:
        # Main daemon loop
        while not _global_stop_requested:
            time.sleep(1.0)
            now = time.time()

            # Periodically check if diary should be updated.
            # Audit round 8 fix I2/I3 + round 12 fix: run the diary
            # update in a worker thread so blocking Ollama calls (can
            # be 30-180s) don't starve the SIGINT poll loop. Round 12
            # adds an is_alive() guard — if Ollama stalls beyond the
            # 60s check interval, the previous worker is still running,
            # and spawning a second one had two failure modes:
            #   1. Two threads with the same SQLite handle racing
            #      writes (database-locked spikes).
            #   2. After 30 stalled minutes you'd have 30 zombie
            #      diary threads each holding a partial summary in
            #      memory.
            if now - last_diary_check >= diary_check_interval:
                last_diary_check = now
                prev = globals().get("_diary_worker")
                if prev is not None and prev.is_alive():
                    debug_log(
                        "diary update skipped: previous worker still running",
                        "memory",
                    )
                else:
                    _t = threading.Thread(
                        target=_check_and_update_diary,
                        args=(db, cfg, False),
                        daemon=True,
                        name="diary-update",
                    )
                    globals()["_diary_worker"] = _t
                    _t.start()

        # Keep voice thread alive (unless stop requested).
        # Audit round 8 fix I3: this loop used to fire diary update
        # EVERY 0.5s with no interval gate — a busy-loop hammering
        # Ollama. Now it just sleeps, waiting for the voice thread or
        # global stop. Diary updates are scheduled by the main loop
        # above, never from here.
        if voice_thread is not None:
            while voice_thread.is_alive() and not _global_stop_requested:
                time.sleep(0.5)

    except KeyboardInterrupt:
        debug_log("daemon received KeyboardInterrupt", "jarvis")
    finally:
        print("🔄 Daemon shutting down - saving memory...", flush=True)
        debug_log("daemon finally block starting - performing cleanup", "jarvis")

        # Clean shutdown - stop dictation first
        if dictation is not None:
            debug_log("stopping dictation engine...", "jarvis")
            dictation.stop()
            debug_log("dictation engine stopped", "jarvis")

        if voice_thread is not None:
            debug_log("stopping voice thread...", "jarvis")
            voice_thread.stop()
            try:
                voice_thread.join(timeout=2.0)
            except Exception:
                pass
            debug_log("voice thread stopped", "jarvis")

        # Final diary update before shutdown
        debug_log("performing final diary update (force=True)...", "jarvis")
        print("📝 Updating diary before shutdown...", flush=True)

        # Check dialogue memory status
        if _global_dialogue_memory is None:
            print("⚠️ Dialogue memory is None - nothing to save", flush=True)
        else:
            # Display-only count; actual save uses the atomic snapshot path.
            pending = _global_dialogue_memory.get_pending_chunks()
            print(f"💬 Found {len(pending)} pending conversation chunks", flush=True)

        # Use callbacks if they were set by desktop app (for live UI updates in bundled mode)
        # Use IPC (stdout events) if callbacks not set (subprocess mode)
        use_callbacks = any(_diary_update_callbacks.values())
        use_ipc = not use_callbacks  # Subprocess mode - emit events to stdout
        _check_and_update_diary(db, cfg, verbose=True, force=True, timeout_sec=SHUTDOWN_DIARY_TIMEOUT_SEC, use_callbacks=use_callbacks, use_ipc=use_ipc)
        print("✅ Diary update complete", flush=True)
        debug_log("diary update complete", "jarvis")

        if tts is not None:
            tts.stop()

        # Tear down persistent MCP sessions so subprocess-launched
        # children (e.g. chrome-devtools-mcp's Chrome) close cleanly.
        try:
            from .tools.external.mcp_runtime import shutdown_runtime
            shutdown_runtime()
        except Exception as _e:
            debug_log(f"MCP runtime shutdown error: {_e}", "jarvis")

        db.close()

        # Drop the warm-profile graph listener so the module registry does
        # not retain a closure pointing at this run's DialogueMemory after
        # shutdown — relevant for tests and any embedder that re-runs the
        # daemon in-process.
        if _warm_profile_graph_listener is not None:
            try:
                from .memory.graph import unregister_graph_mutation_listener
                unregister_graph_mutation_listener(_warm_profile_graph_listener)
            except Exception:
                pass
            _warm_profile_graph_listener = None

        # Audit round 12 fix: shut down the action pool so the 4 worker
        # threads don't leak across in-process daemon restarts (test
        # runner, self-upgrade reload). `wait=False` matches the
        # fail-fast shutdown semantics — we don't want to block on a
        # stuck action.
        try:
            from .listening.action_dispatcher import shutdown_action_pool
            shutdown_action_pool(wait=False)
        except Exception:
            pass

        debug_log("daemon stopped", "jarvis")
        print("👋 Daemon stopped", flush=True)

        # Final typed event so HUD knows daemon went down cleanly
        # (vs SIGKILL — observers can distinguish absence-of-event).
        try:
            from .ipc import get_stream
            stream = get_stream()
            stream.emit("daemon_shutdown", reason="orderly")
            stream.disable()  # quiet rotations / writes after this point
        except Exception:
            pass

        # R34-S40 — explicit flush of the audit + facts stores BEFORE
        # _exit. Previous shutdown skipped atexit (to avoid torch race)
        # but that also skipped audit_store.close() → up to 100 events
        # lost on every shutdown. Drain them synchronously now.
        try:
            from .audit import get_audit_store as _gas
            _store = _gas()
            if hasattr(_store, "close"):
                _store.close()
        except Exception:
            pass
        try:
            from .memory.facts import get_facts_store as _gfs
            _facts = _gfs()
            if hasattr(_facts, "close"):
                _facts.close()
        except Exception:
            pass

        # Skip Python's normal exit path to avoid torch destructor race.
        # PyTorch's c10::Dispatcher::deregisterFallback_ has a known
        # SIGSEGV at interpreter shutdown when atexit handlers run in
        # the "wrong" order across libtorch_cpu.dylib and Python's
        # module teardown. We've already done our orderly cleanup
        # above (dictation, voice thread, diary, MCP, db) so it's
        # safe to bypass the rest of Python's teardown. _exit(0) calls
        # _exit(2) syscall directly — no Python finalizers, no torch
        # destructors, no segfault in crash logs.
        import os as _os
        _os.stdout.flush() if hasattr(_os, 'stdout') else None
        try:
            import sys as _sys
            _sys.stdout.flush()
            _sys.stderr.flush()
        except Exception:
            pass
        _os._exit(0)


if __name__ == "__main__":
    main()

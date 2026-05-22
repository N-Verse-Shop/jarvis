"""State management for listening modes (wake word, collection, hot window)."""

import time
import threading
from typing import Optional
from enum import Enum
from datetime import datetime

from ..debug import debug_log


def _elapsed(now: float, then: float) -> float:
    """Compute non-negative elapsed seconds between two wall-clock samples.

    Audit round 17 fix: ``time.time()`` is wall-clock and can jump
    BACKWARDS (NTP step after laptop wake, manual time change, DST
    transition on misconfigured systems). When that happens
    ``now - then`` goes negative, every ``>= timeout`` check becomes
    permanently False, and the collection/hot-window state machine
    refuses to time out until the user manually intervenes.

    Clamping to >= 0 means a backward jump LOOKS like "no time passed"
    which is the right semantics here: we'd rather hold collection
    open for a tick longer than have it never finalise. Forward jumps
    can fire timeouts early — the user just re-triggers, which is
    a much milder failure mode than a hung listener.

    All timeout checks in this module route through this helper so
    the wall-clock-jump posture is consistent.
    """
    diff = now - then
    return diff if diff > 0 else 0.0


class ListeningState(Enum):
    """Possible listening states."""
    WAKE_WORD = "wake_word"      # Waiting for wake word
    COLLECTING = "collecting"    # Accumulating query text
    HOT_WINDOW = "hot_window"    # Listening without wake word after TTS


class StateManager:
    """Manages listening state transitions and timing."""

    def __init__(self, hot_window_seconds: float = 3.0, echo_tolerance: float = 0.3,
                 voice_collect_seconds: float = 2.0, max_collect_seconds: float = 60.0,
                 hot_window_persistent: bool = False,
                 hot_window_max_session_seconds: float = 1800.0,
                 hot_window_max_idle_seconds: float = 180.0):
        """
        Initialize state manager.

        Args:
            hot_window_seconds: Duration of hot window listening
            echo_tolerance: Delay before activating hot window (for echo suppression)
            voice_collect_seconds: Silence timeout for query collection
            max_collect_seconds: Maximum time to collect a single query
            hot_window_persistent: If True, hot window never auto-expires;
                stays active until user says a stop command or right-clicks
                the HUD coin to "End session". Disables the expiry timer
                entirely and makes `_should_expire_hot_window()` always
                return False.
            hot_window_max_session_seconds: Absolute ceiling for a single
                persistent session. After this many seconds since the FIRST
                hot-window activation (without an intervening stop), the
                session force-ends regardless of activity. Safety net
                against runaway echo loops that "talk to themselves"
                forever. Default 1800 = 30 min. Only active when
                `hot_window_persistent=True`.
            hot_window_max_idle_seconds: Idle-timeout safety net for
                persistent sessions. If no real user wake-word has been
                heard for this long (echo/hallucination dispatches don't
                count), force-end the session. Default 180s = 3 min.
                Set 0 to disable. Only active when
                `hot_window_persistent=True`.
        """
        self.hot_window_seconds = hot_window_seconds
        self.echo_tolerance = echo_tolerance
        self.voice_collect_seconds = voice_collect_seconds
        self.max_collect_seconds = max_collect_seconds
        self.hot_window_persistent = hot_window_persistent
        self.hot_window_max_session_seconds = hot_window_max_session_seconds
        self.hot_window_max_idle_seconds = hot_window_max_idle_seconds

        # Current state
        self._state = ListeningState.WAKE_WORD
        self._state_lock = threading.Lock()

        # Collection state
        self._pending_query: str = ""
        self._last_voice_time: float = 0.0
        self._collect_start_time: float = 0.0

        # Hot window state
        self._hot_window_start_time: float = 0.0
        self._hot_window_span_start: float = 0.0  # When window span began (schedule time)
        self._hot_window_span_end: float = 0.0     # When window span ended (expiry time)

        # Persistent-session safety net state
        #   _session_started_at: timestamp of the first hot_window activation
        #     in a persistent session. Cleared on force_end_session. Used to
        #     enforce hot_window_max_session_seconds.
        #   _last_user_wake_at: timestamp of the most recent CONFIRMED user
        #     wake-word detection (not an echo/hallucination dispatch). Used
        #     to enforce hot_window_max_idle_seconds. Bumped via
        #     mark_user_wake().
        self._session_started_at: float = 0.0
        self._last_user_wake_at: float = 0.0

        # Timer-based hot window management
        self._hot_window_activation_timer: Optional[threading.Timer] = None
        self._hot_window_expiry_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()
        self._voice_debug: bool = False  # Cache for use in timer callbacks

        # Stop flag for background threads
        self._should_stop = False

    def get_state(self) -> ListeningState:
        """Get current listening state."""
        with self._state_lock:
            return self._state

    def is_collecting(self) -> bool:
        """Check if currently in collection mode."""
        return self.get_state() == ListeningState.COLLECTING

    def is_hot_window_active(self) -> bool:
        """Check if hot window is currently active."""
        return self.get_state() == ListeningState.HOT_WINDOW

    def start_collection(self, initial_text: str = "") -> None:
        """
        Start query collection mode.

        Args:
            initial_text: Optional initial text to seed the collection
        """
        with self._state_lock:
            self._state = ListeningState.COLLECTING
            self._pending_query = initial_text.strip()
            self._last_voice_time = time.time()
            self._collect_start_time = self._last_voice_time

        start_time_str = datetime.fromtimestamp(self._collect_start_time).strftime('%H:%M:%S.%f')[:-3]
        debug_log(f"collection started at {start_time_str}: '{initial_text}'", "state")

        # Set face state to LISTENING
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            face_state_manager = get_jarvis_state()
            face_state_manager.set_state(JarvisState.LISTENING)
            debug_log("face state set to LISTENING (collection started)", "state")
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"failed to set face state to LISTENING: {e}", "state")

    def add_to_collection(self, text: str) -> None:
        """
        Add text to current collection.

        Args:
            text: Text to append to pending query
        """
        if not self.is_collecting():
            return

        with self._state_lock:
            self._pending_query = (self._pending_query + " " + text).strip()
            self._last_voice_time = time.time()

        debug_log(f"added to collection: '{text}' -> '{self._pending_query}'", "state")

    def get_pending_query(self) -> str:
        """Get the current pending query text."""
        with self._state_lock:
            return self._pending_query

    def clear_collection(self) -> str:
        """
        Clear and return the current pending query.

        Returns:
            The query that was being collected
        """
        with self._state_lock:
            query = self._pending_query
            collect_start_time = self._collect_start_time
            self._pending_query = ""
            if self._state == ListeningState.COLLECTING:
                self._state = ListeningState.WAKE_WORD

        if query and collect_start_time > 0:
            end_time = time.time()
            duration = end_time - collect_start_time
            start_time_str = datetime.fromtimestamp(collect_start_time).strftime('%H:%M:%S.%f')[:-3]
            end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"collection cleared: '{query}' (started: {start_time_str}, ended: {end_time_str}, duration: {duration:.2f}s)", "state")
        else:
            debug_log(f"collection cleared: '{query}'", "state")

        # Note: Don't set face state here - it will be set to THINKING or ASLEEP by caller

        return query

    def check_collection_timeout(self) -> bool:
        """
        Check if collection should timeout due to silence or max duration.

        Returns:
            True if collection should be finalized
        """
        if not self.is_collecting():
            return False

        current_time = time.time()
        # Audit round 17 fix: route both elapsed-time computations
        # through ``_elapsed`` so a backward wall-clock jump cannot
        # wedge the timeout permanently.
        silence_timeout = _elapsed(current_time, self._last_voice_time) >= self.voice_collect_seconds
        max_timeout = _elapsed(current_time, self._collect_start_time) >= self.max_collect_seconds

        if silence_timeout or max_timeout:
            timeout_type = "silence" if silence_timeout else "max"

            end_time = time.time()
            duration = end_time - self._collect_start_time
            start_time_str = datetime.fromtimestamp(self._collect_start_time).strftime('%H:%M:%S.%f')[:-3]
            end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]

            debug_log(f"collection timeout ({timeout_type}): '{self._pending_query}' (started: {start_time_str}, ended: {end_time_str}, duration: {duration:.2f}s)", "state")
            return True

        return False

    def force_collection_timeout(self) -> None:
        """Force the collection timeout to fire on the next check.

        Audit round 24 (F47): used by the HUD ``force_finalize``
        control action when the user presses ✓ while we're in the
        middle of collecting (waiting for them to keep speaking).
        Mutates ``_last_voice_time`` to a value far enough in the
        past that ``check_collection_timeout`` returns True on its
        very next call — which the voice loop polls every iteration.
        Effect: the dispatch fires immediately instead of waiting
        out ``voice_collect_seconds`` of additional silence.
        """
        with self._state_lock:
            if self._state != ListeningState.COLLECTING:
                debug_log("force_collection_timeout: not collecting — no-op", "state")
                return
            # Drag _last_voice_time backwards by the full silence
            # window + 1s headroom. Next check_collection_timeout
            # will see elapsed >= voice_collect_seconds and return
            # True.
            self._last_voice_time = time.time() - (self.voice_collect_seconds + 1.0)
            debug_log(
                f"force_collection_timeout: pulled _last_voice_time back so next "
                f"check fires immediately (pending='{self._pending_query[:60]}')",
                "state",
            )

    def was_speech_during_hot_window(self, utterance_start_time: float,
                                     utterance_end_time: float = 0.0) -> bool:
        """Check if speech overlapped with the hot window time span.

        Uses timestamps instead of a mutable boolean flag. This eliminates
        race conditions between the hot window expiry timer and slow Whisper
        transcription — the check works regardless of when the transcript arrives.

        Args:
            utterance_start_time: When VAD detected voice onset (time.time()).
                                  If 0, falls back to current state check.
            utterance_end_time: When the utterance ended (time.time()).
                                Used to detect overlap when the utterance started
                                before the span (e.g. mic picked up TTS echo)
                                but extended into the hot window period.

        Returns:
            True if:
            - Hot window is currently active, OR
            - Hot window activation is pending (echo_tolerance delay), OR
            - Speech started during the window span (even if window has since expired)
            - Speech started before the span but ended during it (overlap)
        """
        with self._state_lock:
            is_active = self._state == ListeningState.HOT_WINDOW
            span_start = self._hot_window_span_start
            span_end = self._hot_window_span_end

        with self._timer_lock:
            is_pending = self._hot_window_activation_timer is not None

        # Currently active — always accept regardless of timing
        if is_active:
            return True

        # No timestamp — refuse to speculate. Previously we returned
        # `is_pending` here, which let Whisper hallucination chunks
        # ("ага", "так", "субтитры") that arrive without a real
        # utterance_start_time get accepted as hot-window follow-ups
        # during the activation_timer's echo_tolerance window. That
        # was a direct trigger for "Jarvis responds to itself" — a
        # hallucinated chunk had no timestamp → treated as in-window
        # → fed to intent judge → judge sometimes said directed=true.
        # Audit round 6 fix: require an actual timing signal.
        if utterance_start_time <= 0:
            return False

        # Pending activation — accept if speech started after scheduling
        if is_pending:
            return span_start <= 0 or utterance_start_time >= span_start

        # Window expired — accept if speech overlapped with the span
        # This handles two cases:
        # 1. Speech started within the span (normal hot window follow-up)
        # 2. Speech started before the span but ended during it (mic picked up
        #    TTS echo during playback, then user spoke during hot window —
        #    Whisper merges both into one chunk)
        if span_start > 0 and span_end > 0:
            if span_start <= utterance_start_time <= span_end:
                return True
            if (utterance_end_time > 0
                    and utterance_start_time < span_start
                    and utterance_end_time >= span_start):
                debug_log(
                    f"utterance overlaps hot window span "
                    f"(start={utterance_start_time:.2f} < span_start={span_start:.2f}, "
                    f"end={utterance_end_time:.2f} >= span_start)", "state"
                )
                return True

        return False

    def cancel_hot_window_activation(self) -> None:
        """Cancel any pending hot window activation timer.

        Call this when user starts a new query to prevent delayed activation
        from interfering with the current interaction.
        """
        with self._timer_lock:
            if self._hot_window_activation_timer is not None:
                self._hot_window_activation_timer.cancel()
                self._hot_window_activation_timer = None
                debug_log("cancelled pending hot window activation", "state")

    def force_end_session(self) -> None:
        """Force-exit hot window and return to wake-word listening.

        Called when user says a stop command ("стоп", "досить", "тиша")
        OR clicks "End session" in the HUD right-click menu. Works even
        when `hot_window_persistent=True` — this is the ONLY way to leave
        a persistent session.

        Cancels both activation and expiry timers, flips state to
        WAKE_WORD, and pushes the HUD to IDLE.
        """
        self.cancel_hot_window_activation()
        self._cancel_hot_window_expiry_timer()
        with self._state_lock:
            was_hot = (self._state == ListeningState.HOT_WINDOW)
            self._state = ListeningState.WAKE_WORD
            self._hot_window_span_end = time.time()
            # Clear persistent-session safety state — next activation
            # starts a fresh session timer.
            self._session_started_at = 0.0
            self._last_user_wake_at = 0.0
        debug_log(f"session force-ended (was_hot={was_hot})", "state")

        # Push HUD to IDLE so the coin disappears.
        try:
            from desktop_app.face_widget import get_jarvis_state, JarvisState
            get_jarvis_state().set_state(JarvisState.IDLE)
        except ImportError:
            pass
        except Exception as e:
            debug_log(f"force_end_session: failed to set HUD IDLE: {e}", "state")
        try:
            print("💤 Session ended — say 'Джарвис' to start again\n", flush=True)
        except Exception:
            pass

    def _cancel_hot_window_expiry_timer(self) -> None:
        """Cancel the hot window expiry timer."""
        with self._timer_lock:
            if self._hot_window_expiry_timer is not None:
                self._hot_window_expiry_timer.cancel()
                self._hot_window_expiry_timer = None

    def reset_hot_window_expiry(self) -> None:
        """Reset the hot window expiry timer to give the user the full window.

        Called when echo is rejected during the hot window, so the time spent
        processing echo doesn't eat into the user's actual follow-up window.

        If the hot window already expired while the echo was being transcribed,
        this reactivates it — the user shouldn't lose their follow-up window
        just because Whisper was slow to produce the echo transcript.
        """
        with self._state_lock:
            if self._state == ListeningState.HOT_WINDOW:
                # Still active — just reset the timer
                self._hot_window_start_time = time.time()
            elif self._state == ListeningState.WAKE_WORD:
                # Expired while processing echo — reactivate
                self._state = ListeningState.HOT_WINDOW
                self._hot_window_start_time = time.time()
                debug_log("hot window reactivated (expired during echo processing)", "state")
                try:
                    print(f"👂 Listening for follow-up ({int(self.hot_window_seconds)}s)...", flush=True)
                except Exception:
                    pass
            else:
                # COLLECTING or another active state — don't interfere
                return

        self._schedule_hot_window_expiry()
        debug_log(f"hot window expiry reset (echo rejected, restarting {self.hot_window_seconds}s timer)", "state")

    def _schedule_hot_window_expiry(self) -> None:
        """Schedule hot window expiry timer.

        This timer guarantees expiry will fire even if no audio is being processed.

        Skips entirely when `hot_window_persistent=True` — user explicitly
        asked for sessions that stay alive until manually terminated.
        """
        self._cancel_hot_window_expiry_timer()
        if self.hot_window_persistent:
            debug_log("hot window expiry skipped (persistent mode)", "state")
            return

        def _expire():
            with self._state_lock:
                if self._state != ListeningState.HOT_WINDOW:
                    return
                self._state = ListeningState.WAKE_WORD
                self._hot_window_span_end = time.time()

            expiry_time = self._hot_window_span_end
            duration = expiry_time - self._hot_window_start_time if self._hot_window_start_time > 0 else 0
            expiry_time_str = datetime.fromtimestamp(expiry_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"hot window expired (timer) at {expiry_time_str} after {duration:.2f}s", "state")

            # Set face state to IDLE
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.IDLE)
                debug_log("face state set to IDLE (hot window timer expiry)", "state")
            except ImportError:
                # Desktop app not available (headless mode)
                pass
            except Exception as e:
                debug_log(f"failed to set face state to IDLE: {e}", "state")

            # Always show user-facing output
            try:
                print("💤 Returning to wake word mode\n", flush=True)
            except Exception:
                pass

        with self._timer_lock:
            self._hot_window_expiry_timer = threading.Timer(self.hot_window_seconds, _expire)
            self._hot_window_expiry_timer.daemon = True
            self._hot_window_expiry_timer.start()

        debug_log(f"scheduled hot window expiry in {self.hot_window_seconds}s", "state")

    def schedule_hot_window_activation(self, voice_debug: bool = False) -> None:
        """
        Schedule hot window activation after echo tolerance delay.

        Uses threading.Timer for reliable activation instead of daemon thread + sleep.

        Args:
            voice_debug: Whether to enable debug logging
        """
        schedule_time_str = datetime.fromtimestamp(time.time()).strftime('%H:%M:%S.%f')[:-3]
        debug_log(f"scheduling hot window activation at {schedule_time_str} (delay={self.echo_tolerance}s, should_stop={self._should_stop})", "state")

        # Cancel any pending activation first
        self.cancel_hot_window_activation()

        # Start a new window span — reset end so old expired spans don't interfere
        with self._state_lock:
            self._hot_window_span_start = time.time()
            self._hot_window_span_end = 0.0

        # Cache voice_debug for use in timer callbacks
        self._voice_debug = voice_debug

        def _activate():
            # Clear the timer reference now that it's fired
            with self._timer_lock:
                self._hot_window_activation_timer = None

            # Check if we should still activate
            if self._should_stop:
                debug_log("hot window activation cancelled (should_stop=True)", "state")
                return

            with self._state_lock:
                # Don't overwrite COLLECTING state - user may have already started a new query
                if self._state == ListeningState.COLLECTING:
                    debug_log("hot window activation cancelled (already collecting)", "state")
                    return
                self._state = ListeningState.HOT_WINDOW
                self._hot_window_start_time = time.time()
                # Stamp session start on the FIRST activation; subsequent
                # re-activations within the same persistent session don't
                # reset the clock — that's the whole point of the ceiling.
                if self.hot_window_persistent and self._session_started_at <= 0.0:
                    self._session_started_at = self._hot_window_start_time
                    # Treat session start as the first "user wake" so the
                    # idle timer doesn't fire immediately.
                    if self._last_user_wake_at <= 0.0:
                        self._last_user_wake_at = self._hot_window_start_time
                    debug_log(
                        f"persistent session started "
                        f"(max_age={self.hot_window_max_session_seconds:.0f}s, "
                        f"max_idle={self.hot_window_max_idle_seconds:.0f}s)",
                        "state",
                    )

            activation_time_str = datetime.fromtimestamp(self._hot_window_start_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"hot window activated at {activation_time_str} for {self.hot_window_seconds}s (after {self.echo_tolerance}s echo delay)", "state")

            # Set face state to LISTENING
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.LISTENING)
                debug_log("face state set to LISTENING (hot window activated)", "state")
            except ImportError:
                pass
            except Exception as e:
                debug_log(f"failed to set face state to LISTENING: {e}", "state")

            # Always show user-facing output
            try:
                print(f"👂 Listening for follow-up ({int(self.hot_window_seconds)}s)...", flush=True)
            except Exception as e:
                debug_log(f"failed to print hot window message: {e}", "state")

            # Schedule the expiry timer now that hot window is active
            self._schedule_hot_window_expiry()

        # Use Timer for more reliable activation
        with self._timer_lock:
            self._hot_window_activation_timer = threading.Timer(self.echo_tolerance, _activate)
            self._hot_window_activation_timer.daemon = True
            self._hot_window_activation_timer.start()

        debug_log("hot window activation timer started", "state")

    def mark_user_wake(self) -> None:
        """Record that a real user wake-word was just detected.

        Bumps the idle-safety timestamp AND resets the session-age
        timestamp so a long healthy session never hits the max-session
        force-end during active conversation. Echo loops and Whisper
        hallucinations don't call this — only confirmed wake-word matches
        (wake_detection.is_wake_word_detected returning True) and
        successful hot-window dispatches that the intent judge confirmed
        are directed at Jarvis.

        Audit round 6 fix: previously only bumped `_last_user_wake_at`.
        That meant `max_session_seconds` (1800s) could fire DURING a
        healthy multi-turn conversation just because the session had
        been running for 30 minutes total. Now both timers reset on
        confirmed wake — `max_session` becomes a "no wake event for 30
        min straight" backstop instead of a 30-min wall-clock ceiling.
        """
        now = time.time()
        with self._state_lock:
            self._last_user_wake_at = now
            # Reset session-age too — only when persistent mode is on
            # AND a session is currently in flight. Outside persistent
            # mode `_session_started_at` is always 0.
            if self.hot_window_persistent and self._session_started_at > 0:
                self._session_started_at = now

    def _persistent_session_should_force_end(self) -> tuple[bool, str]:
        """Check the persistent-session safety nets.

        Returns (should_end, reason). Only applies when persistent=True
        AND the hot window is currently active AND the session has been
        started. Reason is a short tag for debug logs.

        Audit round 12 fix: snapshot ``_session_started_at`` and
        ``_last_user_wake_at`` under ``_state_lock`` into locals so a
        concurrent ``force_end_session()`` (which resets them to 0)
        can't slip between the `<= 0.0` guard and the subtraction. The
        prior unlocked read could see ``_session_started_at`` reset
        mid-check, producing a huge ``session_age`` (≈ now) and
        force-ending a freshly-cleared session by accident.
        """
        if not self.hot_window_persistent:
            return (False, "")
        if not self.is_hot_window_active():
            return (False, "")

        with self._state_lock:
            session_started_at = self._session_started_at
            last_user_wake_at = self._last_user_wake_at
        if session_started_at <= 0.0:
            return (False, "")

        now = time.time()
        # Audit round 17 fix: backward wall-clock jumps would make
        # ``session_age``/``idle_age`` negative, defeating the safety
        # nets and letting a runaway echo loop persist past either
        # ceiling. ``_elapsed`` clamps to 0 — equivalent to "no time
        # passed" for the duration of the jump.
        session_age = _elapsed(now, session_started_at)
        if self.hot_window_max_session_seconds > 0 and \
                session_age >= self.hot_window_max_session_seconds:
            return (True, f"max_age={session_age:.0f}s>="
                          f"{self.hot_window_max_session_seconds:.0f}s")

        if self.hot_window_max_idle_seconds > 0 and last_user_wake_at > 0:
            idle_age = _elapsed(now, last_user_wake_at)
            if idle_age >= self.hot_window_max_idle_seconds:
                return (True, f"max_idle={idle_age:.0f}s>="
                              f"{self.hot_window_max_idle_seconds:.0f}s")

        return (False, "")

    def _should_expire_hot_window(self) -> bool:
        """Check if hot window should expire due to timeout.

        Note: With timer-based expiry, this is now mainly a fallback check.
        The timer should handle expiry automatically.

        In persistent mode, the normal hot_window_seconds expiry is
        disabled, BUT the two safety nets (max_session, max_idle) can
        still trigger a force-end here. This is the runaway-echo-loop
        kill switch.
        """
        if self.hot_window_persistent:
            should_end, reason = self._persistent_session_should_force_end()
            if should_end:
                debug_log(
                    f"persistent session hit safety ceiling ({reason}) — force-ending",
                    "state",
                )
                # We mutate state outside the lock; the wrapper
                # check_hot_window_expiry() will handle the cleanup.
                return True
            return False
        if not self.is_hot_window_active():
            return False
        current_time = time.time()
        # Audit round 17 fix: route through ``_elapsed`` so a
        # backward wall-clock jump cannot keep the hot window
        # alive past its real timeout window.
        return _elapsed(current_time, self._hot_window_start_time) >= self.hot_window_seconds

    def check_hot_window_expiry(self, voice_debug: bool = False) -> bool:
        """
        Check and handle hot window expiry.

        Note: With timer-based expiry, this is now a fallback check.
        The timer should handle expiry automatically, but this method
        provides a synchronous check for the main audio processing loop.

        Args:
            voice_debug: Whether to enable debug logging

        Returns:
            True if hot window was expired
        """
        if self._should_expire_hot_window():
            # Persistent-mode safety-net trip needs the FULL cleanup path
            # (clears _session_started_at, pushes HUD to IDLE, prints the
            # user-visible "session ended" message). Going through
            # force_end_session is the only way to leave a persistent
            # session cleanly.
            if self.hot_window_persistent:
                self.force_end_session()
                return True

            # Cancel expiry timer since we're handling it here
            self._cancel_hot_window_expiry_timer()

            with self._state_lock:
                self._state = ListeningState.WAKE_WORD
                self._hot_window_span_end = time.time()

            debug_log("hot window expired (poll)", "state")

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.IDLE)
                debug_log("face state set to IDLE (hot window poll expiry)", "state")
            except ImportError:
                pass
            except Exception as e:
                debug_log(f"failed to set face state to IDLE: {e}", "state")

            # Always show user-facing output
            try:
                print("💤 Returning to wake word mode\n", flush=True)
            except Exception:
                pass

            return True
        return False

    def expire_hot_window(self, voice_debug: bool = False) -> None:
        """
        Manually expire the hot window.

        Args:
            voice_debug: Whether to enable debug logging
        """
        # R34-S58.3 C3.1: persistent mode opts out of all auto-expiry —
        # matches the guard in `_schedule_hot_window_expiry` and
        # `_should_expire_hot_window`. Without this, an external caller
        # invoking expire_hot_window() (e.g. stop-command path) would
        # close a window the user explicitly asked to stay open.
        if getattr(self, "hot_window_persistent", False):
            debug_log("expire_hot_window: skipped, persistent mode", "state")
            return
        # Cancel expiry timer since we're manually expiring
        self._cancel_hot_window_expiry_timer()

        if self.is_hot_window_active():
            with self._state_lock:
                self._state = ListeningState.WAKE_WORD
                self._hot_window_span_end = time.time()

            debug_log("hot window manually expired", "state")

            # Set face state to IDLE (awake and ready, waiting for wake word)
            try:
                from desktop_app.face_widget import get_jarvis_state, JarvisState
                face_state_manager = get_jarvis_state()
                face_state_manager.set_state(JarvisState.IDLE)
                debug_log("face state set to IDLE (hot window manually expired)", "state")
            except ImportError:
                pass
            except Exception as e:
                debug_log(f"failed to set face state to IDLE: {e}", "state")

            # Always show user-facing output
            try:
                print("💤 Returning to wake word mode", flush=True)
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the state manager and cancel all timers."""
        self._should_stop = True

        # Cancel all timers
        self.cancel_hot_window_activation()
        self._cancel_hot_window_expiry_timer()

        with self._state_lock:
            self._state = ListeningState.WAKE_WORD

"""Common types and result classes for tools.

Audit round 12 fix: previous version had no contract — tool
implementations split four ways on (success, reply_text, error_message)
and the engine read ``if result.reply_text:`` as the success signal,
which silently masked failures from tools that returned
``success=False, reply_text=<error message>``. ``ToolExecutionResult``
now enforces the invariant via ``__post_init__`` and the
``user_text`` / ``error_text`` accessor properties give the engine one
canonical place to read the message regardless of which legacy code
path produced the result.

Contract:
  * ``success=True``  ⇒ ``reply_text`` MUST be non-empty,
    ``error_message`` MUST be empty/None.
  * ``success=False`` ⇒ ``error_message`` MUST be non-empty;
    ``reply_text`` SHOULD be None (legacy code that stuffs the error
    into ``reply_text`` is auto-migrated by ``__post_init__`` so the
    engine sees a consistent shape).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolExecutionResult:
    """Result object for tool execution."""
    success: bool
    reply_text: Optional[str]
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        # Auto-migrate legacy `success=False, reply_text=<error>` shape
        # into the canonical `success=False, error_message=<error>`
        # shape. This lets us land the contract incrementally without
        # touching every tool implementation in lockstep — the engine
        # reads ``error_text`` and gets the right thing regardless.
        if not self.success and not (self.error_message or "").strip():
            if (self.reply_text or "").strip():
                self.error_message = self.reply_text
                self.reply_text = None

    @property
    def user_text(self) -> str:
        """Non-error text intended for the user / LLM. Empty on failure."""
        if self.success:
            return self.reply_text or ""
        return ""

    @property
    def error_text(self) -> str:
        """Human-readable failure reason. Empty when ``success`` is True."""
        if self.success:
            return ""
        return (self.error_message or self.reply_text or "").strip()

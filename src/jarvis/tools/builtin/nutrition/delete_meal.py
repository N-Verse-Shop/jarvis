"""Delete meal tool for nutrition tracking."""

from typing import Dict, Any, Optional, Callable

from ....debug import debug_log
from ...base import Tool, ToolContext
from ...types import ToolExecutionResult


class DeleteMealTool(Tool):
    """Tool for deleting meals from the nutrition database."""
    
    @property
    def name(self) -> str:
        return "deleteMeal"
    
    @property
    def description(self) -> str:
        return "Delete a meal from the nutrition database by ID."
    
    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "ID of the meal to delete"}
            },
            "required": ["id"]
        }
    
    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the delete meal tool."""
        context.user_print("🗑️ Deleting the meal…")
        mid = None
        if args and isinstance(args, dict):
            try:
                mid = int(args.get("id"))
            except Exception:
                mid = None
        is_deleted = False
        if mid is not None:
            try:
                is_deleted = context.db.delete_meal(mid)
            except Exception:
                is_deleted = False
        debug_log(f"DELETE_MEAL: id={mid} deleted={is_deleted}", "nutrition")
        # Audit round 13 fix: round 12's types.py auto-migration moves
        # ``success=False, reply_text=<text>`` into ``error_message``,
        # which means the engine no longer surfaces the polite "Sorry"
        # message as a tool result — it appears as an error and the
        # LLM gets back "(no result)". Split the two cases explicitly
        # so the failure path passes the human message through
        # ``error_message=`` (the engine reads ``error_text`` for it).
        if is_deleted:
            context.user_print("✅ Meal deleted.")
            return ToolExecutionResult(success=True, reply_text="Meal deleted.")
        context.user_print("⚠️ I couldn't delete that meal.")
        return ToolExecutionResult(
            success=False,
            reply_text=None,
            error_message=(
                "Could not delete meal "
                f"id={mid if mid is not None else '(missing)'} — "
                "row not found or DB error. Ask the user to confirm the id."
            ),
        )

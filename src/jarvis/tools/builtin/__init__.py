"""Builtin tools module.

This module contains all the built-in tools available to the Jarvis system.
Each tool is implemented using the common Tool interface for consistency.
"""

# Import all tool classes
from .screenshot import ScreenshotTool
from .web_search import WebSearchTool
from .local_files import LocalFilesTool
from .fetch_web_page import FetchWebPageTool
from .nutrition.log_meal import LogMealTool
from .nutrition.fetch_meals import FetchMealsTool
from .nutrition.delete_meal import DeleteMealTool
from .weather import WeatherTool
from .stop import StopTool
# Audit round 20 P3 — Mac-control capability.
from .mac_control import MacControlTool
# R35-S1 — n8n workflow automation tool.
from .n8n import N8NTool
# R35-S7 — Claude Code bridge (delegates dev tasks to subscription, not API).
from .claude_bridge import ClaudeCodeSpawnTool

# Import supporting functions that may still be used elsewhere

__all__ = [
    # Tool classes
    'ScreenshotTool',
    'WebSearchTool',
    'LocalFilesTool',
    'FetchWebPageTool',
    'LogMealTool',
    'FetchMealsTool',
    'DeleteMealTool',
    'WeatherTool',
    'StopTool',
    'MacControlTool',
    'N8NTool',
    'ClaudeCodeSpawnTool',
]

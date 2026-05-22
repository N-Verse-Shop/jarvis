"""External-service integrations for Jarvis.

R35-S1 onwards. Each integration lives in its own module exposing a
small client class with explicit auth + scrubbed error handling.

Currently:
- ``n8n_client`` — REST client for self-hosted n8n at
  ``n8n.nexus-studio-innovation.com`` (or any compatible instance).
  Authenticates via ``JARVIS_N8N_API_KEY`` env var.
"""

from .n8n_client import (
    N8NClient,
    N8NError,
    N8NAuthError,
    N8NNotConfiguredError,
    get_n8n_client,
)

__all__ = [
    "N8NClient",
    "N8NError",
    "N8NAuthError",
    "N8NNotConfiguredError",
    "get_n8n_client",
]

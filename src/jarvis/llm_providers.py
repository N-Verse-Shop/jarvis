"""Multi-provider LLM router for Jarvis voice path.

R35-S25. NVIDIA NIM alone hits TTFT 10s+ on cold-cache; running 3
OpenAI-compatible free providers in a tier ladder gives consistent
0.2-3s replies because at least ONE provider is always warm.

Providers (all free tier, OpenAI-compatible):

  ┌─────────────┬──────────────────────────────────┬─────────────┬─────────────────────┐
  │ provider    │ endpoint                         │ free tier   │ best model          │
  ├─────────────┼──────────────────────────────────┼─────────────┼─────────────────────┤
  │ NVIDIA NIM  │ integrate.api.nvidia.com         │ ~5K cred/mo │ llama-3.3-70b       │
  │ Groq        │ api.groq.com                     │ 30 RPM      │ llama-3.3-70b       │
  │ Cerebras    │ api.cerebras.ai                  │ 1M tok/day  │ llama-3.3-70b       │
  │ SambaNova   │ api.sambanova.ai                 │ 1M tok/day  │ llama-3.3-70b       │
  └─────────────┴──────────────────────────────────┴─────────────┴─────────────────────┘

Routing order (sequential, on TTFT priority):
  Groq (fastest) → Cerebras → NVIDIA NIM → SambaNova → Hetzner Ollama

Activation: drop the corresponding env var in ``~/.config/jarvis/.env``.
A provider with no key is silently skipped (returns ``None`` from
``call_streaming``). Tier-ladder picks next in line.

Privacy gate: if user prompt matches a privacy keyword
(``config.privacy_keywords``), ALL cloud providers are skipped and the
turn falls straight to Hetzner Ollama (EU-only).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .debug import debug_log


@dataclass(frozen=True)
class _Provider:
    """OpenAI-compatible LLM provider descriptor."""
    name: str
    base_url: str            # without trailing /chat/completions
    env_key_var: str
    default_model: str


# Order = priority. Groq is FASTEST in production (300+ tok/s),
# Cerebras second (250 tok/s), NIM third (50 tok/s). SambaNova last
# because the free tier shows occasional outages.
PROVIDERS: Tuple[_Provider, ...] = (
    _Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        env_key_var="JARVIS_GROQ_KEY",
        default_model="llama-3.3-70b-versatile",
    ),
    _Provider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        env_key_var="JARVIS_CEREBRAS_KEY",
        # R35-S26: Cerebras free tier only exposes gpt-oss-120b + zai-glm-4.7
        # (no Llama 3.3 on personal tier). gpt-oss-120b benchmarks: TTFT
        # 21ms total on second call (basically instant). Wins the ladder.
        default_model="gpt-oss-120b",
    ),
    _Provider(
        name="nim",
        base_url="https://integrate.api.nvidia.com/v1",
        env_key_var="JARVIS_NVIDIA_NIM_KEY",
        default_model="meta/llama-3.3-70b-instruct",
    ),
    _Provider(
        name="sambanova",
        base_url="https://api.sambanova.ai/v1",
        env_key_var="JARVIS_SAMBANOVA_KEY",
        default_model="Meta-Llama-3.3-70B-Instruct",
    ),
)


# R35-S29: Mistral AI — EU-hosted (Paris), GDPR-compliant, OpenAI-compatible.
# Deliberately NOT in the PROVIDERS ladder above: that ladder serves
# NON-private turns where Groq (US, 0.3s) is the right default. Mistral is
# reserved for the PRIVACY path — client-sensitive turns (IBONS/Werkvertrag)
# that must avoid US clouds. Its EU GPU API answers in ~0.5-1s (vs the
# self-hosted Hetzner CPU box's 40-100s). Activated by ``JARVIS_MISTRAL_KEY``.
# Data residency note: this sends the (keyword-flagged) prompt to Mistral,
# a third-party EU processor — more private than US clouds, less private
# than the fully self-hosted Hetzner Ollama. The privacy_llm config knob
# chooses between them.
MISTRAL_PROVIDER: _Provider = _Provider(
    name="mistral",
    base_url="https://api.mistral.ai/v1",
    env_key_var="JARVIS_MISTRAL_KEY",
    default_model="mistral-small-latest",
)


def _get_key(provider: _Provider) -> Optional[str]:
    """Read API key for ``provider`` from environment. None if absent."""
    val = os.environ.get(provider.env_key_var, "").strip()
    return val or None


def list_enabled_providers() -> List[str]:
    """Return names of providers with non-empty API keys in env.

    Useful for ``/jarvis doctor`` to show which tiers are live.
    """
    return [p.name for p in PROVIDERS if _get_key(p) is not None]


def call_provider_streaming(
    provider: _Provider,
    chat_model: str,
    system_prompt: str,
    user_content: str,
    messages_history: Optional[List[Dict[str, str]]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    timeout_sec: float = 8.0,
    temperature: float = 0.3,
    max_tokens: int = 120,
    abort_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """Stream a chat completion from a single OpenAI-compatible provider.

    Returns the assembled reply text on success, ``None`` on auth /
    rate-limit / network / parse error. The caller falls back to the
    next tier.

    Timeout is per-provider; default 8 s is the right balance — if a
    provider hasn't returned the first token in 8 s it's cold and the
    next tier will likely serve faster.
    """
    key = _get_key(provider)
    if not key:
        return None

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if messages_history:
        messages.extend(messages_history)
    messages.append({"role": "user", "content": user_content})

    payload: Dict[str, Any] = {
        "model": chat_model,
        "messages": messages,
        "stream": True,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    # R35-S26: gpt-oss-* and nemotron-3-super-* default to high reasoning
    # budget which eats the user-visible output budget. Force low so the
    # final answer comes back in standard ``content`` within max_tokens.
    # Llama / Mistral / etc. ignore this param silently.
    if "gpt-oss" in chat_model.lower() or "nemotron-3-super" in chat_model.lower():
        payload["reasoning_effort"] = "low"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    url = f"{provider.base_url}/chat/completions"
    _connect_timeout = min(3.0, max(1.5, float(timeout_sec)))

    chunks: List[str] = []
    started = time.monotonic()
    try:
        with requests.post(
            url, json=payload, headers=headers,
            timeout=(_connect_timeout, timeout_sec),
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                debug_log(
                    f"{provider.name}: HTTP {resp.status_code} {resp.text[:100]}",
                    "llm",
                )
                return None
            for line in resp.iter_lines(decode_unicode=False):
                if abort_event is not None and abort_event.is_set():
                    return None
                if not line or line.startswith(b":"):
                    continue
                if line.startswith(b"data: "):
                    payload_str = line[6:].decode("utf-8", errors="replace")
                else:
                    continue
                if payload_str.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                tok = delta.get("content")
                if tok:
                    chunks.append(tok)
                    if on_token is not None:
                        try:
                            on_token(tok)
                        except Exception:
                            pass
    except requests.Timeout:
        elapsed = time.monotonic() - started
        debug_log(f"{provider.name}: timeout after {elapsed:.1f}s", "llm")
        return None
    except requests.RequestException as exc:
        debug_log(f"{provider.name}: {type(exc).__name__}: {exc}", "llm")
        return None

    full = "".join(chunks).strip()
    return full or None


def call_tier_ladder(
    system_prompt: str,
    user_content: str,
    messages_history: Optional[List[Dict[str, str]]] = None,
    on_token: Optional[Callable[[str], None]] = None,
    per_provider_timeout_sec: float = 8.0,
    temperature: float = 0.3,
    max_tokens: int = 120,
    abort_event: Optional[threading.Event] = None,
    model_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Try each enabled provider in order. Return (reply, provider_name).

    First provider with non-empty key + 200 OK + non-empty content wins.
    A failure (no key, auth, timeout, 5xx, empty body) silently advances
    to the next tier. If ALL providers fail, returns ``(None, None)``
    and the caller should fall back to Hetzner Ollama.

    Args:
        system_prompt: System message
        user_content: User message
        messages_history: Optional prior turns (Ollama-format dict list)
        on_token: Optional token callback for streaming TTS
        per_provider_timeout_sec: Max wait per provider before advancing
        temperature: Sampling temperature
        max_tokens: Output cap
        abort_event: Optional Event to short-circuit mid-stream
        model_overrides: Optional ``{provider_name: model_id}`` map
    """
    for provider in PROVIDERS:
        key = _get_key(provider)
        if not key:
            continue
        model = (model_overrides or {}).get(provider.name, provider.default_model)
        reply = call_provider_streaming(
            provider=provider,
            chat_model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            messages_history=messages_history,
            on_token=on_token,
            timeout_sec=per_provider_timeout_sec,
            temperature=temperature,
            max_tokens=max_tokens,
            abort_event=abort_event,
        )
        if reply:
            return reply, provider.name
    return None, None

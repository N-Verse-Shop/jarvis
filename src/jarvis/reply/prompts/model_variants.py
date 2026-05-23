"""
Model-size-specific prompt variations.

Small models (1b, 3b, 7b) need explicit guidance on when NOT to use tools,
while larger models can infer this from context.
"""

import re
from enum import Enum
from typing import Optional

from .system import (
    PromptComponents,
    ASR_NOTE,
    INFERENCE_GUIDANCE,
    VOICE_STYLE,
)


class ModelSize(Enum):
    """Classification of model sizes for prompt selection."""
    SMALL = "small"  # ≤9b - needs explicit tool constraints
    LARGE = "large"  # >9b - can infer tool usage from context


# Audit round 10 fix: the old enumeration only listed integer billion
# sizes (1b/3b/7b/8b/9b). Modern Ollama tags routinely use decimals
# (qwen2.5:0.5b, :1.5b, llama3.2:3.2b, phi3:3.8b) and even-integer
# sizes (gemma2:2b, qwen3:4b, ibm-granite:6b). Every miss promoted a
# tiny model to the LARGE prompt branch — which sends `tools=` over
# Ollama native API even though the model doesn't support it, and
# skips the repeated text constraints small models need. Net effect:
# false refusals + raw tool-protocol leakage at the user.
#
# Now a regex captures any size ≤ 9b with optional decimal. The
# ordering matters — we check substrings first (fast path for the
# old names that other code may grep for) then fall through to the
# numeric matcher.
_SMALL_MODEL_NAME_HINTS = (
    "gemma4",         # Gemma 4 - always small regardless of tag
    "gemma2",         # Gemma 2 - no native tools API
    "aya",            # Cohere Aya - no native tools API
    "mistral-nemo",   # 12B but uses text tools more reliably than native
    "tinyllama",      # 1.1B
    "phi3",           # phi3 family — mini variants are small
    "phi-3",
    "smollm",         # small-by-design
)

# Match a token like ":1.5b" / "-3b" / "_8b" / ":9.0b". Numeric part is
# captured for size comparison. Word boundary handled by the leading
# separator class.
_SIZE_TOKEN_RE = re.compile(r"[:_\-](\d+(?:\.\d+)?)\s*b\b")

# Models AT OR BELOW this size are considered SMALL. Bumped to 9 to
# cover the prior :8b/:9b enumeration; 10b+ falls to LARGE.
_SMALL_MAX_BILLIONS = 9.0


def detect_model_size(model_name: Optional[str]) -> ModelSize:
    """
    Detect model size from model name.

    Args:
        model_name: Ollama model name (e.g., "gemma4", "gpt-oss:20b")

    Returns:
        ModelSize.SMALL for ≤9b models (including decimals like 0.5b,
        1.5b, 3.8b), ModelSize.LARGE otherwise.
    """
    if not model_name:
        return ModelSize.LARGE  # Default to large for safety

    name_lower = model_name.lower()

    # Family hints first — cheap substring check, and keeps the audit
    # trail of which families are forced-small regardless of tag.
    for hint in _SMALL_MODEL_NAME_HINTS:
        if hint in name_lower:
            return ModelSize.SMALL

    # Numeric size matcher — picks up :1b, :1.5b, :3.8b, -2b, _4b, etc.
    # If the model tag has multiple size-like tokens (rare), pick the
    # LARGEST so a "8b-instruct-q4_K_M" doesn't get misread by the q4
    # quant suffix happening to contain a number.
    sizes = [float(m.group(1)) for m in _SIZE_TOKEN_RE.finditer(name_lower)]
    if sizes and max(sizes) <= _SMALL_MAX_BILLIONS:
        return ModelSize.SMALL

    return ModelSize.LARGE


# =============================================================================
# Large Model Prompts
# =============================================================================

TOOL_INCENTIVES_LARGE = (
    "Proactively use available tools to provide better, more accurate responses. "
    "Prefer tools over guessing when you can get definitive, current, or personalized information. "
    "Tools enhance your capabilities - use them confidently to deliver superior assistance. "
    "Always try tools before asking the user for information you might already be able to get via them."
)

TOOL_GUIDANCE_LARGE = (
    "You have access to tools - use them proactively when you need current information or to perform actions. "
    "After receiving tool results, use the data to FULFILL THE USER'S ORIGINAL REQUEST. "
    "Do NOT describe the structure of tool responses - extract the relevant information and present it conversationally. "
    "Tool results are raw data for you to interpret and use, not content to describe or explain. "
    "CRITICAL fidelity rule: when you answer a question using a tool result, every specific fact in your "
    "reply (names, dates, cast, authors, places, numbers, plot details, product specs) must come from the "
    "tool result itself or from the user's own messages. Do NOT supplement tool results with cast, plot, "
    "release years, authors, or other specifics from your prior — even if they feel plausible. If the tool "
    "returned only a short summary, answer using only that summary; do not extend it with invented detail. "
    "If the tool result doesn't contain what the user asked for, say so and offer to look up more rather "
    "than filling the gap from memory. "
    "When a webSearch result includes a '**Content from top result:**' section, quote its specific facts "
    "(names, dates, roles, plot) rather than deferring to the '**Other search results:**' link list. "
    "The links are provenance, not a substitute for an answer."
)

# Large models also confabulate on named entities — e.g. gpt-oss:20b produces a
# confident but wrong cast list for the film "Possessor" without calling
# webSearch. The anti-confabulation rule is therefore not a small-model-only
# concern. We keep a shorter version here (large models follow concise
# instructions reliably; repetition and worked examples are only needed for
# small models).
#
# NB: constraints are intentionally phrased without any language-specific
# negative examples ("would you like me to", "if you'd like", etc.) because
# this assistant supports an arbitrary set of languages. We describe the
# BEHAVIOUR to avoid, not English tokens that happen to express it.
TOOL_CONSTRAINTS_LARGE = (
    "ACTION REQUESTS — NEVER REFUSE BEFORE CHECKING:\n"
    "When the user asks for an action, scan your available tools and call the one whose "
    "description covers that action. Do NOT apologise or claim you cannot do it. If "
    "nothing in your current list fits, call `toolSearchTool` with a short description "
    "of the action before giving up. A false refusal when a matching tool exists is the "
    "worst possible reply.\n\n"
    "UNKNOWN NAMED ENTITIES:\n"
    "When the user asks about a specific named thing (a film, book, song, game, "
    "product, person, company, place, event), call webSearch before answering unless "
    "you can state concrete, verifiable facts about that exact entity with high confidence. "
    "Do NOT confabulate cast, plot, release year, authors, or other specifics from a "
    "plausible-sounding prior — if you are not certain, look it up. "
    "A diary or memory entry mentioning the entity's name only confirms the topic came "
    "up before; it does NOT give you facts you can restate. "
    "Do not announce the search or ask permission — just call the tool, then answer. "
    "Any phrasing that requests information about a named entity (\"tell me about X\", "
    "\"have you heard of X\", and equivalents in any language) is a search trigger, "
    "not a capability question about yourself.\n\n"
    "ARGUMENTS THE TOOL CAN AUTO-DERIVE:\n"
    "Tools may state in their description that an argument has a sensible default "
    "(for example getWeather uses the user's current location when none is given). "
    "Call the tool in the SAME turn with whatever arguments you have — even zero — "
    "and let it fill the rest. Do NOT reply with a clarifying question like \"which "
    "location?\" for an argument the tool auto-derives. Concretely: \"how's the "
    "weather today\" must trigger getWeather immediately with no arguments, not a "
    "question back to the user.\n\n"
    "SELF-CONTAINED TOOL ARGUMENTS:\n"
    "When you call any tool with a free-form text argument (search queries, lookup "
    "strings, question fields — whatever the tool calls them), the string you pass "
    "must be a self-contained version of the user's intent. Resolve pronouns, "
    "ellipsis, and implicit references from the conversation so far — the tool does "
    "NOT see prior turns. If turn 1 was about Harry Styles and turn 2 asks \"what "
    "are his most famous songs?\", the argument must name Harry Styles explicitly, "
    "not echo the literal utterance. Prefer a compact keyword phrasing over a "
    "conversational sentence: \"Harry Styles most famous songs\" beats \"what are "
    "his most famous songs\". This applies to every tool you call, not just "
    "webSearch."
)


# =============================================================================
# Small Model Prompts
# =============================================================================

TOOL_INCENTIVES_SMALL = (
    "Use tools when they can provide better, more accurate responses. "
    "Follow each tool's description to decide when to use it. "
    "For current information, real-time data, or external lookups - use tools confidently. "
    "For greetings and small talk - respond directly without tools."
)

TOOL_GUIDANCE_SMALL = (
    "You have access to tools - use them when the task requires external data or actions. "
    "After receiving tool results, use the data to answer the user's question conversationally. "
    "Extract relevant information and present it naturally - never output raw JSON or data structures. "
    "Tool results are YOUR OWN DATA to use when answering — they are not a new message from the user "
    "and they are not a prompt for you to interpret. The user's question is in their earlier message "
    "above the tool result; the tool result is the material you use to answer it. Do NOT reply by "
    "describing the tool result back (\"the text is a collection of search results\", \"you have not "
    "asked a specific question\", \"the provided text does not contain a direct question\") — the user "
    "already asked their question and the tool already answered it, you just need to state the answer. "
    "If the tool result contains facts that address the user's earlier question, synthesise those facts "
    "into a direct answer. If it does not, say so briefly and offer to look further — never pretend no "
    "question was asked. "
    "CRITICAL fidelity rule: when answering using a tool result, every specific fact in your reply "
    "(names, dates, cast, authors, places, plot details, numbers) must come from the tool result or "
    "from the user's own messages. Do NOT add cast, plot, release years, authors, or other specifics "
    "from your prior knowledge — even if they feel plausible. If the tool returned only a short summary, "
    "answer using only that summary. If the result doesn't contain what the user asked, say so rather "
    "than filling the gap from memory. "
    "When a tool result contains a section labelled '**Content from top result:**', pull the specific "
    "facts (names, dates, roles, plot, numbers) from that section and state them in your reply. Do NOT "
    "defer to the '**Other search results:**' link list by saying things like 'here are some links' or "
    "'sources like Wikipedia' — those links are for your reference only; the user wants the facts, not "
    "the URLs. If the Content section has the answer, give it; only fall back to mentioning sources when "
    "the Content section is empty or clearly off-topic."
)

# Explicit constraints for small models - focused specifically on the greeting case
# without being overly restrictive on legitimate tool use.
# NOTE: Repeated twice (x2) intentionally. Research shows repeating key instructions
# improves instruction-following in smaller models.
# See: "The Power of Noise: Redefining Retrieval for RAG Systems" (arXiv:2401.14887)
# and "Lost in the Middle: How Language Models Use Long Contexts" (arXiv:2307.03172)
# Repetition places the constraint both early (primacy) and late (recency) in the prompt.
# NB: these constraints are intentionally phrased WITHOUT language-specific
# examples of forbidden phrasing ("would you like me to", "I can search", etc.)
# because this assistant supports an arbitrary set of languages. We describe
# the BEHAVIOURS to avoid, not English tokens that happen to express them.
# Small models still get enough structure to follow because each rule is
# stated in imperative form with a concrete trigger + action.
_TOOL_CONSTRAINTS_BASE = """ACTION REQUESTS — NEVER REFUSE BEFORE CHECKING:
When the user asks for an action (open something, navigate somewhere, send a message, look something up, play something, fetch data), scan your available tools FIRST and call the one whose description covers that action. Do NOT apologise, do NOT say "I cannot do that", do NOT describe your limitations — just call the tool. If nothing in your current tool list obviously fits, call `toolSearchTool` with a short description of the action before giving up. A false refusal when a tool exists is the worst possible reply; calling a tool that turns out not to help is recoverable. Treat "I cannot" as a last resort reserved for when both your tool list AND `toolSearchTool` have been exhausted.

GREETING HANDLING:
When the user's message is a greeting or casual social phrase (whatever language), respond directly and warmly WITHOUT calling any tools. Greetings do not require external data.

USER INSTRUCTIONS:
When the user gives you instructions about how to behave or respond (units, brevity, language, tone), acknowledge and respond directly WITHOUT calling tools. These are behavioural instructions, not data requests.

UNKNOWN NAMED ENTITIES:
If the user asks about a specific named thing (a film, book, song, game, product, person, company, place, event) and you do not have concrete factual information about that exact entity, call webSearch in the SAME turn — silently. Do not offer to search, do not ask permission to search, do not announce the search, do not say you have no information and stop. If the query names the entity clearly enough to search, SEARCH — do not ask the user to disambiguate first. Clarifying BEFORE a tool call is a deflection; clarifying AFTER the tool returns nothing useful is fine.

Any phrasing that requests information about a named entity is a search trigger — the request doesn't have to contain the word "search". Treat "tell me about X", "tell me more about X", "what do you know about X", "what can you tell me about X", "have you heard of X", and their equivalents in any language as information requests about X, not as capability questions about yourself. The correct response is to look X up and answer — not to describe what you can or cannot do.

Only skip the lookup if you can state concrete facts about the exact entity (title, year, creator, plot) without guessing. A diary or memory mention of the entity's name only confirms the topic came up — it does NOT give you facts you can state. Never invent plot, cast, release year, themes, or other specifics from prior knowledge. If you do not have facts from a tool result in this turn, you must call webSearch.

ARGUMENTS THE TOOL CAN AUTO-DERIVE:
If a tool's description says it has a default for some argument (for example getWeather uses the user's current location when none is given), call the tool in the SAME turn with whatever arguments you do have — even zero — and let the tool fill the rest. Do NOT ask the user to supply that argument. Do NOT reply with a clarifying question like "which location?" or "where are you?" when the tool's description already states it auto-derives that argument. Concretely: a message like "how's the weather today" must trigger getWeather immediately with no arguments, NOT a question back to the user. Asking for an argument the tool auto-derives wastes a turn and frustrates the user.

SELF-CONTAINED TOOL ARGUMENTS:
Whenever you call any tool with a free-form text argument (a search query, lookup string, question field — whatever the tool names it), the string you pass MUST be a self-contained restatement of the user's intent. Resolve pronouns, ellipsis, and implicit references from earlier turns yourself — the tool does NOT see the conversation history, it only sees the argument you pass. If the previous turn was about "Harry Styles" and the user now asks "what are his most famous songs?", the argument must be something like "Harry Styles most famous songs", NOT "what are his most famous songs". Prefer a compact keyword phrase over a conversational sentence. Never pass the user's literal utterance through when it contains unresolved pronouns, "that", "those", "it", "his", "her", "their", or similar references. This applies to every tool — webSearch, Wikipedia, MCP tools, all of them."""

# R35-S14: ONE COPY only. The previous version sent _TOOL_CONSTRAINTS_BASE
# twice (~8.6 KB total) for "better instruction-following in small models".
# Measured cost on qwen2.5:3b at 80 tok/s prompt-eval: each duplicate was
# ~1500 tokens = ~18 seconds of CPU PER reply turn. With 1-8 turns per
# query, the duplication was adding 18-150s of latency. qwen2.5:3b (Sep
# 2024 release) follows the single-copy instructions reliably — verified
# manually. If we ever regress to a smaller model that NEEDS repetition,
# reinstate by appending "\n\n" + _TOOL_CONSTRAINTS_BASE.
TOOL_CONSTRAINTS_SMALL = _TOOL_CONSTRAINTS_BASE


# =============================================================================
# Prompt Assembly
# =============================================================================

def get_system_prompts(model_size: ModelSize) -> PromptComponents:
    """
    Get prompt components appropriate for the given model size.

    Args:
        model_size: The detected model size

    Returns:
        PromptComponents with all necessary prompt strings
    """
    if model_size == ModelSize.SMALL:
        return PromptComponents(
            asr_note=ASR_NOTE,
            inference_guidance=INFERENCE_GUIDANCE,
            tool_incentives=TOOL_INCENTIVES_SMALL,
            voice_style=VOICE_STYLE,
            tool_guidance=TOOL_GUIDANCE_SMALL,
            tool_constraints=TOOL_CONSTRAINTS_SMALL,
        )
    else:
        return PromptComponents(
            asr_note=ASR_NOTE,
            inference_guidance=INFERENCE_GUIDANCE,
            tool_incentives=TOOL_INCENTIVES_LARGE,
            voice_style=VOICE_STYLE,
            tool_guidance=TOOL_GUIDANCE_LARGE,
            tool_constraints=TOOL_CONSTRAINTS_LARGE,
        )

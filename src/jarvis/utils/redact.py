from __future__ import annotations
import re

# Deterministic structural scrub patterns. Order matters: specific
# vendor-shaped tokens are matched before generic catches so the more
# informative label wins (e.g. "[REDACTED_AWS_KEY]" beats "[REDACTED_HEX]").
#
# Audit round 11 fixes:
#   * Card-number rule now requires Luhn validity — was eating E.164 phones
#     ("+380 67 123 45 67" → 13+ digits with separators) as `[REDACTED_CARD]`.
#   * Phone/IBAN/URL-userinfo patterns added — none were redacted before.
#   * OpenAI/GitHub patterns broadened — old `sk-[A-Za-z0-9]{32,}` missed
#     `sk-proj-…` (hyphens in body), `gh[pousr]_` missed
#     `github_pat_<82-char>` fine-grained tokens entirely.
#   * JWT pattern capped to `{8,8000}` to bound regex backtracking on
#     adversarial 100KB tool-output payloads (latent ReDoS).


def _looks_like_card(match: re.Match[str]) -> str:
    """Luhn-validate a digit run before masking as a card number.

    macOS / E.164 phones, IMEIs, long ISBNs, account numbers, and ID-like
    timestamps all hit 13–19 digits with separators. Without a Luhn check
    the diary used to mask legitimate Ukrainian phone numbers as
    ``[REDACTED_CARD]`` — strictly worse than no redaction (the user
    can't tell what was supposed to be there). Luhn keeps real card
    numbers masked while letting phone numbers through unchanged.
    """
    raw = match.group(0)
    digits = [int(c) for c in raw if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return raw
    # Standard Luhn: from rightmost, double every second digit.
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return "[REDACTED_CARD]" if checksum % 10 == 0 else raw


_REDACTION_RULES: list[tuple[re.Pattern[str], object]] = [
    # Audit round 12 fix: ordering changed — URL-userinfo BEFORE email
    # so `https://user:pass@host` doesn't get its `pass@host` eaten by
    # the email rule. Vendor-token rules (GH/OpenAI/Stripe) BEFORE the
    # keyword-anchored credential rule so `TOKEN=ghp_…` gets the
    # `[REDACTED_GH_TOKEN]` label rather than the generic `[REDACTED]`.
    # Phone separator class no longer includes `.` so `+1.2.3.4.5.6.7.8`
    # version strings aren't masked as phone numbers.

    # URL with embedded user:password — first, before email could eat
    # the userinfo segment.
    (re.compile(r"://[^/@\s:]+:[^/@\s]+@"), "://[REDACTED_USERINFO]@"),

    # Email — only after URL-userinfo. Negative lookbehind for `:` is
    # cheap insurance against edge cases the URL rule above missed
    # (mailto:user:pass@x style).
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE), "[REDACTED_EMAIL]"),

    # Vendor-specific access keys MOVED UP — the previous ordering let
    # `TOKEN=ghp_…` collapse to `TOKEN=[REDACTED]` via the keyword rule
    # before the precise label could fire.
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED_STRIPE_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,255}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bsk-(?:proj-|ant-|svcacct-)?[A-Za-z0-9_\-]{20,255}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOG_KEY]"),
    (re.compile(r"\b(AWS|GH|GCP|AZURE|xox[abpcr]-)[A-Za-z0-9_\-]{10,}\b", re.IGNORECASE), "[REDACTED_TOKEN]"),

    # JWT — must be 3 dot-separated Base64URL segments. Audit round 16
    # fix: the previous bound (``eyJ[0-9A-Za-z._\-]{8,2000}``) matched
    # any ``eyJ…`` followed by 8+ Base64URL chars even WITHOUT the two
    # `.` separators. A random Base64URL blob starting with ``eyJ``
    # (rare but seen in cache keys, opaque tool tokens, even tracebacks)
    # was mis-redacted as a JWT. Real JWTs always have shape
    # ``<header>.<payload>.<signature>``; require both dots explicitly.
    # Per-segment 4-1500 chars keeps ReDoS bounded on adversarial input
    # (the previous {8,2000} was a flat range across the whole token).
    (
        re.compile(
            r"\beyJ[A-Za-z0-9_\-]{4,1500}"
            r"\.[A-Za-z0-9_\-]{4,1500}"
            r"\.[A-Za-z0-9_\-]{4,1500}\b"
        ),
        "[REDACTED_JWT]",
    ),

    # Authorisation headers — Bearer/Basic carry credentials in line.
    (re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE), "Authorization: Bearer [REDACTED]"),
    (re.compile(r"Authorization:\s*Basic\s+[A-Za-z0-9+/=]+", re.IGNORECASE), "Authorization: Basic [REDACTED]"),

    # Card numbers — Luhn-checked via callback to avoid masking phones.
    (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _looks_like_card),

    # IBAN — country (2 letters) + check (2 digits) + 11-30 alphanumerics.
    # Audit round 16 fix: ``IGNORECASE`` so lowercased pasted IBANs
    # (``de89 3704 0044 0532 0130 00``) still match — real IBAN spec
    # is uppercase but users paste either case. The space-aware shape
    # was already correct (``[ ]?[A-Z0-9]`` lets one space precede each
    # body char). Range bumped to 31 to cover the full 34-char IBANs
    # used by Malta and a few others (4-char prefix + 30 body chars
    # was 1 short).
    (
        re.compile(
            r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,31}\b", re.IGNORECASE
        ),
        "[REDACTED_IBAN]",
    ),

    # Phone numbers (E.164-ish). Round 12 fix: dropped `.` from the
    # separator class. Version strings (`+1.2.3.4.5.6.7.8`), IP-port
    # combos, semver, and similar dotted tokens that happened to start
    # with `+` no longer get masked as phones. Real phone formatters
    # use space, dash, paren — not dot.
    #
    # Audit round 16 fix (Cyrillic/East-European formats): the previous
    # inner separator was ``[\s()\-]?`` — at most ONE separator char
    # between digit groups. ``+38 (067) 123-45-67`` (the standard
    # Ukrainian shape) needs ``) `` between groups (`)` + space, two
    # chars), so the regex failed to match the most common UA phone
    # format AT ALL. Same failure for ``+7 (495) 123-45-67`` (RU) and
    # any ``+CC (XXX) ...`` shape. Switching the inner separator to
    # ``[\s()\-]+`` (one OR MORE) covers the parenthesised area-code
    # pattern. Iteration cap dropped to ``{2,8}`` (was {2,6}) so longer
    # 10-11 digit phones split into 3-4-digit groups still match.
    (
        re.compile(
            r"\+\d{1,3}(?:[\s()\-]+\d{2,4}){2,8}"
        ),
        "[REDACTED_PHONE]",
    ),

    # Keyword-anchored credentials — LAST among credentials so it only
    # catches generic `password=…` / `token=…` after the vendor labels
    # had a chance to fire on shaped tokens.
    (re.compile(
        r"\b(pass(?:word)?|secret|token|apikey|api_key|"
        r"(?:refresh|access|id|oauth)_?token|session(?:_?id)?|sid)"
        r"\s*[:=]\s*\S+?(?=[&\s\"';]|$)",
        re.IGNORECASE,
    ), r"\1=[REDACTED]"),

    # Audit round 9 fix #1: anchored to an EXPLICIT redaction keyword so
    # plain commits/checksums in diary content are preserved.
    (re.compile(
        r"\b(?:hash|digest|hmac|sig|signature)\s*[:=]\s*[0-9A-Fa-f]{32,}\b",
        re.IGNORECASE,
    ), "[REDACTED_HEX]"),
    (re.compile(r"\b\d{6}\b(?=.*(otp|2fa|code))", re.IGNORECASE), "[REDACTED_OTP]"),
]


def _apply_rules(text: str) -> str:
    """Apply all redaction rules. Callable replacements are dispatched by re.sub itself."""
    scrubbed = text
    for pattern, repl in _REDACTION_RULES:
        scrubbed = pattern.sub(repl, scrubbed)  # type: ignore[arg-type]
    return scrubbed


def redact(text: str, max_len: int = 8000) -> str:
    scrubbed = _apply_rules(text)
    scrubbed = " ".join(scrubbed.split())
    if len(scrubbed) > max_len:
        scrubbed = scrubbed[:max_len]
    return scrubbed


def scrub_secrets(text: str) -> str:
    """Apply the structural scrub rules without whitespace collapse or length cap.

    Use for structured content (tool output, multi-line payloads) where
    preserving newlines matters but tokens/emails/etc. must still be masked.
    """
    return _apply_rules(text)

"""PII redaction for log lines (ticket 27).

Admissions chat queries can contain a citizen ID, a phone number or an
email. Those must not be written to logs, which are retained, shipped and
read by people who have no business seeing a student's identity documents.

The detection is NOT re-implemented here: it reuses
``GuardrailService.PII_PATTERNS`` so a pattern added to the guardrail is
redacted in logs the same day. Two detectors would drift, and the drift
would be silent in exactly the case that matters.

Redaction is applied by the caller at the log site. The module has no
side effects and does not read settings itself, so it is safe to call from
anywhere, including import time.
"""
from __future__ import annotations

import re
from typing import Final

from src.services.guardrail import GuardrailService

#: Replaces a matched span. Deliberately opaque: a masked value such as
#: ``0987***321`` still leaks most of the digits.
MASK: Final[str] = "[REDACTED]"

#: Guardrail keyword hits carry no span to mask (they are bare words like
#: "otp"), so they cannot be redacted in place. They only raise a flag.
_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(k) for k in GuardrailService.PII_KEYWORDS),
    re.IGNORECASE,
)

#: Compiled once, in a stable order, from the guardrail's own patterns.
_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    GuardrailService.PII_PATTERNS[name] for name in sorted(GuardrailService.PII_PATTERNS)
)


def redact_pii(text: object) -> str:
    """Return ``text`` with every PII pattern replaced by :data:`MASK`.

    Accepts any object so a log site can pass a value of unknown type
    (the call sites interpolate ints, None and enums). ``None`` becomes
    the empty string rather than the literal ``"None"``.

    Idempotent: redacting an already-redacted string is a no-op, because
    :data:`MASK` matches none of the patterns.
    """
    if text is None:
        return ""
    result = str(text)
    for pattern in _PATTERNS:
        result = pattern.sub(MASK, result)
    return result


def contains_pii(text: object) -> bool:
    """Whether ``text`` looks like it holds PII.

    True for either a pattern match (email/phone/citizen ID) or a guardrail
    keyword hit. Use it to decide whether a line needs redaction at all —
    redaction itself is cheap, but the flag is what a reviewer greps for.
    """
    if text is None:
        return False
    value = str(text)
    if _KEYWORD_RE.search(value):
        return True
    return any(pattern.search(value) is not None for pattern in _PATTERNS)


def safe_query(query: object, *, max_chars: int = 80) -> str:
    """Redact and truncate a query for logging.

    Truncation happens AFTER redaction. The other order would cut a
    citizen ID in half and leave the first digits in the log, which is
    worse than useless: it looks redacted but is not.
    """
    redacted = redact_pii(query)
    if max_chars > 0 and len(redacted) > max_chars:
        return f"{redacted[:max_chars]}…"
    return redacted

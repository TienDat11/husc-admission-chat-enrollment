"""Canonical PII redaction used by the SQL unanswered-question queue.

Masks are intentionally *specific* (``[SĐT]``, ``[CCCD]``, ``[EMAIL]``) so an
administrator can tell *which class* of identifier was present without
recovering the raw value.  This differs from :mod:`src.services.log_redaction`,
whose opaque ``[REDACTED]` mask is retained for the file-backed queue and the
raw access logs.
"""

from __future__ import annotations

import re

#: A Vietnamese mobile number, optionally prefixed with +84 / 84 / 0.
_PHONE = re.compile(r"\b(?:\+?84|0)(?:[\s.-]?\d){8,10}\b")

#: A 12-digit CCCD/CMND (canonical citizen identifier).
_CITIZEN_ID = re.compile(r"\b\d{12}\b")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_PHONE_MASK = "[SĐT]"
_CITIZEN_ID_MASK = "[CCCD]"
_EMAIL_MASK = "[EMAIL]"


def redact_pii(text: object) -> str:
    """Return ``text`` with phones/CCCDs/emails replaced by class masks.

    Non-string input is coerced to ``str`` first so callers never have to
    guard the call.  Replacements are applied in an order that keeps a
    12-digit citizen id from being partially consumed by the phone pattern.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _CITIZEN_ID.sub(_CITIZEN_ID_MASK, text)
    text = _PHONE.sub(_PHONE_MASK, text)
    text = _EMAIL.sub(_EMAIL_MASK, text)
    return text

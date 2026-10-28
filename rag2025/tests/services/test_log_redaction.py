"""Ticket 27: queries must not reach logs with PII intact.

The regression this defends against is a new log line (or an edited one)
that interpolates the raw query. A redaction helper nobody calls is
worthless, so the last test here is a source scan over the whole of
``src/`` rather than a behavioural test of one call site.

Vietnamese queries are used verbatim because the guardrail's phone
pattern is written against Vietnamese number shapes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.services.log_redaction import MASK, contains_pii, redact_pii, safe_query

REPO_SRC = Path(__file__).resolve().parents[2] / "src"

CITIZEN_ID = "123456789012"
PHONE = "0987654321"
EMAIL = "a.b@husc.edu.vn"


class TestRedaction:
    def test_redacts_each_pii_kind(self) -> None:
        assert CITIZEN_ID not in redact_pii(f"cccd của em là {CITIZEN_ID}")
        assert PHONE not in redact_pii(f"sđt em {PHONE} nhé")
        assert EMAIL not in redact_pii(f"mail em là {EMAIL}")

    def test_leaves_admission_content_alone(self) -> None:
        """Over-redaction would gut the logs' diagnostic value."""
        query = "điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?"
        assert redact_pii(query) == query

    def test_is_idempotent(self) -> None:
        once = redact_pii(f"{EMAIL} {PHONE} {CITIZEN_ID}")
        assert redact_pii(once) == once
        assert MASK in once

    def test_redacts_every_occurrence_not_just_the_first(self) -> None:
        result = redact_pii(f"{PHONE} và {PHONE}")
        assert PHONE not in result
        assert result.count(MASK) == 2

    def test_none_becomes_empty_string(self) -> None:
        """Log sites interpolate optional fields; None must not print."""
        assert redact_pii(None) == ""

    def test_contains_pii_flags_keyword_without_a_span(self) -> None:
        """'otp' has no span to mask, but its presence is still notable."""
        assert contains_pii("mã otp của em là gì") is True
        assert contains_pii("điểm chuẩn 2025") is False


class TestSafeQuery:
    def test_truncates_after_redacting(self) -> None:
        """Truncating first would cut a citizen ID in half and leave digits.

        This is the whole reason ``safe_query`` exists instead of callers
        doing ``query[:80]`` themselves.
        """
        result = safe_query(f"cccd {CITIZEN_ID} " + "x" * 200, max_chars=30)
        assert CITIZEN_ID not in result
        assert len(result) <= 31  # 30 chars + the ellipsis

    def test_keeps_short_queries_intact(self) -> None:
        assert safe_query("học phí ngành Y khoa?") == "học phí ngành Y khoa?"


class TestNoRawQueryLogSites:
    """Source scan: the redaction must actually be wired in.

    Log calls are extracted with balanced-paren scanning, not a line regex.
    A line-based scan silently misses a call whose f-string wraps onto the
    next line — which is exactly how the worst leak in this codebase was
    written (``v2_pii_block``), so a line scanner would have reported a
    clean tree while the raw query was still being logged.
    """

    LOG_CALL = re.compile(r"logger\.(?:debug|info|warning|error|exception)\s*\(")
    #: An f-string INTERPOLATION whose expression is a query variable.
    #: Requiring the braces is what keeps prose like "Unified query failed"
    #: or "sub-query retrieve failed" from being reported as a leak — those
    #: interpolate something else (``{exc}``, ``{res}``) and carry no query.
    RAW_QUERY_INTERP = re.compile(
        r"\{[^{}]*\b(?:user_query|original_query|normalized_query|query|request\.query|q)\b[^{}]*\}"
    )
    REDACTED = re.compile(r"safe_query\(|redact_pii\(")

    @classmethod
    def _log_call_bodies(cls, source: str) -> list[tuple[int, str]]:
        """Return ``(line_number, call_text)`` for every logger call."""
        calls: list[tuple[int, str]] = []
        for match in cls.LOG_CALL.finditer(source):
            start = match.start()
            depth = 0
            for offset, char in enumerate(source[start:]):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        body = source[start : start + offset + 1]
                        calls.append((source.count("\n", 0, start) + 1, body))
                        break
        return calls

    def test_no_log_site_interpolates_a_raw_query(self) -> None:
        offenders: list[str] = []
        for path in sorted(REPO_SRC.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for lineno, body in self._log_call_bodies(source):
                if not self.RAW_QUERY_INTERP.search(body):
                    continue
                if self.REDACTED.search(body):
                    continue
                flattened = " ".join(body.split())
                offenders.append(f"{path.relative_to(REPO_SRC.parent)}:{lineno}: {flattened}")

        assert not offenders, (
            "log site interpolates a raw query without redaction:\n  "
            + "\n  ".join(offenders)
        )

    def test_scanner_detects_a_multiline_leak(self) -> None:
        """Falsification: the scanner must not pass a wrapped f-string.

        Without this, the scanner can regress to a line-based one and keep
        reporting success while leaking.
        """
        leaky = (
            "def f(request):\n"
            "    logger.info(\n"
            "        f\"pii_block query='{request.query[:80]}'\"\n"
            "    )\n"
        )
        calls = self._log_call_bodies(leaky)
        assert len(calls) == 1, calls
        _, body = calls[0]
        assert self.RAW_QUERY_INTERP.search(body)
        assert not self.REDACTED.search(body)

    def test_scanner_accepts_a_redacted_multiline_call(self) -> None:
        safe = (
            "def f(request):\n"
            "    logger.info(\n"
            "        f\"pii_block query='{safe_query(request.query, max_chars=80)}'\"\n"
            "    )\n"
        )
        _, body = self._log_call_bodies(safe)[0]
        assert self.REDACTED.search(body)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.services.query_router",
        "src.services.query_enhancer",
    ],
)
def test_modules_import_cleanly(module_name: str) -> None:
    """Guards the import added to each module against a cycle."""
    __import__(module_name)

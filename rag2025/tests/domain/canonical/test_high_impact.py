"""TDD: rag2025/src/domain/canonical/high_impact.py — ticket 25 criterion 2.

The reviewer's diff must flag the fields that change what a student is
served, identically for every client. Acceptance probed here:
- every field in ``HIGH_IMPACT_FIELDS`` reports high-impact;
- every field in ``NON_HIGH_IMPACT_FIELDS`` reports low-impact;
- an *unknown* field defaults to high-impact — a field added later must not
  silently downgrade out of the reviewer's view;
- the classification is versioned: ``high_impact_version()`` exposes the
  stamp a diff carries, and the two lists are disjoint (one field cannot
  answer both ways).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.high_impact import (  # noqa: E402
    HIGH_IMPACT_FIELDS,
    HIGH_IMPACT_VERSION,
    NON_HIGH_IMPACT_FIELDS,
    high_impact_version,
    is_high_impact,
)


@pytest.mark.parametrize("field", sorted(HIGH_IMPACT_FIELDS))
def test_declared_high_impact_fields_report_high_impact(field: str):
    assert is_high_impact(field) is True


@pytest.mark.parametrize("field", sorted(NON_HIGH_IMPACT_FIELDS))
def test_declared_low_impact_fields_report_low_impact(field: str):
    assert is_high_impact(field) is False


def test_unknown_field_defaults_to_high_impact():
    """A field nobody has classified yet is *shown*, never silently downgraded.

    This is the property that makes the list safe to version: forgetting to
    classify a new field must fail toward visibility, not toward hiding the
    change from the reviewer.
    """
    assert is_high_impact("may_moi_them_ngay_mai") is True


def test_the_two_lists_are_disjoint():
    overlap = HIGH_IMPACT_FIELDS & NON_HIGH_IMPACT_FIELDS
    assert overlap == frozenset(), f"field classified both ways: {overlap}"


def test_version_string_is_exposed_and_stable():
    """The diff stamps its classification with this version; two diffs with
    the same stamp are comparable, so the value must not drift per call."""
    assert HIGH_IMPACT_VERSION == "1"
    assert high_impact_version() == HIGH_IMPACT_VERSION
    assert high_impact_version() == high_impact_version()

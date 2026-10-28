"""Server-side diff between two versions of one canonical record - ticket 25.

The diff is computed here, not in the admin client, for the reason the ticket
states: the same diff is replayed by API consumers, the audit trail, and the
release notes, and the high-impact flag must be enforced over every client
rather than requested of it. The classification itself lives in
``high_impact.py`` (versioned); this module only decides *what changed*.

Comparing two versions of *different* records is refused: a diff that quietly
spans two facts would mislabel every field as changed and is a caller bug,
not a review finding.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from src.domain.canonical.high_impact import (
    HIGH_IMPACT_VERSION,
    is_high_impact,
)
from src.domain.canonical.record import CanonicalRecord


@dataclass(frozen=True)
class FieldChange:
    """One field whose value differs between the two versions.

    ``old``/``new`` are JSON-safe (dates and enums are dumped to their wire
    strings) so the whole diff serializes without a second conversion step.
    """

    field: str
    old: object
    new: object
    high_impact: bool

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "old": self.old,
            "new": self.new,
            "high_impact": self.high_impact,
        }


@dataclass(frozen=True)
class RecordDiff:
    """The whole difference between a published version and a draft version.

    ``baseline_version`` is ``None`` when the record has no published version
    yet - a brand-new fact has no diff to show, and the preview says so
    instead of diffing against an invented empty record.
    """

    record_id: str
    baseline_version: int | None
    draft_version: int
    changes: tuple[FieldChange, ...]
    high_impact_version: str = HIGH_IMPACT_VERSION

    @property
    def high_impact_changes(self) -> tuple[FieldChange, ...]:
        return tuple(c for c in self.changes if c.high_impact)

    @property
    def has_high_impact(self) -> bool:
        return any(c.high_impact for c in self.changes)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "baseline_version": self.baseline_version,
            "draft_version": self.draft_version,
            "high_impact_version": self.high_impact_version,
            "has_high_impact": self.has_high_impact,
            "changes": [c.to_dict() for c in self.changes],
        }


def _json_safe(value: object) -> object:
    """The value as it appears on the wire (dates/enums as their strings).

    ``model_dump(mode="json")`` already does this per record; this is the
    fallback for values the model kept as extras of arbitrary type.
    """
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def diff_records(
    baseline: CanonicalRecord | None, draft: CanonicalRecord
) -> RecordDiff:
    """Field-by-field difference from ``baseline`` (published) to ``draft``.

    Every field the two versions carry is compared (``extra="allow"`` means
    the compared shape is the record's own, not a fixed schema); ``version``
    is excluded - it is identity bookkeeping, and *every* edit bumps it, so
    reporting it would bury the fields the reviewer came to read. Changes are
    ordered by field name so the diff is deterministic regardless of dump
    order.
    """
    if baseline is not None and baseline.record_id != draft.record_id:
        raise ValueError(
            f"cannot diff {baseline.record_id} against {draft.record_id}: "
            "a review diff compares versions of one record"
        )
    old_dump = {} if baseline is None else baseline.model_dump(mode="json")
    new_dump = draft.model_dump(mode="json")

    changes: list[FieldChange] = []
    for field in sorted(set(old_dump) | set(new_dump)):
        if field == "version":
            continue
        old_value = _json_safe(old_dump.get(field))
        new_value = _json_safe(new_dump.get(field))
        if old_value == new_value:
            continue
        changes.append(
            FieldChange(
                field=field,
                old=old_value,
                new=new_value,
                high_impact=is_high_impact(field),
            )
        )
    return RecordDiff(
        record_id=draft.record_id,
        baseline_version=baseline.version if baseline is not None else None,
        draft_version=draft.version,
        changes=tuple(changes),
    )

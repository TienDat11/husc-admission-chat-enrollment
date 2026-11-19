"""Record identity: coordinate keys and record-type priority (plan §PL-6.6).

A coordinate key names one slot a record occupies for a given admission cycle:
``RECORD_TYPE:scope:subject:year:round:methods_hash``. Two versions of the same
slot share a key, which is what supersession (§PL-6.7) joins on.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from src.domain.canonical.record import RecordType

NO_ROUND = "none"
ALL_METHODS = "all"

# §PL-6.6: when one block carries several record types, the higher one wins.
RECORD_TYPE_PRIORITY: tuple[RecordType, ...] = (
    RecordType.DIEM_CHUAN,
    RecordType.CHI_TIEU,
    RecordType.HOC_PHI,
    RecordType.TO_HOP,
    RecordType.NGANH,
    RecordType.THOI_GIAN,
)


def methods_hash(methods: Iterable[str] | None) -> str:
    """``"all"`` when the record is not limited to specific methods, else a 12-hex digest."""
    cleaned = [unicodedata.normalize("NFC", m.strip()) for m in (methods or ()) if m and m.strip()]
    if not cleaned:
        return ALL_METHODS
    return hashlib.sha256("|".join(sorted(cleaned)).encode("utf-8")).hexdigest()[:12]


def coordinate_key(
    *,
    record_type: RecordType,
    scope: str,
    subject: str,
    admission_year: int,
    admission_round: str | None = None,
    methods_hash_value: str = ALL_METHODS,
) -> str:
    for part, name in ((scope, "scope"), (subject, "subject")):
        if not part or ":" in part:
            raise ValueError(f"coordinate {name} must be non-empty and contain no ':': {part!r}")
    return ":".join(
        (
            record_type.value,
            scope,
            subject,
            str(admission_year),
            admission_round or NO_ROUND,
            methods_hash_value,
        )
    )


@dataclass(frozen=True)
class CoordinateParts:
    record_type: RecordType
    scope: str
    subject: str
    admission_year: int
    admission_round: str
    methods_hash: str


def parse_coordinate_key(key: str) -> CoordinateParts:
    parts = key.split(":")
    if len(parts) != 6:
        raise ValueError(f"malformed coordinate key: {key!r}")
    record_type, scope, subject, year, admission_round, mhash = parts
    return CoordinateParts(
        record_type=RecordType(record_type),
        scope=scope,
        subject=subject,
        admission_year=int(year),
        admission_round=admission_round,
        methods_hash=mhash,
    )


def pick_record_type(types: Iterable[RecordType]) -> RecordType | None:
    """The highest-priority record type present, per §PL-6.6."""
    present = set(types)
    for candidate in RECORD_TYPE_PRIORITY:
        if candidate in present:
            return candidate
    return None

"""Versioned canonical admissions record — one fact, with provenance.

Source of truth for the vocabulary every admissions fact must carry before it
can be materialized into retrieval. See:
  - .scratch/husc-rag-admin-upgrade/issues/02-canonical-record-schema.md
  - .scratch/husc-rag-admin-upgrade/spec.md (Implementation Decisions)

Design notes
------------
- ``record_type`` reuses ``src.domain.entities.EntityType`` for the fact kinds
  the graph already knows (same string value, so a canonical record and a graph
  node agree on the kind). The extra kinds below are canonical-only facts named
  as high-impact in the spec and have no ``EntityType`` counterpart yet.
- ``status`` values are the lifecycle states named verbatim in the spec's state
  machine. The transition *rules* are not here — ticket 12 owns enforcement.
- Text is stored and re-emitted byte-for-byte: no ``\\uXXXX`` escaping.
  Unicode normalisation happens at the *authoring* boundary (ticket 03,
  ``admin_lifecycle.create`` / ``edit``) — by the time a record reaches the
  store its ``text`` is already NFC, so the bytes written here are stable.
  Legacy 1.0 records stored before ticket 03 may carry NFD bytes; reading
  them is unaffected (the store does not re-normalise). ``chunk_identity``
  deliberately does not normalise Unicode, so NFD-era chunks keep their ids.
- Additive evolution: every field added after 1.0 MUST carry a default, so a
  payload written by an older writer still validates. Unknown keys written by a
  newer writer are preserved verbatim (``extra="allow"``) instead of dropped.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.domain.entities import EntityType

SCHEMA_VERSION = "1.1"


class RecordType(str, Enum):
    """Kind of admissions fact a canonical record states.

    First group: mirrors ``EntityType`` values verbatim (shared vocabulary).
    Second group: canonical-only fact kinds (no graph entity counterpart).

    **Rollback constraint (1.1, ticket 03):** ``THONG_BAO`` and ``HOI_DAP``
    are 1.1 additions. Once a store contains either value, reverting the
    reader to 1.0 code will raise ``ValidationError`` on load. A code
    rollback therefore requires restoring a store snapshot taken *before*
    1.1 was activated (spec 02 §2); a release-only rollback (keep code,
    point back at an older release) is always safe.
    """

    NGANH = EntityType.NGANH.value
    TO_HOP = EntityType.TO_HOP.value
    DIEM_CHUAN = EntityType.DIEM_CHUAN.value
    HOC_PHI = EntityType.HOC_PHI.value
    THOI_GIAN = EntityType.THOI_GIAN.value
    TO_CHUC = EntityType.TO_CHUC.value
    CHINH_SACH = EntityType.CHINH_SACH.value
    # canonical-only kinds — high-impact facts the graph does not model yet
    THONG_BAO = "THONG_BAO"
    HOI_DAP = "HOI_DAP"
    CHI_TIEU = "CHI_TIEU"
    DIEU_KIEN = "DIEU_KIEN"


class RecordStatus(str, Enum):
    """Lifecycle state of a record version (spec state machine, verbatim).

    ``draft --submit--> review --approve--> approved --publish--> published
    --supersede--> archived``.
    """

    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class AuthorityLevel(str, Enum):
    """Source authority tier of the fact.

    Ordered by the spec's proposed source priority ("Thứ tự ưu tiên nguồn"),
    highest first. That order still needs HUSC sign-off, so it lives in exactly
    one place (``_AUTHORITY_RANK``) instead of being re-derived per caller.
    """

    HUSC_CANONICAL = "husc_canonical"
    HUSC_OFFICIAL = "husc_official"
    HUE_UNIVERSITY = "hue_university"
    MINISTRY = "ministry"
    PRIOR_YEAR_OFFICIAL = "prior_year_official"
    DERIVED_FAQ = "derived_faq"

    @property
    def rank(self) -> int:
        """1 = highest authority. Ties mean the spec treats them as one tier."""
        return _AUTHORITY_RANK[self]


_AUTHORITY_RANK: dict[AuthorityLevel, int] = {
    AuthorityLevel.HUSC_CANONICAL: 1,
    AuthorityLevel.HUSC_OFFICIAL: 2,
    AuthorityLevel.HUE_UNIVERSITY: 3,
    AuthorityLevel.MINISTRY: 3,
    AuthorityLevel.PRIOR_YEAR_OFFICIAL: 4,
    AuthorityLevel.DERIVED_FAQ: 5,
}


class CanonicalRecord(BaseModel):
    """One versioned admissions fact: effective window + provenance + text.

    A record cannot exist with a reversed effective window, and a ``published``
    record cannot exist without a source document. Both are domain invariants
    enforced at the boundary, not in a retrieval prompt.

    ``source_document_id`` / ``source_url`` / ``source_page`` are nullable so a
    draft can be authored before its source document is uploaded; only the
    published state requires provenance to be present.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: str = SCHEMA_VERSION
    record_id: str = Field(min_length=1)
    record_type: RecordType
    version: int = Field(ge=1)
    admission_year: int = Field(ge=2000, le=2100)
    effective_from: date
    effective_to: date | None = None
    status: RecordStatus
    authority_level: AuthorityLevel
    source_document_id: str | None = None
    source_url: str | None = None
    source_page: int | None = Field(default=None, ge=1)
    published_at: datetime | None = None
    # Who wrote this version, as the server authenticated them (ticket 13). It
    # is stamped by the lifecycle service from the caller's identity — never
    # taken from the request body — and it is what the four-eyes rule compares
    # against the approver. Recorded per *version* rather than per event because
    # the rule must survive a restart: the event ledger (ticket 14) is a log,
    # while this field is part of the version being approved.
    authored_by: str | None = None
    supersedes: str | None = None
    locale: Literal["vi"] = "vi"
    text: str
    # 1.1 additive fields — all carry defaults so a 1.0 payload still validates.
    admission_round: str | None = None
    major_codes: list[str] = Field(default_factory=list)
    admission_methods: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_domain_invariants(self) -> "CanonicalRecord":
        if self.effective_to is not None and self.effective_from > self.effective_to:
            raise ValueError(
                f"effective_from ({self.effective_from}) must not be after "
                f"effective_to ({self.effective_to})"
            )
        if self.status is RecordStatus.PUBLISHED and not (
            self.source_document_id or ""
        ).strip():
            raise ValueError(
                "a published record requires source_document_id — "
                "no admissions fact reaches students without provenance"
            )
        return self

    @property
    def record_version_key(self) -> str:
        """Stable identity of this record version: ``record_id@version``.

        This is the input ticket 06 hashes for deterministic chunk identity.
        """
        return f"{self.record_id}@{self.version}"


def to_canonical_json(record: CanonicalRecord) -> bytes:
    """Deterministic UTF-8 bytes for one record version.

    Byte-faithful by construction: ``ensure_ascii=False`` keeps Vietnamese text
    as UTF-8 (never ``\\uXXXX`` escapes) and nothing normalizes Unicode.
    ``sort_keys=True`` means equal records produce equal bytes, so a record hash
    is reproducible across processes and runs.
    """
    payload = record.model_dump(mode="json")
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def from_canonical_json(data: bytes | str) -> CanonicalRecord:
    """Read back bytes written by :func:`to_canonical_json`.

    Tolerates additive evolution in both directions: fields added in a later
    schema version have defaults (so an older payload validates), and unknown
    keys from a newer writer are preserved rather than dropped.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return CanonicalRecord.model_validate(json.loads(data))

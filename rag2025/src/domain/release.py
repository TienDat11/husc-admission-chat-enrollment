"""Domain models for canonical release manifests and version transitions.
See rag2025/.omc/plans/data_lifecycle_plan_v2.md M1 lines 244-313.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

SCHEMA_VERSION = "2.0"


class ScopeLevel(str, Enum):
    national = "national"
    hue_university = "hue_university"
    husc = "husc"


class TransitionKind(str, Enum):
    REPLACES = "REPLACES"
    AMENDS = "AMENDS"
    CORRECTS = "CORRECTS"
    REPEALS = "REPEALS"
    EXPIRES = "EXPIRES"


def compute_corpus_sha256(doc_sha256_list: list[str]) -> str:
    """Deterministic corpus hash: sha256 of sorted doc hashes joined by
    newline. MUST be identical across calls regardless of input order."""
    joined = "\n".join(sorted(doc_sha256_list))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VersionTransition:
    """Bản ghi supersession TƯỜNG MINH. KHÔNG suy ra từ filename / chunk_id /
    cosine similarity. Mọi transition phải trace về evidence_span — do đó
    __post_init__ raise ValueError nếu evidence_span rỗng."""

    transition_id: str
    kind: TransitionKind
    subject_doc_id: str
    object_doc_id: str
    effective_from: date
    evidence_doc_id: str
    evidence_span: str
    evidence_offset: tuple[int, int]
    detector: Literal["rule", "manual", "portal_metadata"]
    detector_confidence: float
    scope: ScopeLevel
    affected_articles: list[str] = field(default_factory=list)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.evidence_span or not self.evidence_span.strip():
            raise ValueError(
                "VersionTransition.evidence_span must not be empty — "
                "every transition must trace to source evidence"
            )


@dataclass(frozen=True)
class ReleaseManifest:
    """Composite artifact — đủ để REPRODUCE bit-perfect một release."""

    release_id: str
    schema_version: str = SCHEMA_VERSION
    created_at: Optional[datetime] = None
    parent_release_id: Optional[str] = None
    corpus_sha256: str = ""
    document_count: int = 0
    chunk_count: int = 0
    transition_count: int = 0
    embedding_model: str = ""
    embedding_dim: int = 0
    chunk_strategy: str = ""
    chunk_strategy_version: str = ""
    pdf_parser: str = ""
    detector_version: str = ""
    lancedb_table: str = ""
    lancedb_tag: str = ""
    lancedb_version: int = 0
    graph_sha256: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.created_at is not None:
            d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReleaseManifest":
        d2 = dict(d)
        if d2.get("created_at"):
            d2["created_at"] = datetime.fromisoformat(d2["created_at"])
        return cls(**d2)

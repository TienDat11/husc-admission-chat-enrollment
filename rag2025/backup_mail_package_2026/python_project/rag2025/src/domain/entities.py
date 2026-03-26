"""
Domain Entities – GraphRAG Layer

Pure domain models (no infrastructure dependencies).
All models use dataclasses for immutability and clarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityType(str, Enum):
    NGANH = "NGANH"                 # Academic major
    TO_HOP = "TO_HOP"               # Subject combination
    DIEM_CHUAN = "DIEM_CHUAN"       # Admission score
    HOC_PHI = "HOC_PHI"             # Tuition fee
    THOI_GIAN = "THOI_GIAN"         # Duration
    TO_CHUC = "TO_CHUC"             # Organization / university
    CHINH_SACH = "CHINH_SACH"       # Policy
    UNKNOWN = "UNKNOWN"


class RelationType(str, Enum):
    CO_TO_HOP = "CO_TO_HOP"         # major HAS subject_combination
    CO_DIEM = "CO_DIEM"             # major HAS admission_score
    THUOC_TRUONG = "THUOC_TRUONG"   # major BELONGS_TO university
    YEU_CAU = "YEU_CAU"             # major REQUIRES policy
    LIEN_QUAN = "LIEN_QUAN"         # generic relation


@dataclass(frozen=True)
class Entity:
    """A named entity extracted from a chunk.

    Attributes:
        text: Raw surface form from the document.
        entity_type: Semantic category.
        normalized: Canonical form for graph node ID.
        chunk_id: Source chunk this entity was extracted from.
    """
    text: str
    entity_type: EntityType
    normalized: str
    chunk_id: str

    @property
    def node_id(self) -> str:
        """Graph node ID: type:normalized_text."""
        return f"{self.entity_type.value}:{self.normalized}"


@dataclass(frozen=True)
class Triple:
    """A (head, relation, tail) knowledge triple.

    Attributes:
        head: Head entity node_id.
        relation: Semantic relation type.
        tail: Tail entity node_id.
        chunk_id: Source chunk this triple was extracted from.
        weight: Edge confidence weight (default 1.0).
    """
    head: str
    relation: RelationType
    tail: str
    chunk_id: str
    weight: float = 1.0


@dataclass
class Chunk:
    """A document chunk from the JSONL corpus.

    Attributes:
        chunk_id: Unique chunk identifier (matches Qdrant point payload).
        text: Clean text content.
        faq_type: Category tag from source JSONL.
        metadata: Additional metadata from source.
    """
    chunk_id: str
    text: str
    faq_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Output of NER extraction for a single chunk.

    Attributes:
        chunk_id: Source chunk identifier.
        entities: Extracted entities.
        triples: Extracted knowledge triples.
        error: Error message if extraction failed.
    """
    chunk_id: str
    entities: List[Entity] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None

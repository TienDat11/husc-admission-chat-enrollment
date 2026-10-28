"""Domain models for canonical release manifests and version transitions.
See rag2025/.omc/plans/data_lifecycle_plan_v2.md M1 lines 244-313.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
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

def compute_file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed.

    The same digest function the release manifest pins an artifact with and the
    rollback verifies it against — one implementation, so "the bytes I published"
    and "the bytes I restored" cannot be measured two different ways. Streamed
    rather than ``read_bytes()`` because the graph artifact is measured in
    megabytes and a publish must not double it in RAM.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Composite artifact — đủ để REPRODUCE bit-perfect một release.

    The release unit is **two** served artifacts, and both are pinned here:

    * the LanceDB table (``lancedb_table``/``lancedb_tag``/``lancedb_version``);
    * the knowledge graph (``graph_path``/``graph_sha256``/``graph_schema_version``).

    Pinning the graph is not bookkeeping. A graph maps entity nodes to the
    ``chunk_id``s they came from, so a graph from release R2 answering over
    release R1's table injects R2 chunk ids into an R1 answer — a mixed-release
    answer built from a *matching* pointer and table. Without the pin there is
    nothing for a rollback to restore the graph *to*, and nothing that fails
    when the two are paired wrongly.
    """

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
    # Release-relative POSIX path of the pinned graph artifact. Relative, not
    # absolute: a manifest that named an absolute path would describe a release
    # reproducible only on the machine that wrote it.
    graph_path: str = ""
    # sha256 of the artifact's bytes — what the rollback verifies before it
    # serves them.
    graph_sha256: str = ""
    # The GraphML schema version inside those bytes. Checked separately from the
    # hash so a release pinned from a graph built under an older schema is
    # refused by name rather than loaded and then misread.
    graph_schema_version: str = ""

    def __post_init__(self) -> None:
        pin = (self.graph_path, self.graph_sha256, self.graph_schema_version)
        pinned = [value for value in pin if value]
        if pinned and len(pinned) != len(pin):
            raise ValueError(
                "a graph pin must be complete or absent: graph_path, "
                "graph_sha256 and graph_schema_version describe one artifact, so a "
                "manifest carrying only some of them pins something it cannot "
                f"verify (got path={self.graph_path!r}, "
                f"sha256={self.graph_sha256!r}, "
                f"schema_version={self.graph_schema_version!r})"
            )

    @property
    def pins_graph(self) -> bool:
        """Whether this release pins a graph artifact at all.

        False for a release published before the graph was part of the release
        unit; such a release cannot be restored *over* a serving graph (see
        :func:`~src.services.admin_rollback.ReleaseGraphRollback.validate`).
        """
        return bool(self.graph_sha256)

    def pinned_graph_path(self, release_dir: str | Path) -> Path | None:
        """The pinned artifact resolved under ``release_dir``, or ``None``.

        Raises:
            ValueError: the pin is absolute, names a drive, or escapes
                ``release_dir``. A pin is store state, and one that resolved
                outside its own release would let a rollback serve another
                release's graph — the exact mixture the pin exists to prevent —
                so it is refused rather than clamped.
        """
        raw = (self.graph_path or "").strip()
        if not raw:
            return None
        normalized = raw.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        if PurePosixPath(normalized).is_absolute() or any(
            part in ("..", "") for part in parts
        ):
            raise ValueError(
                f"release {self.release_id}'s graph_path {raw!r} is not a relative "
                "path inside its own release directory"
            )
        root = Path(release_dir).resolve()
        resolved = (root / normalized).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"release {self.release_id}'s graph_path {raw!r} escapes its release "
                f"directory ({root})"
            )
        return resolved

    def with_graph_pin(self, *, path: str, sha256: str, schema_version: str) -> "ReleaseManifest":
        """This manifest with the graph artifact pinned. Frozen, so a copy."""
        return replace(
            self, graph_path=path, graph_sha256=sha256, graph_schema_version=schema_version
        )

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

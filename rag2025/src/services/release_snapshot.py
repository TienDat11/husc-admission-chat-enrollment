"""Immutable in-RAM snapshot of the active release — U8 (§PL-5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReleaseSnapshot:
    """Frozen picture of the release every serving path reads.

    ``chunks_table`` is the LanceDB table/version pointer for this release;
    serving code reads chunks through it, never via ``data/major_codes/*.json``.
    """

    release_id: str
    recommender_snapshot: dict[str, Any]
    major_codes: frozenset[str]
    cutoff_scores: dict[str, float]
    quota: dict[str, int]
    tuition: dict[str, float]
    priority_rules: dict[str, Any]
    major_whitelist: frozenset[str]
    chunks_table: str
    corpus_stamp: list[dict[str, Any]]


@dataclass(frozen=True)
class ReleaseSnapshotLoadError(Exception):
    """The active release has no usable snapshot (serving must refuse)."""

    release_id: str
    reason: str

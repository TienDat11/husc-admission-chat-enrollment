"""Comparison contract for multi-major same-year queries (ticket 18).

A teacher asks to compare several majors in one admission year. The answer is a
table: one row per major, one column per high-impact metric (cutoff score /
tuition / quota / ...). The contract makes three guarantees that the LLM is
specifically tempted to violate:

1. **Same effective year, only.** All compared majors share one admission
   year; a major carrying a different year is *split out* (returned in
   ``refused``) rather than merged into the same table. "Effective year" is the
   canonical record's ``admission_year`` — the same coordinate the conflict
   detect, the abstention policy and the year-banner expose.
2. **Per-cell provenance.** Every high-impact cell carries its OWN citation —
   a ``CellProvenance`` traced back to a real canonical-record version via
   ``CanonicalRecordStore.resolve_provenance``. One shared ``source_id`` for the
   whole table is rejected by the model validator: each cell's citation must
   point at its own ``record_version_key``.
3. **Missing cells are marked, never inferred.** When no record carries a
   metric for a major, the cell is ``state="missing"`` with no ``value`` and no
   citation — the generator must not copy a sibling cell's value across.

The model classes here are intentionally pure dataclasses/Pydantic with no I/O;
the building of a table (reading records, resolving provenance, deciding the
route) lives in :mod:`src.services.comparison_service`.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# Cell state literals (constants vs. free-form strings so a typo like "misng"
# can never produce a silent cell that bypasses the validators below).
CELL_STATE_PRESENT: str = "present"
CELL_STATE_MISSING: str = "missing"


class ComparisonTableError(ValueError):
    """Raised when a comparison table violates a contract invariant."""


# A cell's provenance is real provenance — the canonical-record version + the
# source document it was extracted from. It is NOT a free-form string. It is the
# same coordinate family as ``services.citation.ClaimCitation`` so the answer
# surface and the admin preview surface cite the same thing.
class CellProvenance(BaseModel):
    """Where one cell's value came from."""

    record_version_key: str = Field(
        ..., description="`record_id@version` of the canonical record version"
    )
    source_document_id: Optional[str] = Field(
        default=None,
        description="Repo-relative document id the record was extracted from",
    )
    source_url: Optional[str] = Field(
        default=None, description="Canonical source URL when available"
    )
    source_page: Optional[int] = Field(
        default=None,
        description="1-based page in the source document; None when unknown",
        ge=1,
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Retrieval chunk that materialized the value, when known",
    )

    model_config = {"extra": "forbid"}


# A column is a high-impact metric the teacher wants compared. The contract is
# agnostic to which metrics appear: the service passes them in as labels, so the
# same table shapes for "so sánh điểm chuẩn" and "so sánh học phí" without
# duplicating the model for every metric kind.
class ComparisonColumn(BaseModel):
    """One metric column header of the table."""

    metric: str = Field(..., description="Field label, e.g. 'điểm chuẩn' or 'học phí'")
    display: str = Field(..., description="Human-readable column header")

    model_config = {"extra": "forbid"}


# A single cell. The validator is the enforcement point for rules 2 and 3:
#   - a missing cell MUST NOT carry a value (never inferred from elsewhere)
#   - a missing cell MUST NOT carry a citation (no provenance for an absent fact)
#   - a present (high-impact) cell MUST carry its OWN provenance
class ComparisonCell(BaseModel):
    """One major x metric cell of the comparison table."""

    major: str = Field(..., description="Major label this cell belongs to")
    metric: str = Field(..., description="Metric this cell reports")
    state: str = Field(..., description="`present` or `missing`")
    value: Optional[str] = Field(
        default=None, description="Rendered value; None when missing"
    )
    source: Optional[CellProvenance] = Field(
        default=None, description="Provenance of the value; None when missing"
    )
    note: Optional[str] = Field(
        default=None, description="Observed caveat, e.g. 'số liệu ước đạo'"
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _enforce_cell_contract(self) -> "ComparisonCell":
        if self.state == CELL_STATE_MISSING:
            # Rule 3: a missing cell is never filled in by inference. No value,
            # no citation — and a present value or a borrowed citation would be
            # exactly the inference the contract forbids.
            if self.value is not None:
                raise ValueError(
                    f"missing cell '{self.major}/{self.metric}' must not carry "
                    "a value (inference forbidden by ticket 18 rule 3)"
                )
            if self.source is not None:
                raise ValueError(
                    f"missing cell '{self.major}/{self.metric}' must not carry "
                    "a citation (no provenance for an absent fact)"
                )
        elif self.state == CELL_STATE_PRESENT:
            # Rule 2: a high-impact cell carries its OWN citation. A present cell
            # with a value but no provenance is a fabricated citation — forbidden.
            if self.value is None:
                raise ValueError(
                    f"present cell '{self.major}/{self.metric}' must carry a value"
                )
            if self.source is None:
                raise ValueError(
                    f"present cell '{self.major}/{self.metric}' must carry its own "
                    "citation (ticket 18 rule 2: per-cell provenance, no shared "
                    "source_id)"
                )
        else:
            raise ValueError(
                f"unknown cell state {self.state!r}; expected 'present' or 'missing'"
            )
        return self


class RefusedMajor(BaseModel):
    """A major split out of the table because its year did not match."""

    major: str = Field(..., description="Major label being refused")
    effective_year: int = Field(..., description="The year the major actually carries")
    reason: str = Field(..., description="Why the major was excluded from the table")

    model_config = {"extra": "forbid"}


class ComparisonRow(BaseModel):
    """One major's row of cells."""

    major: str = Field(..., description="Major label, matches its cells")
    effective_year: int = Field(..., description="Admission year this row's facts carry")
    record_version_key: str = Field(
        ..., description="Latest record version backing this major"
    )
    cells: List[ComparisonCell] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ComparisonTable(BaseModel):
    """The comparison answer: one table per effective year.

    Rule 1 lives in :meth:`assert_year_uniformity`: every row MUST share the
    table's ``effective_year``, otherwise the table is invalid. The service
    splits differing-year majors into ``refused`` instead of producing such a
    table, so a constructed table is always year-uniform — but the validator
    keeps that invariant true even for hand-built inputs (the contract is the
    source of truth, not the service).
    """

    effective_year: int = Field(..., description="Admission year all rows share")
    query: str = Field(..., description="The comparison question this table answers")
    columns: List[ComparisonColumn]
    rows: List[ComparisonRow] = Field(default_factory=list)
    refused: List[RefusedMajor] = Field(
        default_factory=list,
        description="Majors excluded because their year did not match",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def assert_year_uniformity(self) -> "ComparisonTable":
        # Rule 1: same effective year only.
        for row in self.rows:
            if row.effective_year != self.effective_year:
                raise ValueError(
                    f"major {row.major!r} carries year {row.effective_year} "
                    f"but this table is year {self.effective_year} — "
                    "different years must be split out, not merged (rule 1)"
                )
        return self

    @model_validator(mode="after")
    def assert_cells_match_columns_and_rows(self) -> "ComparisonTable":
        # Shape integrity: every cell references a declared column and its table
        # row's major; every (row, column) pair has exactly one cell. This is
        # what makes the table iterable as rows x columns on the FE.
        labels = {c.metric for c in self.columns}
        expected = {(r.major, c.metric) for r in self.rows for c in self.columns}
        got: set[tuple[str, str]] = set()
        for row in self.rows:
            row_labels: set[str] = set()
            for cell in row.cells:
                if cell.major != row.major:
                    raise ValueError(
                        f"cell {cell.major!r}/{cell.metric!r} major does not match "
                        f"its row {row.major!r}"
                    )
                if cell.metric not in labels:
                    raise ValueError(
                        f"cell {cell.major!r}/{cell.metric!r} metric is not a "
                        "declared column"
                    )
                key = (cell.major, cell.metric)
                if key in got:
                    raise ValueError(
                        f"duplicate cell {key!r}: a comparison table has one cell "
                        "per major x metric"
                    )
                got.add(key)
                row_labels.add(cell.metric)
            missing = labels - row_labels
            if missing:
                raise ValueError(
                    f"row {row.major!r} is missing cells for metric(s) "
                    f"{sorted(missing)}"
                )
        if got != expected:
            missing = expected - got
            raise ValueError(
                f"table cells do not cover the full grid; missing {sorted(missing)}"
            )
        return self

    @model_validator(mode="after")
    def assert_one_citation_per_major(self) -> "ComparisonTable":
        # Rule 2, table-level: a SINGLE citation (record_version_key) shared
        # across the WHOLE table is forbidden — i.e. present cells spanning 2+
        # distinct majors all resolving to the same record_version_key. That is
        # the failure mode this contract stops: a table filled by one source_id
        # rather than per-cell provenance. A single major's row legitimately
        # shares its own record_version_key across its metrics (one canonical
        # fact sheet backs that major's row); what is illegal is one citation
        # standing in for multiple majors, so the rejection only fires when
        # present cells touch more than one distinct row.
        versions_by_major: dict[str, set[str]] = {}
        for row in self.rows:
            for cell in row.cells:
                if cell.state == CELL_STATE_PRESENT and cell.source:
                    versions_by_major.setdefault(row.major, set()).add(
                        cell.source.record_version_key
                    )
        distinct_versions = {v for vers in versions_by_major.values() for v in vers}
        if len(versions_by_major) > 1 and len(distinct_versions) == 1:
            only = next(iter(distinct_versions))
            raise ValueError(
                f"all present cells across {len(versions_by_major)} majors share "
                f"one citation ({only}) — per-cell provenance required "
                "(ticket 18 rule 2: never one source_id for the whole table)"
            )
        return self


class ComparisonResponse(BaseModel):
    """Top-level comparison answer returned by /v2/compare.

    Exactly one primary ``table`` (the same-year group); majors that could not
    be grouped (different effective year) are listed in ``refused`` so the FE
    can surface them instead of silently dropping or guessing. The
    ``route``/``intent`` echo the router's decision so the client can confirm
    the question was served by the multihop path (rule 4).
    """

    query: str = Field(..., description="The comparison question answered")
    route: str = Field(..., description="Pipeline route — must be 'graph_rag' for a comparison")
    intent: str = Field(..., description="Router intent label")
    effective_year: int = Field(..., description="Year all table rows share")
    table: ComparisonTable
    refused: List[RefusedMajor] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def assert_routed_to_multihop(self) -> "ComparisonResponse":
        # Rule 4: comparison questions are served by the multihop path. A
        # response built from a non-graph route is structurally invalid.
        if self.route != "graph_rag":
            raise ValueError(
                f"comparison response built with route={self.route!r}; ticket 18 "
                "requires the multihop (graph_rag) path"
            )
        return self

    @model_validator(mode="after")
    def assert_year_matches_table(self) -> "ComparisonResponse":
        if self.effective_year != self.table.effective_year:
            raise ValueError("response effective_year does not match table")
        if self.effective_year != self.table.effective_year:
            raise ValueError("year mismatch between response and table")
        return self

    def cell_count(self) -> Dict[tuple[str, str], str]:
        """`(major, metric) -> state` quick-view used by tests."""
        out: Dict[tuple[str, str]] = {}
        for row in self.table.rows:
            for cell in row.cells:
                out[(row.major, cell.metric)] = cell.state
        return out

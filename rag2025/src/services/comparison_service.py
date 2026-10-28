"""Comparison service for multi-major same-year queries (ticket 18).

Reads canonical records out of the canonical record store, groups them by the
``admission_year`` effective-year field, and builds a
:class:`~src.domain.comparison.ComparisonResponse` whose high-impact cells each
carry real per-cell provenance (no shared ``source_id`` for the whole table)
and whose missing cells are marked ``missing`` rather than inferred.

The service consumes the router's decision — it does NOT own routing — but it
enforces *contract 4*: a comparison response is only produced when the router's
route is ``graph_rag`` (the multihop path). If the router sent the question to a
non-graph route, the service refuses to build, because serving a comparison
table from a single-hop path is exactly the regression this ticket prevents.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from src.domain.canonical.record import CanonicalRecord, RecordStatus
from src.domain.comparison import (
    CELL_STATE_MISSING,
    CELL_STATE_PRESENT,
    CellProvenance,
    ComparisonCell,
    ComparisonColumn,
    ComparisonResponse,
    ComparisonRow,
    ComparisonTable,
    ComparisonTableError,
    RefusedMajor,
)
from src.infrastructure.canonical_record_store import (
    CanonicalRecordStore,
    RecordProvenance,
)
from src.services.source_document_service import SourceDocumentService


class ComparisonRoutingError(ValueError):
    """The router did not send this question to the multihop path (rule 4)."""


class ComparisonYearMismatchError(ValueError):
    """Caller asked to merge majors across two different effective years."""


# Metrics the comparison surface knows how to render. Each entry is a
# `(record.payload.field_name, human_label)` pair. These are the high-impact fact
# kinds a teacher compares between majors -- exactly the field surface that the
# per-cell citation rule (rule 2) protects. New metrics are additive.
DEFAULT_COMPARISON_METRICS: tuple[tuple[str, str], ...] = (
    ("diem_chuan", "điểm chuẩn"),
    ("hoc_phi", "học phí"),
    ("chi_tieu", "chỉ tiêu"),
)


class ComparisonService:
    """Builds comparison tables from canonical records + real provenance.

    ``store`` is the canonical record store (the source of
    ``resolve_provenance``, which turns a ``record_version_key`` into real
    ``source_document_id`` / ``source_url`` / ``source_page`` -- never a
    string). ``source_document`` optionally vouches for those citations against
    the registry; when present, a cell whose provenance points at an unregistered
    document is downgraded to ``missing`` rather than trusted blindly (ticket 39's
    blind-citation seam).
    """

    def __init__(
        self,
        store: CanonicalRecordStore,
        source_document: Optional[SourceDocumentService] = None,
        metrics: Sequence[tuple[str, str]] = DEFAULT_COMPARISON_METRICS,
    ) -> None:
        self._store = store
        self._source_document = source_document
        self._metrics = list(metrics)

    # ------------------------------------------------------------------ #
    # Public contract
    # ------------------------------------------------------------------ #
    def build(
        self,
        *,
        query: str,
        route: str,
        intent: str,
        effective_year: int,
        record_ids: Sequence[str],
    ) -> ComparisonResponse:
        """Build a same-year comparison response.

        Args:
            query: the comparison question.
            route: the router's route decision (``"graph_rag"`` required).
            intent: the router's intent label (expected ``"so_sanh"``).
            effective_year: the admission year all compared majors must share.
            record_ids: candidate canonical record_ids to read. The service
                reads the latest *published* version of each; records whose
                ``admission_year`` differs are refused (rule 1).

        Raises:
            ComparisonRoutingError: ``route`` is not ``graph_rag`` (rule 4).
            ComparisonTableError: a constructed table failed a model invariant
                (year uniformity / per-cell provenance / missing-cell rules).
        """
        # Rule 4: the question must have been routed to the multihop path.
        if route != "graph_rag":
            raise ComparisonRoutingError(
                f"comparison requires route 'graph_rag' (multihop); got {route!r}"
            )

        rows, refused = self._collect_rows(effective_year, record_ids)

        columns = [
            ComparisonColumn(metric=field, display=label)
            for field, label in self._metrics
        ]

        table = ComparisonTable(
            effective_year=effective_year,
            query=query,
            columns=columns,
            rows=rows,
            refused=refused,
        )
        return ComparisonResponse(
            query=query,
            route=route,
            intent=intent,
            effective_year=effective_year,
            table=table,
            refused=refused,
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _collect_rows(
        self, effective_year: int, record_ids: Sequence[str]
    ) -> tuple[list[ComparisonRow], list[RefusedMajor]]:
        """Group records by admission_year; refuse the year-mismatched ones.

        Only PUBLISHED versions are compared (drafts are not student-facing
        facts). A record whose year differs is *split out* into ``refused``
        with a stable reason -- it is never silently merged into a table whose
        year it does not carry.
        """
        rows: list[ComparisonRow] = []
        refused: list[RefusedMajor] = []

        seen: set[str] = set()
        for record_id in record_ids:
            version = self._store.get_record(record_id).latest
            if version.status is not RecordStatus.PUBLISHED:
                # Unpublished records carry no student-facing fact; leave them
                # out of the table rather than guessing a value for them.
                continue

            year = int(version.admission_year)
            if year != effective_year:
                # Rule 1: different effective year -> refused, not merged.
                refused.append(
                    RefusedMajor(
                        major=_record_label(version),
                        effective_year=year,
                        reason=(
                            f"effective year {year} differs from the requested "
                            f"comparison year {effective_year}; split out instead "
                            "of merged (ticket 18 rule 1)"
                        ),
                    )
                )
                continue

            # Real provenance, resolved from the store (never a hand-written
            # string). May be None when the document is not in the registry.
            provenance = self._store.resolve_provenance(version.record_version_key)
            cell_source = self._cell_source_for(version, provenance)

            major_label = _record_label(version)
            row = ComparisonRow(
                major=major_label,
                effective_year=year,
                record_version_key=version.record_version_key,
                cells=self._cells_for(version, major_label, cell_source),
            )

            if row.major in seen:
                # Two record ids for the same major in one year: keep one row,
                # don't duplicate - the canonical store key is (major, year).
                continue
            seen.add(row.major)
            rows.append(row)

        rows.sort(key=lambda r: r.major)
        refused.sort(key=lambda r: (r.effective_year, r.major))
        return rows, refused

    def _cells_for(
        self,
        record: CanonicalRecord,
        major_label: str,
        cell_source: Optional[CellProvenance],
    ) -> list[ComparisonCell]:
        """Render one cell per declared metric for a major's record.

        Each present cell takes its value + provenance from the SAME record --
        so different metrics in the same major resolve to different
        ``record_version_key`` / document provenance, and a metric that the
        record does not carry is ``missing`` (never copied from another major).

        When ``cell_source`` is None (the document is not in the registry, or
        the registry check downgraded it), every populated metric is downgraded
        to a ``missing`` cell instead of being emitted with a fabricated
        citation (rule 3 + ticket 39 blind-citation seam).
        """
        values: Mapping[str, object] = _record_values(record)
        cells: list[ComparisonCell] = []
        for field, _label in self._metrics:
            raw = values.get(field)
            if raw is None or raw == "" or _is_blank(raw):
                # Rule 3: missing cell -- never infer, never copy a sibling.
                cells.append(
                    ComparisonCell(
                        major=major_label,
                        metric=field,
                        state=CELL_STATE_MISSING,
                        value=None,
                        source=None,
                        note=None,
                    )
                )
                continue

            # Rule 2 + ticket 39: an unregistered/untrusted source document
            # makes the cell's provenance untrustworthy -- downgrade to missing
            # rather than stamp a bogus citation. A present cell must carry its
            # own provenance, so None provenance forces a missing cell.
            if cell_source is None:
                cells.append(
                    ComparisonCell(
                        major=major_label,
                        metric=field,
                        state=CELL_STATE_MISSING,
                        value=None,
                        source=None,
                        note="source document not in the registry",
                    )
                )
                continue

            cells.append(
                ComparisonCell(
                    major=major_label,
                    metric=field,
                    state=CELL_STATE_PRESENT,
                    value=_render_value(raw),
                    source=cell_source,
                )
            )
        return cells

    def _cell_source_for(
        self, version: CanonicalRecord, provenance: RecordProvenance
    ) -> Optional[CellProvenance]:
        """Build real per-cell provenance, or None when untrusted.

        A citation is real provenance resolved via the store -- never a
        hand-written string. When a source-document service is wired, a document
        id the registry does not vouch for (ticket 39 blind-citation surface) is
        treated as untrustworthy: provenance becomes None so the cell is
        downgraded to ``missing`` rather than handed a bogus citation. Without
        the service, the store's provenance is carried as-is; the model
        validator rejects a ``present`` cell that lost its citation.
        """
        if (
            self._source_document is not None
            and not self._source_document.is_registered(provenance.source_document_id)
        ):
            return None
        return CellProvenance(
            record_version_key=version.record_version_key,
            source_document_id=provenance.source_document_id,
            source_url=provenance.source_url,
            source_page=provenance.source_page,
            chunk_id=None,
        )


def _record_values(record: CanonicalRecord) -> Mapping[str, object]:
    """The record's structured payload, or an empty mapping.

    Mirrors the canonical schema's additive payload (`values` / `fields` /
    `facts`, see semantic_units._STRUCTURED_PAYLOAD_KEYS) without importing the
    chunker: the comparison contract cares which metric keys exist, not how the
    record was chunked.
    """
    for key in ("values", "fields", "facts"):
        payload = getattr(record, key, None)
        if isinstance(payload, Mapping):
            return payload
    return {}


def _record_label(record: CanonicalRecord) -> str:
    """A human label for a record's major.

    Canonical record_ids are stable identifiers, not display strings; the
    display label is taken from the record's structured payload when present,
    else the record_id. The label is cosmetic -- the contract's invariants ride
    on ``record_version_key`` and ``effective_year``, not on this string.
    """
    payload = _record_values(record)
    name = payload.get("ten_nganh") or payload.get("ma_nganh") or record.record_id
    return str(name)


def _render_value(raw: object) -> str:
    """Render a structured metric value to a stable display string."""
    if isinstance(raw, bool):
        return "có"
    if isinstance(raw, float):
        # Drop trailing-zero float formatting so 24.0 -> "24" for scores.
        if raw.is_integer():
            return str(int(raw))
        return repr(raw)
    return str(raw)


def _is_blank(value: object) -> bool:
    return isinstance(value, str) and value.strip() == ""

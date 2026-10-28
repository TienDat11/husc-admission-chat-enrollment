"""Unanswered-question queue — ticket 7.

When a question cannot be answered (out of scope, no evidence, or the
generator failed), the system queues it instead of silently dropping it. The
queue is the intake for the human-review batches the admission team runs
(T15), and it is also the anchor the student-facing ``POST /v2/feedback``
endpoint resolves against.

Rules that hold by construction:

* No raw query text is ever persisted. Queries are redacted through
  ``log_redaction.redact_pii`` — the same detectors as the guardrail — before
  anything touches the queue file. The redacted value is the only value ever
  stored, and no auxiliary field carries the unredacted text.
* ``resolved_at`` / ``resolved_by_record_id`` mark completion. T15 calls
  ``UnansweredQueueService.resolve``; nothing else rewrites the queue.
* JSONL-backed so append-only writes never truncate an in-flight line: one
  complete JSON object per line. Each file is written with a ``.tmp``
  sibling + ``os.replace`` so a crash leaves either the old or the new file
  whole, never a torn file.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.services.log_redaction import redact_pii

__all__ = [
    "FEEDBACK_FILENAME",
    "Item",
    "ItemNotFoundError",
    "QUEUE_DIRNAME",
    "UnansweredQueueError",
    "UnansweredQueueService",
]

QUEUE_DIRNAME = "unanswered"
QUEUE_FILENAME = "queue.jsonl"
FEEDBACK_FILENAME = "feedback.jsonl"


@dataclass
class Item:
    """One unanswered question plus its resolution state.

    ``question`` is already redacted at write time — never the raw user text,
    and never a field carrying the raw value. ``resolved`` is true only once
    an officer has resolved the item through ``UnansweredQueueService.resolve``.
    """

    question: str
    trace_id: str
    asked_at: str  # ISO-8601 UTC
    route: str
    status_code: str
    resolved_at: str | None = None
    resolved_by_record_id: str | None = None
    # K4 (T16): triage marker for the 48h-SLA denominator. An item
    # the team judges spam or outside HUSC's scope still counts as
    # "created"; the flag only removes it from the K4 denominator
    # (02 §9: "mục tạo trong tuần (trừ mục bị đánh spam/ngoài
    # phạm vi)"). False by default, so every existing row and every
    # older writer round-trips unchanged; ``from_dict`` defaults a
    # missing key to False for the same reason.
    out_of_scope: bool = False
    @property
    def resolved(self) -> bool:
        return self.resolved_at is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Item":
        return cls(
            question=str(row["question"]),
            trace_id=str(row["trace_id"]),
            asked_at=str(row["asked_at"]),
            route=str(row.get("route") or ""),
            status_code=str(row["status_code"]),
            resolved_at=(str(row["resolved_at"]) if row.get("resolved_at") else None),
            resolved_by_record_id=(
                str(row["resolved_by_record_id"]) if row.get("resolved_by_record_id") else None
            ),
            out_of_scope=bool(row.get("out_of_scope", False)),
        )


class UnansweredQueueError(Exception):
    """Base class for unanswered-queue failures."""


class ItemNotFoundError(UnansweredQueueError):
    """No queued item (open or resolved) carries that ``trace_id``."""


class UnansweredQueueService:
    """Append-only queue of questions that could not be answered.

    All mutations to the queue file happen under a single thread lock, so a
    review batch (T15) and a student feedback call can arrive concurrently
    without the file ever ending up with a torn line.
    """

    def __init__(
        self,
        queue_path: str | Path | None = None,
        *,
        data_dir: str | Path | None = None,
    ) -> None:
        """``data_dir`` is the parent directory the app gives at startup
        (the canonical-store root); the queue lives in ``<data_dir>/unanswered/``.
        ``queue_path`` overrides the full path, which is what tests use."""
        if queue_path is not None:
            self._queue_path = Path(queue_path)
            self._feedback_path = self._queue_path.with_name(FEEDBACK_FILENAME)
        elif data_dir is not None:
            self._queue_path = Path(data_dir) / QUEUE_DIRNAME / QUEUE_FILENAME
            self._feedback_path = self._queue_path.with_name(FEEDBACK_FILENAME)
        else:
            raise ValueError(
                "unanswered queue requires queue_path or data_dir — "
                "attach the service to app.state from a configured location"
            )
        self._lock = threading.Lock()

    @property
    def queue_path(self) -> Path:
        return self._queue_path

    @property
    def feedback_path(self) -> Path:
        return self._feedback_path

    def _ensure_dir(self) -> None:
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_lines(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _write_lines(self, path: Path, rows: list[dict[str, Any]]) -> None:
        self._ensure_dir()
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def push(self, question: str, trace_id: str, route: str, status_code: str) -> Item:
        """Queue an unanswered question.

        The raw ``question`` is redacted through ``log_redaction`` before it is
        written: the unredacted text is never persisted anywhere in this
        module, and no field carries it.

        Idempotent by ``trace_id``: a question already queued under that id is
        returned unchanged and nothing is appended, so a retried request does
        not duplicate the same item in the review batch.
        """
        if not trace_id.strip():
            raise ValueError("trace_id must not be empty")
        with self._lock:
            rows = self._read_lines(self._queue_path)
            for row in rows:
                if str(row["trace_id"]) == trace_id:
                    return Item.from_dict(row)
            item = Item(
                question=redact_pii(question),
                trace_id=trace_id,
                asked_at=datetime.now(timezone.utc).isoformat(),
                route=str(route),
                status_code=str(status_code),
            )
            rows.append(item.to_dict())
            self._write_lines(self._queue_path, rows)
            return item
    def resolve(self, trace_id: str, resolved_by_record_id: str) -> Item:
        """Mark an item as resolved.

        T15 calls this when an officer resolves a batch. It fills
        ``resolved_at`` (UTC ISO-8601) and ``resolved_by_record_id``, leaves
        everything else untouched, and preserves the redacted ``question``.
        Raises ``ItemNotFoundError`` when ``trace_id`` does not exist — a
        review batch must never record a resolution against a phantom item.
        """
        with self._lock:
            rows = self._read_lines(self._queue_path)
            for row in rows:
                if str(row["trace_id"]) == trace_id:
                    row["resolved_at"] = datetime.now(timezone.utc).isoformat()
                    row["resolved_by_record_id"] = resolved_by_record_id
                    self._write_lines(self._queue_path, rows)
                    return Item.from_dict(row)
            raise ItemNotFoundError(f"no unanswered item with trace_id {trace_id!r}")

    def mark_out_of_scope(self, trace_id: str, out_of_scope: bool = True) -> Item:
        """Flag an item as spam / out of scope (K4 triage).

        The flag only removes the item from the K4 SLA denominator;
        it never resolves the item and never rewrites anything else.
        Raises ``ItemNotFoundError`` for an unknown ``trace_id``,
        same refusal channel as ``resolve``.
        """
        with self._lock:
            rows = self._read_lines(self._queue_path)
            for row in rows:
                if str(row["trace_id"]) == trace_id:
                    row["out_of_scope"] = bool(out_of_scope)
                    self._write_lines(self._queue_path, rows)
                    return Item.from_dict(row)
            raise ItemNotFoundError(f"no unanswered item with trace_id {trace_id!r}")


    def items(self) -> list[Item]:
        """All items ever queued, in write order."""
        with self._lock:
            return [Item.from_dict(row) for row in self._read_lines(self._queue_path)]

    def unanswered(self) -> list[Item]:
        """Open items still waiting for review, in write order."""
        return [item for item in self.items() if not item.resolved]


    def feedback(
        self,
        trace_id: str,
        helpful: bool,
        reason: str | None = None,
        *,
        question: str | None = None,
        route: str = "served",
        status_code: str = "SUCCESS",
    ) -> dict[str, Any]:
        """Record a student's feedback on a served answer.

        The feedback is recorded regardless of whether the trace_id is present
        in the queue, so the public endpoint never rejects a student's input;
        the record carries a ``linked_item`` flag when a queued item exists for
        the same trace_id. Idempotent by ``trace_id``: a second call for the
        same answer returns the existing record rather than appending a
        duplicate.

        Negative feedback on a SUCCESS trace is promoted into the queue (the
        T07 loop closes): when ``question`` is supplied the trace is queued
        under it so the officer triage queue sees it.  Without ``question``
        (the served path never retains raw query text) the queue gets a
        truthful ``linked_item`` marker with an empty question string — no
        invented text.

        Bounds: ``trace_id`` must be 1–128 characters; ``reason`` must be
        at most 4096 characters.  Both are raised as ``ValueError`` (the
        file's existing refusal channel for malformed input, cf. ``push``).
        The service write path redacts ``reason`` through ``redact_pii`` so
        every caller gets the guarantee, not just the HTTP route.
        """
        if not trace_id.strip():
            raise ValueError("trace_id must not be empty")
        if len(trace_id) > 128:
            raise ValueError(f"trace_id exceeds 128 characters (got {len(trace_id)})")
        clean_reason = (reason or "").strip()
        if len(clean_reason) > 4096:
            raise ValueError(
                f"reason exceeds 4096 characters (got {len(clean_reason)})"
            )
        with self._lock:
            feedback_rows = self._read_lines(self._feedback_path)
            for row in feedback_rows:
                if str(row["trace_id"]) == trace_id:
                    return dict(row)
            queue_rows = self._read_lines(self._queue_path)
            matched = next(
                (row for row in queue_rows if str(row["trace_id"]) == trace_id),
                None,
            )
            # Close the T07 loop: a negative rating on a served answer is
            # triage-worthy.  Promote into the queue only when the trace is
            # not already there.  A supplied question is redacted and
            # stored; without one the queue gets a truthful empty-question
            # marker so the trace still appears in `unanswered()`.
            if not helpful and matched is None:
                if question is not None:
                    item = Item(
                        question=redact_pii(question),
                        trace_id=trace_id,
                        asked_at=datetime.now(timezone.utc).isoformat(),
                        route=route,
                        status_code=status_code,
                    )
                    queue_rows.append(item.to_dict())
                    self._write_lines(self._queue_path, queue_rows)
                    matched = item.to_dict()
                else:
                    row: dict[str, Any] = {
                        "question": "",
                        "trace_id": trace_id,
                        "asked_at": datetime.now(timezone.utc).isoformat(),
                        "route": route,
                        "status_code": status_code,
                        "resolved_at": None,
                        "resolved_by_record_id": None,
                    }
                    queue_rows.append(row)
                    self._write_lines(self._queue_path, queue_rows)
                    matched = row
            record = {
                "trace_id": trace_id,
                "helpful": helpful,
                "reason": redact_pii(clean_reason) if clean_reason else "",
                "answered_at": datetime.now(timezone.utc).isoformat(),
                "linked_item": matched is not None,
            }
            feedback_rows.append(record)
            self._write_lines(self._feedback_path, feedback_rows)
            return record

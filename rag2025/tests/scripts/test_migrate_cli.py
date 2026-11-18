"""U3 migration CLI: build_report (dry-run report), canonical hash, main (argparse entry).

Fixtures are written with the real writers (CanonicalRecordStore, UnansweredQueueService,
IdentityStore, AuditLedger); corruption cases edit the real bytes afterwards.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import scripts.migrate_file_stores as migrate
from scripts.migrate_file_stores import (
    RowsLoad,
    _canonical_hash,
    build_report,
    load_audit,
    load_identities,
    main,
)
from src.domain.admin_identity import Role
from src.domain.audit import AuditAction, AuditEvent, TargetKind
from src.domain.canonical.record import (
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from src.infrastructure.identity_store import IdentityEntry, IdentityStore
from src.services.unanswered_queue import UnansweredQueueService

RAG_ROOT = Path(__file__).resolve().parents[2]
CLI = RAG_ROOT / "scripts" / "migrate_file_stores.py"
STORE_ORDER = [
    "records", "retractions", "releases", "outbox", "conflicts", "idempotency",
    "unanswered", "feedback", "source_documents", "identities", "audit",
]
_PLAIN_ALICE = "alice-plaintext-credential-XYZ"
_PLAIN_BOB = "bob-plaintext-credential-QRS"
_PWHASH = "scrypt$16384$8$1$c2FsdA==$aGFzaA=="
_QUESTION_MARKER = "QUESTION-PLAINTEXT-MARKER"


def _rec(record_id: str, year: int = 2026) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=record_id,
        record_type=RecordType.CHI_TIEU,
        version=1,
        admission_year=year,
        effective_from=date(year, 1, 1),
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="doc-1",
        published_at=None,
        admission_round=None,
        admission_methods=[],
        text="noi dung",
    )


def _identity_fixture(root: Path) -> None:
    IdentityStore(root).seed([
        IdentityEntry(
            subject_id="6f1c2a4e-9d3b-4c1a-8e2f-1a2b3c4d5e6f", actor_id="alice", role=Role.EDITOR,
            credential=_PLAIN_ALICE, password_hash=_PWHASH,
        ),
        IdentityEntry(
            subject_id="bob-legacy", actor_id="bob", role=Role.ADMIN,
            credential=_PLAIN_BOB, disabled=True,
        ),
    ])


def _event(n: int, **overrides: object) -> AuditEvent:
    values = dict(
        action=AuditAction.PUBLISH,
        actor_id="alice",
        role="editor",
        target=f"rec-{n}",
        target_kind=TargetKind.RELEASE,
        timestamp=datetime(2026, 1, n, tzinfo=timezone.utc),
        before_version=str(n),
        after_version=str(n + 1),
        reason=f"reason {n}",
        release_id="rel-1",
    )
    values.update(overrides)
    return AuditEvent(**values)


def _ledger_fixture(root: Path) -> None:
    ledger = AuditLedger(root / "audit" / "ledger.jsonl")
    for n in (1, 2, 3):
        ledger.append(_event(n))


def _push_question(root: Path, question: str, trace: str = "trace-1") -> None:
    UnansweredQueueService(queue_path=root / "unanswered" / "queue.jsonl").push(
        question, trace, "route", "OUT_OF_SCOPE"
    )


def _push_unredacted_question(root: Path, question: str, trace: str) -> None:
    # The real push redacts before writing, so PII reaches the loader only if the
    # stored line is edited afterwards (as a pre-redaction legacy file would be).
    _push_question(root, "placeholder", trace)
    path = root / "unanswered" / "queue.jsonl"
    line = json.loads(path.read_text(encoding="utf-8").split("\n")[0])
    line["question"] = question
    path.write_text(json.dumps(line, ensure_ascii=False) + "\n", encoding="utf-8")


def _multi_store(root: Path) -> None:
    CanonicalRecordStore(root).write_record(_rec("CHI_TIEU:nganh:7220104:2026:x"))
    _push_question(root, f"hoc phi bao nhieu {_QUESTION_MARKER}")
    _identity_fixture(root)
    _ledger_fixture(root)


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ── 1. stores list: fixed order, entry shape, clean mismatch ─────────────


def test_report_lists_every_store_in_fixed_order_with_zero_mismatch(tmp_path):
    _multi_store(tmp_path)
    report = build_report(tmp_path)
    assert [s["name"] for s in report["stores"]] == STORE_ORDER
    for entry in report["stores"]:
        assert set(entry) == {"name", "source_rows", "target_rows", "mismatch", "hash"}
        assert entry["target_rows"] == entry["source_rows"]
        assert entry["mismatch"] == 0
    by_name = {s["name"]: s for s in report["stores"]}
    assert by_name["records"]["target_rows"] == 1
    assert by_name["unanswered"]["target_rows"] == 1
    assert by_name["identities"]["target_rows"] == 2
    assert by_name["audit"]["target_rows"] == 3
    assert by_name["releases"]["target_rows"] == 0


# ── 2. mismatch when source_rows != len(rows) ────────────────────────────


def test_mismatch_is_nonzero_when_loader_source_rows_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        migrate, "load_releases", lambda root: RowsLoad(rows=[{"a": 1}], source_rows=2)
    )
    by_name = {s["name"]: s for s in build_report(tmp_path)["stores"]}
    assert by_name["releases"]["source_rows"] == 2
    assert by_name["releases"]["target_rows"] == 1
    assert by_name["releases"]["mismatch"] == 1


# ── 3. idempotence ───────────────────────────────────────────────────────


def test_build_report_twice_is_equal_with_equal_report_hash(tmp_path):
    _multi_store(tmp_path)
    first = build_report(tmp_path)
    second = build_report(tmp_path)
    assert first == second
    assert first["report_hash"] == second["report_hash"]


# ── 4. canonical hash ────────────────────────────────────────────────────


def test_canonical_hash_golden_value_for_bytes_uuid_datetime_date():
    rows = [{
        "b": b"\x01\xff",
        "u": uuid.UUID(int=1),
        "d": datetime(2026, 1, 2, 3, 4, 5),
        "dt": date(2026, 1, 2),
    }]
    text = (
        '[{"b":"hex:01ff","d":"2026-01-02T03:04:05","dt":"2026-01-02",'
        '"u":"00000000-0000-0000-0000-000000000001"}]'
    )
    assert _canonical_hash(rows) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_canonical_hash_of_identities_and_audit_fixtures_is_stable(tmp_path):
    _identity_fixture(tmp_path)
    _ledger_fixture(tmp_path)
    identities = load_identities(tmp_path).rows
    assert any(isinstance(v, bytes) for row in identities for v in row.values())
    assert _canonical_hash(identities) == _canonical_hash(load_identities(tmp_path).rows)
    audit = load_audit(tmp_path).rows
    assert _canonical_hash(audit) == _canonical_hash(load_audit(tmp_path).rows)


def test_canonical_hash_rejects_set_and_unknown_objects():
    with pytest.raises(TypeError):
        _canonical_hash([{"x": {1, 2}}])
    with pytest.raises(TypeError):
        _canonical_hash([{"x": object()}])


# ── 5. report fields: shape, counts only, no row contents ────────────────


def test_report_top_level_shape_and_contains_no_row_contents(tmp_path):
    _multi_store(tmp_path)
    report = build_report(tmp_path)
    assert set(report) == {
        "stores", "redacted_rows", "quarantined_rows", "id_map", "unfillable", "normalised", "dry_run", "report_hash",
    }
    assert report["dry_run"] is True
    assert report["quarantined_rows"] == 0
    assert report["redacted_rows"] == 0
    assert "records" in report["id_map"]
    assert "retractions" not in report["id_map"]
    assert "identities" not in report["id_map"]
    assert "identities" in report["unfillable"]
    assert all(type(v) is int for v in report["unfillable"].values())

    text = json.dumps(report)
    assert _PLAIN_ALICE not in text
    assert _PLAIN_BOB not in text
    assert _QUESTION_MARKER not in text
    assert str(tmp_path) not in text


def test_report_hash_is_sha256_of_canonical_report_without_hash_key(tmp_path):
    _multi_store(tmp_path)
    report = build_report(tmp_path)
    body = {k: v for k, v in report.items() if k != "report_hash"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert report["report_hash"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── 6. redaction is counted (in-process; subprocess variant below) ───────


_RAW_CCCD = "012345678901"
_RAW_EMAIL = "nguoi@example.com"


def test_redaction_of_cccd_and_email_is_counted_and_never_printed(tmp_path, capsys):
    root = tmp_path / "root"
    _push_unredacted_question(root, f"CCCD {_RAW_CCCD} lien he {_RAW_EMAIL}", "trace-r")
    report = build_report(root)
    assert report["redacted_rows"] == 1
    assert _RAW_CCCD not in json.dumps(report)
    assert _RAW_EMAIL not in json.dumps(report)

    report_path = tmp_path / "out" / "r.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(root)]) == 0
    captured = capsys.readouterr()
    assert _RAW_CCCD not in captured.out + captured.err
    assert _RAW_EMAIL not in captured.out + captured.err
    assert _RAW_CCCD not in report_path.read_text(encoding="utf-8")
    assert _RAW_EMAIL not in report_path.read_text(encoding="utf-8")


def test_redaction_leaks_nothing_through_the_subprocess(tmp_path):
    root = tmp_path / "root"
    _push_unredacted_question(root, f"CCCD {_RAW_CCCD} lien he {_RAW_EMAIL}", "trace-r")
    report_path = tmp_path / "r.json"
    proc = _run_cli(root, report_path, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["redacted_rows"] == 1
    assert _RAW_CCCD not in proc.stdout + proc.stderr
    assert _RAW_EMAIL not in proc.stdout + proc.stderr
    assert _RAW_CCCD not in report_path.read_text(encoding="utf-8")
    assert _RAW_EMAIL not in report_path.read_text(encoding="utf-8")


def test_identity_subject_email_stays_out_of_report_and_output(tmp_path, capsys):
    root = tmp_path / "root"
    IdentityStore(root).seed([
        IdentityEntry(
            subject_id="someone@example.com", actor_id="someone", role=Role.EDITOR,
            credential=_PLAIN_ALICE, password_hash=_PWHASH,
        ),
    ])
    report = build_report(root)
    assert "identities" not in report["id_map"]
    assert "someone@example.com" not in json.dumps(report)
    report_path = tmp_path / "out" / "r.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(root)]) == 0
    captured = capsys.readouterr()
    assert "someone@example.com" not in captured.out + captured.err
    assert "someone@example.com" not in report_path.read_text(encoding="utf-8")


def test_drift_on_second_load_counts_as_mismatch(tmp_path, monkeypatch, capsys):
    def drifting_loader():
        answers = [RowsLoad(rows=[{"a": 1}], source_rows=1), RowsLoad(rows=[{"a": 2}], source_rows=1)]
        return lambda root: answers.pop(0)

    monkeypatch.setattr(migrate, "load_releases", drifting_loader())
    by_name = {s["name"]: s for s in build_report(tmp_path)["stores"]}
    assert by_name["releases"]["mismatch"] == 1

    monkeypatch.setattr(migrate, "load_releases", drifting_loader())
    report_path = tmp_path / "out" / "r.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(tmp_path)]) == 0
    assert "mismatch total 1" in capsys.readouterr().out


def test_report_hash_is_equal_for_identical_content_under_different_roots(tmp_path):
    first = tmp_path / "a"
    _multi_store(first)
    second = tmp_path / "b"
    shutil.copytree(first, second)
    assert build_report(first)["report_hash"] == build_report(second)["report_hash"]


def _denied(root):
    raise PermissionError(13, "denied", "/x/y")


def test_os_error_from_a_loader_exits_2_naming_the_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(migrate, "load_releases", _denied)
    report_path = tmp_path / "r.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(tmp_path)]) == 2
    assert "/x/y" in capsys.readouterr().err
    assert not report_path.exists()


def test_type_error_from_serializing_a_row_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        migrate, "load_releases", lambda root: RowsLoad(rows=[{"x": {1, 2}}], source_rows=1)
    )
    report_path = tmp_path / "r.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(tmp_path)]) == 2
    assert capsys.readouterr().err.strip() != ""
    assert not report_path.exists()


def test_failed_replace_exits_2_leaves_no_tmp_and_keeps_old_report(tmp_path, monkeypatch, capsys):
    report_path = tmp_path / "r.json"
    report_path.write_bytes(b"previous-report")

    def broken_replace(src, dst):
        raise OSError(28, "disk full", str(dst))

    monkeypatch.setattr(migrate.os, "replace", broken_replace)
    assert main(["--dry-run", "--report", str(report_path), "--root", str(tmp_path / "absent")]) == 2
    assert report_path.read_bytes() == b"previous-report"
    assert not Path(str(report_path) + ".tmp").exists()
    assert capsys.readouterr().err.strip() != ""


# ── 7. main writes the report; corrupt input exits 2 and writes nothing ──


def test_main_dry_run_writes_json_equal_to_build_report(tmp_path, capsys):
    root = tmp_path / "root"
    _multi_store(root)
    report_path = tmp_path / "out" / "report.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(root)]) == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == build_report(root)
    assert not Path(str(report_path) + ".tmp").exists()
    out = capsys.readouterr().out
    assert f"wrote {report_path}" in out
    assert "mismatch total 0" in out


@pytest.mark.parametrize("which", ["record", "queue"])
def test_main_corrupt_file_exits_2_names_path_and_writes_nothing(tmp_path, capsys, which):
    root = tmp_path / "root"
    _multi_store(root)
    bad = _corrupt(root, which)
    report_path = tmp_path / "report.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(root)]) == 2
    assert str(bad) in capsys.readouterr().err
    assert not report_path.exists()


def test_main_corrupt_file_leaves_preexisting_report_untouched(tmp_path, capsys):
    root = tmp_path / "root"
    _multi_store(root)
    _corrupt(root, "queue")
    report_path = tmp_path / "report.json"
    report_path.write_bytes(b"previous-report")
    assert main(["--dry-run", "--report", str(report_path), "--root", str(root)]) == 2
    assert report_path.read_bytes() == b"previous-report"


def _corrupt(root: Path, which: str) -> Path:
    if which == "record":
        bad = next((root / "records").rglob("v*.json"))
        data = bad.read_bytes()
        bad.write_bytes(data[: len(data) // 2])
        return bad
    bad = root / "unanswered" / "queue.jsonl"
    bad.write_text(bad.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    return bad


# ── 8. subprocess: sys.path bootstrap and exit codes through the real CLI ─


def _run_cli(root: Path, report_path: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), "--dry-run", "--report", str(report_path), "--root", str(root)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_subprocess_dry_run_writes_correct_json(tmp_path):
    root = tmp_path / "root"
    _multi_store(root)
    report_path = tmp_path / "r.json"
    proc = _run_cli(root, report_path, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(report_path.read_text(encoding="utf-8")) == build_report(root)


def test_cli_subprocess_corrupt_root_exits_2_and_writes_nothing(tmp_path):
    root = tmp_path / "root"
    _multi_store(root)
    bad = _corrupt(root, "queue")
    report_path = tmp_path / "r.json"
    proc = _run_cli(root, report_path, tmp_path)
    assert proc.returncode == 2
    assert str(bad) in proc.stderr
    assert not report_path.exists()


def test_cli_subprocess_counts_redaction(tmp_path):
    root = tmp_path / "root"
    _push_unredacted_question(root, "CCCD 012345678901 lien he nguoi@example.com", "trace-r")
    report_path = tmp_path / "r.json"
    proc = _run_cli(root, report_path, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(report_path.read_text(encoding="utf-8"))["redacted_rows"] >= 1


# ── 9. real run refused; missing --report is an argparse error ───────────


def test_real_run_is_refused_and_writes_nothing(tmp_path, capsys):
    report_path = tmp_path / "r.json"
    assert main(["--report", str(report_path), "--root", str(tmp_path)]) == 2
    assert "real run not implemented" in capsys.readouterr().err
    assert not report_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_missing_report_is_argparse_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--dry-run", "--root", str(tmp_path)])
    assert exc.value.code == 2


# ── 10. dry run writes nothing under root ────────────────────────────────


def test_dry_run_leaves_root_tree_unchanged(tmp_path):
    root = tmp_path / "root"
    _multi_store(root)
    before = _tree(root)
    assert before
    assert main(["--dry-run", "--report", str(tmp_path / "out" / "r.json"), "--root", str(root)]) == 0
    assert _tree(root) == before


# ── 11. absent or empty root: every store is empty and clean ─────────────


@pytest.mark.parametrize("mode", ["absent", "empty"])
def test_absent_or_empty_root_reports_zero_rows_and_exits_0(tmp_path, mode):
    root = tmp_path / "root"
    if mode == "empty":
        root.mkdir()
    report = build_report(root)
    assert [s["name"] for s in report["stores"]] == STORE_ORDER
    for s in report["stores"]:
        assert s["source_rows"] == 0 and s["target_rows"] == 0 and s["mismatch"] == 0
    report_path = tmp_path / "r.json"
    assert main(["--dry-run", "--report", str(report_path), "--root", str(root)]) == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

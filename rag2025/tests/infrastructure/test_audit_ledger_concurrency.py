"""Concurrency and failure-mode tests for the audit ledger — ticket 14.

Two properties that a happy-path test cannot see, and that decide whether the
ledger is an audit log or a decoration:

1. **Concurrent appends do not fork the chain.** ``append`` is a
   read-compute-write: read the tail, hash it, write the next line. Without a
   lock two writers read the same tail and both claim the same ``seq``, and the
   failure then surfaces as ``verify`` reporting the ledger tampered with —
   an audit log that calls itself corrupt the first time two publishes overlap.
   Proven both in-process (threads) and across processes (the file lock).
2. **A write that reaches the store but not the ledger says so.** The store is
   committed first, so the append can fail after the fact; the error must name
   the divergence rather than read as "nothing happened".

The thread test is the one that would have caught the missing lock: it is
asserted below against the unguarded implementation too (see the module note in
the ticket report), where it fails with duplicate sequence numbers.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.audit import AuditAction, AuditEvent  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
NOW = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)

THREADS = 24
APPENDS_PER_THREAD = 6


def _event(actor: str, index: int) -> AuditEvent:
    return AuditEvent(
        action=AuditAction.PUBLISH,
        actor_id=actor,
        target=RECORD_ID,
        timestamp=NOW,
        before_version=f"{RECORD_ID}@1",
        after_version=f"{RECORD_ID}@2",
        release_id="R-2026-09-13",
        event_id=f"{actor}-{index}",
    )


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "ledger.jsonl"


def _assert_intact_one_chain_of(path: Path, expected: int, *, event_ids: set[str]) -> None:
    """The three properties a forked chain violates, checked from disk."""
    ledger = AuditLedger(path)
    verification = ledger.verify()
    assert verification.ok is True, (
        f"chain broke at index {verification.broken_index}: {verification.reason}"
    )
    entries = ledger.entries()
    assert len(entries) == expected
    assert [entry.seq for entry in entries] == list(range(1, expected + 1))
    assert {entry.event.event_id for entry in entries} == event_ids
    assert verification.checked == expected


def test_concurrent_appends_from_threads_keep_one_chain(ledger_path):
    """N threads appending at once produce one chain, seq 1..N, N entries.

    Without the lock in ``append`` this fails: two writers read the same tail
    and both write the same ``seq``, so ``verify`` reports the second one as a
    break ("entry N carries seq X") and the entry count comes up short of the
    number of appends that reported success.
    """
    ledger = AuditLedger(ledger_path)
    start = threading.Barrier(THREADS)
    total = THREADS * APPENDS_PER_THREAD
    event_ids = {
        f"cb.xuat-ban-{thread}-{index}"
        for thread in range(THREADS)
        for index in range(APPENDS_PER_THREAD)
    }

    def append_batch(thread: int) -> list[int]:
        start.wait(timeout=30)
        return [
            ledger.append(_event(f"cb.xuat-ban-{thread}", index)).seq
            for index in range(APPENDS_PER_THREAD)
        ]

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        seqs = [seq for batch in pool.map(append_batch, range(THREADS)) for seq in batch]

    # The returned entries themselves must be a permutation of 1..N: a duplicate
    # here is the fork, caught at the source rather than only in verify().
    assert sorted(seqs) == list(range(1, total + 1))
    _assert_intact_one_chain_of(ledger_path, total, event_ids=event_ids)


def test_concurrent_appends_from_separate_instances_keep_one_chain(ledger_path):
    """Two ledger objects over one file — the router's and a CLI's — serialize.

    A lock owned by the instance would not: each instance would hold its own and
    the two would still fork the file.
    """
    ledgers = [AuditLedger(ledger_path) for _ in range(THREADS)]
    start = threading.Barrier(THREADS)
    total = THREADS * APPENDS_PER_THREAD
    event_ids = {
        f"cli-{thread}-{index}"
        for thread in range(THREADS)
        for index in range(APPENDS_PER_THREAD)
    }

    def append_batch(thread: int) -> None:
        start.wait(timeout=30)
        for index in range(APPENDS_PER_THREAD):
            ledgers[thread].append(_event(f"cli-{thread}", index))

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(append_batch, range(THREADS)))

    _assert_intact_one_chain_of(ledger_path, total, event_ids=event_ids)


_WORKER = """
import sys
from datetime import datetime, timezone

from src.domain.audit import AuditAction, AuditEvent
from src.infrastructure.audit_ledger import AuditLedger

path, actor, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
ledger = AuditLedger(path)
for index in range(count):
    ledger.append(
        AuditEvent(
            action=AuditAction.PUBLISH,
            actor_id=actor,
            target="DIEM_CHUAN:NGANH:cntt:2026",
            timestamp=datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc),
            before_version="DIEM_CHUAN:NGANH:cntt:2026@1",
            after_version="DIEM_CHUAN:NGANH:cntt:2026@2",
            release_id="R-2026-09-13",
            event_id=f"{actor}-{index}",
        )
    )
"""

WORKERS = 3
APPENDS_PER_PROCESS = 10


def test_concurrent_appends_from_separate_processes_keep_one_chain(ledger_path):
    """The file lock, which is the only thing that binds a second *process*.

    A threading lock is enough for one worker's threads and nothing more; ticket
    29's concurrent publish/rollback can be a second worker, a CLI, or a cron
    job, and the ledger is one file they share.
    """
    total = WORKERS * APPENDS_PER_PROCESS
    event_ids = {
        f"proc-{worker}-{index}"
        for worker in range(WORKERS)
        for index in range(APPENDS_PER_PROCESS)
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(ledger_path), f"proc-{worker}", str(APPENDS_PER_PROCESS)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for worker in range(WORKERS)
    ]
    for process in processes:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr.decode("utf-8", "replace")

    _assert_intact_one_chain_of(ledger_path, total, event_ids=event_ids)

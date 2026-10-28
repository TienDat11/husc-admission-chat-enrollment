"""Subprocess-isolated document parsing — ticket T04.

The chatbot must keep answering queries while an admin uploads a hostile
file, so a PDF/DOCX/XLSX parse never runs in-process: the parent spawns this
module as a worker in a *subprocess* and collects a JSON result. Three
isolation layers, each the strongest one its platform allows:

1. **Process isolation.** The worker is a separate ``python`` process; a
   hostile file that corrupts the parser's memory cannot take the chatbot
   down.
2. **Timeout.** The parent kills the whole process tree after
   ``timeout_s`` — a file that makes the parser spin (deeply nested
   objects, zip-slip loops) cannot hang the upload route forever.
3. **Soft memory bound.** The parent polls the worker's RSS with psutil
   and kills the tree at ``ram_limit_mb``. On Windows there is no
   ``RLIMIT_AS``/cgroup equivalent, so this is a *soft* limit — the worker
   can exceed it in the gap between polls before the kill lands. The Docker
   hard-limit upgrade path is T13's (container ``--memory``), which makes
   the OS itself enforce a real ceiling; until then the poll is the
   best the host OS allows without new infrastructure.

The worker also enforces a **decompressed-byte cap** per document:
PDF content streams and DOCX/XLSX zip members are tallied as they are
read, and a file whose content inflates past ``max_decompressed_mb`` is
refused with a named error — that is the decompression-bomb path, and
because it runs in the worker process the chatbot's event loop is
unaffected either way (acceptance criterion 5).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.services.magic_bytes import DOCX_KIND, PDF_KIND, XLSX_KIND

WorkerKind = Literal[PDF_KIND, DOCX_KIND, XLSX_KIND]

#: Poll interval for the RSS soft-check; short enough to bound the overshoot,
#: long enough to keep the poll itself negligible.
_RSS_POLL_S = 0.25

#: A page (PDF) or the whole document (DOCX) counts as *scanned* — image
#: only, no extractable text — when it yields fewer characters than this.
SCAN_PAGE_CHAR_THRESHOLD = 30


@dataclass(frozen=True)
class DocumentParseResult:
    """What the parser read out of one document, and how loudly it complained.

    ``error`` is ``None`` on success; a set ``error`` names the refusal
    (timeout, memory cap, decompression cap, unreadable file) and ``text``
    is ``""`` — the caller treats an errored result as "nothing was read",
    which is what keeps a scan-only or hostile document from producing an
    empty draft record.
    """

    text: str
    page_count: int
    chars_per_page: float
    parser: str
    scan_pages: tuple[int, ...]
    pii_detected: bool
    table_count: int
    headings: tuple[dict, ...] = ()
    #: One dict per spreadsheet data row (header -> value); empty for
    #: PDF/DOCX, populated by the xlsx parser (ticket T17).
    excel_rows: tuple[dict, ...] = ()
    refused: bool = False
    error: str | None = None
    warnings: tuple[str, ...] = ()


def parse_document(
    path: str | Path,
    kind: WorkerKind,
    *,
    ram_limit_mb: int = 1024,
    timeout_s: float = 60.0,
    max_decompressed_mb: int = 512,
) -> DocumentParseResult:
    """Parse ``path`` in a subprocess and report back.

    Never raises for a *parse* failure — a hostile or corrupt file comes
    back as ``error`` set. The only way this function raises is the
    worker interpreter itself failing to start (no Python on
    ``sys.executable``), which is a deployment error, not an input error.
    """
    document = Path(path)
    if not document.is_file():
        return _error_result(f"file not found: {document}")

    command = [
        sys.executable,
        "-m",
        "src.services.document_parser_worker",
        str(document),
        kind,
        str(max_decompressed_mb),
    ]
    env = _worker_env()
    try:
        process = _spawn(command, env)
    except OSError as exc:
        return _error_result(f"failed to start parser subprocess: {exc}")

    try:
        deadline = timeout_s
        while True:
            try:
                stdout, stderr = process.communicate(
                    timeout=_RSS_POLL_S,
                )
                break
            except subprocess.TimeoutExpired:
                deadline -= _RSS_POLL_S
                if deadline <= 0:
                    _kill_process_tree(process.pid)
                    process.wait(timeout=10.0)
                    return _error_result(
                        f"parse timed out after {timeout_s:.0f}s "
                        "(the process tree was killed)",
                        refused=True,
                    )
                if not _rss_within(process.pid, ram_limit_mb):
                    _kill_process_tree(process.pid)
                    process.wait(timeout=10.0)
                    return _error_result(
                        f"parser exceeded the {ram_limit_mb} MB soft memory "
                        "cap (killed); retry with a smaller document or "
                        "run inside the Docker hard-limit (ticket T13)",
                        refused=True,
                    )
        return _read_worker_result(process, stdout, stderr)
    finally:
        if process.poll() is None:
            _kill_process_tree(process.pid)


def _read_worker_result(
    process: subprocess.Popen,
    stdout: bytes,
    stderr: bytes,
) -> DocumentParseResult:
    """Turn the worker's last JSON line into a result; any deviation errors.

    The worker prints exactly one JSON object on the last stdout line;
    everything before it (progress, library noise) is ignored. A
    non-zero exit *with* a JSON payload is still reported (the payload
    carries the error), a non-zero exit *without* one surfaces the
    stderr tail — the parser crashing is an error, not a silent empty.
    """
    payload: dict[str, Any] | None = None
    for line in reversed(stdout.decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
            break
    if payload is not None and ("error" in payload or "text" in payload):
        return DocumentParseResult(
            text=str(payload.get("text", "")),
            page_count=int(payload.get("page_count", 0)),
            chars_per_page=float(payload.get("chars_per_page", 0.0)),
            parser=str(payload.get("parser", "none")),
            scan_pages=tuple(
                int(page) for page in payload.get("scan_pages", ())
            ),
            pii_detected=bool(payload.get("pii_detected", False)),
            table_count=int(payload.get("table_count", 0)),
            headings=tuple(
                {"text": h["text"], "offset": int(h["offset"])}
                for h in payload.get("headings", ())
            ),
            excel_rows=tuple(
                dict(row) for row in payload.get("excel_rows", ())
            ),
            refused=bool(payload.get("refused", False)),
            error=payload.get("error"),
            warnings=tuple(str(w) for w in payload.get("warnings", ())),
        )
    if process.returncode == 0:
        return _error_result(
            "parser subprocess exited successfully but produced no result"
        )
    detail = stderr.decode("utf-8", "replace").strip().splitlines()
    tail = detail[-1] if detail else "no stderr output"
    return _error_result(f"parser subprocess failed (exit {process.returncode}): {tail}")


def _error_result(message: str, *, refused: bool = False) -> DocumentParseResult:
    return DocumentParseResult(
        text="",
        page_count=0,
        chars_per_page=0.0,
        parser="none",
        scan_pages=(),
        pii_detected=False,
        table_count=0,
        headings=(),
        refused=refused,
        error=message,
    )


def _worker_env() -> dict[str, str]:
    """An environment in which ``python -m src.services...`` resolves.

    The worker imports both ``src.*`` and the bare ``services.*`` /
    ``config.*`` aliases (the source root, not its parent), so BOTH the
    rag2025 package root and ``rag2025/src`` must be on ``PYTHONPATH``.

    In production ``PYTHONPATH=src`` is exported (Dockerfile, CI), so the
    bare aliases resolve and the package root is added here for ``src.*``.
    Under a bare local ``pytest`` (no PYTHONPATH), ``conftest.py`` puts both
    roots on the parent's ``sys.path`` — but ``sys.path`` is not inherited
    by a subprocess, so the worker would fail on the first ``services.*``
    import. Both roots are therefore guaranteed here, which makes the
    worker independent of how the parent was launched.
    """
    env = dict(os.environ)
    package_root = Path(__file__).resolve().parents[2]
    source_root = package_root / "src"
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    for root in (str(source_root), str(package_root)):
        if root not in parts:
            parts.insert(0, root)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _spawn(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    """Spawn the worker in its own process group where the OS allows it.

    A new group/session is what makes *killing the tree* reliable on
    POSIX (``kill -PGID`` semantics via the session) and on Windows
    (``CREATE_NEW_PROCESS_GROUP`` detaches the worker from the parent's
    console, so a console event cannot take it down halfway through a
    parse). Without it the tree-kill below would only reach the worker
    itself.
    """
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )


def _kill_process_tree(pid: int) -> None:
    """Kill ``pid`` and every process it spawned, parent first.

    psutil reaches the children on every platform; if psutil is
    unavailable the direct kill still lands, and the worker spawns no
    children of its own, so the fallback degrades gracefully.
    """
    try:
        import psutil
    except ImportError:
        _kill_single(pid)
        return
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return  # already gone
    try:
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                continue
        parent.kill()
    except psutil.Error:
        _kill_single(pid)


def _kill_single(pid: int) -> None:
    if sys.platform == "win32":
        # /F /T kills the whole tree; this is the psutil-free Windows path.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _rss_within(pid: int, ram_limit_mb: int) -> bool:
    """Soft memory bound: the worker's RSS stays under the cap.

    A failed RSS read (process already reaped) reads as *within* — the
    ``communicate`` that follows will finish immediately. Only a
    measured overshoot triggers the kill.
    """
    try:
        import psutil

        rss = psutil.Process(pid).memory_info().rss
    except ImportError:
        return True
    except psutil.Error:
        return True
    return rss <= ram_limit_mb * 1024 * 1024

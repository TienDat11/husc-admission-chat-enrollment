"""One-shot re-index run for the operator - ticket 15 (closes B15-1's manual path).

The production auto-runner is ``src.services.reindex_runner.ReindexRunner``
started by the app lifecycle; this script is the *same runner object* driven
once by hand. It constructs the runner with the configured store, outbox, and
a real (lazy) embedder, calls ``run_once()`` exactly once, and reports what the
runner decided. It never constructs a ``ReindexJob`` itself - the runner is the
single construction site, so the manual path and the automatic path cannot
drift into two different notions of "what is due".

Exit codes:
  0  nothing was due, or the run published a serving corpus
  1  a run was attempted and failed, or work was due but could not be done
  2  the runner could not be constructed (bad configuration)

Usage (from rag2025/):
  python scripts/run_reindex.py
  python scripts/run_reindex.py --store-root D:/data/canonical_store
  python scripts/run_reindex.py --corpus-dir D:/data/canonical_corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from config.settings import RAGSettings  # noqa: E402
from src.services.embedding import EmbeddingService  # noqa: E402
from src.services.reindex_runner import ReindexRunner  # noqa: E402


def _build_runner(settings: RAGSettings, args: argparse.Namespace) -> ReindexRunner:
    """The runner over this deployment's real paths, mirroring app wiring.

    Store root, releases root, and outbox path are resolved exactly the way
    ``src/app/lifecycle.py::configure_canonical_record_store`` resolves them,
    so a manual run consumes the same outbox the app writes on publish.
    ``corpus_dir`` is only passed when the operator names one; otherwise the
    runner resolves ``release_status.DEFAULT_CORPUS_DIR`` lazily, which is the
    same directory the index-jobs read model reports.
    """
    from src.infrastructure.canonical_record_store import CanonicalRecordStore
    from src.services.admin_rollback import RELEASES_DIRNAME
    from src.services.reindex_outbox import OUTBOX_FILENAME, ReindexOutbox

    root = Path(args.store_root) if args.store_root else Path(settings.CANONICAL_STORE_ROOT)

    store = CanonicalRecordStore(root)
    outbox = ReindexOutbox(root / "outbox" / OUTBOX_FILENAME)
    embedder = EmbeddingService(settings)
    return ReindexRunner(
        store=store,
        outbox=outbox,
        embedder_provider=lambda: embedder,
        corpus_dir=args.corpus_dir,
        releases_root=root / RELEASES_DIRNAME,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_reindex",
        description="Run the incremental re-index once (operator path over ReindexRunner).",
    )
    parser.add_argument("--store-root", help="canonical store root (default: CANONICAL_STORE_ROOT)")
    parser.add_argument(
        "--corpus-dir",
        help="serving corpus dir to publish into (default: the runner's DEFAULT_CORPUS_DIR)",
    )
    args = parser.parse_args(argv)

    # RAGSettings' env_file is the relative ".env" (ingest_lancedb.py:29 does
    # the same): an operator invoking this from outside rag2025/ would
    # otherwise silently get defaults for the store root.
    load_dotenv()
    settings = RAGSettings()
    try:
        runner = _build_runner(settings, args)
    except Exception as exc:  # noqa: BLE001 - a runner that cannot exist is exit 2
        print(f"run_reindex: cannot construct runner: {exc}", file=sys.stderr)
        return 2

    outcome = runner.run_once()

    if outcome.error is not None:
        print(
            f"run_reindex: FAILED ({outcome.reason or 'no reason'}): {outcome.error}",
            file=sys.stderr,
        )
        return 1
    if not outcome.due:
        print(f"run_reindex: nothing due ({outcome.reason or 'no reason'})")
        return 0
    if outcome.action == "noop":
        print(f"run_reindex: no-op ({outcome.reason or 'nothing to re-index'})")
        return 0
    if outcome.action == "skipped":
        print(
            f"run_reindex: work was due but the run was skipped "
            f"({outcome.reason or 'no reason'})",
            file=sys.stderr,
        )
        return 1
    result = outcome.result
    if result is None or not result.serving:
        state = getattr(result, "state", None)
        print(f"run_reindex: ran but the corpus is not serving (state={state})", file=sys.stderr)
        return 1
    print(
        "run_reindex: corpus published and serving "
        f"(mode={result.plan.mode.value}, embedded={result.embedded_count}, "
        f"reused={len(result.reused_chunk_ids)}, removed={len(result.removed_chunk_ids)}, "
        f"cursor_advanced={result.outbox_cursor_advanced})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

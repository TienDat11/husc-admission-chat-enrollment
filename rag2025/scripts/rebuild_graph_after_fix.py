"""@spec(S18.6-T7) Wrapper script that rebuilds the graphRAG knowledge graph
from data sạch (post-fix JSONL). Calls scripts/build_graph.py with safe
defaults so auto_ingest_yearly.py (T6) can invoke it as the final step.

Idempotent: re-running is a no-op (build_graph.py is incremental-safe).

Usage:
    python scripts/rebuild_graph_after_fix.py           # full rebuild (slow)
    python scripts/rebuild_graph_after_fix.py --quiet   # only summary
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild knowledge_graph.graphml + entity_index.json from clean data")
    p.add_argument("--quiet", action="store_true", help="Only print summary line")
    p.add_argument("--incremental", action="store_true", help="Only process new chunks")
    p.add_argument("--limit", type=int, default=0, help="Max chunks (0=all)")
    args = p.parse_args()

    build_script = REPO_ROOT / "scripts" / "build_graph.py"
    if not build_script.exists():
        print(f"ERROR: build_graph.py not found at {build_script}", file=sys.stderr)
        return 2

    graph_path = REPO_ROOT / "data" / "graph" / "knowledge_graph.graphml"
    entity_path = REPO_ROOT / "data" / "graph" / "entity_index.json"
    mtime_before = graph_path.stat().st_mtime if graph_path.exists() else 0.0

    cmd = [sys.executable, str(build_script)]
    if args.incremental:
        cmd.append("--incremental")
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    started = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - started

    if proc.returncode != 0:
        print(f"rebuild_graph_after_fix: build_graph.py exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    mtime_after = graph_path.stat().st_mtime if graph_path.exists() else 0.0
    import json
    ent_count = 0
    if entity_path.exists():
        try:
            ent_count = len(json.loads(entity_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            ent_count = -1

    summary = {
        "graph_mtime_before": mtime_before,
        "graph_mtime_after": mtime_after,
        "mtime_changed": mtime_after > mtime_before,
        "entity_count": ent_count,
        "elapsed_seconds": round(elapsed, 1),
        "incremental": args.incremental,
    }
    if not args.quiet:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"rebuild_graph_after_fix: entities={ent_count} "
            f"elapsed={elapsed:.1f}s mtime_changed={summary['mtime_changed']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
# @spec(S13.1) snapshot legacy chunks before reingest
"""Snapshot rag2025/data/chunked/ to a timestamped legacy directory before destructive reingest."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_chunks(source_dir: Path, target_root: Path, timestamp: str | None = None) -> Path:
    """Copy *.jsonl + .ingest_manifest.json from source_dir to target_root/<timestamp>/.

    Args:
        source_dir: directory containing chunked_*.jsonl files.
        target_root: parent directory where the timestamped snapshot dir will be created.
        timestamp: explicit ISO-safe UTC timestamp (e.g. "20260525T120000Z"). When None,
            auto-generated from datetime.now(timezone.utc).

    Returns:
        Path to the created snapshot subdirectory.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    source_dir = Path(source_dir)
    target_root = Path(target_root)
    target_subdir = target_root / timestamp
    # Idempotent uniqueness: refuse to overwrite existing snapshot.
    target_subdir.mkdir(parents=True, exist_ok=False)

    files_meta: list[dict[str, Any]] = []
    total_bytes = 0

    for jsonl in sorted(source_dir.glob("*.jsonl")):
        dest = target_subdir / jsonl.name
        shutil.copy2(jsonl, dest)
        size = dest.stat().st_size
        files_meta.append({"name": jsonl.name, "size_bytes": size, "sha256": _compute_sha256(dest)})
        total_bytes += size

    manifest = source_dir / ".ingest_manifest.json"
    if manifest.exists():
        dest = target_subdir / ".ingest_manifest.json"
        shutil.copy2(manifest, dest)
        size = dest.stat().st_size
        files_meta.append({"name": ".ingest_manifest.json", "size_bytes": size, "sha256": _compute_sha256(dest)})
        total_bytes += size

    meta = {
        "source_dir": str(source_dir.resolve()),
        "timestamp": timestamp,
        "file_count": len(files_meta),
        "total_bytes": total_bytes,
        "files": files_meta,
    }
    (target_subdir / "_snapshot_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Snapshot complete: {target_subdir} ({len(files_meta)} files, {total_bytes} bytes)")
    return target_subdir


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot legacy chunks before reingest")
    parser.add_argument("--source", default="rag2025/data/chunked", help="Source chunks dir")
    parser.add_argument("--target-root", default="rag2025/data/chunked_legacy_2025", help="Target root dir")
    parser.add_argument("--timestamp", default=None, help="Explicit UTC timestamp; auto-generate if omitted")
    args = parser.parse_args()

    try:
        out = snapshot_chunks(Path(args.source), Path(args.target_root), args.timestamp)
        print(str(out))
        return 0
    except Exception as exc:
        logger.exception(f"Snapshot failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

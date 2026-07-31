"""Immutable canonical release store: manifest I/O + atomic pointer swap +
immutability enforcement. See data_lifecycle_plan_v2.md M1, 0.2 D5.
This is the layer of last resort -- LanceDB/graph are rebuildable VIEWS,
this is the source of truth.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from src.domain.release import ReleaseManifest


def write_manifest(release_dir: str | Path, manifest: ReleaseManifest) -> Path:
    """Write release_manifest.json into release_dir (created if missing).
    Returns the Path of the written file."""
    d = Path(release_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "release_manifest.json"
    out.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def read_manifest(release_dir: str | Path) -> ReleaseManifest:
    """Read release_manifest.json from release_dir back into a ReleaseManifest."""
    d = Path(release_dir)
    data = json.loads((d / "release_manifest.json").read_text(encoding="utf-8"))
    return ReleaseManifest.from_dict(data)


def publish_pointer(pointer_path: str | Path, release_id: str) -> None:
    """Atomically swap the active-release pointer to point at release_id.
    Writes a .tmp sibling then os.replace() onto pointer_path -- atomic on
    the same NTFS volume. If the process dies between the tmp write and
    the replace, pointer_path is left completely untouched and valid."""
    p = Path(pointer_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"release_id": release_id}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def mark_release_immutable(release_dir: str | Path) -> None:
    """Make every file under release_dir read-only (0o444). Any later
    write attempt to a file in this tree raises PermissionError."""
    d = Path(release_dir)
    for f in d.rglob("*"):
        if f.is_file():
            os.chmod(f, stat.S_IREAD)

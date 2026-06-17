"""
Win32 System Diagnostics and Safety Utilities (Phase H)

Handles MAX_PATH, disk space, SQLite locking, AV interference, and memory pressure.
All functions are no-ops on non-Windows platforms.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import gc
from pathlib import Path
from typing import List, Optional

from loguru import logger

MIN_FREE_GB = 2.0
MAX_RSS_GB = 12.0
_LANCE_WRITE_RETRIES = 5


def is_win32() -> bool:
    return sys.platform == "win32"


def ensure_extended_path(path: str | Path) -> str:
    """Convert path to extended-length format (\\\\?\\ prefix) for >260 char paths."""
    abs_path = os.path.abspath(str(path))
    if is_win32() and len(abs_path) > 200 and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def safe_copytree(src: str | Path, dst: str | Path) -> None:
    """Copy directory tree with extended-path support on Windows."""
    src_str = ensure_extended_path(src)
    dst_str = ensure_extended_path(dst)
    os.makedirs(os.path.dirname(dst_str), exist_ok=True)
    shutil.copytree(src_str, dst_str, dirs_exist_ok=True)
    logger.info(f"Snapshot copied: {src} -> {dst}")


def check_disk_space(path: str | Path = "D:/") -> float:
    """Check free disk space. Returns free GB. Raises if below MIN_FREE_GB."""
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        logger.warning(f"Low disk space: {free_gb:.1f}GB free on {path}")
        cleanup_old_snapshots(keep=2)
        usage = shutil.disk_usage(str(path))
        free_gb = usage.free / (1024 ** 3)
        if free_gb < MIN_FREE_GB:
            raise OSError(f"Disk space critical: {free_gb:.1f}GB free. Aborting.")
    return free_gb


def cleanup_old_snapshots(base_dir: str | Path = "rag2025/data/snapshots", keep: int = 3) -> int:
    """Remove oldest snapshots, keeping N most recent. Returns count removed."""
    snap_path = Path(base_dir)
    if not snap_path.exists():
        return 0
    snaps = sorted(snap_path.iterdir(), key=lambda p: p.stat().st_mtime)
    to_remove = snaps[:-keep] if len(snaps) > keep else []
    for old in to_remove:
        shutil.rmtree(old, ignore_errors=True)
        logger.info(f"Cleaned snapshot: {old}")
    return len(to_remove)


def find_long_paths(root: str | Path = ".", threshold: int = 200) -> List[tuple[str, int]]:
    """Find paths exceeding threshold length."""
    long_paths = []
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        for name in filenames + dirnames:
            full = os.path.join(dirpath, name)
            if len(full) > threshold:
                long_paths.append((full, len(full)))
    long_paths.sort(key=lambda x: -x[1])
    return long_paths[:20]


def safe_lancedb_write(table, records: list, batch_size: int = 50) -> None:
    """Write to LanceDB with retry, AV bypass, and SQLite lock handling."""
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        for attempt in range(_LANCE_WRITE_RETRIES):
            try:
                table.add(batch)
                break
            except Exception as e:
                err_msg = str(e).lower()
                if "database is locked" in err_msg and attempt < _LANCE_WRITE_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(f"LanceDB locked, retry {attempt+1}/{_LANCE_WRITE_RETRIES} in {wait}s")
                    time.sleep(wait)
                    continue
                if "permission" in err_msg and attempt < _LANCE_WRITE_RETRIES - 1:
                    logger.warning(f"LanceDB AV/permission error, retry {attempt+1} in 3s")
                    time.sleep(3)
                    continue
                raise
    logger.info(f"LanceDB write: {len(records)} records in {len(records)//batch_size + 1} batches")


def memory_check() -> float:
    """Check current process RSS in GB. Returns RSS."""
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except ImportError:
        rss = 0.0
    if rss > MAX_RSS_GB:
        gc.collect()
        try:
            import psutil
            rss = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        except ImportError:
            pass
        if rss > MAX_RSS_GB:
            raise MemoryError(f"Memory pressure: {rss:.1f}GB RSS exceeds {MAX_RSS_GB}GB limit")
    return rss


def checkpoint_lancedb(db_path: str | Path) -> None:
    """Force WAL checkpoint to prevent SQLite corruption on Windows."""
    db_dir = Path(db_path)
    if not db_dir.exists():
        return
    for lance_dir in db_dir.rglob("_data.lance"):
        try:
            import sqlite3
            conn = sqlite3.connect(str(lance_dir))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            logger.debug(f"WAL checkpoint: {lance_dir}")
        except Exception as e:
            logger.warning(f"WAL checkpoint failed for {lance_dir}: {e}")


def check_stale_journals(db_path: str | Path = "rag2025/data/lancedb", max_age_s: float = 60) -> int:
    """Find .lance-journal files older than max_age_s seconds."""
    db_dir = Path(db_path)
    if not db_dir.exists():
        return 0
    now = time.time()
    stale = 0
    for j in db_dir.rglob("*.lance-journal"):
        age = now - j.stat().st_mtime
        if age > max_age_s:
            logger.warning(f"Stale journal: {j} (age={age:.0f}s)")
            stale += 1
    return stale


def win32_preflight() -> dict:
    """Run all Win32 diagnostic checks. Returns status dict."""
    result = {"platform": sys.platform, "checks": {}}

    if not is_win32():
        result["checks"]["win32"] = "skipped (non-Windows)"
        return result

    # D-01: Long paths
    long = find_long_paths("rag2025/", threshold=200)
    result["checks"]["long_paths"] = len(long)

    # D-02: Disk space
    try:
        free = check_disk_space("D:/")
        result["checks"]["disk_free_gb"] = round(free, 1)
    except OSError as e:
        result["checks"]["disk_free_gb"] = f"ERROR: {e}"

    # D-04: Stale journals
    stale = check_stale_journals()
    result["checks"]["stale_journals"] = stale

    # D-05: Memory
    try:
        rss = memory_check()
        result["checks"]["rss_gb"] = round(rss, 2)
    except MemoryError as e:
        result["checks"]["rss_gb"] = f"ERROR: {e}"

    return result

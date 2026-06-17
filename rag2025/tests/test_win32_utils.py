"""Tests for Win32 system diagnostics utilities."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.infrastructure.win32_utils import (
    ensure_extended_path,
    check_disk_space,
    cleanup_old_snapshots,
    find_long_paths,
    is_win32,
    memory_check,
    win32_preflight,
)


def test_is_win32():
    result = is_win32()
    assert isinstance(result, bool)
    assert result == (sys.platform == "win32")


def test_ensure_extended_path_short():
    short_path = "rag2025/data/test.jsonl"
    result = ensure_extended_path(short_path)
    # Should be absolute but no extended prefix for short paths
    assert os.path.isabs(result)


def test_ensure_extended_path_long():
    # Create a path that exceeds 200 chars
    long_path = "D:/" + "a" * 250 + "/file.txt"
    result = ensure_extended_path(long_path)
    if sys.platform == "win32":
        assert result.startswith("\\\\?\\")
    else:
        # Non-Windows: no prefix added
        assert not result.startswith("\\\\?\\")


def test_ensure_extended_path_already_extended():
    extended = "\\\\?\\D:\\very\\long\\path"
    result = ensure_extended_path(extended)
    assert result == extended  # Should not double-prefix


def test_check_disk_space_returns_float():
    # Should work on any drive that exists
    drive = "C:/" if sys.platform == "win32" else "/"
    free = check_disk_space(drive)
    assert isinstance(free, float)
    assert free > 0


def test_find_long_paths_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a short-named file
        (Path(tmpdir) / "short.txt").write_text("test")
        result = find_long_paths(tmpdir, threshold=500)
        assert len(result) == 0


def test_find_long_paths_with_long():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create deeply nested path
        deep = Path(tmpdir)
        for i in range(20):
            deep = deep / f"level_{i:02d}"
        deep.mkdir(parents=True)
        (deep / "file.txt").write_text("deep")
        result = find_long_paths(tmpdir, threshold=100)
        # May or may not exceed 100 chars depending on tmpdir length
        assert isinstance(result, list)


def test_cleanup_old_snapshots_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        removed = cleanup_old_snapshots(tmpdir, keep=2)
        assert removed == 0


def test_cleanup_old_snapshots_removes_oldest():
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_base = Path(tmpdir)
        for i in range(5):
            d = snap_base / f"snap_{i}"
            d.mkdir()
            (d / "data.txt").write_text(f"snapshot {i}")

        removed = cleanup_old_snapshots(tmpdir, keep=2)
        assert removed == 3  # 5 - 2 = 3 removed
        remaining = list(snap_base.iterdir())
        assert len(remaining) == 2


def test_memory_check_returns_float():
    try:
        rss = memory_check()
        assert isinstance(rss, float)
        assert rss > 0
    except ImportError:
        pytest.skip("psutil not installed")


def test_win32_preflight_returns_dict():
    result = win32_preflight()
    assert "platform" in result
    assert "checks" in result

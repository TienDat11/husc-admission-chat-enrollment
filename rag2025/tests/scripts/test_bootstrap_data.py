"""Tests for scripts/bootstrap_data.py: completion marker, safe extract, sha check."""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bootstrap_data as bd  # noqa: E402

ENV = {
    "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
    "R2_BUCKET": "husc-admission-data",
    "R2_ACCESS_KEY_ID": "test-key-id",
    "R2_SECRET_ACCESS_KEY": "test-secret",
}


def _make_bundle(path: Path, members: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local_download(bundle: Path):
    def download(_cfg, dest: Path) -> str:
        shutil.copyfile(bundle, dest)
        return hashlib.sha256(dest.read_bytes()).hexdigest()

    return download


def test_empty_preexisting_lancedb_still_installs(tmp_path: Path) -> None:
    bundle = tmp_path / "src.tar.gz"
    digest = _make_bundle(bundle, {"lancedb/husc_chunks.lance/data": b"abc"})
    data_dir = tmp_path / "data"
    (data_dir / "lancedb").mkdir(parents=True)
    (data_dir / "lancedb" / "stale.txt").write_text("old")

    env = {**ENV, "DATA_BUNDLE_SHA256": digest}
    assert bd.bootstrap(env, data_dir, download=_local_download(bundle)) == "installed"
    assert (data_dir / "lancedb" / "husc_chunks.lance" / "data").read_bytes() == b"abc"
    assert not (data_dir / "lancedb" / "stale.txt").exists()
    assert not any(p.name.startswith(".retired-") for p in data_dir.iterdir())


def test_marker_present_skips_without_download(tmp_path: Path) -> None:
    (tmp_path / bd.MARKER).write_text("a" * 64 + "\n")

    def must_not_download(_cfg, _dest):
        raise AssertionError("download must not run when install is marked complete")

    assert bd.bootstrap(ENV, tmp_path, download=must_not_download) == "skipped"
    env = {**ENV, "DATA_BUNDLE_SHA256": "a" * 64}
    assert bd.bootstrap(env, tmp_path, download=must_not_download) == "skipped"


def test_marker_with_different_sha_reinstalls(tmp_path: Path) -> None:
    bundle = tmp_path / "src.tar.gz"
    digest = _make_bundle(bundle, {"lancedb/data.lance": b"new"})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / bd.MARKER).write_text("0" * 64 + "\n")

    env = {**ENV, "DATA_BUNDLE_SHA256": digest}
    assert bd.bootstrap(env, data_dir, download=_local_download(bundle)) == "installed"
    assert (data_dir / "lancedb" / "data.lance").read_bytes() == b"new"
    assert (data_dir / bd.MARKER).read_text().strip() == digest


def test_installs_bundle_and_verifies_sha(tmp_path: Path) -> None:
    bundle = tmp_path / "src.tar.gz"
    digest = _make_bundle(bundle, {"lancedb/data.lance": b"abc", "graph/g.json": b"{}"})
    data_dir = tmp_path / "data"

    env = {**ENV, "DATA_BUNDLE_SHA256": digest}
    assert bd.bootstrap(env, data_dir, download=_local_download(bundle)) == "installed"
    assert (data_dir / "lancedb" / "data.lance").read_bytes() == b"abc"
    assert (data_dir / "graph" / "g.json").read_bytes() == b"{}"
    assert (data_dir / bd.MARKER).read_text().strip() == digest
    assert not (data_dir / ".data-bundle.partial").exists()


def test_sha_mismatch_fails_and_installs_nothing(tmp_path: Path) -> None:
    bundle = tmp_path / "src.tar.gz"
    _make_bundle(bundle, {"lancedb/data.lance": b"abc"})
    data_dir = tmp_path / "data"

    env = {**ENV, "DATA_BUNDLE_SHA256": "0" * 64}
    with pytest.raises(bd.BootstrapError, match="sha256 mismatch"):
        bd.bootstrap(env, data_dir, download=_local_download(bundle))
    assert not (data_dir / "lancedb").exists()
    assert not (data_dir / bd.MARKER).exists()


@pytest.mark.parametrize("evil", ["../escape.txt", "/abs/escape.txt", "lancedb/../../escape.txt"])
def test_rejects_path_traversal(tmp_path: Path, evil: str) -> None:
    bundle = tmp_path / "src.tar.gz"
    _make_bundle(bundle, {"lancedb/ok.lance": b"ok", evil: b"pwned"})
    data_dir = tmp_path / "data"

    with pytest.raises(bd.BootstrapError, match="unsafe path"):
        bd.bootstrap(ENV, data_dir, download=_local_download(bundle))
    assert not (tmp_path / "escape.txt").exists()
    assert not (data_dir / "lancedb").exists()


def test_missing_env_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(bd.BootstrapError, match="R2_BUCKET"):
        bd.bootstrap({"R2_ENDPOINT": "x"}, tmp_path, download=lambda c, d: "")


def test_installs_local_bundle_without_r2(tmp_path: Path) -> None:
    bundle = tmp_path / "data-bundle.tar.gz"
    digest = _make_bundle(bundle, {"lancedb/data.lance": b"abc"})
    data_dir = tmp_path / "data"

    def must_not_download(_cfg, _dest):
        raise AssertionError("R2 must not be used when DATA_BUNDLE_LOCAL exists")

    env = {"DATA_BUNDLE_LOCAL": str(bundle), "DATA_BUNDLE_SHA256": digest}
    assert bd.bootstrap(env, data_dir, download=must_not_download) == "installed-local"
    assert (data_dir / "lancedb" / "data.lance").read_bytes() == b"abc"
    assert (data_dir / bd.MARKER).read_text().strip() == digest
    assert bundle.exists()


def test_local_path_missing_falls_through_to_r2_env_error(tmp_path: Path) -> None:
    env = {"DATA_BUNDLE_LOCAL": str(tmp_path / "absent.tar.gz")}
    with pytest.raises(bd.BootstrapError, match="missing required env"):
        bd.bootstrap(env, tmp_path / "data", download=lambda c, d: "")

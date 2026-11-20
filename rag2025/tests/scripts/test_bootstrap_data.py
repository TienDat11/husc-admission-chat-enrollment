"""Tests for scripts/bootstrap_data.py: idempotent skip, safe extract, sha check."""

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


def _make_bundle(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _local_download(bundle: Path):
    def download(_cfg, dest: Path) -> str:
        shutil.copyfile(bundle, dest)
        return hashlib.sha256(dest.read_bytes()).hexdigest()

    return download


def test_skips_when_lancedb_already_present(tmp_path: Path) -> None:
    (tmp_path / "lancedb").mkdir()

    def must_not_download(_cfg, _dest):
        raise AssertionError("download must not run when data already exists")

    assert bd.bootstrap(ENV, tmp_path, download=must_not_download) == "skipped"


def test_installs_bundle_and_verifies_sha(tmp_path: Path) -> None:
    bundle = tmp_path / "src.tar.gz"
    _make_bundle(bundle, {"lancedb/data.lance": b"abc", "graph/g.json": b"{}"})
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    data_dir = tmp_path / "data"

    env = {**ENV, "DATA_BUNDLE_SHA256": digest}
    assert bd.bootstrap(env, data_dir, download=_local_download(bundle)) == "installed"
    assert (data_dir / "lancedb" / "data.lance").read_bytes() == b"abc"
    assert (data_dir / "graph" / "g.json").read_bytes() == b"{}"
    assert not (data_dir / ".data-bundle.partial").exists()


def test_sha_mismatch_fails_and_installs_nothing(tmp_path: Path) -> None:
    bundle = tmp_path / "src.tar.gz"
    _make_bundle(bundle, {"lancedb/data.lance": b"abc"})
    data_dir = tmp_path / "data"

    env = {**ENV, "DATA_BUNDLE_SHA256": "0" * 64}
    with pytest.raises(bd.BootstrapError, match="sha256 mismatch"):
        bd.bootstrap(env, data_dir, download=_local_download(bundle))
    assert not (data_dir / "lancedb").exists()


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

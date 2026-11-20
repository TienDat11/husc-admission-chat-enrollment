"""Populate DATA_DIR from a data bundle (idempotent).

Install is complete only when DATA_DIR/.bootstrap-complete exists, written as the
LAST step of a successful install and holding the bundle sha256. A pre-existing
lancedb/ directory alone does not count as installed (it may be empty or partial).

Skip when the marker exists and (DATA_BUNDLE_SHA256 unset or equal to the marker).
Otherwise install DATA_BUNDLE_LOCAL when that file exists, else download
DATA_BUNDLE_KEY from R2_BUCKET. Either way verify DATA_BUNDLE_SHA256 when set,
validate every tar member, then install into DATA_DIR. Stdlib only: the R2 GET is
signed with SigV4 by hand.

Exit codes: 0 = populated or already present, 1 = failure (message on stderr).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

SENTINEL = "lancedb"
MARKER = ".bootstrap-complete"
REQUIRED_ENV = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
DEFAULT_BUNDLE_KEY = "data-bundle.tar.gz"
REGION = "auto"
CHUNK_SIZE = 1 << 20
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class BootstrapError(RuntimeError):
    """Raised for any condition that must fail the container start."""


# download(cfg, dest) writes the object to dest and returns its sha256 hex digest.
Download = Callable[[Mapping[str, str], Path], str]


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _sigv4_headers(
    endpoint: str,
    bucket: str,
    key: str,
    access_key: str,
    secret_key: str,
    now: dt.datetime,
) -> dict[str, str]:
    host = urllib.parse.urlsplit(endpoint).netloc
    canonical_uri = "/" + urllib.parse.quote(f"{bucket}/{key}", safe="/-_.~")
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = "UNSIGNED-PAYLOAD"
    signed_values = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    signed_headers = ";".join(sorted(signed_values))
    canonical_headers = "".join(f"{name}:{signed_values[name]}\n" for name in sorted(signed_values))
    canonical_request = "\n".join(
        ["GET", canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )
    scope = f"{date_stamp}/{REGION}/s3/aws4_request"
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()]
    )
    signing_key = _hmac(("AWS4" + secret_key).encode(), date_stamp)
    for part in (REGION, "s3", "aws4_request"):
        signing_key = _hmac(signing_key, part)
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return {
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }


def _r2_config(env: Mapping[str, str]) -> dict[str, str]:
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        raise BootstrapError(f"missing required env: {', '.join(missing)}")
    return {
        "endpoint": env["R2_ENDPOINT"].rstrip("/"),
        "bucket": env["R2_BUCKET"],
        "access_key": env["R2_ACCESS_KEY_ID"],
        "secret_key": env["R2_SECRET_ACCESS_KEY"],
        "key": env.get("DATA_BUNDLE_KEY") or DEFAULT_BUNDLE_KEY,
    }


def _download_r2(cfg: Mapping[str, str], dest: Path) -> str:
    headers = _sigv4_headers(
        cfg["endpoint"],
        cfg["bucket"],
        cfg["key"],
        cfg["access_key"],
        cfg["secret_key"],
        dt.datetime.now(dt.timezone.utc),
    )
    url = f"{cfg['endpoint']}/{cfg['bucket']}/{urllib.parse.quote(cfg['key'], safe='/-_.~')}"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as resp:
            with dest.open("wb") as out:
                while chunk := resp.read(CHUNK_SIZE):
                    digest.update(chunk)
                    out.write(chunk)
    except urllib.error.HTTPError as exc:
        raise BootstrapError(f"R2 download failed: HTTP {exc.code} for key {cfg['key']!r}") from exc
    except urllib.error.URLError as exc:
        raise BootstrapError(f"R2 download failed: {exc.reason}") from exc
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sha(actual: str, expected: str, name: str) -> None:
    if expected and actual != expected:
        raise BootstrapError(f"sha256 mismatch for {name!r}: expected {expected}, got {actual}")


def _validated_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for member in members:
        name = member.name
        path = PurePosixPath(name)
        if (
            "\\" in name
            or name.startswith("/")
            or _WINDOWS_DRIVE.match(name)
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise BootstrapError(f"unsafe path in bundle: {name!r}")
        if not (member.isreg() or member.isdir()):
            raise BootstrapError(f"unsupported member type in bundle: {name!r}")
    return members


def _discard(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _swap_in(source: Path, target: Path, data_dir: Path) -> None:
    """Replace target with source: rename the old target aside, move source in, delete old."""
    retired = data_dir / f".retired-{target.name}"
    _discard(retired)
    if target.exists() or target.is_symlink():
        os.replace(target, retired)
    os.replace(source, target)
    _discard(retired)


def _install(bundle: Path, data_dir: Path) -> None:
    """Extract into a staging dir, then swap top-level entries into place.

    SENTINEL is swapped last so an interrupted install never looks complete.
    """
    staging = Path(tempfile.mkdtemp(dir=data_dir, prefix=".bootstrap-"))
    try:
        with tarfile.open(bundle, "r:gz") as tar:
            members = _validated_members(tar)
            tar.extractall(staging, members=members, filter="data")
        if not (staging / SENTINEL).is_dir():
            raise BootstrapError(f"bundle has no top-level {SENTINEL}/ directory")
        entries = sorted(staging.iterdir(), key=lambda p: p.name == SENTINEL)
        for entry in entries:
            _swap_in(entry, data_dir / entry.name, data_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _is_installed(data_dir: Path, expected: str) -> bool:
    marker = data_dir / MARKER
    if not marker.is_file():
        return False
    return not expected or marker.read_text().strip().lower() == expected


def _write_marker(data_dir: Path, sha: str) -> None:
    (data_dir / MARKER).write_text(sha + "\n")


def bootstrap(
    env: Mapping[str, str],
    data_dir: Path,
    download: Download = _download_r2,
) -> str:
    """Return "skipped", "installed-local" or "installed" (R2)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = (env.get("DATA_BUNDLE_SHA256") or "").strip().lower()
    if _is_installed(data_dir, expected):
        return "skipped"
    (data_dir / MARKER).unlink(missing_ok=True)

    local_raw = (env.get("DATA_BUNDLE_LOCAL") or "").strip()
    if local_raw and Path(local_raw).is_file():
        local = Path(local_raw)
        actual = _sha256_file(local)
        _check_sha(actual, expected, str(local))
        _install(local, data_dir)
        _write_marker(data_dir, actual)
        return "installed-local"

    cfg = _r2_config(env)
    partial = data_dir / ".data-bundle.partial"
    try:
        actual = download(cfg, partial)
        _check_sha(actual, expected, cfg["key"])
        _install(partial, data_dir)
    finally:
        partial.unlink(missing_ok=True)
    _write_marker(data_dir, actual)
    return "installed"


def main() -> int:
    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    try:
        outcome = bootstrap(os.environ, data_dir)
    except (BootstrapError, OSError, tarfile.TarError) as exc:
        print(f"bootstrap_data: ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"bootstrap_data: {outcome} ({data_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

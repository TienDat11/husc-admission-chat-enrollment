"""T10 wiring — the refusal trailer's "last updated" date comes from the
release the app is actually serving, not from a value a test injected.

The acceptance criterion this file pins is the one a unit test of
``refusal_footer`` cannot see: that the *composition root*
(``configure_canonical_record_store``) registers the serving-release provider,
and that a published release makes the date appear on the footer through the
real module the app renders it from.

Two identities of the same file matter here. ``abstention_policy`` is reachable
as both ``services.abstention_policy`` (when ``src`` is on ``sys.path``) and
``src.services.abstention_policy`` (main.py's absolute import). They are the
same file but distinct module objects, and both render footers in production —
so the wiring must reach both. This test asserts exactly that.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.release import ReleaseManifest  # noqa: E402
from src.infrastructure.canonical_store import (  # noqa: E402
    publish_pointer,
    write_manifest,
)
from src.services.admin_rollback import (  # noqa: E402
    POINTER_FILENAME,
    RELEASES_DIRNAME,
    release_tag,
)

# The two module identities the app renders footers through.
from services import abstention_policy as ap_services  # noqa: E402
from src.services import abstention_policy as ap_src  # noqa: E402

RELEASE_ID = "R-2026-09-01"
RELEASE_DATE = datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc)


@pytest.fixture
def clean_source():
    """Restore "no date known" on both identities after the test.

    ``configure_release_date_source`` is global module state; leaking it would
    pin another test's footer to this test's release.
    """
    yield
    ap_src.configure_release_date_source(None)
    ap_services.configure_release_date_source(None)


def _configure(tmp_path: Path):
    app = FastAPI()
    configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "canonical_store")
    )
    return app


def _publish(store_root: Path, *, created_at: datetime | None) -> None:
    releases = store_root / RELEASES_DIRNAME
    write_manifest(
        releases / RELEASE_ID,
        ReleaseManifest(
            release_id=RELEASE_ID,
            created_at=created_at,
            lancedb_tag=release_tag(RELEASE_ID),
        ),
    )
    publish_pointer(releases / POINTER_FILENAME, RELEASE_ID)


def test_no_release_means_no_date_but_still_a_hotline(tmp_path, clean_source):
    """Before the first publish the footer shows the hotline, not a date."""
    _configure(tmp_path)

    assert ap_src.release_date() is None
    assert ap_services.release_date() is None

    footer = ap_src.refusal_footer()
    assert "Hotline" in footer
    assert "cập nhật lần cuối" not in footer


def test_a_published_release_makes_the_date_appear(tmp_path, clean_source):
    """The core wiring claim: publish → the footer names the serving date."""
    store_root = tmp_path / "canonical_store"
    _configure(tmp_path)
    _publish(store_root, created_at=RELEASE_DATE)

    # The provider is read per call, so a release published after configure is
    # still reflected — no restart, no re-configuration.
    assert ap_src.release_date() == RELEASE_DATE.isoformat()
    assert ap_services.release_date() == RELEASE_DATE.isoformat()

    for footer in (ap_src.refusal_footer(), ap_services.refusal_footer()):
        assert "Hotline" in footer
        assert "Thông tin cập nhật lần cuối: 01/09/2026" in footer


def test_the_date_tracks_the_active_pointer_not_the_first_release(
    tmp_path, clean_source
):
    """A later publish moves the date: the provider reads the pointer live."""
    store_root = tmp_path / "canonical_store"
    _configure(tmp_path)
    _publish(store_root, created_at=RELEASE_DATE)

    newer_id = "R-2026-10-01"
    newer = datetime(2026, 10, 1, tzinfo=timezone.utc)
    releases = store_root / RELEASES_DIRNAME
    write_manifest(
        releases / newer_id,
        ReleaseManifest(
            release_id=newer_id,
            created_at=newer,
            parent_release_id=RELEASE_ID,
            lancedb_tag=release_tag(newer_id),
        ),
    )
    publish_pointer(releases / POINTER_FILENAME, newer_id)

    assert ap_src.release_date() == newer.isoformat()


def test_a_release_with_no_recorded_date_omits_the_line(tmp_path, clean_source):
    """Fail-soft: a manifest without created_at is "no date", not a crash."""
    store_root = tmp_path / "canonical_store"
    _configure(tmp_path)
    _publish(store_root, created_at=None)

    assert ap_src.release_date() is None
    assert "cập nhật lần cuối" not in ap_src.refusal_footer()

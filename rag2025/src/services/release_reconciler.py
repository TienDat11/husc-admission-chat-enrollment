"""Release reconciler — polls PG every 2s to hot-swap the in-RAM snapshot (§PL-5)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.services.release_snapshot import ReleaseSnapshot, ReleaseSnapshotLoadError

logger = logging.getLogger(__name__)


class ReleaseReconciler:
    """Polls ``releases`` for the active row and swaps ``app.state.release_snapshot``.

    Starts lazily via :meth:`start` (called from the app lifespan); stops on
    shutdown. A failed load keeps the previous snapshot and logs, so serving
    never regresses to reading disk files.
    """

    def __init__(
        self,
        app: Any,
        *,
        poll_seconds: float = 2.0,
        load_snapshot: Any | None = None,
    ) -> None:
        self._app = app
        self._poll_seconds = poll_seconds
        self._load_snapshot = load_snapshot or self._default_load
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    async def _default_load(self, release_id: str) -> ReleaseSnapshot:
        """Default loader: build a snapshot from the active release row."""
        from services.major_code_validator import get_whitelist

        # Prefer PG-backed values when the release row exposes them; fall back
        # to the year's whitelist so no serving path reads data/major_codes/*.json.
        major_codes = frozenset(get_whitelist())
        from src.services.major_code_validator import set_snapshot_whitelist
        set_snapshot_whitelist(major_codes)
        from src.services.major_recommender import set_snapshot as _set_rec_snapshot
        snap = ReleaseSnapshot(
            release_id=release_id,
            recommender_snapshot={},
            major_codes=major_codes,
            cutoff_scores={},
            quota={},
            tuition={},
            priority_rules={},
            major_whitelist=major_codes,
            chunks_table="husc_chunks",
            corpus_stamp=[],
        )
        _set_rec_snapshot(snap)
        from src.services.guardrail import set_snapshot_allowed as _set_guardrail_allowed
        _set_guardrail_allowed(major_codes)
        return snap

    async def start(self) -> None:
        self._stopped = False
        self._task = asyncio.create_task(self._loop(), name="release-reconciler")

    async def stop(self) -> None:
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while not self._stopped:
            try:
                await self._tick()
            except Exception:  # pragma: no cover - defensive, keep polling
                logger.exception("release reconciler tick failed")
            await asyncio.sleep(self._poll_seconds)

    async def _tick(self) -> None:
        engine = getattr(self._app.state, "db_engine", None)
        if engine is None:
            return
        # Read the active release id.
        row = None
        with engine.connect() as conn:
            result = conn.execute(
                "SELECT release_id, corpus_stamp FROM releases "
                "WHERE status = 'active' LIMIT 1"
            )
            row = result.mappings().first()
        if row is None:
            return
        release_id = str(row["release_id"])
        current = getattr(self._app.state, "release_snapshot", None)
        if current is not None and current.release_id == release_id:
            return
        try:
            snapshot = await self._load_snapshot(release_id)
        except ReleaseSnapshotLoadError as exc:
            logger.warning("release %s has no snapshot: %s", release_id, exc.reason)
            return
        self._app.state.release_snapshot = snapshot
        logger.info("release reconciler hot-swapped snapshot -> %s", release_id)

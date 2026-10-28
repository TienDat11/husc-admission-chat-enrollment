"""Request-level idempotency for the mutating admin routes — ticket 27.

Why a request key is needed at all
----------------------------------
Publish and rollback are both guarded against *state* replay already:

* ``publish`` cannot run twice because ``_TRANSITIONS`` has no self-loop, so a
  second publish of a published record is refused 409 (``IllegalTransitionError``);
* an explicit-target rollback cannot run twice because the second call finds
  its target already active and is refused (``TargetIsActiveError``).

Neither guard covers the case this module exists for: **the implicit-target
rollback**. With no ``target_release_id`` the target is resolved from the
*current* pointer, so a replayed request resolves a *different* target —
R3→R2 succeeds, then the retry resolves R2→R1 and succeeds again, walking back
two releases on what the caller believes was one action. The replayed call is
not a duplicate by any state comparison (different from/to pair); only the
request's own identity distinguishes it. That is what an idempotency key is.

The contract
------------
``Idempotency-Key`` is optional. Absent, every route behaves exactly as it did
before this module existed — the state guards above remain the protection, and
nothing about existing callers changes. Present, the key buys the stronger
guarantee:

* first request with a key: the claim is taken, the route runs, the response is
  stored;
* replay with the same key and the same body: the stored response is returned
  and the route does **not** run again (marked ``Idempotent-Replay: true``);
* same key, *different* body: refused 422 — a key reused for a different
  intention is a client bug, and answering it with the first request's response
  would report a change that never happened;
* same key while the first is still in flight: refused 409, because the
  alternative is running the operation twice;
* a request that *fails* releases its claim, so a retry after a 5xx is allowed.
  A failure that permanently poisoned its key would leave the operator unable
  to retry the very action that failed.

Storage
-------
One file per key, named by a digest of ``(actor_id, key)``, so the key never
reaches the filesystem as a path — no traversal, and no case-folding collision
between two keys that differ only in case. The claim is an ``O_EXCL`` create,
which is atomic across processes on Windows and POSIX alike, so two concurrent
duplicates cannot both win. The response is written by ``.tmp`` + ``os.replace``,
the same atomic-write idiom the store and the pointer already use.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = [
    "IDEMPOTENCY_DIRNAME",
    "IdempotencyConflict",
    "IdempotencyError",
    "IdempotencyInProgress",
    "IdempotencyKeyInvalid",
    "IdempotencyStore",
    "ReplayResponse",
    "request_fingerprint",
]

IDEMPOTENCY_DIRNAME = "idempotency"

# A key longer than this is refused rather than truncated: two long keys that
# share a prefix must not collide onto one bucket.
_MAX_KEY_LENGTH = 200

# Records are kept for this long, then pruned on the next claim. The window has
# to outlive any realistic retry — a client retrying a request from last week is
# not retrying, it is issuing a new one.
_RETENTION_DAYS = 7

class IdempotencyError(Exception):
    """Base class for idempotency failures."""

class IdempotencyKeyInvalid(IdempotencyError):
    """The supplied key is empty or longer than ``_MAX_KEY_LENGTH``."""

class IdempotencyConflict(IdempotencyError):
    """The key was already used for a different request.

    Refused, never answered from the first request's stored response: a caller
    who reused a key for a different intention has a bug, and returning the
    older response would tell them a change happened that did not.
    """

class IdempotencyInProgress(IdempotencyError):
    """Another request with this key has claimed it and not yet finished."""

@dataclass(frozen=True)
class ReplayResponse:
    """A stored response being returned for a replayed request."""

    status_code: int
    body: dict

def request_fingerprint(payload: object) -> str:
    """Stable digest of a request body, for same-key-different-body detection.

    ``sort_keys`` so field order in the JSON body does not read as a different
    request; ``default=str`` so a body carrying a date or an enum fingerprints
    rather than raising.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class IdempotencyStore:
    """File-backed idempotency records, one file per ``(actor_id, key)``."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root) / IDEMPOTENCY_DIRNAME
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, actor_id: str, key: str) -> Path:
        """The record path for one actor's key.

        The digest covers the actor id as well as the key, so two actors that
        happen to pick the same key cannot read or collide with each other's
        records — one tenant's key is not another's.
        """
        digest = hashlib.sha256(
            f"{actor_id}\x00{key}".encode("utf-8")
        ).hexdigest()
        return self._root / f"{digest}.json"

    def _read(self, path: Path) -> dict | None:
        """The record at ``path``, or ``None`` when unreadable/incomplete.

        A half-written or corrupt record reads as ``None``, which the caller
        treats as still-in-progress. That is the safe direction: the alternative
        — treating it as finished — would either run the operation twice or
        invent a response.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def begin(
        self,
        *,
        actor_id: str,
        key: str,
        operation: str,
        body: object,
    ) -> ReplayResponse | None:
        """Claim ``key``, or return the stored response if it is a replay.

        Returns ``None`` when the caller now owns the claim and must run the
        operation, then call :meth:`complete` or :meth:`abandon`.
        """
        key = (key or "").strip()
        if not key:
            raise IdempotencyKeyInvalid("idempotency key must not be blank")
        if len(key) > _MAX_KEY_LENGTH:
            raise IdempotencyKeyInvalid(
                f"idempotency key must be at most {_MAX_KEY_LENGTH} characters"
            )

        path = self._path(actor_id, key)
        fingerprint = request_fingerprint(body)
        claim = {
            "state": "in_progress",
            "actor_id": actor_id,
            "operation": operation,
            "request_fingerprint": fingerprint,
            "created_at": _utcnow().isoformat(),
        }

        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = self._read(path)
            if existing is None or existing.get("state") != "complete":
                raise IdempotencyInProgress(
                    "a request with this idempotency key is still in flight"
                ) from None
            if (
                existing.get("operation") != operation
                or existing.get("request_fingerprint") != fingerprint
            ):
                raise IdempotencyConflict(
                    "this idempotency key was already used for a different request"
                ) from None
            return ReplayResponse(
                status_code=int(existing.get("status_code", 200)),
                body=existing.get("response") or {},
            )

        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(claim, stream, ensure_ascii=False)
        self._prune()
        return None

    def complete(
        self, *, actor_id: str, key: str, status_code: int, response: object
    ) -> None:
        """Record the finished response against the claim."""
        path = self._path(actor_id, key.strip())
        existing = self._read(path) or {}
        record = {
            **{k: v for k, v in existing.items() if k != "state"},
            "state": "complete",
            "status_code": status_code,
            "response": response,
            "completed_at": _utcnow().isoformat(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def abandon(self, *, actor_id: str, key: str) -> None:
        """Release a claim whose request failed, so a retry is allowed.

        Best-effort by design: if the file cannot be removed the claim stands
        and the retry gets a 409 — an inconvenience, where the alternative
        (raising here) would replace the real error with a bookkeeping one.
        """
        path = self._path(actor_id, key.strip())
        try:
            if (self._read(path) or {}).get("state") == "in_progress":
                path.unlink(missing_ok=True)
        except OSError:
            pass

    def _prune(self) -> None:
        """Drop records past the retention window. Best-effort, never raises.

        Admin traffic is human-scale, so this is not a hot path; it exists so
        the directory cannot grow without bound on a long-lived deployment.
        """
        cutoff = _utcnow() - timedelta(days=_RETENTION_DAYS)
        try:
            entries = list(self._root.glob("*.json"))
        except OSError:
            return
        for entry in entries:
            try:
                modified = datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc)
                if modified < cutoff:
                    entry.unlink(missing_ok=True)
            except OSError:
                continue

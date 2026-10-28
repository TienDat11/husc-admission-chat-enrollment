"""Admin identity configuration — ticket 13.

The registry is configuration, not code: it lives in the environment so an
actor can be added or revoked without a deploy, and so no actor id is ever
hardcoded. Shape of ``ADMIN_IDENTITIES`` (JSON array, one object per actor)::

    ADMIN_IDENTITIES='[
      {"token": "<secret>", "actor_id": "nv.binh@husc", "role": "editor"},
      {"token": "<secret>", "actor_id": "nv.cuong@husc", "role": "reviewer"}
    ]'

Three deliberate choices:

- **The credential is the key, the role is the value.** A caller proves *who*
  it is; the role is looked up server-side. Nothing in a request body or header
  names a role, so a client cannot assert one.
- **The registry is read from the environment at request time, cached by its
  raw text.** That matches how the admin surface was configured before
  (``ADMIN_API_TOKEN`` was read with ``os.getenv``) and keeps tests and ops able
  to rotate a credential without restarting the process. The cache is keyed on
  the raw strings, so it never serves a stale parse.
- **Misconfiguration is loud and denies.** Invalid JSON, a bad role name or a
  duplicate credential raises :class:`AdminIdentityConfigError`, which the HTTP
  layer turns into 503 — an admin surface nobody can enter, rather than one
  somebody can enter with the wrong role.

``ADMIN_API_TOKEN`` keeps working, as a **machine** credential with role
``ops``: the spec's decision 7 keeps the shared token for ops endpoints but
forbids it from granting interactive publish, and role ``ops`` is granted no
admissions action at all (see :mod:`src.domain.admin_identity`). A duplicate
between the two variables is rejected rather than resolved, because "the shared
token is also an editor" is precisely the configuration that decision 7 bans.
"""
from __future__ import annotations

import hmac
import json
import os
from typing import Any

from src.domain.admin_identity import Identity, Role

# The header a caller presents its credential in. Reusing the header the admin
# surface already used (and that the app's CORS allow-list already names) means
# ticket 13 replaces the *meaning* of the credential without a cross-cutting
# CORS change, which is what the ticket-12 comment warned a new header would
# require.
ADMIN_CREDENTIAL_HEADER = "X-Admin-Token"

# Registry of human actors: JSON array of {token, actor_id, role}.
IDENTITIES_ENV = "ADMIN_IDENTITIES"
# Legacy shared token. Registered as the machine credential (role ``ops``).
OPS_TOKEN_ENV = "ADMIN_API_TOKEN"
# Four-eyes switch. Absent/blank keeps the rule (see ``four_eyes_enabled``).
FOUR_EYES_ENV = "ADMIN_FOUR_EYES"

# Audit name for the shared ops credential. Constant by design: it names a
# machine, and the point of the registry is that *human* callers are not.
OPS_ACTOR_ID = "ops-shared-token"

_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

class AdminIdentityConfigError(Exception):
    """The identity registry is misconfigured. Denies every request (503)."""

def parse_identities(raw: str) -> tuple[tuple[str, Identity], ...]:
    """Parse the ``ADMIN_IDENTITIES`` JSON text into ``(token, identity)`` pairs.

    Blank text is an empty registry, not an error — an app with no actors
    configured is a valid (if useless) state. Anything else must be well formed:
    a half-valid registry would silently drop an actor, so one bad entry fails
    the whole parse.
    """
    if not raw.strip():
        return ()

    try:
        entries: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdminIdentityConfigError(f"{IDENTITIES_ENV} is not valid JSON: {exc}") from exc

    if not isinstance(entries, list):
        raise AdminIdentityConfigError(
            f"{IDENTITIES_ENV} must be a JSON array of "
            '{"token", "actor_id", "role"} objects'
        )

    parsed: list[tuple[str, Identity]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AdminIdentityConfigError(f"{IDENTITIES_ENV}[{index}] is not an object")

        token = entry.get("token")
        actor_id = entry.get("actor_id")
        role_text = entry.get("role")

        if not isinstance(token, str) or not token:
            raise AdminIdentityConfigError(f"{IDENTITIES_ENV}[{index}].token must be a non-empty string")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise AdminIdentityConfigError(
                f"{IDENTITIES_ENV}[{index}].actor_id must be a non-empty string"
            )
        try:
            role = Role(role_text)
        except (ValueError, TypeError) as exc:
            raise AdminIdentityConfigError(
                f"{IDENTITIES_ENV}[{index}].role must be one of "
                f"{sorted(r.value for r in Role)} (got {role_text!r})"
            ) from exc

        parsed.append((token, Identity(actor_id=actor_id, role=role)))

    return tuple(parsed)

def _configured() -> tuple[str, str]:
    """The two raw environment values the registry is derived from."""
    return (
        os.environ.get(IDENTITIES_ENV, ""),
        os.environ.get(OPS_TOKEN_ENV, ""),
    )

_cache_key: tuple[str, str] | None = None
_cache: tuple[tuple[str, Identity], ...] = ()

def load_identities() -> tuple[tuple[str, Identity], ...]:
    """The whole registry: configured actors plus the machine credential.

    Cached against the raw configuration text, so an env change (a rotated
    token, a test's ``monkeypatch.setenv``) is picked up on the next call and a
    repeated call within one request does no work.
    """
    global _cache_key, _cache

    key = _configured()
    if key == _cache_key:
        return _cache

    entries = list(parse_identities(key[0]))

    ops_token = key[1].strip()
    if ops_token:
        if any(hmac.compare_digest(ops_token, token) for token, _ in entries):
            raise AdminIdentityConfigError(
                f"{OPS_TOKEN_ENV} is also registered in {IDENTITIES_ENV}: the shared "
                "ops credential must not double as a human actor"
            )
        entries.append((ops_token, Identity(actor_id=OPS_ACTOR_ID, role=Role.OPS)))

    _cache_key, _cache = key, tuple(entries)
    return _cache

def resolve_identity(credential: str) -> Identity | None:
    """The identity ``credential`` maps to, or ``None`` if it maps to none.

    Compares every registered credential with :func:`hmac.compare_digest` and
    does not return early: an unknown credential is rejected after the same
    amount of work regardless of how much of it matches a real one.
    """
    matched: Identity | None = None
    for token, identity in load_identities():
        if hmac.compare_digest(token, credential):
            matched = identity
    return matched

def four_eyes_enabled() -> bool:
    """Whether an approver must differ from the version's author.

    Default is ON, and only an explicit off-value turns it off: a missing or
    blank variable keeps the control, so a deployment that forgets to configure
    it gets the safe behaviour rather than the permissive one.
    """
    raw = (os.environ.get(FOUR_EYES_ENV) or "").strip().lower()
    return raw not in _DISABLED_VALUES

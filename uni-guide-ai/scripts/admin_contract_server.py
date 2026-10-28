"""
admin_contract_server.py - minimal real FastAPI server for the FE cross-stack smoke.

Ticket 42: the frontend smoke test must drive the REAL client functions in
``uni-guide-ai/src/lib/adminApi.ts`` over a REAL HTTP socket against a REAL
FastAPI app, with auth enforced. This launcher builds exactly that: the admin
router mounted on a bare FastAPI app, configured with the same minimal store
fixture the backend's own contract test uses
(``rag2025/tests/routers/test_admin_records_api.py``) - no LanceDB, no LLM, no
retrieval warmup. The app object is the same class of object ``TestClient``
sees there; the only thing added here is a real uvicorn socket so a Node
process can talk to it.

Credentials are NOT hardcoded here. The orchestrator
(``uni-guide-ai/scripts/admin-crud-smoke.mjs``) mints random tokens, writes
them to a JSON file (``[{"token": ..., "actor_id": ..., "role": ...}, ...]``)
and passes the path via ``ADMIN_SMOKE_IDENTITIES``. This process parses the
file into the ``ADMIN_IDENTITIES`` JSON that ``src/config/admin_identity.py``
already reads, so the normal four-eyes / RBAC advisory path stays active.

Protocol with the orchestrator:
- on a successful bind, print exactly ``ADMIN_SMOKE_READY <port>`` to stdout
  (uvicorn's own startup log is suppressed so the line is unambiguous);
- any failure before that line exits non-zero with the reason on stderr,
  so the orchestrator can fail loudly instead of skipping.

Usage (normally spawned, not run by hand):
    ADMIN_SMOKE_IDENTITIES=/path/identities.json \
    ADMIN_SMOKE_STORE_ROOT=/path/store \
    python uni-guide-ai/scripts/admin_contract_server.py [--port 0]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

# This file lives under uni-guide-ai/scripts/ but boots the backend, so the
# rag2025 root is resolved from the repo root (two parents up), not alongside.
RAG_ROOT = Path(os.environ.get(
    "RAG_SRC_ROOT", str(Path(__file__).resolve().parents[2] / "rag2025")
))

# The admin router pulls modules under three import conventions at once:
# `src.*` and `config.*` (rag2025 + rag2025/src on path) and `rag2025.src.*`
# (repo root on path, namespace package). That is exactly the set conftest.py
# installs for the backend's own contract test, so mirror it here.
for root in (RAG_ROOT.parent, RAG_ROOT, RAG_ROOT / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.config.admin_identity import IDENTITIES_ENV  # noqa: E402  (sys.path set above)


def _fail(reason: str) -> None:
    print(f"[admin-contract-server] FATAL: {reason}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _identities_json_from_file(path: Path) -> str:
    """Translate the orchestrator's token file into ADMIN_IDENTITIES JSON.

    Accepted shape: a JSON array of objects with ``token`` / ``actor_id`` /
    ``role``. The roles are passed through verbatim - ``admin_identity.py``
    owns role validation, and this launcher must not pre-approve anything.
    """
    if not path.is_file():
        _fail(f"ADMIN_SMOKE_IDENTITIES file not found: {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        _fail(f"identity file {path} must hold a non-empty JSON array")
    out = []
    for entry in entries:
        for key in ("token", "actor_id", "role"):
            if key not in entry:
                _fail(f"identity entry missing '{key}': {entry!r}")
        out.append(
            {
                "token": entry["token"],
                "actor_id": entry["actor_id"],
                "role": entry["role"],
                "ops": entry.get("ops", []),
                "actions": entry.get("actions", []),
            }
        )
    return json.dumps(out, ensure_ascii=False)


def build_app():
    from fastapi import FastAPI

    from config.settings import RAGSettings
    from src.app.lifecycle import configure_canonical_record_store
    from src.routers.admin import router as admin_router

    identities_path = os.environ.get("ADMIN_SMOKE_IDENTITIES", "")
    if not identities_path:
        _fail(
            "ADMIN_SMOKE_IDENTITIES is not set - refusing to start a server "
            "whose admin credentials would come from anywhere else"
        )
    os.environ[IDENTITIES_ENV] = _identities_json_from_file(Path(identities_path))

    store_root = os.environ.get("ADMIN_SMOKE_STORE_ROOT", "")
    if not store_root:
        _fail("ADMIN_SMOKE_STORE_ROOT is not set - the smoke must never write real data")

    app = FastAPI(title="HUSC admin contract smoke")
    app.include_router(admin_router)
    # Same wiring the backend contract test performs: canonical store, audit
    # ledger, outbox, releases, and the idempotency guard all hang off the
    # configured root. configure_canonical_record_store raises on a store it
    # cannot build - that propagates and kills the process, which is the
    # fail-loud behaviour the orchestrator waits for.
    configure_canonical_record_store(app, RAGSettings(CANONICAL_STORE_ROOT=Path(store_root)))
    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0, help="0 = ephemeral port")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    try:
        app = build_app()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - startup must be loud, not partial
        _fail(f"cannot build the FastAPI app: {type(exc).__name__}: {exc}")

    # Bind the socket here so the ephemeral port is known before announcing
    # readiness; uvicorn 0.46 has no bind_threads(), but Server.run(sockets=)
    # accepts a pre-bound socket and closes it on shutdown.
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            access_log=False,
        )
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as exc:
        _fail(f"cannot bind {args.host}:{args.port}: {type(exc).__name__}: {exc}")
    port = sock.getsockname()[1]
    print(f"ADMIN_SMOKE_READY {port}", flush=True)
    try:
        server.run(sockets=[sock])
    except BaseException as exc:  # noqa: BLE001
        _fail(f"server died after announcing readiness: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

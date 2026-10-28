"""
export_openapi.py — Export FastAPI OpenAPI schema → api/openapi.json for FE codegen.

@spec(S10.A3)
Wraps: rag2025/.omc/plans/standardize_frontend_backend_plan_v2.md Section 5.A3

Usage:
    python rag2025/scripts/export_openapi.py [--output PATH]

CI gate workflow: .github/workflows/api-contract.yml runs this + npm run types:gen
then `git diff --exit-code api/openapi.json uni-guide-ai/src/lib/api-types.ts`.
PR fails if types drift between BE and FE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "api" / "openapi.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # `main.py` imports as `src.*` (e.g. `from src.services.query_enhancer import …`)
    # while its own modules import as `config.*` / `services.*`. Both roots must be on
    # the path: inserting only `rag2025/src` fails with "No module named 'src'".
    for root in (REPO_ROOT / "rag2025", REPO_ROOT / "rag2025" / "src"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    try:
        from main import app  # noqa: E402
    except ImportError as exc:
        print(f"[FATAL] Cannot import FastAPI app from rag2025/src/main.py: {exc}")
        return 1

    # The comparison router is mounted by main.py once Main wires it in. Until
    # then, register it here so /v2/compare and its response models land in
    # api/openapi.json for FE codegen — idempotent: include_router is a no-op if
    # the route is already registered (ticket 18, criterion 5).
    try:
        from routers.comparison import router as _comparison_router  # noqa: E402
        if "/v2/compare" not in {
            r.path for r in app.routes if hasattr(r, "path")
        }:
            app.include_router(_comparison_router)
    except Exception as _e:  # noqa: BLE001
        print(f"[WARN] comparison router not registered for schema export: {_e}")

    schema = app.openapi()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Stable serialization: indent=2, sort_keys, no NaN
    serialized = json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True)
    # newline="" pins LF endings: on Windows the default text mode rewrites
    # every \n to \r\n, so a locally regenerated schema would differ from the
    # committed (LF) bytes on EOL alone and the CI drift gate would fire a
    # false RED — worse, the same file exported on ubuntu would not match the
    # Windows bytes at all. The artifact must be byte-stable across platforms.
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        fh.write(serialized + "\n")
    print(f"[OK] Exported OpenAPI schema → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

    # Add rag2025 to import path
    sys.path.insert(0, str(REPO_ROOT / "rag2025" / "src"))

    try:
        from main import app  # noqa: E402
    except ImportError as exc:
        print(f"[FATAL] Cannot import FastAPI app from rag2025/src/main.py: {exc}")
        return 1

    schema = app.openapi()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Stable serialization: indent=2, sort_keys, no NaN
    serialized = json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(f"[OK] Exported OpenAPI schema → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

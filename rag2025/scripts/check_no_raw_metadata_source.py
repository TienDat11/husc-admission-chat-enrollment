# @spec(S13.2) grep gate — forbid raw metadata["source"] / metadata.get("source") access
"""Pre-commit / CI gate that ensures no source code in `rag2025/src/` accesses
the legacy `metadata["source"]` or `metadata.get("source")` patterns directly.

All such access MUST go through `services._metadata_helpers.get_source_label()`
(or its siblings). The only file allowed to read the legacy `source` key is
`_metadata_helpers.py` itself.

Exit code 0: no violations. Exit code 1: violations found (printed to stderr).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

from loguru import logger


# Match metadata.get("source") or metadata.get('source') and metadata["source"]/['source'].
# Use non-capturing groups to keep the pattern simple.
_PATTERN = re.compile(
    r"""metadata\s*\.\s*get\(\s*["']source["']\s*\)|metadata\s*\[\s*["']source["']\s*\]""",
    re.VERBOSE,
)

# Files allowed to contain the pattern (definition sites, tests).
_ALLOWLIST_BASENAMES = {
    "_metadata_helpers.py",
    "test_metadata_helpers.py",
    "check_no_raw_metadata_source.py",  # this script's own pattern lines
    "test_check_no_raw_metadata_source.py",
}


def find_violations(root: Path) -> list[tuple[Path, int, str]]:
    """Walk *.py files under `root`, collect (path, line_number, line) for matches."""
    violations: list[tuple[Path, int, str]] = []
    for py in sorted(root.rglob("*.py")):
        if py.name in _ALLOWLIST_BASENAMES:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                violations.append((py, line_no, line.rstrip()))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Forbid raw metadata['source'] access")
    parser.add_argument(
        "--root",
        default="rag2025/src",
        help="Directory to scan (default: rag2025/src)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        logger.error(f"Root path does not exist: {root}")
        return 1

    violations = find_violations(root)
    if not violations:
        logger.info(f"No raw metadata['source'] access found under {root}")
        return 0

    logger.error(f"Found {len(violations)} forbidden raw metadata['source'] access(es):")
    for path, line_no, line in violations:
        sys.stderr.write(f"{path}:{line_no}: {line}\n")
    sys.stderr.write(
        "\nUse services._metadata_helpers.get_source_label() instead.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

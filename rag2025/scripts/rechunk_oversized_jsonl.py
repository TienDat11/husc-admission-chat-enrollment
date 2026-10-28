"""T1.8: re-chunk oversized chunks in chunked JSONL files into <=512-token children.

Operates on data/chunked/chunked_*.jsonl (the files read by
scripts/test_chunk_quality.py). Splits any chunk above HARD_MAX_TOKENS using the
same row-aware / balanced splitter as chunk_table_aware_v2. Text-only — no
embedding. Idempotent: chunks already <= HARD_MAX_TOKENS are untouched.

Usage:
    cd rag2025
    python scripts/rechunk_oversized_jsonl.py [--max-tokens 512] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunk_table_aware_v2 import _split_oversized_preserving  # noqa: E402

CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"
DEFAULT_MAX_TOKENS = 512


def _split_chunk(obj: dict, text: str, enc, max_tokens: int) -> list[dict]:
    """Split one chunk dict into <=max_tokens children (suffix _s{n})."""
    pieces = _split_oversized_preserving(text, enc, max_tokens)
    if len(pieces) <= 1:
        return [obj]
    orig_id = obj.get("id")
    base_meta = dict(obj.get("metadata") or {})
    children = []
    for si, piece in enumerate(pieces):
        child = dict(obj)
        child["text"] = piece
        child["text_plain"] = piece
        child["id"] = f"{orig_id}_s{si}"
        meta = dict(base_meta)
        meta["token_count"] = len(enc.encode(piece))
        meta["split_index"] = si
        child["metadata"] = meta
        children.append(child)
    return children


def _process_lines(lines: list[str], enc, max_tokens: int) -> tuple[list[str], int]:
    """Return (new_lines, new_children_count)."""
    new_lines: list[str] = []
    total_children = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        text = obj.get("text") or ""
        if len(enc.encode(text)) <= max_tokens:
            new_lines.append(line)
            continue
        children = _split_chunk(obj, text, enc, max_tokens)
        if len(children) <= 1:
            new_lines.append(line)
            continue
        total_children += len(children)
        new_lines.extend(json.dumps(c, ensure_ascii=False) for c in children)
    return new_lines, total_children


def main() -> int:
    parser = argparse.ArgumentParser(description="T1.8 re-chunk oversized JSONL chunks")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    files = sorted(CHUNKED_DIR.glob("chunked_*.jsonl"))
    total_children = 0
    any_changed = False

    for path in files:
        if "enhanced" in path.name or "test" in path.name or ".bak" in path.name:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines, n_children = _process_lines(lines, enc, args.max_tokens)
        if len(new_lines) == len(lines) and n_children == 0:
            continue
        any_changed = True
        total_children += n_children
        print(
            f"  {path.name}: {len(lines)} -> {len(new_lines)} lines "
            f"(+{n_children} children)"
        )
        if not args.dry_run:
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    if not any_changed:
        print("No oversized chunks found; nothing to re-chunk.")
        return 0

    print(f"Total new children: {total_children}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

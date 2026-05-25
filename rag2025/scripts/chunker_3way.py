# @spec(S13.3) 3-way chunker orchestrator (system + Haiku API + Claude CLI subprocess)
"""Run system_v2, Haiku, and Claude CLI chunkers in parallel for one HUSC notification.

Outputs:
  data/chunked_3way/{nid}/system_v2.jsonl
  data/chunked_3way/{nid}/haiku_v1.jsonl
  data/chunked_3way/{nid}/claude_v1.jsonl
  data/chunked_3way/{nid}/manifest.sha256

Determinism: temperature=0, top_p=1, seed=42 for both LLM lanes. Mocked in tests.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


HAIKU_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_CLI_MODEL = "claude-opus-4-7"
SEED = 42
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
HAIKU_TIMEOUT_S = 120
CLAUDE_CLI_TIMEOUT_S = 300


# ---------- system_v2 (placeholder, deterministic) ----------

def _run_system_v2_chunker(html_text: str) -> list[dict[str, Any]]:
    """Deterministic split by paragraph for system_v2 baseline.

    Real system_v2 lives in rag2025/src/chunker.py (chunk_table_aware_v2). For
    P2-3 scope we ship a placeholder that produces stable output; the real
    chunker is wired in a follow-up integration task.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", html_text) if p.strip()]
    chunks = []
    for i, p in enumerate(paragraphs):
        chunks.append({
            "text": p,
            "suggested_title": f"Đoạn {i + 1}",
            "semantic_topic": "unknown",
            "info_type": "unknown",
        })
    return chunks


# ---------- haiku_v1 (Anthropic API via httpx) ----------

def _run_haiku_chunker(html_text: str, prompt_template: str, *, api_key: str) -> list[dict[str, Any]]:
    """Call Claude Haiku API with strict determinism flags."""
    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": 8000,
        "temperature": 0,
        "top_p": 1,
        "metadata": {"user_id": "chunker_haiku_seed42"},
        "system": prompt_template,
        "messages": [
            {"role": "user", "content": f"<HTML>\n{html_text}\n</HTML>\n\nTrả JSON theo schema."}
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    with httpx.Client(timeout=HAIKU_TIMEOUT_S) as client:
        resp = client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = "".join(block.get("text", "") for block in data.get("content", []))
    return _parse_chunker_json(text)


# ---------- claude_v1 (Claude Code CLI subprocess) ----------

def _run_claude_cli_chunker(html_text: str, prompt_template: str) -> list[dict[str, Any]]:
    """Invoke `claude --print --output-format json --model claude-opus-4-7` via subprocess."""
    # Compose the user prompt to feed via stdin.
    user_input = f"{prompt_template}\n\n<HTML>\n{html_text}\n</HTML>\n\nTrả JSON theo schema."
    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--permission-mode", "acceptEdits",
        "--model", CLAUDE_CLI_MODEL,
    ]
    proc = subprocess.run(
        cmd,
        input=user_input,
        capture_output=True,
        text=True,
        timeout=CLAUDE_CLI_TIMEOUT_S,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exit {proc.returncode}: {proc.stderr}")
    # CLI in --output-format json returns wrapper {"result": "...", ...}.
    try:
        wrapper = json.loads(proc.stdout)
        text = wrapper.get("result", proc.stdout)
    except json.JSONDecodeError:
        text = proc.stdout
    return _parse_chunker_json(text)


# ---------- shared parsing ----------

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_chunker_json(text: str) -> list[dict[str, Any]]:
    """Parse strict-JSON chunker output, stripping fences if present."""
    if not text:
        raise ValueError("Empty chunker response")
    cleaned = _JSON_FENCE.sub("", text.strip())
    data = json.loads(cleaned)
    chunks = data.get("chunks") if isinstance(data, dict) else None
    if not isinstance(chunks, list):
        raise ValueError(f"Expected 'chunks' list in JSON, got: {type(chunks).__name__}")
    return chunks


# ---------- orchestrator ----------

def run_3way(
    nid: int,
    html_text: str,
    output_dir: Path,
    prompt_haiku: str,
    prompt_claude: str,
    *,
    api_key: str | None = None,
    haiku_runner=None,
    claude_runner=None,
    system_runner=None,
) -> dict[str, Any]:
    """Run all 3 chunkers in parallel and write outputs.

    Runners are injectable for testing. In production they default to the
    real implementations defined above.
    """
    haiku_runner = haiku_runner or (lambda html, prompt: _run_haiku_chunker(html, prompt, api_key=api_key or os.environ["ANTHROPIC_API_KEY"]))
    claude_runner = claude_runner or _run_claude_cli_chunker
    system_runner = system_runner or _run_system_v2_chunker

    output_dir = Path(output_dir) / str(nid)
    output_dir.mkdir(parents=True, exist_ok=True)

    lanes = {
        "system_v2": (system_runner, html_text),
        "haiku_v1": (haiku_runner, html_text, prompt_haiku),
        "claude_v1": (claude_runner, html_text, prompt_claude),
    }

    results: dict[str, list[dict[str, Any]] | None] = {}
    errors: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {}
        for name, args in lanes.items():
            futs[pool.submit(*args)] = name
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            try:
                results[name] = fut.result()
            except Exception as exc:
                logger.error(f"lane={name} failed: {exc}")
                errors[name] = str(exc)
                results[name] = None

    manifest_lines = []
    for name, chunks in results.items():
        out_path = output_dir / f"{name}.jsonl"
        if chunks is None:
            out_path.write_text(f"# ERROR: {errors.get(name)}\n", encoding="utf-8")
            sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        else:
            with out_path.open("w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
            sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        manifest_lines.append(f"{sha}  {name}.jsonl")

    manifest = output_dir / "manifest.sha256"
    manifest.write_text("\n".join(sorted(manifest_lines)) + "\n", encoding="utf-8")

    return {
        "id": nid,
        "output_dir": str(output_dir),
        "lanes": {name: (len(c) if c else 0) for name, c in results.items()},
        "errors": errors,
        "manifest": str(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-way chunker for one notification HTML")
    parser.add_argument("--nid", type=int, required=True)
    parser.add_argument("--html-path", required=True)
    parser.add_argument("--output-dir", default="rag2025/data/chunked_3way")
    parser.add_argument("--prompt-haiku", default="rag2025/prompts/chunker_haiku.txt")
    parser.add_argument("--prompt-claude", default="rag2025/prompts/chunker_claude.txt")
    args = parser.parse_args()

    html_text = Path(args.html_path).read_text(encoding="utf-8")
    prompt_haiku = Path(args.prompt_haiku).read_text(encoding="utf-8")
    prompt_claude = Path(args.prompt_claude).read_text(encoding="utf-8")

    res = run_3way(args.nid, html_text, Path(args.output_dir), prompt_haiku, prompt_claude)
    logger.info(json.dumps(res, ensure_ascii=False))
    return 0 if not res["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())

# @spec(S13.3) chunker arbiter — boundary IoU + faithfulness judge (deepseek-v4-pro)
"""Adjudicate between 3-way chunker outputs (system_v2, haiku_v1, claude_v1).

Scoring:
- Boundary agreement: Jaccard / IoU on token offsets pairwise.
- Faithfulness: LLM-as-judge (default deepseek-v4-pro per C12) scores each chunk
  against the source HTML.

Winner rule (P4 weighted):
- Default: system_v2 wins (weight 1.0).
- Override: ensemble wins iff IoU(system, ensemble) < 0.5
  AND faithfulness(ensemble) - faithfulness(system) >= 0.1.

Hard escalation rule:
- If boundary IoU(system, haiku) < 0.5 for >= 3 consecutive notifications,
  return winner=None for this notification (manual decision required).

Outputs:
- data/chunked_2026.jsonl  (canonical chunks for 2026 notifications)
- data/chunked_2025_history.jsonl  (canonical chunks for 2025 history)
- data/chunker_decision.jsonl  (per-doc rationale)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _is_jsonl_data_line(line: str) -> bool:
    """Skip empty + comment lines when reading JSONL written by chunker_3way."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


WINNER_DEFAULT = "system_v2"
IOU_THRESHOLD_OVERRIDE = 0.5
FAITHFULNESS_DELTA_OVERRIDE = 0.1
IOU_THRESHOLD_ESCALATE = 0.5
ESCALATION_CONSEC_DOCS = 3


# ---------- IoU computation ----------

def _chunk_token_span(chunk: dict[str, Any]) -> tuple[int, int]:
    """Return (start_offset, end_offset) for one chunk.

    If the chunk has explicit offsets, use them; else derive from text length
    and a running cursor (caller's responsibility to maintain).
    """
    if "start_offset" in chunk and "end_offset" in chunk:
        return int(chunk["start_offset"]), int(chunk["end_offset"])
    # Fallback: cumulative-length window (callers should pre-compute and pass offsets).
    text = chunk.get("text", "")
    return 0, len(text)


def _spans_with_cursor(chunks: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Compute (start, end) spans by running cursor over chunks lacking offsets."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for c in chunks:
        if "start_offset" in c and "end_offset" in c:
            spans.append((int(c["start_offset"]), int(c["end_offset"])))
        else:
            length = len(c.get("text", ""))
            spans.append((cursor, cursor + length))
            cursor += length
    return spans


def compute_iou(spans_a: list[tuple[int, int]], spans_b: list[tuple[int, int]]) -> float:
    """Compute IoU on union-of-intervals between two boundary partitions.

    Both inputs are lists of (start, end) intervals. We compute the intersection
    length and union length over the entire span, then return intersection/union.

    Edge cases: empty inputs return 0.0; identical inputs return 1.0.
    """
    if not spans_a or not spans_b:
        return 0.0
    set_a = _to_charset(spans_a)
    set_b = _to_charset(spans_b)
    intersect = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersect / union) if union > 0 else 0.0


def _to_charset(spans: list[tuple[int, int]]) -> set[int]:
    """Convert intervals to a set of integer positions for IoU computation."""
    out: set[int] = set()
    for s, e in spans:
        out.update(range(s, e))
    return out


# ---------- Faithfulness judge ----------

def faithfulness_judge(
    chunk_text: str,
    source_html: str,
    *,
    judge_runner: Callable[[str], float] | None = None,
) -> float:
    """Score chunk faithfulness in [0, 1] vs source HTML.

    judge_runner: callable accepting the prompt string and returning a float
    score. In production this calls deepseek-v4-pro via UnifiedLLMClient with
    temperature=0 (per C12). For tests, inject a deterministic stub.
    """
    if judge_runner is None:
        # Lazy default — only used if caller doesn't inject (production path).
        judge_runner = _default_deepseek_judge

    prompt = (
        "Đánh giá độ trung thực (faithfulness) của CHUNK so với SOURCE HTML.\n"
        "Trả về DUY NHẤT một số thực trong [0.0, 1.0]: "
        "1.0 = chunk faithful 100%, 0.0 = chunk fabricated.\n\n"
        f"<SOURCE>\n{source_html}\n</SOURCE>\n\n"
        f"<CHUNK>\n{chunk_text}\n</CHUNK>\n\n"
        "Score (chỉ số):"
    )
    return judge_runner(prompt)


def _default_deepseek_judge(prompt: str) -> float:
    """Production path placeholder — callers MUST inject judge_runner.

    The previous implementation used asyncio.run() to drive an async
    UnifiedLLMClient call, which crashes inside any already-running event
    loop (FastAPI route, pytest-asyncio test, async pipeline). To avoid
    silent fallback to 0.0 (which then short-circuits the override rule
    and locks the system to system_v2), we require explicit injection.

    To wire deepseek-v4-pro in production, build a sync judge_runner using
    httpx.Client (UnifiedLLMClient OpenAI-compatible endpoint) and pass it
    via faithfulness_judge(judge_runner=...).
    """
    raise RuntimeError(
        "faithfulness_judge requires explicit judge_runner injection. "
        "The default async path was removed because asyncio.run() inside "
        "an already-running event loop crashes. See arbiter docs."
    )


# ---------- Arbiter ----------

class _ConsecutiveEscalation:
    """Track consecutive low-IoU docs across calls for hard-escalate rule."""

    def __init__(self) -> None:
        self.streak = 0

    def update(self, system_haiku_iou: float) -> bool:
        """Return True when escalation threshold reached."""
        if system_haiku_iou < IOU_THRESHOLD_ESCALATE:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= ESCALATION_CONSEC_DOCS


def arbitrate_one(
    *,
    nid: int,
    source_html: str,
    system_chunks: list[dict[str, Any]],
    haiku_chunks: list[dict[str, Any]],
    claude_chunks: list[dict[str, Any]],
    judge_runner: Callable[[str], float] | None = None,
    escalation_state: _ConsecutiveEscalation | None = None,
) -> dict[str, Any]:
    """Pick a winner for one notification.

    Returns dict with: nid, winner, iou_pairs, faithfulness, escalated, rationale.
    """
    state = escalation_state or _ConsecutiveEscalation()

    spans_sys = _spans_with_cursor(system_chunks)
    spans_haiku = _spans_with_cursor(haiku_chunks)
    spans_claude = _spans_with_cursor(claude_chunks)

    iou_sys_haiku = compute_iou(spans_sys, spans_haiku)
    iou_sys_claude = compute_iou(spans_sys, spans_claude)
    iou_haiku_claude = compute_iou(spans_haiku, spans_claude)

    # Hard escalation check.
    escalated = state.update(iou_sys_haiku)
    if escalated:
        return {
            "nid": nid,
            "winner": None,
            "escalated": True,
            "rationale": f"IoU(system,haiku)={iou_sys_haiku:.3f} < {IOU_THRESHOLD_ESCALATE} for {state.streak} consecutive docs",
            "iou_pairs": {
                "system_vs_haiku": iou_sys_haiku,
                "system_vs_claude": iou_sys_claude,
                "haiku_vs_claude": iou_haiku_claude,
            },
        }

    # Faithfulness scoring (mean over chunks per lane).
    def _mean_faithfulness(chunks: list[dict[str, Any]]) -> float:
        if not chunks:
            return 0.0
        return sum(faithfulness_judge(c.get("text", ""), source_html, judge_runner=judge_runner) for c in chunks) / len(chunks)

    f_sys = _mean_faithfulness(system_chunks)
    f_haiku = _mean_faithfulness(haiku_chunks)
    f_claude = _mean_faithfulness(claude_chunks)
    f_ensemble = max(f_haiku, f_claude)
    iou_sys_ensemble = max(iou_sys_haiku, iou_sys_claude)

    # Default winner.
    winner = WINNER_DEFAULT
    rationale = "system_v2 default (weight 1.0)"

    # Override rule.
    if iou_sys_ensemble < IOU_THRESHOLD_OVERRIDE and (f_ensemble - f_sys) >= FAITHFULNESS_DELTA_OVERRIDE:
        winner = "haiku_v1" if f_haiku >= f_claude else "claude_v1"
        rationale = (
            f"Ensemble override: iou(sys,ensemble)={iou_sys_ensemble:.3f} < {IOU_THRESHOLD_OVERRIDE}, "
            f"faithfulness gap={f_ensemble - f_sys:.3f} >= {FAITHFULNESS_DELTA_OVERRIDE}"
        )

    return {
        "nid": nid,
        "winner": winner,
        "escalated": False,
        "rationale": rationale,
        "iou_pairs": {
            "system_vs_haiku": iou_sys_haiku,
            "system_vs_claude": iou_sys_claude,
            "haiku_vs_claude": iou_haiku_claude,
        },
        "faithfulness": {
            "system_v2": f_sys,
            "haiku_v1": f_haiku,
            "claude_v1": f_claude,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunker 3-way arbiter")
    parser.add_argument("--input-dir", required=True, help="Path to chunked_3way/{nid}/")
    parser.add_argument("--source-html", required=True, help="Path to original HTML")
    parser.add_argument("--nid", type=int, required=True)
    parser.add_argument("--decision-out", default="rag2025/data/chunker_decision.jsonl")
    args = parser.parse_args()

    indir = Path(args.input_dir)
    sys_chunks = [json.loads(l) for l in (indir / "system_v2.jsonl").read_text(encoding="utf-8").splitlines() if _is_jsonl_data_line(l)]
    haiku_chunks = [json.loads(l) for l in (indir / "haiku_v1.jsonl").read_text(encoding="utf-8").splitlines() if _is_jsonl_data_line(l)]
    claude_chunks = [json.loads(l) for l in (indir / "claude_v1.jsonl").read_text(encoding="utf-8").splitlines() if _is_jsonl_data_line(l)]
    source_html = Path(args.source_html).read_text(encoding="utf-8")

    # NOTE: This CLI processes one notification per invocation. Multi-notification
    # callers MUST maintain a single _ConsecutiveEscalation() across calls to
    # honor the "3 consecutive low-IoU docs → escalate" hard rule (MED-5 fix).
    decision = arbitrate_one(
        nid=args.nid,
        source_html=source_html,
        system_chunks=sys_chunks,
        haiku_chunks=haiku_chunks,
        claude_chunks=claude_chunks,
    )

    decision_path = Path(args.decision_out)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    with decision_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info(json.dumps(decision, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

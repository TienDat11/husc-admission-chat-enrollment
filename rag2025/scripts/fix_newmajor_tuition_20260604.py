"""One-shot corpus correction (2026-06-04), user-approved fast fix.

Fixes 3 issues found in the LanceDB rag2025 audit:
  1. Old-wrong tuition in aggregate chunks: Khối VII 525.000 -> 490.000,
     Khối IV 540.000 -> 555.000 (Khối V 600.000 unchanged). NĐ 238/2025.
  2. Fabricated tuition on the 3 NEW 2026 majors (7440102SC, 7460108,
     7510302IC) -> replaced with "chưa công bố (ngành mới 2026...)".
  3. Wrong Hán-Nôm code 7485104 -> 7220104 (chunked_24_chunk_356).

Updates BOTH on-disk jsonl (re-ingest safety) and the live LanceDB table
in place via t.update(where=...) so the embedding vector is preserved.
Idempotent: re-running is a no-op once corrected.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import lancedb

ROOT = Path(__file__).resolve().parents[1]
DB_URI = str(ROOT / "data" / "lancedb")
TABLE = "rag2025"

NEW_MAJOR_CODES = ("7440102SC", "7460108", "7510302IC")
BLOCK_FIX_CHUNKS = (
    "liet_ke_28_nganh_2026_v2",
    "liet_ke_26_nganh_2026",
    "thay_doi_tuyensinh_2026",
)
HANNOM_CHUNK = "chunked_24_chunk_356"

# segment regex: "học phí Khối VII 525.000đ/TC" etc.
_HP_SEG = re.compile(r"học phí\s+Khối\s+(VII|IV|V)\s+([\d.]+)đ/TC", re.IGNORECASE)
_NEW_CODE_RE = re.compile("|".join(re.escape(c) for c in NEW_MAJOR_CODES))


def _fix_block_line(line: str) -> str:
    """Fix one major-list line: strip tuition for new majors, else map number."""
    if _NEW_CODE_RE.search(line):
        # new major: replace the whole "học phí Khối X NNN.000đ/TC" with placeholder
        return _HP_SEG.sub("học phí chưa công bố (ngành mới 2026, liên hệ trường)", line)

    def _map(m: re.Match) -> str:
        block, num = m.group(1).upper(), m.group(2)
        if block == "VII" and num == "525.000":
            num = "490.000"
        elif block == "IV" and num == "540.000":
            num = "555.000"
        return f"học phí Khối {m.group(1)} {num}đ/TC"

    return _HP_SEG.sub(_map, line)


def fix_block_text(text: str) -> str:
    return "\n".join(_fix_block_line(ln) for ln in text.split("\n"))


def fix_hannom_text(text: str) -> str:
    return text.replace("7485104", "7220104")


def correct_disk() -> list[str]:
    """Patch on-disk jsonl sources. Returns list of human-readable actions."""
    log: list[str] = []
    for jf in ROOT.glob("data/**/*.jsonl"):
        try:
            lines = jf.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        changed = False
        out: list[str] = []
        for ln in lines:
            if not ln.strip():
                out.append(ln)
                continue
            try:
                o = json.loads(ln)
            except Exception:
                out.append(ln)
                continue
            cid = str(o.get("chunk_id") or o.get("id") or "")
            new_txt = None
            if cid in BLOCK_FIX_CHUNKS and isinstance(o.get("text"), str):
                new_txt = fix_block_text(o["text"])
            elif cid == HANNOM_CHUNK and isinstance(o.get("text"), str):
                new_txt = fix_hannom_text(o["text"])
            if new_txt is not None and new_txt != o["text"]:
                o["text"] = new_txt
                out.append(json.dumps(o, ensure_ascii=False))
                changed = True
                log.append(f"disk: {jf.name}:{cid} patched")
            else:
                out.append(ln)
        if changed:
            jf.write_text("\n".join(out) + "\n", encoding="utf-8")
    return log


def correct_lancedb() -> list[str]:
    log: list[str] = []
    db = lancedb.connect(DB_URI)
    t = db.open_table(TABLE)
    df = t.to_pandas()
    seen: set[str] = set()
    for _, row in df.iterrows():
        cid = str(row["chunk_id"])
        if cid in seen:
            continue
        text = row["text"]
        if not isinstance(text, str):
            continue
        new_txt = None
        if cid in BLOCK_FIX_CHUNKS:
            new_txt = fix_block_text(text)
        elif cid == HANNOM_CHUNK:
            new_txt = fix_hannom_text(text)
        if new_txt is not None and new_txt != text:
            esc = new_txt.replace("'", "''")
            t.update(where=f"chunk_id = '{cid}'", values={"text": new_txt})
            log.append(f"lancedb: {cid} updated ({len(text)}->{len(new_txt)} chars)")
        seen.add(cid)
    return log


def verify() -> None:
    db = lancedb.connect(DB_URI)
    t = db.open_table(TABLE)
    df = t.to_pandas()
    bad525 = df["text"].astype(str).str.contains("525.000", na=False).sum()
    bad540 = df["text"].astype(str).str.contains("540.000", na=False).sum()
    bad748 = df["text"].astype(str).str.contains("7485104", na=False).sum()
    # new majors must NOT carry a numeric tuition on their list line
    m = df["chunk_id"].astype(str).eq("liet_ke_28_nganh_2026_v2")
    new_fab = 0
    if m.any():
        tx = df[m].iloc[0]["text"]
        for ln in tx.split("\n"):
            if _NEW_CODE_RE.search(ln) and _HP_SEG.search(ln):
                new_fab += 1
    print(f"VERIFY rag2025: 525.000 rows={bad525} (chk_375 range ok if 1) "
          f"540.000 rows={bad540} 7485104 rows={bad748} new-major-fabricated-lines={new_fab}")
    assert bad748 == 0, "Hán-Nôm 7485104 still present"
    assert new_fab == 0, "new major still has fabricated tuition number"


if __name__ == "__main__":
    actions = correct_disk() + correct_lancedb()
    for a in actions:
        print(a)
    print(f"TOTAL_ACTIONS={len(actions)}")
    verify()
    print("DONE")

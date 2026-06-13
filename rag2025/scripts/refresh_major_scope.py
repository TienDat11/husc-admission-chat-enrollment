# @spec(S19) refresh_major_scope — regenerate the crawler-writable major-scope file
"""Regenerate ``rag2025/data/major_codes/husc_majors_current.json`` from the
freshest available source. Idempotent: re-running with no upstream changes
produces the same file. Supports ``--dry-run`` to print a diff.

Source priority (offline-safe):
  1. Live crawl of tuyensinh.husc.edu.vn/nganhdaotao-dh.php (best-effort;
     if a clean major-list endpoint isn't reachable, falls through to step 2)
  2. Canonical ``husc_tuition_2026_official.json`` (PDF-grounded, always present)
  3. ``2026.json`` (code list) to fill gaps where the tuition file has no
     name (rare)

The KEY deliverable: a script that, when invoked (cron / post-crawl hook),
re-stamps ``husc_majors_current.json`` so the guardrail's mtime-cache picks
up new majors automatically without restart.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

# Make rag2025/src importable so we can reuse the major_code_validator infra
_HERE = Path(__file__).resolve().parent
_RAG_ROOT = _HERE.parent
_SRC = _RAG_ROOT / "src"
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.major_code_validator import (
    WHITELIST_2026,
    is_well_formed_major_code,
    get_whitelist,
)  # noqa: E402

OUT_PATH = _RAG_ROOT / "data" / "major_codes" / "husc_majors_current.json"
TUITION_PATH = _RAG_ROOT / "data" / "major_codes" / "husc_tuition_2026_official.json"
CODES_PATH = _RAG_ROOT / "data" / "major_codes" / "2026.json"


# ---------------------------------------------------------------------------
# Alias helpers (no external NLP — same shape as the existing guardrail logic)
# ---------------------------------------------------------------------------

# Common alias templates keyed by Vietnamese major name fragment. Kept
# minimal and conservative — only the highest-value aliases that catch
# real user phrasing.
ALIAS_TEMPLATES = {
    "vật lý": ["vật lý", "vat ly", "vat ly hoc"],
    "công nghệ bán dẫn": ["công nghệ bán dẫn", "cong nghe ban dan", "bán dẫn", "ban dan", "ctdt bán dẫn"],
    "công nghệ thông tin": ["cntt", "cong nghe thong tin", "công nghệ thông tin", "it"],
    "công nghệ thông tin - ctđt cử nhân việt-nhật": ["cntt việt nhật", "cntt viet nhat", "cntt vn"],
    "khoa học dữ liệu": ["khdl", "khoa hoc du lieu", "khoa học dữ liệu", "data science"],
    "khoa học dữ liệu (mã 7460108)": ["khdl", "khoa hoc du lieu", "khoa học dữ liệu", "data science"],
    "7460108": ["khdl", "khoa hoc du lieu", "khoa học dữ liệu", "data science", "khoa hoc du", "hoc du lieu"],
    "kỹ thuật phần mềm": ["ktpm", "ky thuat phan mem", "kỹ thuật phần mềm"],
    "công nghệ sinh học": ["sinh học", "cong nghe sinh hoc", "sinh hoc", "cnsh"],
    "khoa học môi trường": ["khmt", "khoa hoc moi truong", "môi trường"],
    "hóa học": ["hoa hoc", "hóa học"],
    "hóa dược": ["hoa duoc", "hóa dược"],
    "toán học": ["toan hoc", "toán", "toan"],
    "toán ứng dụng": ["toan ung dung", "toán ứng dụng"],
    "kiến trúc": ["kien truc", "kiến trúc"],
    "địa kỹ thuật xây dựng": ["dia ky thuat xay dung", "địa kỹ thuật"],
    "kỹ thuật trắc địa - bản đồ": ["trac dia ban do", "trắc địa bản đồ"],
    "cnkt điện tử - viễn thông": ["điện tử viễn thông", "dien tu vien thong", "cnkt dtvt"],
    "cnkt hóa học": ["cnkt hoa hoc", "công nghệ kỹ thuật hóa học"],
    "triết học": ["triet hoc", "triết học"],
    "lịch sử": ["lich su", "lịch sử"],
    "văn học": ["van hoc", "văn học", "văn"],
    "quản lý văn hóa": ["quan ly van hoa", "qlvh"],
    "quản lý nhà nước": ["qlnn", "quan ly nha nuoc"],
    "xã hội học": ["xa hoi hoc", "xhh"],
    "đông phương học": ["dong phuong hoc", "đông phương học"],
    "đông nam á học": ["dong nam a hoc", "đông nam á"],
    "báo chí": ["bao chi", "báo chí", "journalism"],
    "truyền thông số": ["truyen thong so", "truyền thông số", "digital media"],
    "truyền thông đa phương tiện": ["truyen thong da phuong tien", "multimedia"],
    "tâm lý học": ["tam ly hoc", "tâm lý học"],
    "công tác xã hội": ["cong tac xa hoi", "ctxh", "social work"],
    "hán - nôm": ["han nom", "hán nôm"],
    "khoa học máy tính": ["khmt", "khoa hoc may tinh", "computer science"],
    "hệ thống thông tin": ["httt", "he thong thong tin", "information systems"],
    "quản lý tài nguyên và môi trường": ["qltnmt", "quan ly tai nguyen va moi truong"],
    "quản lý an toàn, sức khỏe và môi trường": ["qlasm", "an toàn sức khỏe môi trường"],
    "y sinh": ["y sinh", "biomedical"],
    "7460108": ["khdl", "khoa hoc du lieu", "khoa học dữ liệu", "data science"],
}


def _aliases_for(ten: str) -> List[str]:
    t_low = ten.lower()
    for key, aliases in ALIAS_TEMPLATES.items():
        if key in t_low:
            return aliases
    # Fallback: at least include the name itself (folded by guardrail later)
    return []


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------


def _load_tuition_majors() -> List[dict]:
    if not TUITION_PATH.exists():
        return []
    try:
        with TUITION_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    majors = data.get("majors", []) if isinstance(data, dict) else []
    return [m for m in majors if isinstance(m, dict)]


def _load_codes_only_majors() -> List[dict]:
    """Codes listed in 2026.json but NOT in the tuition file (rare gap)."""
    if not CODES_PATH.exists():
        return []
    try:
        with CODES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    tuition = {m.get("ma_nganh") for m in _load_tuition_majors()}
    out: List[dict] = []
    for code in data.get("codes", []):
        if not isinstance(code, str) or not is_well_formed_major_code(code):
            continue
        if code in tuition:
            continue
        out.append({"ma_nganh": code, "ten": code})  # fallback name = code
    return out


def _try_live_crawl() -> Optional[List[dict]]:
    """Best-effort live crawl of the major list page.

    Intentionally lightweight: a real implementation would scrape
    tuyensinh.husc.edu.vn/nganhdaotao-dh.php. We attempt a network call
    but stay fail-soft — any failure returns None and the caller falls
    back to the canonical tuition file (which is the source of truth for
    the 2026 cycle).
    """
    try:
        import httpx  # local import — only required for the live path
    except Exception:
        return None
    try:
        url = "https://tuyensinh.husc.edu.vn/nganhdaotao-dh.php"
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(
                url,
                headers={"User-Agent": "HUSC-Admission-Chat/1.0 (+refresh-major-scope)"},
            )
        if resp.status_code != 200:
            return None
        # We intentionally do NOT parse the HTML here — the page layout
        # changes frequently and a brittle parser would silently produce
        # a worse file than the canonical tuition json. The live path
        # is a STUB: when a parser is available, plug it in here and
        # return [{"ma_nganh": "...", "ten": "..."}, ...].
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build / write
# ---------------------------------------------------------------------------


def collect_majors() -> List[dict]:
    """Build the unified majors list from the freshest available source."""
    live = _try_live_crawl()
    if live:
        return live
    merged: List[dict] = list(_load_tuition_majors())
    seen_codes = {m.get("ma_nganh") for m in merged}
    for m in _load_codes_only_majors():
        if m.get("ma_nganh") not in seen_codes:
            merged.append(m)
            seen_codes.add(m.get("ma_nganh"))
    return merged


def build_document(majors: Iterable[dict]) -> dict:
    majors_list: List[dict] = []
    for m in majors:
        if not isinstance(m, dict):
            continue
        code = m.get("ma_nganh")
        ten = m.get("ten")
        if not code or not ten:
            continue
        aliases = _aliases_for(ten)
        majors_list.append({"ma_nganh": code, "ten": ten, "aliases": aliases})
    return {
        "_meta": {
            "source": "scripts/refresh_major_scope.py — auto-stamped, crawler-overwritable",
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "crawler overwrites this; guardrail reads it via mtime-reload.",
        },
        "majors": majors_list,
    }


def _existing_majors(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [m for m in data.get("majors", []) if isinstance(m, dict)]


def _diff(old: List[dict], new: List[dict]) -> Tuple[Set[str], Set[str]]:
    old_codes = {m.get("ma_nganh") for m in old}
    new_codes = {m.get("ma_nganh") for m in new}
    return (new_codes - old_codes), (old_codes - new_codes)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Refresh husc_majors_current.json from canonical sources")
    p.add_argument("--dry-run", action="store_true", help="Print diff without writing")
    args = p.parse_args(argv)

    majors = collect_majors()
    if not majors:
        print("[refresh_major_scope] no majors collected; aborting", file=sys.stderr)
        return 1

    doc = build_document(majors)
    new_codes = {m["ma_nganh"] for m in doc["majors"]}
    old = _existing_majors(OUT_PATH)
    added, removed = _diff(old, doc["majors"])

    print(f"[refresh_major_scope] source: scripts/refresh_major_scope.py")
    print(f"[refresh_major_scope] canonical: {len(_load_tuition_majors())} majors from husc_tuition_2026_official.json")
    print(f"[refresh_major_scope] codes-only gaps filled from 2026.json: {len(_load_codes_only_majors())}")
    print(f"[refresh_major_scope] final major count: {len(doc['majors'])}")
    if added:
        print(f"[refresh_major_scope] + added ({len(added)}): {sorted(added)}")
    if removed:
        print(f"[refresh_major_scope] - removed ({len(removed)}): {sorted(removed)}")
    if not added and not removed:
        print("[refresh_major_scope] no diff vs current file")

    if args.dry_run:
        print("[refresh_major_scope] --dry-run: not writing")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[refresh_major_scope] wrote {OUT_PATH} ({len(doc['majors'])} majors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

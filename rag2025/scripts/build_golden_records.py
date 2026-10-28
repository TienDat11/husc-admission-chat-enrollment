"""Derive expected_record_ids for husc_86q_test.json from question text + store facts.

Hard rule (T00 gap 1): expected ids come ONLY from
  (a) an explicit 7-digit major code in the question text,
  else (b) a single longest-first match against the alias table — which is
  DERIVED from the store's own NGANH records' `ten_nganh` values (NFC-normalised,
  bracketed/parenthesised qualifiers and OCR-space artifacts stripped, near-
  duplicate names consolidated) PLUS an explicit, auditable override map for
  abbreviations that never appear as a full store name (CNTT, KTPM, CNKH, ...),
  intersected with
  (c) fact-type keywords (HOC_PHI, DIEM_CHUAN, CHI_TIEU, TO_HOP, NGANH, THOI_GIAN),
  and
  (d) the admission year — an explicit 4-digit year IN the question text; when
  the text names no year at all the store's default year (2026) applies. An
  explicit year with NO records in the store (e.g. 2023, 2027) must NOT fall
  back to 2026: it yields an empty expected set.

Expected ids are NEVER taken from baseline retrieval output. The T00 baseline
sidecar JSONL is optionally joined by question id to record what T00 *actually*
retrieved into a separate, clearly-labelled field ``baseline_observed_record_ids``
(still expressed in terms of the store's ``source_chunk_id`` -> ``record_id``
join, never merged into the expected set).

Deterministic: sorted keys, sorted id lists, no timestamps.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
_HERE = _SCRIPTS_DIR.parent  # rag2025/
for _p in (str(_HERE), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_YEAR: int = 2026

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_BLOCK_BRACKETS_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def _strip_accents(s: str) -> str:
    """NFC-normalise then drop diacritics, including 'd' with stroke."""
    s = unicodedata.normalize("NFC", s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\u0111", "d").replace("\u0110", "d")  # đ / Đ already casefolded-safe
    return s


def _clean_text(s: str) -> str:
    """NFC-normalise, drop bracketed qualifiers, strip accents, casefold, collapse
    whitespace + punctuation. Every pattern below (alias OR fact-type) runs on
    THIS output — so patterns are written in accent-stripped, lowercase form."""
    s = _BLOCK_BRACKETS_RE.sub(" ", s)
    s = _strip_accents(s).casefold()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _clean_name(name: str) -> str:
    """Same clean applied to the store's noisy `ten_nganh` values."""
    return _clean_text(name)


# ---------------------------------------------------------------------------
# Alias table — store-derived + hand-reviewed overrides
# ---------------------------------------------------------------------------

# Consolidation map: every NGANH record's `ten_nganh` (after _clean_name) must
# fold into exactly ONE canonical pattern+code entry below. Keys are the
# CLEANED forms of the store names we have actually observed; values are the
# (pattern, ma_nganh) pair they fold into. This is the auditable decision
# log — see README.
_CONSOLIDATION: Dict[str, Tuple[str, str]] = {
    # 7220104
    "han nom": ("han nom", "7220104"),
    # 7229001
    "triet hoc": ("triet hoc", "7229001"),
    # 7229010
    "lich su": ("lich su", "7229010"),
    # 7229030
    "van hoc": ("van hoc", "7229030"),
    # 7229042
    "quan ly van hoa": ("quan ly van hoa", "7229042"),
    # 7310205
    "quan ly nha nuoc": ("quan ly nha nuoc|quan ly cong", "7310205"),
    # 7310301
    "xa hoi hoc": ("xa hoi hoc", "7310301"),
    # 7310608
    "dong phuong hoc": ("dong phuong hoc", "7310608"),
    # 7320101
    "bao chi": ("bao chi", "7320101"),
    # 7320111
    "truyen thong so": ("truyen thong so", "7320111"),
    # 7420101
    "sinh hoc": ("sinh hoc", "7420101"),
    # 7420201
    "cong nghe sinh hoc": ("cong nghe sinh hoc", "7420201"),
    # 7440102SC is an override (see _ALIAS_OVERRIDES): both SC and base clean to 'vat ly hoc', so it must be anchored on question text, not store names
    "vat ly hoc": ("vat ly", "7440102"),
    # 7440112
    "hoa hoc": ("hoa hoc", "7440112"),
    # 7440301
    "khoa hoc moi truong": ("khoa hoc moi truong", "7440301"),
    # 7460108
    "khoa hoc du lieu": ("khoa hoc du lieu", "7460108"),
    # 7480103
    "ky thuat phan mem": ("ky thuat phan mem|ktpm|phan mem", "7480103"),
    # 7480107TD
    "quan tri va phan tich du lieu": ("quan tri va phan tich du lieu", "7480107TD"),
    # 7480201 / 7480201VJ / 7510302IC — store names clean to the bare major name
    # (program qualifiers are in parentheses, dropped by the cleaner), so they
    # cannot be anchored on store names: the distinguishing pattern lives in the
    # question text.
    # 7480201: 7480201VJ ("ky su viet nhat") is anchored on text below; the
    # base code is reached by plain "cong nghe thong tin" / "cntt".
    "cong nghe thong tin": ("cong nghe thong tin", "7480201"),
    # 7510302IC ("thiet ke vi mach") anchored on text; base "ky thuat dien tu"
    # family reaches 7510302.
    "cong nghe ky thuat dien tu vien thong": (
        "cong nghe ky thuat dien tu|dien tu - vien thong|ky thuat dien tu",
        "7510302",
    ),
    # 7510401
    "cong nghe ky thuat hoa hoc": ("cong nghe ky thuat hoa hoc|cnkh", "7510401"),
    # 7520503
    "ky thuat trac dia ban do": ("trac dia|ban do", "7520503"),
    # 7580101
    "kien truc": ("kien truc", "7580101"),
    # 7580211
    "dia ky thuat xay dung": ("dia ky thuat|ky thuat xay dung", "7580211"),
    # 7760101
    "cong tac xa hoi": ("cong tac xa hoi", "7760101"),
    # 7850101
    "quan ly tai nguyen va moi truong": ("quan ly tai nguyen va moi truong", "7850101"),
    # 7850105
    "quan ly an toan suc khoe va moi truong": ("quan ly an toan suc khoe va moi truong", "7850105"),
}

# Overrides: abbreviations / truncated forms that NEVER appear as a full
# `ten_nganh` in the store, added so the mapping lives in one reviewable
# place. (pattern, ma_nganh, why)
_ALIAS_OVERRIDES: List[Tuple[str, str, str]] = [
    ("ktpm", "7480103", "KTPM = abbreviation of ky thuat phan mem, no store name ever uses it"),
    ("cntt", "7480201", "CNTT = abbreviation of cong nghe thong tin, no store name ever uses it"),
    ("cnkh", "7510401", "CNKH = abbreviation of cong nghe ky thuat hoa hoc, no store name ever uses it"),
    ("ban dan", "7440102SC", "both 7440102 and 7440102sc clean to 'vat ly hoc'; the SC programme is distinguished by its 'ban dan' name in the question text ('vi mach' is excluded: it also names the 7510302IC programme, so it alone is ambiguous and yields no expected set)"),
    ("ky su viet nhat", "7480201VJ", "the VJ programme qualifies the base CNTT major only in parentheses (dropped by the cleaner); reached from the question text"),
    ("thiet ke vi mach", "7510302IC", "the IC programme qualifies the base DTVT major only in parentheses (dropped by the cleaner); anchored on the full phrase so bare 'vi mach' stays ambiguous and yields no expected set"),
]


@dataclass
class _CompiledAlias:
    pattern: re.Pattern
    ma_nganh: str
    note: str
    key_len: int


def build_alias_table(
    nganh_records: List[dict],
    overrides: Optional[List[Tuple[str, str, str]]] = None,
) -> List[_CompiledAlias]:
    """Alias table = _CONSOLIDATION (store-derived, keyed by cleaned ten_nganh)
    PLUS _ALIAS_OVERRIDES (hand-curated abbreviations). Every consolidation key
    MUST correspond to at least one actual store record — i.e. the table is
    anchored to the store, not invented. The result is compiled to longest-
    first so specific patterns (e.g. 'cong nghe sinh hoc') win over generic
    ones ('sinh hoc')."""
    overrides = _ALIAS_OVERRIDES if overrides is None else overrides

    store_clean_names: Set[str] = set()
    for r in nganh_records:
        ten = r.get("values", {}).get("ten_nganh")
        if ten:
            store_clean_names.add(_clean_name(ten))

    entries: List[Tuple[str, str, str]] = []
    for clean_key, (pat, code) in _CONSOLIDATION.items():
        # Anchor to the store: at least one store record must clean to exactly
        # this key, otherwise it's a stale/fabricated entry.
        if clean_key not in store_clean_names:
            # Allow known multi-line OCR variants to still be recognised:
            # fall back to checking whether ANY store record's cleaned form
            # STARTS with the key (OCR split a single name into 2 records).
            prefix_hit = any(n.startswith(clean_key) for n in store_clean_names)
            if not prefix_hit:
                raise ValueError(
                    f"alias key {clean_key!r} matches no NGANH ten_nganh in the store"
                )
            entries.append((pat, code, f"store-derived (OCR variant of {clean_key!r})"))
        else:
            entries.append((pat, code, f"store-derived from ten_nganh {clean_key!r}"))

    for pat, code, note in overrides:
        entries.append((pat, code, note))

    # De-duplicate by pattern (first wins) then sort longest-first.
    seen: Set[str] = set()
    deduped: List[Tuple[str, str, str]] = []
    for pat, code, note in entries:
        key = (pat, code)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((pat, code, note))

    out = [
        _CompiledAlias(
            pattern=re.compile(
                "|".join(f"(?<![a-z0-9]){part}(?![a-z0-9])" for part in p.split("|"))
            ),
            ma_nganh=code,
            note=note,
            key_len=max(len(part) for part in p.split("|")),
        )
        for p, code, note in deduped
    ]
    out.sort(key=lambda a: -a.key_len)
    return out


# ---------------------------------------------------------------------------
# Fact-type keyword patterns — ALL written against _clean_text output
# (accent-stripped, casefolded), consistent with the alias patterns.
# ---------------------------------------------------------------------------

_FACT_PATTERNS: Dict[str, re.Pattern] = {
    "TO_HOP": re.compile(
        r"(?<![a-z0-9])(to hop|khoi [a-dt]\d{2}|tổ hợp)(?![a-z0-9])"
    ),
    "NGANH": re.compile(
        r"(?<![a-z0-9])(ma nganh|danh sach nganh)(?![a-z0-9])"
    ),
    "HOC_PHI": re.compile(
        r"(?<![a-z0-9])(hoc phi|phi tin chi|fees?)(?![a-z0-9])"
    ),
    "DIEM_CHUAN": re.compile(
        r"(?<![a-z0-9])(diem chuan|diem trung tuy)(?![a-z0-9])"
    ),
    "CHI_TIEU": re.compile(
        r"(?<![a-z0-9])(chi tieu)(?![a-z0-9])"
    ),
    "THOI_GIAN": re.compile(
        r"(?<![a-z0-9])(thoi gian dao tao|nam dao tao|dao tao .*? nam)(?![a-z0-9])"
    ),
}

# Years present in the store (admission_year values observed). Explicit years
# in the question that are NOT in this set yield NO records (no fallback).
_KNOWN_YEARS: frozenset = frozenset({2024, 2025, 2026})

_MAJOR_CODE_RE = re.compile(r"(?<![0-9])(\d{7}(?:[a-z]{2})?)(?![0-9a-z])")
_YEAR_RE = re.compile(r"(?<![0-9])(20[12]\d)(?![0-9])")


@dataclass
class _Triple:
    major_codes: List[str]
    fact_types: List[str]
    years: List[int]
    basis: Optional[str]
    expected_record_ids: List[str] = field(default_factory=list)
    reason: Optional[str] = None


def match_question(
    question: str,
    alias_table: List[_CompiledAlias],
) -> Dict[str, object]:
    t = _clean_text(question)
    raw_codes = _MAJOR_CODE_RE.findall(t)
    # normalise suffix codes to uppercase (store ids use upper-case suffixes)
    codes = sorted({c.upper() if not c.isupper() else c for c in raw_codes})
    basis: Optional[str] = "code" if codes else None
    if not codes:
        found: List[str] = []
        for a in alias_table:
            if a.pattern.search(t) and a.ma_nganh not in found:
                found.append(a.ma_nganh)
        if len(found) == 1:
            codes = found
            basis = "alias"
        elif len(found) > 1:
            codes = sorted(found)
            basis = "ambiguous-alias"
        else:
            basis = None
    fact_types = sorted(k for k, pat in _FACT_PATTERNS.items() if pat.search(t))
    years = sorted({int(y) for y in _YEAR_RE.findall(t)})
    return {
        "major_codes": codes,
        "fact_types": fact_types,
        "years": years,
        "basis": basis,
    }


def derive_expected_ids(
    question: str,
    alias_table: List[_CompiledAlias],
    store_by_cell: Dict[Tuple[str, str, int], List[str]],
) -> _Triple:
    triple = match_question(question, alias_table)
    codes = list(triple["major_codes"])  # type: ignore[arg-type]
    fact_types = list(triple["fact_types"])  # type: ignore[arg-type]
    years_explicit = list(triple["years"])  # type: ignore[arg-type]
    basis = triple["basis"]

    out = _Triple(
        major_codes=codes,
        fact_types=fact_types,
        years=years_explicit,
        basis=basis,
    )

    if len(codes) == 1 and basis in ("code", "alias"):
        pass
    elif basis == "ambiguous-alias":
        out.reason = "multiple majors named"
        return out
    elif len(codes) == 0:
        out.reason = "no major derivable from text"
        return out
    else:
        out.reason = "explicit code not recognisable"
        return out

    if not fact_types:
        out.reason = "no fact type in text"
        return out

    # Year resolution: use explicit years that actually exist in the store;
    # if the text names no year at all, fall back to DEFAULT_YEAR; if the
    # text names years but none of them exist in the store, NO fallback —
    # empty set (this is the 2023/2027 trap the spec calls out).
    years_with_records = [y for y in years_explicit if y in _KNOWN_YEARS]
    if years_explicit:
        years = years_with_records
        out.years = years_explicit  # report what the text actually said
    else:
        years = [DEFAULT_YEAR]
        out.years = [DEFAULT_YEAR]

    code = codes[0]
    ids: Set[str] = set()
    for ft in fact_types:
        for y in years:
            ids.update(store_by_cell.get((ft, code, y), ()))
    if not ids:
        out.reason = f"no store record for ({code}, {fact_types}, {sorted(years)})"
        return out

    out.expected_record_ids = sorted(ids)
    return out


# ---------------------------------------------------------------------------
# Store + sidecar loading
# ---------------------------------------------------------------------------


def load_store(store_root: Path) -> List[dict]:
    out: List[dict] = []
    for d in sorted(os.listdir(store_root)):
        p = store_root / d / "v0001.json"
        if not p.is_file():
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def load_sidecar_records(path: Path) -> Dict[str, List[dict]]:
    """T00 sidecar JSONL: question id -> list of retrieved chunk dicts."""
    out: Dict[str, List[dict]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = rec.get("id")
        if not qid:
            continue
        out.setdefault(qid, []).extend(rec.get("retrieved_chunks") or [])
    return out


def _chunk_id_to_record_id(chunks: List[dict], record_ids_by_chunk: Dict[str, List[str]]) -> List[str]:
    """Join observed chunk ids -> store record ids via source_chunk_id, so the
    'observed' field is in the same id space as the expected set (record ids),
    not raw chunk slugs. Chunks that are NOT store chunks (corpus-only) are
    still listed, labelled, since the field is explicitly 'observed' data,
    not expected data."""
    ids: List[str] = []
    seen: Set[str] = set()
    for ch in chunks:
        cid = ch.get("chunk_id")
        if not cid:
            continue
        for rid in record_ids_by_chunk.get(cid, ()):
            if rid not in seen:
                seen.add(rid)
                ids.append(rid)
        # Also keep the raw chunk id itself (labelled) so nothing is silently
        # dropped when a chunk has no store record.
        raw = f"chunk:{cid}"
        if raw not in seen:
            seen.add(raw)
            ids.append(raw)
    return sorted(ids)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    data_root = _HERE / "data"
    dataset_path = data_root / "eval" / "husc_86q_test.json"
    store_root = data_root / "canonical_store" / "records"
    out_path = data_root / "eval" / "husc86q_golden_records.json"
    sidecar_path = _HERE / "results" / "baseline_T00" / "husc86q_records_t00.jsonl"

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    sidecar_chunks = load_sidecar_records(sidecar_path)

    store = load_store(store_root)
    nganh_records = [r for r in store if r.get("record_type") == "NGANH"]
    alias_table = build_alias_table(nganh_records)

    store_by_cell: Dict[Tuple[str, str, int], List[str]] = {}
    record_ids_by_chunk: Dict[str, List[str]] = {}
    all_ids: Set[str] = set()
    for r in store:
        all_ids.add(r["record_id"])
        cell = (
            r["record_type"],
            str(r.get("values", {}).get("ma_nganh", "")),
            r["admission_year"],
        )
        store_by_cell.setdefault(cell, []).append(r["record_id"])
        if r.get("source_chunk_id"):
            record_ids_by_chunk.setdefault(r["source_chunk_id"], []).append(r["record_id"])
    for cell in store_by_cell:
        store_by_cell[cell].sort()
    for c in record_ids_by_chunk:
        record_ids_by_chunk[c].sort()

    schema_versions = sorted({r.get("schema_version") for r in store})

    questions_out: List[Dict[str, object]] = []
    with_expected = 0
    no_exp_ambiguous = 0
    no_exp_unmatched_major = 0
    no_exp_no_fact_type = 0
    distinct_expected: Set[str] = set()

    for q in dataset:
        qid = q["id"]
        text = q["question"]
        t = derive_expected_ids(text, alias_table, store_by_cell)

        entry: Dict[str, object] = {"id": qid, "question": text}
        if t.expected_record_ids:
            for rid in t.expected_record_ids:
                if rid not in all_ids:
                    raise AssertionError(f"expected id {rid!r} not in store")
            with_expected += 1
            distinct_expected.update(t.expected_record_ids)
            entry["expected_record_ids"] = t.expected_record_ids
        else:
            if t.reason == "multiple majors named":
                no_exp_ambiguous += 1
            elif t.reason and "no store record" in t.reason:
                no_exp_unmatched_major += 1
            elif t.reason and "no major derivable" in t.reason:
                no_exp_unmatched_major += 1
            else:
                no_exp_no_fact_type += 1
            entry["expected_record_ids"] = []
            if t.reason:
                entry["reason"] = t.reason

        entry["derivation"] = {
            "major_codes": t.major_codes,
            "fact_types": t.fact_types,
            "years": t.years,
            "basis": t.basis,
        }

        obs_chunks = sidecar_chunks.get(qid, [])
        if obs_chunks:
            obs = _chunk_id_to_record_id(obs_chunks, record_ids_by_chunk)
            if obs:
                entry["baseline_observed_record_ids"] = obs

        questions_out.append(entry)

    coverage = {
        "questions": len(dataset),
        "with_expected": with_expected,
        "no_expected_ambiguous": no_exp_ambiguous,
        "no_expected_unmatched_major": no_exp_unmatched_major,
        "no_expected_no_fact_type": no_exp_no_fact_type,
        "distinct_expected_records": len(distinct_expected),
    }

    derivation_rule = (
        "expected_record_ids is derived EXCLUSIVELY from (a) an explicit "
        "7-digit major code in the question text, else (b) a single "
        "longest-first match against the alias table — itself DERIVED from the "
        "store's NGANH records' `ten_nganh` values (NFC-normalised, qualifier "
        "and OCR-space artifacts stripped, near-duplicate names consolidated "
        "via an explicit, logged decision table, plus a hand-curated "
        "abbreviation-override map for forms that never appear as a store "
        "name) — intersected with (c) fact-type keywords (HOC_PHI, "
        "DIEM_CHUAN, CHI_TIEU, TO_HOP, NGANH, THOI_GIAN) and (d) the "
        "admission year: an explicit 4-digit year in the text is used as-is "
        "(years with no store records, e.g. 2023/2027, yield an empty set, "
        "NO fallback); only when the text names no year at all does the "
        "store default (2026) apply. A non-empty expected set is emitted "
        "ONLY when exactly one major code AND at least one fact type AND at "
        "least one matching store cell all hold. Baseline observed retrieval "
        "(T00) is recorded in a SEPARATE, clearly-labelled field "
        "(`baseline_observed_record_ids`, joined via source_chunk_id) and "
        "NEVER feeds the expected set."
    )

    artifact: Dict[str, object] = {
        "artifact_version": 1,
        "generated_from": {
            "dataset": "data/eval/husc_86q_test.json",
            "dataset_sha256": _sha256_file(dataset_path),
            "store_root": "data/canonical_store/records",
            "record_count": len(store),
            "schema_version": schema_versions[0] if len(schema_versions) == 1 else schema_versions,
        },
        "derivation_rule": derivation_rule,
        "coverage": coverage,
        "alias_table": [
            {"pattern": a.pattern.pattern, "ma_nganh": a.ma_nganh, "note": a.note}
            for a in alias_table
        ],
        "questions": questions_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"coverage: {coverage}")


if __name__ == "__main__":
    main()

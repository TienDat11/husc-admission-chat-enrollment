"""
Graph Entity Pruner — Remove non-HUSC entities from entity_index.json & knowledge_graph.graphml.

HUSC 2026 uses code prefixes: 748xxxx, 751xxxx, 742xxxx, 746xxxx, 731xxxx, 758xxxx, 744xxxx
Non-HUSC prefixes: 714xxxx, 721xxxx, 722xxxx, etc.

Also prunes NGANH entities that are not in HUSC's 26-program catalog.

Usage:
    python scripts/prune_graph_entities.py [--dry-run]
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

# HUSC 2026 26-program canonical list (normalized)
HUSC_PROGRAMS_NORMALIZED = {
    "cong_nghe_thong_tin", "ky_thuat_phan_mem", "khoa_hoc_may_tinh",
    "quan_tri_va_phan_tich_du_lieu", "ky_thuat_dien_tu_vien_thong",
    "cong_nghe_sinh_hoc", "ky_thuat_hoa_hoc", "khoa_hoc_vat_lieu",
    "kien_truc", "dia_ky_thuat_xay_dung", "ky_thuat_trac_dia_ban_do",
    "quan_ly_tai_nguyen_va_moi_truong", "khoa_hoc_moi_truong",
    "toan_hoc", "vat_ly_hoc", "hoa_hoc", "sinh_hoc",
    "dia_ly_hoc", "lich_su", "van_hoc", "han_nom", "triet_hoc",
    "dong_phuong_hoc", "bao_chi", "truyen_thong_so",
    "xa_hoi_hoc", "cong_tac_xa_hoi", "quan_ly_van_hoa",
}

# Also accept common abbreviations / variant spellings
HUSC_PROGRAMS_ALIASES = {
    "cntt": "cong_nghe_thong_tin",
    "ktpm": "ky_thuat_phan_mem",
    "khmt": "khoa_hoc_may_tinh",
    "cnkt_dien_tu_vien_thong": "ky_thuat_dien_tu_vien_thong",
    "cnkt_hoa_hoc": "ky_thuat_hoa_hoc",
    "han_nom": "han_nom",  # variant without dash
    "han_nhom": "han_nom",  # typo variant
}

# HUSC-valid code prefixes (from actual data)
HUSC_CODE_PREFIXES = {"748", "751", "742", "746", "731", "758", "744"}

# Known non-HUSC programs that commonly hallucinate
NON_HUSC_PROGRAMS_NORMALIZED = {
    "giao_duc_mam_non", "giao_duc_tieu_hoc", "giao_duc_trung_hoc",
    "y_khoa", "duoc_hoc", "rang_ham_mat",
    "kinh_te", "quan_tri_kinh_doanh", "luat",
    "du_lich", "quan_ly_khach_san", "ke_toan",
    "ngon_ngu_anh", "ngon_ngu_phap", "ngon_ngu_trung_quoc",
    "su_pham", "gdct", "gdsh",
    "quoc_te_hoc", "chinh_tri_hoc", "quan_tri_cong",
    "ky_thuat_co_khien", "ky_thuat_dien",
}


def is_husc_nganh(entity_key: str, entity_data: Dict) -> bool:
    """Check if a NGANH entity belongs to HUSC's 26-program catalog."""
    if not entity_key.startswith("NGANH:"):
        return True  # Not a NGANH entity, keep it

    normalized = entity_data.get("normalized", "")

    # Check against HUSC catalog
    if normalized in HUSC_PROGRAMS_NORMALIZED:
        return True

    # Check aliases
    if normalized in HUSC_PROGRAMS_ALIASES:
        return True

    # Check known non-HUSC
    if normalized in NON_HUSC_PROGRAMS_NORMALIZED:
        return False

    # Check code prefix directly from entity key (e.g. NGANH:7140201)
    key_suffix = entity_key.split(":", 1)[1] if ":" in entity_key else ""
    m = re.match(r'^(\d{3})\d{4}', key_suffix)
    if m:
        prefix = m.group(1)
        if prefix in HUSC_CODE_PREFIXES:
            return True
        return False  # Code prefix in key is not HUSC

    # Check code prefix from chunk_ids
    chunk_ids = entity_data.get("chunk_ids", [])
    for cid in chunk_ids:
        # Match patterns like "husc_nganh_7480107_2025" or "748xxxx"
        m = re.search(r'(\d{3})\d{4}', cid)
        if m:
            prefix = m.group(1)
            if prefix in HUSC_CODE_PREFIXES:
                return True
            return False  # Code prefix not in HUSC set

    # If no chunk_ids with codes, check text for known non-HUSC keywords
    text = entity_data.get("text", "").lower()
    for prog in ["giáo dục mầm non", "giáo dục tiểu học", "sư phạm",
                 "du lịch", "khách sạn", "quản trị kinh doanh",
                 "y khoa", "dược", "kế toán", "luật",
                 "giáo dục thể chất", "tâm lý học giáo dục",
                 "kinh tế chính trị", "kinh tế quốc tế", "thống kê kinh tế"]:
        if prog in text:
            return False

    # Check normalized name for extended non-HUSC programs
    extended_non_husc = {
        "giao_duc_the_chat", "tam_ly_hoc_giao_duc",
        "kinh_te_chinh_tri", "kinh_te_quoc_te", "thong_ke_kinh_te",
        "quan_ly_an_toan_suc_khoe", "suc_khoe_cong_cong",
        "dieu_duong", "y_hoc_co_truyen", "y_hoc_duoc",
        "khoa_hoc_giao_duc", "quan_ly_giao_duc",
        "ngon_ngu_hoc", "tieng_anh", "tieng_phap", "tieng_trung",
        "tin_hoc", "he_thong_thong_tin",
        "ky_thuat_moi_truong", "ky_thuat_co_khi",
    }
    if normalized in extended_non_husc:
        return False

    # Default: keep (safer to keep than to remove incorrectly)
    return True


def prune_entity_index(index_path: str, dry_run: bool = False) -> Dict:
    """Prune non-HUSC NGANH entities from entity_index.json."""
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    removed_keys: List[str] = []
    removed_entity_types: Dict[str, int] = {}

    # Identify entities to remove
    keys_to_remove = []
    for key, data in index.items():
        if not is_husc_nganh(key, data):
            keys_to_remove.append(key)

    # Also check UNKNOWN entities that are actually non-HUSC codes
    for key, data in index.items():
        if key.startswith("UNKNOWN:"):
            # Check if it's a non-HUSC code
            text = data.get("text", "")
            m = re.match(r'^(\d{3})\d{4}', text)
            if m and m.group(1) not in HUSC_CODE_PREFIXES:
                if key not in keys_to_remove:
                    keys_to_remove.append(key)

    for key in keys_to_remove:
        etype = key.split(":")[0] if ":" in key else "UNKNOWN"
        removed_entity_types[etype] = removed_entity_types.get(etype, 0) + 1
        removed_keys.append(key)
        if not dry_run:
            del index[key]

    # Clean up references: remove pruned entity keys from triples/chunk_ids
    if not dry_run:
        remaining_refs_cleaned = 0
        for key, data in index.items():
            chunk_ids = data.get("chunk_ids", [])
            if isinstance(chunk_ids, list):
                original_len = len(chunk_ids)
                data["chunk_ids"] = [c for c in chunk_ids if c not in removed_keys]
                remaining_refs_cleaned += original_len - len(data["chunk_ids"])

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    return {
        "total_entities_before": len(index) + len(removed_keys),
        "removed_count": len(removed_keys),
        "removed_keys": removed_keys[:20],  # First 20 for display
        "removed_by_type": removed_entity_types,
        "dry_run": dry_run,
    }


def prune_graphml(graphml_path: str, removed_entity_keys: Set[str], dry_run: bool = False) -> Dict:
    """Remove nodes and edges referencing pruned entities from knowledge_graph.graphml."""
    with open(graphml_path, encoding="utf-8") as f:
        content = f.read()

    # Simple text-based pruning for GraphML
    # Remove <node> elements whose id is in removed_entity_keys
    # Remove <edge> elements whose source or target is in removed_entity_keys

    original_size = len(content)

    # Build pattern for removed keys (escape special chars)
    escaped_keys = [re.escape(k) for k in removed_entity_keys]
    if not escaped_keys:
        return {"nodes_removed": 0, "edges_removed": 0, "size_change": 0}

    key_pattern = "|".join(escaped_keys)

    # Remove nodes
    node_pattern = re.compile(
        rf'<node[^>]*id="(?:{key_pattern})"[^>]*>.*?</node>',
        re.DOTALL,
    )
    nodes_removed = len(node_pattern.findall(content))
    content = node_pattern.sub("", content)

    # Also catch self-closing nodes
    node_sc_pattern = re.compile(
        rf'<node[^>]*id="(?:{key_pattern})"[^>]*/>',
    )
    nodes_removed += len(node_sc_pattern.findall(content))
    content = node_sc_pattern.sub("", content)

    # Remove edges referencing removed nodes
    edge_pattern = re.compile(
        rf'<edge[^>]*(?:source|target)="(?:{key_pattern})"[^>]*>.*?</edge>',
        re.DOTALL,
    )
    edges_removed = len(edge_pattern.findall(content))
    content = edge_pattern.sub("", content)

    # Also catch self-closing edges
    edge_sc_pattern = re.compile(
        rf'<edge[^>]*(?:source|target)="(?:{key_pattern})"[^>]*/>',
    )
    edges_removed += len(edge_sc_pattern.findall(content))
    content = edge_sc_pattern.sub("", content)

    new_size = len(content)

    if not dry_run:
        with open(graphml_path, "w", encoding="utf-8") as f:
            f.write(content)

    return {
        "nodes_removed": nodes_removed,
        "edges_removed": edges_removed,
        "size_change": original_size - new_size,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prune non-HUSC entities from graph")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    args = parser.parse_args()

    base = Path(__file__).parent.parent / "data" / "graph"
    index_path = str(base / "entity_index.json")
    graphml_path = str(base / "knowledge_graph.graphml")

    print(f"=== Graph Entity Pruner (dry_run={args.dry_run}) ===")
    print(f"Entity index: {index_path}")
    print(f"GraphML: {graphml_path}")
    print()

    # Step 1: Prune entity_index.json
    result = prune_entity_index(index_path, dry_run=args.dry_run)
    print(f"Entity index: {result['removed_count']} entities removed "
          f"(of {result['total_entities_before']} total)")
    print(f"  By type: {result['removed_by_type']}")
    if result['removed_keys']:
        print(f"  Sample removed: {result['removed_keys'][:10]}")
    print()

    # Step 2: Prune knowledge_graph.graphml
    if Path(graphml_path).exists():
        removed_keys = set(result.get("removed_keys", []))
        # Read full removed list from the index if not dry_run
        if not args.dry_run:
            with open(index_path, encoding="utf-8") as f:
                pass  # already pruned, keys are gone

        # For graphml, we need the actual removed keys
        # Re-read to get ALL removed keys (not just first 20)
        if args.dry_run:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            all_removed = set()
            for key, data in index.items():
                if not is_husc_nganh(key, data):
                    all_removed.add(key)
            for key, data in index.items():
                if key.startswith("UNKNOWN:"):
                    text = data.get("text", "")
                    m = re.match(r'^(\d{3})\d{4}', text)
                    if m and m.group(1) not in HUSC_CODE_PREFIXES:
                        all_removed.add(key)
            removed_keys = all_removed

        graphml_result = prune_graphml(graphml_path, removed_keys, dry_run=args.dry_run)
        print(f"GraphML: {graphml_result['nodes_removed']} nodes, "
              f"{graphml_result['edges_removed']} edges removed "
              f"({graphml_result['size_change']} bytes)")
    else:
        print("GraphML file not found, skipping")

    print()
    if args.dry_run:
        print("DRY RUN — no files were modified. Remove --dry-run to apply.")
    else:
        print("Done. Graph pruned successfully.")


if __name__ == "__main__":
    main()

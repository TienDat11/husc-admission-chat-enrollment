"""
Generate Admissions Office Report from 86Q evaluation results.

Creates 2 outputs:
1. Technical bottleneck verification table (86q_bang_xac_minh.md)
2. Non-technical admissions office report (86q_bao_cao_tuyen_sinh.md)

Usage:
    cd rag2025
    python scripts/gen_admissions_report.py
    python scripts/gen_admissions_report.py --input results/86q_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RESULTS_PATH = Path(__file__).parent.parent / "results" / "86q_results.json"
OUTPUT_DIR = Path(__file__).parent.parent / "results"


def load_results(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: Results file not found: {path}")
        print("Run eval_86q.py first to generate results.")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def diagnose_bottleneck(r: dict) -> str:
    """Auto-diagnose bottleneck type from result fields."""
    if r["passed"]:
        return "OK"
    if not r["has_answer"]:
        return "NO_ANSWER"
    route = r.get("route", "")
    num_chunks = r.get("num_chunks", 0)
    ner_seeds = r.get("ner_seed_count", r.get("ppr_seed_count", 0))
    ppr_hits = r.get("ppr_hits", 0)
    has_answer = r.get("has_answer", False)

    if num_chunks < 3:
        return "RETRIEVAL"
    if route == "graph_rag" and ner_seeds == 0:
        return "NER_FAIL"
    if route == "graph_rag" and ppr_hits == 0:
        return "GRAPH_DISCONNECT"
    if has_answer and not r["passed"]:
        return "GENERATION"
    return "NO_ANSWER"


BOTTLENECK_EMOJI = {
    "OK": "✅",
    "GENERATION": "⚠️",
    "RETRIEVAL": "🔍",
    "NER_FAIL": "🧠",
    "GRAPH_DISCONNECT": "📊",
    "NO_ANSWER": "❌",
}

BOTTLENECK_LABEL = {
    "OK": "✅ OK",
    "GENERATION": "⚠️ GEN",
    "RETRIEVAL": "🔍 RET",
    "NER_FAIL": "🧠 NER",
    "GRAPH_DISCONNECT": "📊 GRP",
    "NO_ANSWER": "❌ NO",
}


def gen_verification_table(data: dict) -> str:
    """Technical bottleneck verification table for teachers/engineers."""
    results = data["results"]
    metadata = data.get("metadata", {})
    summary = data.get("summary", {})

    lines = []
    lines.append("# BẢNG XÁC MINH KỸ THUẬT — PHÂN TÍCH BOTTLENECK HỆ THỐNG RAG")
    lines.append("")
    lines.append(f"Ngày kiểm tra: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    lines.append(f"Model NER: `{metadata.get('ner_model', 'N/A')}`")
    lines.append(f"Model HYDE: `{metadata.get('hyde_model', 'N/A')}`")
    lines.append(f"Model Generation: `{metadata.get('gen_model', 'N/A')}`")
    lines.append(f"Embedding: `{metadata.get('embedding_model', 'N/A')}` (dim={metadata.get('embedding_dim', 'N/A')})")
    lines.append(f"GraphRAG threshold: `{metadata.get('graphrag_threshold', 'N/A')}`")
    lines.append("")

    # Annotate each result with bottleneck diagnosis
    for r in results:
        r["_bottleneck"] = diagnose_bottleneck(r)

    # ── Summary stats ──────────────────────────────────────────────────────────
    total = metadata.get("total_questions", len(results))
    route_dist = metadata.get("route_distribution", {})
    total_pass = sum(1 for r in results if r["passed"])
    total_has_ans = sum(1 for r in results if r["has_answer"])

    lines.append("## I. Tóm tắt thống kê tổng quan")
    lines.append("")
    lines.append("| Chỉ số | Giá trị |")
    lines.append("|--------|---------|")
    lines.append(f"| Tổng câu hỏi | {total} |")
    lines.append(f"| Câu đạt (passed) | {total_pass} ({total_pass/total*100:.1f}%) |")
    lines.append(f"| Câu có câu trả lời | {total_has_ans} ({total_has_ans/total*100:.1f}%) |")
    lines.append(f"| Route PaddedRAG | {route_dist.get('padded_rag', 0)} |")
    lines.append(f"| Route GraphRAG | {route_dist.get('graph_rag', 0)} |")
    lines.append(f"| Route Error | {route_dist.get('error', 0)} |")
    lines.append("")

    # ── Bottleneck count breakdown ─────────────────────────────────────────────
    bottleneck_counts: dict[str, int] = defaultdict(int)
    for r in results:
        bottleneck_counts[r["_bottleneck"]] += 1

    lines.append("## II. Phân tích Bottleneck theo loại")
    lines.append("")
    lines.append("| Bottleneck | Ký hiệu | Số câu | Tỷ lệ | Ý nghĩa |")
    lines.append("|-----------|---------|--------|-------|---------|")
    bottleneck_desc = {
        "OK":               "Trả lời đúng, không có vấn đề",
        "GENERATION":       "Có thông tin trong chunks nhưng LLM bỏ sót",
        "RETRIEVAL":        "Retrieval trả về < 3 chunks — thiếu ngữ cảnh",
        "NER_FAIL":         "GraphRAG: NER không trích xuất được entity nào",
        "GRAPH_DISCONNECT": "GraphRAG: PPR không lan truyền được (đồ thị rời rạc)",
        "NO_ANSWER":        "Hệ thống không tạo được câu trả lời",
    }
    for bt in ["OK", "GENERATION", "RETRIEVAL", "NER_FAIL", "GRAPH_DISCONNECT", "NO_ANSWER"]:
        cnt = bottleneck_counts.get(bt, 0)
        pct = cnt / total * 100 if total else 0
        lines.append(f"| {BOTTLENECK_LABEL[bt]} | {BOTTLENECK_EMOJI[bt]} | {cnt} | {pct:.1f}% | {bottleneck_desc[bt]} |")
    lines.append("")

    # ── Worst weakness_type categories ────────────────────────────────────────
    wt_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0, "bottlenecks": defaultdict(int)})
    for r in results:
        wt = r.get("weakness_type") or r.get("category", "unknown")
        wt_stats[wt]["total"] += 1
        if r["passed"]:
            wt_stats[wt]["pass"] += 1
        wt_stats[wt]["bottlenecks"][r["_bottleneck"]] += 1

    lines.append("## III. Hiệu suất theo weakness_type")
    lines.append("")
    lines.append("| Weakness Type | Số câu | Pass | Tỷ lệ | Bottleneck chính |")
    lines.append("|--------------|--------|------|-------|-----------------|")
    for wt, s in sorted(wt_stats.items(), key=lambda x: x[1]["pass"] / x[1]["total"] if x[1]["total"] else 1):
        acc = s["pass"] / s["total"] * 100 if s["total"] else 0
        top_bt = max(s["bottlenecks"], key=s["bottlenecks"].get) if s["bottlenecks"] else "N/A"
        lines.append(f"| {wt} | {s['total']} | {s['pass']} | {acc:.1f}% | {BOTTLENECK_LABEL.get(top_bt, top_bt)} |")
    lines.append("")

    # ── Route accuracy comparison ──────────────────────────────────────────────
    route_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0, "latency": []})
    for r in results:
        rt = r.get("route", "error")
        route_stats[rt]["total"] += 1
        if r["passed"]:
            route_stats[rt]["pass"] += 1
        route_stats[rt]["latency"].append(r.get("latency_s", 0))

    lines.append("## IV. So sánh Route")
    lines.append("")
    lines.append("| Route | Số câu | Pass | Tỷ lệ | Avg Latency |")
    lines.append("|-------|--------|------|-------|-------------|")
    for rt, s in route_stats.items():
        acc = s["pass"] / s["total"] * 100 if s["total"] else 0
        avg_lat = sum(s["latency"]) / len(s["latency"]) if s["latency"] else 0
        lines.append(f"| {rt} | {s['total']} | {s['pass']} | {acc:.1f}% | {avg_lat:.2f}s |")
    lines.append("")

    # ── NER failure list ───────────────────────────────────────────────────────
    ner_fails = [r for r in results if r.get("ner_seed_count", r.get("ppr_seed_count", 0)) == 0 and r.get("route") == "graph_rag"]
    if ner_fails:
        lines.append("## V. Câu hỏi NER thất bại (seeds=0, route=graph_rag)")
        lines.append("")
        lines.append("| ID | Câu hỏi | Bottleneck |")
        lines.append("|----|---------|-----------|")
        for r in ner_fails:
            q_short = r["question"][:80].replace("|", "/")
            lines.append(f"| {r['id']} | {q_short} | {BOTTLENECK_LABEL[r['_bottleneck']]} |")
        lines.append("")

    # ── Per-question detail table ──────────────────────────────────────────────
    lines.append("## VI. Chi tiết từng câu hỏi (Bottleneck Table)")
    lines.append("")
    lines.append(
        "| # | ID | Weakness | Route | Cmplx | Intent | NER | PPR | Chunks | Conf | Lat(s) | Provider | Pass | Missing | Bottleneck |"
    )
    lines.append(
        "|---|-----|---------|-------|-------|--------|-----|-----|--------|------|--------|----------|------|---------|-----------|"
    )

    for i, r in enumerate(results, 1):
        bt = r["_bottleneck"]
        emoji = BOTTLENECK_EMOJI[bt]
        route_short = "Graph" if r.get("route") == "graph_rag" else ("Padded" if r.get("route") == "padded_rag" else "Err")
        ner_seeds = r.get("ner_seed_count", r.get("ppr_seed_count", 0))
        ppr_hits = r.get("ppr_hits", 0)
        conf = r.get("top_confidence", 0.0)
        wt = (r.get("weakness_type") or r.get("category", ""))[:20]
        intent_short = str(r.get("intent", ""))[:15]
        missing = ", ".join(r.get("must_include", [])) if not r["passed"] else ""
        missing = missing[:40].replace("|", "/")
        pass_icon = "✅" if r["passed"] else ("⚠️" if r["has_answer"] else "❌")
        ner_flag = f"{ner_seeds}" if ner_seeds > 0 else "**0⚠**"
        ppr_flag = f"{ppr_hits}" if ppr_hits > 0 else ("**0⚠**" if r.get("route") == "graph_rag" else "—")
        lines.append(
            f"| {i} | {r['id']} | {wt} | {route_short} | {r.get('complexity',0)} | {intent_short} "
            f"| {ner_flag} | {ppr_flag} | {r.get('num_chunks',0)} | {conf:.3f} "
            f"| {r.get('latency_s',0)} | {str(r.get('provider',''))[:12]} "
            f"| {pass_icon} | {missing} | {emoji} {bt} |"
        )

    lines.append("")

    # ── Legend ─────────────────────────────────────────────────────────────────
    lines.append("## VII. Chú giải")
    lines.append("")
    lines.append("| Ký hiệu | Ý nghĩa |")
    lines.append("|---------|---------|")
    lines.append("| ✅ OK | Câu trả lời đúng, đủ thông tin |")
    lines.append("| ⚠️ GEN | Có thông tin trong chunks nhưng LLM bỏ sót (generation bottleneck) |")
    lines.append("| 🔍 RET | Retrieval trả về < 3 chunks — thiếu ngữ cảnh để trả lời |")
    lines.append("| 🧠 NER | NER không trích xuất được entity nào (seeds=0) — GraphRAG không khởi động được |")
    lines.append("| 📊 GRP | PPR không lan truyền được — đồ thị tri thức rời rạc |")
    lines.append("| ❌ NO | Hệ thống không tạo được câu trả lời |")
    lines.append("| **0⚠** | Giá trị = 0, đây là dấu hiệu bottleneck |")
    lines.append("| Cmplx | Router complexity score (1–4) |")
    lines.append("| NER | Số NER seed entities được trích xuất |")
    lines.append("| PPR | Số chunks có PPR score > 0 |")
    lines.append("| Conf | Top chunk confidence score |")
    lines.append("")
    lines.append(f"*Báo cáo tự động tạo bởi hệ thống RAG2025 — {datetime.now().strftime('%d/%m/%Y %H:%M')}*")

    return "\n".join(lines)


def gen_admissions_report(data: dict) -> str:
    """Non-technical report for admissions office staff."""
    results = data["results"]
    metadata = data.get("metadata", {})

    total = metadata.get("total_questions", len(results))
    total_pass = sum(1 for r in results if r["passed"])
    total_has_ans = sum(1 for r in results if r["has_answer"])
    total_partial = total_has_ans - total_pass
    total_no_ans = total - total_has_ans
    overall_acc = total_pass / total * 100 if total else 0

    def score_icon(r: dict) -> str:
        if r["passed"]:
            return "🟢"
        if r["has_answer"]:
            return "🟡"
        return "🔴"

    def score_label(r: dict) -> str:
        if r["passed"]:
            return "Trả lời đầy đủ"
        if r["has_answer"]:
            return "Trả lời nhưng thiếu thông tin"
        return "Không trả lời được"

    def format_sources(r: dict) -> str:
        sources = r.get("chunk_sources", [])
        unique = list(dict.fromkeys(s for s in sources if s))
        if not unique:
            return "_Không rõ nguồn_"
        return ", ".join(f"`{s}`" for s in unique[:3])

    # Group by weakness_type (fall back to category)
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        key = r.get("weakness_type") or r.get("category", "Khác")
        groups[key].append(r)

    # Vietnamese group descriptions for common weakness types
    group_intro = {
        "simple": "Câu hỏi tra cứu đơn giản — tìm một thông tin cụ thể như điểm chuẩn, học phí, mã ngành.",
        "multihop": "Câu hỏi nhiều bước — cần kết hợp nhiều thông tin, so sánh hoặc liệt kê.",
        "temporal": "Câu hỏi theo thời gian — liên quan đến thay đổi theo năm hoặc hạn chót.",
        "score_lookup": "Tra cứu điểm chuẩn theo ngành và năm.",
        "tuition": "Câu hỏi về học phí, chi phí đào tạo.",
        "major_info": "Thông tin về ngành học, chương trình đào tạo.",
        "admission_method": "Phương thức xét tuyển, điều kiện đăng ký.",
        "deadline": "Hạn nộp hồ sơ, lịch thi, thời gian xét tuyển.",
        "contact": "Thông tin liên hệ, địa chỉ, số điện thoại.",
        "scholarship": "Học bổng, hỗ trợ tài chính.",
        "comparison": "So sánh giữa các ngành hoặc trường.",
        "enumeration": "Liệt kê danh sách ngành, môn học, điều kiện.",
    }

    lines = []
    lines.append("# BÁO CÁO ĐÁNH GIÁ HỆ THỐNG TƯ VẤN TUYỂN SINH")
    lines.append("## Đại học Khoa học Huế (HUSC)")
    lines.append("")
    lines.append(f"*Ngày đánh giá: {datetime.now().strftime('%d/%m/%Y')}*")
    lines.append("")
    lines.append("> Báo cáo này dành cho cán bộ phòng Tuyển sinh. Không chứa thuật ngữ kỹ thuật.")
    lines.append("> Mục đích: kiểm tra xem hệ thống AI trả lời đúng và đủ thông tin chưa.")
    lines.append("")

    # ── Tổng quan ──────────────────────────────────────────────────────────────
    lines.append("## I. Kết quả tổng quan")
    lines.append("")
    lines.append(f"Hệ thống được kiểm tra với **{total} câu hỏi** thực tế mà phụ huynh và học sinh thường hỏi.")
    lines.append("")
    lines.append("| Kết quả | Số câu | Tỷ lệ |")
    lines.append("|---------|--------|-------|")
    lines.append(f"| 🟢 Trả lời đầy đủ | {total_pass} | {total_pass/total*100:.1f}% |")
    lines.append(f"| 🟡 Trả lời nhưng thiếu thông tin | {total_partial} | {total_partial/total*100:.1f}% |")
    lines.append(f"| 🔴 Không trả lời được | {total_no_ans} | {total_no_ans/total*100:.1f}% |")
    lines.append(f"| **Tổng** | **{total}** | **100%** |")
    lines.append("")

    # ── Per-group sections ─────────────────────────────────────────────────────
    lines.append("## II. Chi tiết theo nhóm câu hỏi")
    lines.append("")

    for group_key, group_results in sorted(groups.items()):
        g_total = len(group_results)
        g_pass = sum(1 for r in group_results if r["passed"])
        g_has = sum(1 for r in group_results if r["has_answer"])
        g_acc = g_pass / g_total * 100 if g_total else 0

        lines.append(f"### Nhóm: {group_key}")
        lines.append("")
        intro = group_intro.get(group_key, f"Câu hỏi thuộc nhóm **{group_key}**.")
        lines.append(intro)
        lines.append("")
        lines.append(f"Kết quả nhóm: **{g_pass}/{g_total}** câu trả lời đầy đủ ({g_acc:.1f}%)")
        lines.append("")

        for r in group_results:
            icon = score_icon(r)
            label = score_label(r)
            sources = format_sources(r)

            lines.append(f"#### {icon} Câu {r['id']} — {label}")
            lines.append("")
            lines.append(f"**Câu hỏi:** {r['question']}")
            lines.append("")
            lines.append(f"**Hệ thống trả lời:** {r['answer'] if r['answer'] else '_Hệ thống không trả lời được._'}")
            lines.append("")
            lines.append(f"**Nguồn thông tin:** {sources}")
            lines.append("")

            if not r["passed"] and r.get("must_include"):
                missing_str = ", ".join(f'"{t}"' for t in r["must_include"])
                lines.append(f"**Thông tin còn thiếu:** {missing_str}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # ── Kết luận ───────────────────────────────────────────────────────────────
    lines.append("## III. Kết luận")
    lines.append("")

    # What the system answers well
    good_groups = [(k, v) for k, v in groups.items()
                   if sum(1 for r in v if r["passed"]) / len(v) >= 0.7]
    bad_groups = [(k, v) for k, v in groups.items()
                  if sum(1 for r in v if r["passed"]) / len(v) < 0.5]

    lines.append("### Hệ thống trả lời tốt")
    lines.append("")
    if good_groups:
        for k, v in good_groups:
            g_pass = sum(1 for r in v if r["passed"])
            lines.append(f"- **{k}**: {g_pass}/{len(v)} câu đúng ({g_pass/len(v)*100:.0f}%)")
    else:
        lines.append("- Chưa có nhóm nào đạt tỷ lệ trả lời đúng ≥ 70%.")
    lines.append("")

    lines.append("### Nhóm cần cải thiện")
    lines.append("")
    if bad_groups:
        for k, v in bad_groups:
            g_pass = sum(1 for r in v if r["passed"])
            lines.append(f"- **{k}**: chỉ {g_pass}/{len(v)} câu đúng ({g_pass/len(v)*100:.0f}%) — cần bổ sung tài liệu hoặc cải thiện hệ thống")
    else:
        lines.append("- Không có nhóm nào dưới 50% — hệ thống hoạt động ổn định.")
    lines.append("")

    lines.append("### Tóm tắt")
    lines.append("")
    lines.append(
        f"Hệ thống trả lời đầy đủ **{total_pass}/{total} câu hỏi** ({overall_acc:.1f}%). "
        f"Có thêm **{total_partial} câu** có câu trả lời nhưng còn thiếu thông tin. "
        f"**{total_no_ans} câu** hệ thống chưa trả lời được."
    )
    lines.append("")

    if total_no_ans > 0:
        no_ans_list = [r for r in results if not r["has_answer"]]
        lines.append("**Các câu hỏi hệ thống chưa trả lời được:**")
        lines.append("")
        for r in no_ans_list[:10]:
            lines.append(f"- {r['id']}: {r['question']}")
        if len(no_ans_list) > 10:
            lines.append(f"- _(và {len(no_ans_list)-10} câu khác)_")
        lines.append("")

    lines.append("**Kiến nghị:**")
    lines.append("")
    lines.append("- Bổ sung tài liệu cho các nhóm câu hỏi có tỷ lệ trả lời thấp.")
    lines.append("- Kiểm tra lại các câu trả lời 🟡 (có trả lời nhưng thiếu thông tin) để xác nhận thông tin nào còn thiếu.")
    lines.append("- Các câu 🔴 cần được ưu tiên xử lý trước khi triển khai hệ thống cho người dùng thực.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Báo cáo tự động tạo bởi hệ thống RAG2025 — {datetime.now().strftime('%d/%m/%Y %H:%M')}*")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(RESULTS_PATH))
    args = parser.parse_args()

    input_path = Path(args.input)
    data = load_results(input_path)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Report 1: Technical bottleneck table
    table = gen_verification_table(data)
    table_path = OUTPUT_DIR / "86q_bang_xac_minh.md"
    table_path.write_text(table, encoding="utf-8")
    print(f"Verification table saved to: {table_path}")

    # Report 2: Admissions office report
    report = gen_admissions_report(data)
    report_path = OUTPUT_DIR / "86q_bao_cao_tuyen_sinh.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Admissions report saved to: {report_path}")

    # Quick summary
    results = data["results"]
    total = len(results)
    total_pass = sum(1 for r in results if r["passed"])
    graph_count = sum(1 for r in results if r.get("route") == "graph_rag")
    print(f"\nSummary: {total_pass}/{total} passed ({total_pass/total*100:.1f}%) | GraphRAG: {graph_count} questions")


if __name__ == "__main__":
    main()

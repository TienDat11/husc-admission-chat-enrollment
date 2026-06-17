"""Generate 2 reports from 86Q eval results:
1. Technical report (báo cáo chuyên môn) — detailed metrics, route distribution, NER seeds, PPR hits
2. Admission staff report (báo cáo cho thầy tuyển sinh) — clean Q&A with citations
"""
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results" / "86q_results.json"
TECH_REPORT = ROOT / "results" / "86q_bao_cao_chuyen_mon.md"
STAFF_REPORT = ROOT / "results" / "86q_bao_cao_thay_tuyen_sinh.md"


def load_results():
    if not RESULTS.exists():
        print(f"[ERROR] Results file not found: {RESULTS}")
        return None
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def calc_lang_pass(answer: str) -> bool:
    """Detect if answer is purely Vietnamese (no English sentences)."""
    import re
    allowed = {"THPT","HUSC","IELTS","TOEFL","VSTEP","VNĐ","KTX","CNTT","KTPM","KHMT",
               "NV","CCCD","DHT","A00","A01","A02","B00","C00","C01","C02","C03",
               "D01","D07","D14","D78","V00","V01","V02","V12","X01","X06","X25",
               "X26","X70","X78","SDH","KH","XTT","HB","NL","PPR"}
    pattern = re.compile(r'(?:^|[\s(])([A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){2,})(?:[\s).,]|$)')
    for m in pattern.finditer(answer or ""):
        phrase = m.group(1)
        words = phrase.split()
        if not all(w in allowed for w in words):
            return False
    return True


def gen_technical_report(data):
    """Generate detailed technical report with metrics, distributions, error analysis."""
    results = data["results"]
    meta = data["metadata"]
    n = len(results)

    # Collect metrics
    pass_count = sum(1 for r in results if r.get("passed"))
    has_ans_count = sum(1 for r in results if r.get("has_answer"))
    lang_ok = sum(1 for r in results if calc_lang_pass(r.get("answer", "")))
    lang_fail = n - lang_ok

    # Route distribution
    routes = {"padded_rag": 0, "graph_rag": 0}
    for r in results:
        routes[r.get("route", "padded_rag")] = routes.get(r.get("route", "padded_rag"), 0) + 1

    # NER seeds & PPR hits
    ner_seeds = [r.get("ner_seed_count", 0) for r in results]
    ppr_hits = [r.get("ppr_hits", r.get("ppr_hit_count", 0)) for r in results]
    latencies = [r.get("latency_s", r.get("latency_ms", 0)/1000) for r in results]

    avg_ner = sum(ner_seeds) / len(ner_seeds) if ner_seeds else 0
    avg_ppr = sum(ppr_hits) / len(ppr_hits) if ppr_hits else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    # Per category
    by_cat = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, {"total": 0, "passed": 0, "has_ans": 0})
        by_cat[cat]["total"] += 1
        if r.get("passed"):
            by_cat[cat]["passed"] += 1
        if r.get("has_answer"):
            by_cat[cat]["has_ans"] += 1

    # GraphRAG-specific PPR analysis
    graph_results = [r for r in results if r.get("route") == "graph_rag"]
    graph_ppr_zero = sum(1 for r in graph_results if r.get("ppr_hits", r.get("ppr_hit_count", 0)) == 0)

    # Failure analysis
    failures = [r for r in results if not r.get("passed")]
    fail_no_chunks = sum(1 for r in failures if r.get("num_chunks", r.get("chunks_used", 0)) == 0)
    fail_with_chunks = len(failures) - fail_no_chunks

    md = []
    md.append("# Báo cáo chuyên môn — Đánh giá hệ thống RAG HUSC 2026")
    md.append("")
    md.append(f"**Ngày tạo**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    md.append(f"**Mô hình NER**: {meta.get('ner_model','?')}  ")
    md.append(f"**Mô hình HyDE**: {meta.get('hyde_model','?')}  ")
    md.append(f"**Mô hình sinh đáp án**: {meta.get('gen_model','?')}  ")
    md.append(f"**Embedding**: {meta.get('embedding_model','?')} (dim={meta.get('embedding_dim','?')})  ")
    md.append(f"**Tổng số câu hỏi đánh giá**: {n}")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 1. Tổng quan kết quả")
    md.append("")
    md.append("| Chỉ số | Giá trị | Tỷ lệ | Ngưỡng | Kết luận |")
    md.append("|--------|---------|-------|--------|----------|")
    md.append(f"| GROUNDED (đúng must_include) | {pass_count}/{n} | {pass_count/n*100:.1f}% | ≥85% | {'✅ ĐẠT' if pass_count/n>=0.85 else '❌ CHƯA ĐẠT'} |")
    md.append(f"| Có câu trả lời | {has_ans_count}/{n} | {has_ans_count/n*100:.1f}% | ≥95% | {'✅ ĐẠT' if has_ans_count/n>=0.95 else '⚠️ CẦN CẢI THIỆN'} |")
    md.append(f"| LANG (100% tiếng Việt) | {lang_ok}/{n} | {lang_ok/n*100:.1f}% | 100% | {'✅ ĐẠT' if lang_ok==n else '❌ CHƯA ĐẠT'} |")
    md.append(f"| Latency trung bình | {avg_lat:.1f} s | — | <30 s | {'✅ ĐẠT' if avg_lat<30 else '⚠️'} |")
    md.append("")

    md.append("## 2. Phân phối routing")
    md.append("")
    md.append("| Route | Số câu | Tỷ lệ |")
    md.append("|-------|--------|-------|")
    md.append(f"| PaddedRAG (1-hop) | {routes.get('padded_rag',0)} | {routes.get('padded_rag',0)/n*100:.1f}% |")
    md.append(f"| GraphRAG (multi-hop/PPR) | {routes.get('graph_rag',0)} | {routes.get('graph_rag',0)/n*100:.1f}% |")
    md.append("")

    md.append("## 3. NER & PPR — Chỉ báo điểm nghẽn")
    md.append("")
    md.append("| Chỉ số | Giá trị | Trước fix | Sau fix |")
    md.append("|--------|---------|-----------|---------|")
    md.append(f"| NER seeds avg | {avg_ner:.1f} | 19.4 | {avg_ner:.1f} |")
    md.append(f"| PPR hits avg | {avg_ppr:.1f} | 8.5 | {avg_ppr:.1f} |")
    if graph_results:
        md.append(f"| GraphRAG có PPR=0 | {graph_ppr_zero}/{len(graph_results)} | ~100% | {graph_ppr_zero/len(graph_results)*100:.1f}% |")
    md.append("")

    md.append("## 4. Phân tích theo loại câu hỏi")
    md.append("")
    md.append("| Loại | Tổng | Pass | Tỷ lệ Pass | Có TL |")
    md.append("|------|------|------|------------|-------|")
    cat_vi = {"simple": "Đơn giản", "multihop": "Nhiều bước", "temporal": "Theo thời gian", "comparative": "So sánh"}
    for cat, counts in by_cat.items():
        name = cat_vi.get(cat, cat)
        rate = counts["passed"] / counts["total"] * 100
        md.append(f"| {name} | {counts['total']} | {counts['passed']} | {rate:.1f}% | {counts['has_ans']} |")
    md.append("")

    md.append("## 5. Phân tích thất bại (FAIL)")
    md.append("")
    md.append(f"- Tổng câu fail: **{len(failures)}** ({len(failures)/n*100:.1f}%)")
    md.append(f"- Fail do không có chunk nào (chunks=0): {fail_no_chunks}")
    md.append(f"- Fail dù có chunks (must_include không match): {fail_with_chunks}")
    md.append(f"- Fail do trả lời tiếng Anh: {n - lang_ok}")
    md.append("")

    md.append("### Chi tiết 30 câu FAIL đầu tiên")
    md.append("")
    md.append("| ID | Câu hỏi | Route | NER seeds | PPR hits | Vấn đề |")
    md.append("|----|---------|-------|-----------|----------|--------|")
    for r in failures[:30]:
        q = r.get("question", r.get("query", ""))[:80].replace("|", "\\|")
        if r.get("num_chunks", 0) == 0:
            issue = "Thiếu chunks"
        elif not calc_lang_pass(r.get("answer", "")):
            issue = "Câu trả lời chứa tiếng Anh"
        else:
            issue = "Must_include không match"
        md.append(f"| {r['id']} | {q} | {r.get('route','?')} | {r.get('ner_seed_count',0)} | {r.get('ppr_hits',r.get('ppr_hit_count',0))} | {issue} |")
    if len(failures) > 30:
        md.append(f"| ... | (còn {len(failures)-30} câu) | | | | |")
    md.append("")

    md.append("## 6. Kết luận chuyên môn")
    md.append("")
    if pass_count / n >= 0.85 and lang_ok == n:
        md.append("✅ **Hệ thống đạt ngưỡng production** cho cả GROUNDED và LANG.")
    else:
        md.append("⚠️ **Hệ thống chưa đạt ngưỡng production**:")
        if pass_count / n < 0.85:
            md.append(f"  - GROUNDED {pass_count/n*100:.1f}% < 85% — cần cải thiện retrieval/grounding")
        if lang_ok < n:
            md.append(f"  - LANG {lang_ok/n*100:.1f}% < 100% — còn {n-lang_ok} câu lẫn tiếng Anh")
    md.append("")
    md.append("**Khuyến nghị tiếp theo**:")
    if avg_ner < 30:
        md.append(f"  - NER seeds avg = {avg_ner:.1f} — cần mở rộng từ vựng và regex pattern")
    if avg_ppr < 20 and routes.get("graph_rag", 0) > 0:
        md.append(f"  - PPR hits avg = {avg_ppr:.1f} — cần tăng độ liên kết của knowledge graph")
    if fail_with_chunks > fail_no_chunks:
        md.append("  - Phần lớn fail có chunks nhưng must_include không match — cần điều chỉnh prompt sinh đáp án")

    return "\n".join(md)


def gen_staff_report(data):
    """Generate clean Q&A report for admission staff with citations."""
    results = data["results"]
    n = len(results)
    pass_count = sum(1 for r in results if r.get("passed"))

    md = []
    md.append("# Báo cáo Tư vấn Tuyển sinh HUSC — Hệ thống chatbot RAG")
    md.append("")
    md.append(f"**Ngày kiểm thử**: {datetime.now().strftime('%d/%m/%Y')}  ")
    md.append(f"**Tổng số câu hỏi**: {n}  ")
    md.append(f"**Số câu trả lời đạt yêu cầu**: {pass_count}/{n} ({pass_count/n*100:.1f}%)  ")
    md.append(f"**Nguồn dữ liệu**: tuyensinh.husc.edu.vn (cập nhật 2026), Thông tư 06/2026/TT-BGDĐT")
    md.append("")
    md.append("---")
    md.append("")

    # Group by category
    by_cat = {}
    for r in results:
        cat = r.get("category", "khac")
        by_cat.setdefault(cat, []).append(r)

    cat_vi = {
        "simple": "Câu hỏi đơn giản (1 thông tin)",
        "multihop": "Câu hỏi tổng hợp (nhiều thông tin)",
        "temporal": "Câu hỏi theo thời gian/năm",
        "comparative": "Câu hỏi so sánh",
    }

    for cat, items in by_cat.items():
        md.append(f"## {cat_vi.get(cat, cat).upper()}")
        md.append("")
        for r in items:
            status = "✅" if r.get("passed") else ("⚠️" if r.get("has_answer") else "❌")
            md.append(f"### {status} {r['id']}")
            md.append("")
            md.append(f"**Câu hỏi**: {r.get('question', r.get('query', ''))}")
            md.append("")
            md.append(f"**Trả lời**: {r.get('answer', '(không có câu trả lời)')}")
            md.append("")
            sources = r.get("chunk_sources", r.get("sources", []))
            unique_sources = list(dict.fromkeys(sources))  # dedupe preserving order
            if unique_sources:
                md.append("**Căn cứ tài liệu**:")
                for s in unique_sources[:5]:
                    md.append(f"  - {s}")
            else:
                md.append("**Căn cứ tài liệu**: (không có)")
            md.append("")
            md.append("---")
            md.append("")

    return "\n".join(md)


def main():
    data = load_results()
    if not data:
        return 1

    tech_md = gen_technical_report(data)
    TECH_REPORT.write_text(tech_md, encoding="utf-8")
    print(f"[OK] Technical report saved to {TECH_REPORT}")

    staff_md = gen_staff_report(data)
    STAFF_REPORT.write_text(staff_md, encoding="utf-8")
    print(f"[OK] Staff report saved to {STAFF_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

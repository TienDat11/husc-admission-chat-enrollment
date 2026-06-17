"""
Sinh 2 báo cáo cho 86 câu thí sinh thực:
1) results/86q_thi_sinh_chuyen_mon.md
2) results/86q_thi_sinh_thay_tuyen_sinh.md
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import statistics

RESULTS = Path(__file__).parent.parent / "results"
DATA = json.loads((RESULTS / "86q_thi_sinh_results.json").read_text(encoding="utf-8"))

meta = DATA["metadata"]
results = DATA["results"]
n = len(results)


def _has_fallback(ans: str) -> bool:
    a = ans.lower()
    return ("không tìm thấy" in a) or ("không cung cấp" in a) or ("không đủ" in a)


total_ans = sum(1 for r in results if r.get("has_answer"))
fallback = sum(1 for r in results if _has_fallback(r.get("answer", "")))
lats = [r["latency_s"] for r in results]
routes = defaultdict(int)
intents = defaultdict(lambda: {"t": 0, "ans": 0})
for r in results:
    routes[r["route"]] += 1
    i = r.get("intent", "general")
    intents[i]["t"] += 1
    if r.get("has_answer"):
        intents[i]["ans"] += 1


# ──────────────────────────────────────────────────────────────────────
# 1. Báo cáo CHUYÊN MÔN — chi tiết kỹ thuật cho dev
# ──────────────────────────────────────────────────────────────────────
chuyen_mon = []
chuyen_mon.append("# Báo cáo chuyên môn — 86 câu thí sinh thực (Phase E)\n")
chuyen_mon.append(f"**Ngày**: 2026-05-30 | **Pipeline**: SmartRouter (HyDE + Step-Back) → PaddedRAG/GraphRAG → {meta.get('gen_model', 'mimo-v2.5-pro')}\n")
chuyen_mon.append(f"**Phase E enhancements**: Anti-fallback retry • Aggregation booster (12 rules) • Router cache (LRU)\n")
chuyen_mon.append("\n---\n")
chuyen_mon.append("## 1. Tổng quan\n")
chuyen_mon.append(f"| Chỉ số | Giá trị |\n|---|---|\n")
chuyen_mon.append(f"| Tổng số câu hỏi | {n} |\n")
chuyen_mon.append(f"| Có câu trả lời (has_answer) | {total_ans}/{n} = **{total_ans/n*100:.1f}%** |\n")
chuyen_mon.append(f"| Fallback ('không tìm thấy') | {fallback}/{n} = **{fallback/n*100:.1f}%** |\n")
chuyen_mon.append(f"| Latency trung bình | **{statistics.mean(lats):.1f}s** |\n")
chuyen_mon.append(f"| Latency median | {statistics.median(lats):.1f}s |\n")
chuyen_mon.append(f"| Routing PaddedRAG | {routes['padded_rag']} câu ({routes['padded_rag']/n*100:.1f}%) |\n")
chuyen_mon.append(f"| Routing GraphRAG | {routes['graph_rag']} câu ({routes['graph_rag']/n*100:.1f}%) |\n")
chuyen_mon.append(f"| Embedding | Qwen3-Embedding-0.6B (1024-dim) |\n")
chuyen_mon.append(f"| Generation model | {meta.get('gen_model', 'mimo-v2.5-pro')} (primary) + gpt-5.4 (fallback) via ramclouds.me |\n")
chuyen_mon.append(f"| NER + HyDE model | gpt-5.3 |\n")

chuyen_mon.append("\n## 2. Phân bố theo intent\n")
chuyen_mon.append("| Intent | Số câu | Có câu trả lời | Tỷ lệ |\n|---|---:|---:|---:|\n")
for it, v in sorted(intents.items(), key=lambda x: -x[1]["t"]):
    if v["t"] >= 1:
        rate = v["ans"]/v["t"]*100 if v["t"] else 0
        chuyen_mon.append(f"| `{it}` | {v['t']} | {v['ans']} | {rate:.1f}% |\n")

chuyen_mon.append("\n## 3. Phân tích thất bại\n")
fails = [r for r in results if not r.get("has_answer")]
chuyen_mon.append(f"Có **{len(fails)}/{n}** câu pipeline trả fallback hoặc không sinh được câu trả lời. Phân bổ:\n\n")
fail_by_intent = defaultdict(list)
for r in fails:
    fail_by_intent[r.get("intent", "general")].append(r)
for it in sorted(fail_by_intent.keys(), key=lambda x: -len(fail_by_intent[x])):
    chuyen_mon.append(f"- **{it}** — {len(fail_by_intent[it])} câu\n")

chuyen_mon.append("\n## 4. Chi tiết từng câu\n\n")
for r in results:
    icon = "✅" if r.get("has_answer") else "❌"
    chuyen_mon.append(f"### {icon} {r['id']} — {r.get('intent','?')} / {r['route']}\n\n")
    chuyen_mon.append(f"**Câu hỏi**: {r['question']}\n\n")
    chuyen_mon.append(f"**Câu trả lời** ({r['provider']}, {r['latency_s']}s):\n\n")
    chuyen_mon.append(f"> {r.get('answer','').replace(chr(10),' ')}\n\n")
    if r.get("chunk_ids"):
        ids = [str(x) for x in r['chunk_ids'][:5] if x]
        if ids:
            chuyen_mon.append(f"**Chunks dùng**: `{', '.join(ids)}`\n\n")
    chuyen_mon.append(f"**Routing**: {r['route']} • complexity={r['complexity']} • intent={r['intent']}\n\n")
    chuyen_mon.append("---\n\n")

(RESULTS / "86q_thi_sinh_chuyen_mon.md").write_text("".join(chuyen_mon), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# 2. Báo cáo cho THẦY TUYỂN SINH — gọn, dễ đọc
# ──────────────────────────────────────────────────────────────────────
thay = []
thay.append("# Báo cáo đánh giá hệ thống tư vấn tuyển sinh HUSC\n")
thay.append("## 86 câu hỏi thực tế của thí sinh\n\n")
thay.append("**Ngày kiểm thử**: 30/05/2026  ·  **Hệ thống**: Chatbot RAG HUSC 2026 (Phase E)\n\n")
thay.append("---\n\n")
thay.append("## I. Tóm tắt kết quả\n\n")
thay.append(f"- **Tổng số câu hỏi**: {n}\n")
thay.append(f"- **Hệ thống có câu trả lời**: {total_ans}/{n} câu (**{total_ans/n*100:.1f}%**)\n")
thay.append(f"- **Hệ thống không tìm thấy thông tin**: {fallback}/{n} câu ({fallback/n*100:.1f}%)\n")
thay.append(f"- **Thời gian trả lời trung bình**: {statistics.mean(lats):.0f} giây mỗi câu\n\n")

thay.append("## II. Phân loại nội dung câu hỏi\n\n")
thay.append("Hệ thống phân loại 86 câu vào các nhóm chủ đề chính (intent):\n\n")
intent_vi = {
    "thu_tuc": "Thủ tục hồ sơ / đăng ký",
    "chinh_sach": "Chính sách ưu tiên / điểm cộng / học bổng",
    "to_hop": "Tổ hợp xét tuyển",
    "diem_chuan": "Điểm chuẩn / điểm sàn",
    "hoc_phi": "Học phí",
    "nganh_hoc": "Thông tin về ngành học",
    "liet_ke": "Liệt kê / đếm số lượng ngành",
    "so_sanh": "So sánh ngành / phương thức",
    "da_hop": "Câu hỏi đa hợp (nhiều thông tin)",
    "general": "Câu hỏi chung",
}
thay.append("| Chủ đề | Số câu | Có câu trả lời | Tỷ lệ |\n|---|---:|---:|---:|\n")
for it, v in sorted(intents.items(), key=lambda x: -x[1]["t"]):
    name = intent_vi.get(it, it)
    rate = v["ans"]/v["t"]*100 if v["t"] else 0
    thay.append(f"| {name} | {v['t']} | {v['ans']} | {rate:.0f}% |\n")

thay.append("\n## III. Câu hỏi nhà trường có thể tin cậy chatbot trả lời\n\n")
ok = [r for r in results if r.get("has_answer")]
thay.append(f"Có **{len(ok)}/{n}** câu hỏi (~{len(ok)/n*100:.0f}%) chatbot trả được trên dữ liệu canonical 2026.\n\n")
thay.append("### Ví dụ câu trả lời TỐT của chatbot\n\n")
sample_ok = [r for r in ok if len(r.get("answer", "")) > 80][:8]
for i, r in enumerate(sample_ok, 1):
    thay.append(f"**{i}. {r['question']}**\n\n")
    ans = r.get("answer", "").replace("\n", " ")
    thay.append(f"> {ans}\n\n")

thay.append("\n## IV. Câu hỏi chatbot CHƯA xử lý tốt\n\n")
thay.append(f"Có **{len(fails)}/{n}** câu (~{len(fails)/n*100:.0f}%) chatbot không tìm thấy hoặc trả lời chưa thỏa đáng. Phân bổ:\n\n")
for it in sorted(fail_by_intent.keys(), key=lambda x: -len(fail_by_intent[x])):
    name = intent_vi.get(it, it)
    thay.append(f"- **{name}**: {len(fail_by_intent[it])} câu\n")

thay.append("\n### Ví dụ câu chatbot chưa trả lời được\n\n")
for i, r in enumerate(fails[:8], 1):
    thay.append(f"**{i}. {r['question']}**\n\n")
    ans = r.get("answer", "").replace("\n", " ")
    thay.append(f"> {ans}\n\n")

thay.append("\n## V. Khuyến nghị bổ sung dữ liệu\n\n")
thay.append("Để chatbot xử lý tốt hơn các câu chưa trả lời được, đề xuất nhà trường cập nhật/bổ sung các nhóm thông tin sau lên cổng tuyển sinh:\n\n")

# Khuyến nghị dựa trên fail-by-intent
recommendations = {
    "thu_tuc": [
        "Hướng dẫn từng bước đăng ký tuyển sinh trên cổng Bộ GD&ĐT",
        "Quy trình xét tuyển dành cho thí sinh tự do (đã tốt nghiệp từ năm trước)",
        "Cách nộp hồ sơ xét tuyển trực tuyến vs giấy",
        "Hạn cuối các đợt xét tuyển 2026 (THPT, học bạ, ĐGNL, kết hợp)",
        "Quy trình chuyển điểm năng khiếu (vẽ mỹ thuật) từ trường khác về HUSC",
    ],
    "chinh_sach": [
        "Bảng quy đổi chứng chỉ tiếng Anh sang điểm xét tuyển",
        "Hướng dẫn cộng điểm cho học sinh trường chuyên / đạt giải HSG cấp tỉnh / thành phố",
        "Chính sách hỗ trợ sinh viên khó khăn / khuyết tật",
        "Quy trình xét học bổng (chung cho mọi phương thức hay tách riêng?)",
    ],
    "diem_chuan": [
        "Bảng điểm chuẩn 2026 dự kiến (ngay khi có thể công bố)",
        "Bảng điểm chuẩn 3 năm gần đây (2023-2024-2025) để thí sinh đối chiếu",
        "Điểm sàn 2026 cho từng phương thức xét tuyển",
    ],
    "hoc_phi": [
        "Học phí cụ thể từng ngành tính theo tín chỉ × số tín chỉ chuẩn / năm",
        "Chính sách chia nộp học phí nhiều đợt",
    ],
    "nganh_hoc": [
        "Mô tả chi tiết cơ hội nghề nghiệp sau tốt nghiệp từng ngành (đặc biệt: Lịch sử, Văn học, Quản lý văn hoá...)",
        "Các nghiệp vụ phụ thí sinh có thể học thêm (sư phạm, tin học, ngoại ngữ)",
    ],
}

for intent_key, items in recommendations.items():
    if intent_key in fail_by_intent and len(fail_by_intent[intent_key]) > 0:
        name = intent_vi.get(intent_key, intent_key)
        thay.append(f"\n### {name} ({len(fail_by_intent[intent_key])} câu chưa xử lý)\n\n")
        for it in items:
            thay.append(f"- {it}\n")

thay.append("\n## VI. Kết luận\n\n")
thay.append(
    f"Hệ thống chatbot HUSC ở phiên bản Phase E đã có thể tự động trả lời "
    f"khoảng **{total_ans/n*100:.0f}%** ({total_ans}/{n}) câu hỏi thực tế của thí sinh "
    f"chỉ dựa trên dữ liệu canonical năm 2026 đã ingest. "
    f"Phần còn lại ({fallback/n*100:.0f}%) chủ yếu thuộc nhóm thủ tục hành chính và chính sách "
    f"chi tiết — những thông tin này thường nằm ngoài tài liệu tuyển sinh chính thức "
    f"và cần nhà trường bổ sung để chatbot phục vụ thí sinh đầy đủ hơn.\n\n"
)
thay.append("Báo cáo này đi kèm với báo cáo chuyên môn (`86q_thi_sinh_chuyen_mon.md`) để đội kỹ thuật theo dõi từng câu một cách chi tiết.\n")

(RESULTS / "86q_thi_sinh_thay_tuyen_sinh.md").write_text("".join(thay), encoding="utf-8")

print(f"Wrote {(RESULTS / '86q_thi_sinh_chuyen_mon.md').name}: {sum(len(s) for s in chuyen_mon)} chars")
print(f"Wrote {(RESULTS / '86q_thi_sinh_thay_tuyen_sinh.md').name}: {sum(len(s) for s in thay)} chars")

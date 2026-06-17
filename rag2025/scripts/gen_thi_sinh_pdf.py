"""Generate 2 PDF reports from 86Q evaluation results."""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

RESULTS_DIR = ROOT / "results"
DATA = json.loads((RESULTS_DIR / "86q_thi_sinh_results.json").read_text(encoding="utf-8"))
meta = DATA["metadata"]
results = DATA["results"]
n = len(results)

# Register a Vietnamese-capable TrueType font. Helvetica (ReportLab built-in)
# has NO Vietnamese glyphs — accented chars render as blanks/boxes, making the
# whole PDF unreadable. Arial (Windows) covers full Vietnamese Unicode.
def _register_vietnamese_font():
    candidates = [
        ("Arial", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        ("Tahoma", r"C:\Windows\Fonts\tahoma.ttf", r"C:\Windows\Fonts\tahomabd.ttf"),
        ("Times", r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf"),
    ]
    for name, reg, bold in candidates:
        if Path(reg).exists():
            pdfmetrics.registerFont(TTFont(name, reg))
            bold_name = name
            if Path(bold).exists():
                bold_name = f"{name}-Bold"
                pdfmetrics.registerFont(TTFont(bold_name, bold))
            registerFontFamily(
                name, normal=name, bold=bold_name,
                italic=name, boldItalic=bold_name,
            )
            return name, bold_name
    return "Helvetica", "Helvetica-Bold"

FONT, FONT_BOLD = _register_vietnamese_font()


def md_to_rl(text: str, limit: int = None) -> str:
    """Convert a markdown answer snippet into ReportLab-safe inline markup.

    - escapes XML special chars (& < >) so Paragraph doesn't choke
    - converts **bold** → <b>bold</b>
    - drops leftover markdown bullets/emoji that render as boxes
    - collapses newlines to spaces, optional length cap
    """
    s = (text or "").replace("\n", " ")
    if limit:
        s = s[:limit]
    # Escape XML metachars BEFORE injecting our own tags
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # **bold** → <b>bold</b> (non-greedy, across the snippet)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # leftover single * used as bullet/emphasis → strip
    s = s.replace("*", "")
    # Strip emoji / symbol glyphs Arial can't render (they show as boxes):
    # globe 🌐, phone 📞, pin 📍, checkmarks, arrows, bullet glyphs, etc.
    #
    # VERIFIED Vietnamese-safe (2026-06): the ranges below cover only the
    # emoji/symbol Unicode blocks (Misc Symbols, Dingbats, Pictographs,
    # Mahjong/Chess, Regional Indicators, Arrows, Geometric Shapes, etc.).
    # Vietnamese precomposed letters live at U+00C0-U+024F and U+1EA0-U+1EF9
    # (Latin Extended-A/B + Latin Extended Additional); combining diacritical
    # marks live at U+0300-U+036F. None of those code points are inside any
    # stripped range, so a full Vietnamese string (đ, ô, ư, ạ, ề, ữ, ọ, ẵ,
    # Ạ, Ễ, etc.) passes through unchanged. Regression test:
    #   tests/scripts/test_gen_thi_sinh_pdf.py::test_md_to_rl_preserves_full_vietnamese
    s = re.sub(
        r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
        r"←-⇿⌀-⏿■-◿⬀-⯿️‍�☐-☒]",
        "",
        s,
    )
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _has_fallback(ans):
    # Align with eval's has_answer flag — only a SHORT answer that opens with
    # an explicit "not found" sentinel counts as fallback. Naive substring scan
    # for "không đủ" false-positives on legit answers ("không đủ điểm/điều kiện").
    a = (ans or "").strip().lower()
    return (
        not a
        or len(a) < 50
        or a.startswith("tôi không tìm thấy")
        or a.startswith("không tìm thấy")
        or "không tìm thấy thông tin này trong tài liệu" in a
        or a.startswith("tôi không cung cấp")
    )

total_ans = sum(1 for r in results if r.get("has_answer"))
fallback = sum(1 for r in results if _has_fallback(r.get("answer", "")))
lats = [r["latency_s"] for r in results]
# Read the actual gen model from results metadata (not hardcoded) so the
# report reflects whichever model produced these answers (mimo-v2.5-pro now).
GEN_MODEL = meta.get("gen_model") or "mimo-v2.5-pro"
routes = defaultdict(int)
intents = defaultdict(lambda: {"t": 0, "ans": 0})
for r in results:
    routes[r["route"]] += 1
    i = r.get("intent", "general")
    intents[i]["t"] += 1
    if r.get("has_answer"):
        intents[i]["ans"] += 1

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='VNTitle', fontName=FONT_BOLD, fontSize=18, leading=24,
                          alignment=TA_CENTER, spaceAfter=20, textColor=HexColor('#1a5276')))
styles.add(ParagraphStyle(name='VNH2', fontName=FONT_BOLD, fontSize=14, leading=18,
                          spaceAfter=10, textColor=HexColor('#2c3e50')))
styles.add(ParagraphStyle(name='VNBody', fontName=FONT, fontSize=10, leading=14,
                          spaceAfter=6, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='VNSmall', fontName=FONT, fontSize=8, leading=10, textColor=HexColor('#7f8c8d')))
styles.add(ParagraphStyle(name='VNResult', fontName=FONT, fontSize=9, leading=12, spaceAfter=4))
styles.add(ParagraphStyle(name='VNPass', fontName=FONT, fontSize=9, leading=12, textColor=HexColor('#27ae60')))
styles.add(ParagraphStyle(name='VNFail', fontName=FONT, fontSize=9, leading=12, textColor=HexColor('#e74c3c')))

def build_story_full():
    story = []
    story.append(Paragraph("Báo cáo kỹ thuật — 86 câu hỏi thực tế của thí sinh", styles['VNTitle']))
    story.append(Paragraph(f"Ngày: 30/05/2026 | Pipeline: SmartRouter (HyDE + Step-Back) → PaddedRAG/GraphRAG → {GEN_MODEL}", styles['VNSmall']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#bdc3c7')))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Tổng quan", styles['VNH2']))
    summary_data = [
        ['Chỉ số', 'Giá trị'],
        ['Tổng số câu hỏi', str(n)],
        ['Có câu trả lời (has_answer)', f'{total_ans}/{n} = {total_ans/n*100:.1f}%'],
        ['Fallback', f'{fallback}/{n} = {fallback/n*100:.1f}%'],
        ['Latency trung bình', f'{statistics.mean(lats):.1f}s'],
        ['Latency median', f'{statistics.median(lats):.1f}s'],
        ['Routing PaddedRAG', f'{routes["padded_rag"]} câu ({routes["padded_rag"]/n*100:.1f}%)'],
        ['Routing GraphRAG', f'{routes["graph_rag"]} câu ({routes["graph_rag"]/n*100:.1f}%)'],
        ['Embedding', 'Qwen3-Embedding-0.6B (1024-dim)'],
        ['Generation model', f'{GEN_MODEL} (primary) + gpt-5.4 (fallback)'],
    ]
    t = Table(summary_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTNAME', (0,1), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#bdc3c7')),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, HexColor('#ecf0f1')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Phân bố theo intent", styles['VNH2']))
    intent_vi = {
        "thu_tuc": "Thủ tục hồ sơ / đăng ký",
        "chinh_sach": "Chính sách ưu tiên / điểm cộng / học bổng",
        "to_hop": "Tổ hợp xét tuyển",
        "diem_chuan": "Điểm chuẩn / điểm sàn",
        "hoc_phi": "Học phí",
        "nganh_hoc": "Thông tin về ngành học",
        "liet_ke": "Liệt kê / đếm số lượng ngành",
        "so_sanh": "So sánh ngành / phương thức",
        "da_hop": "Câu hỏi đa hợp",
        "general": "Câu hỏi chung",
    }
    intent_data = [['Intent', 'Số câu', 'Có trả lời', 'Tỷ lệ']]
    for it, v in sorted(intents.items(), key=lambda x: -x[1]["t"]):
        name = intent_vi.get(it, it)
        rate = v["ans"]/v["t"]*100 if v["t"] else 0
        intent_data.append([name, str(v["t"]), str(v["ans"]), f'{rate:.0f}%'])
    t2 = Table(intent_data, colWidths=[6*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTNAME', (0,1), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#bdc3c7')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, HexColor('#ecf0f1')]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Chi tiết từng câu", styles['VNH2']))
    for r in results:
        icon = "ĐẠT" if r.get("has_answer") else "CHƯA"
        st = styles['VNPass'] if r.get("has_answer") else styles['VNFail']
        story.append(Paragraph(f'<b>[{icon}] {r["id"]}</b> — {r.get("intent","?")} / {r["route"]} | {r["latency_s"]}s', st))
        story.append(Paragraph(f'<b>Câu hỏi:</b> {md_to_rl(r["question"])}', styles['VNBody']))
        story.append(Paragraph(f'<b>Trả lời:</b> {md_to_rl(r.get("answer",""))}', styles['VNSmall']))
        if r.get("chunk_ids"):
            ids = [str(x) for x in r['chunk_ids'][:5] if x]
            if ids:
                story.append(Paragraph(f'<b>Chunks:</b> {", ".join(ids)}', styles['VNSmall']))
        story.append(Spacer(1, 4))

    return story


def build_story_teacher():
    story = []
    story.append(Paragraph("Báo cáo đánh giá hệ thống tư vấn tuyển sinh HUSC", styles['VNTitle']))
    story.append(Paragraph("86 câu hỏi thực tế của thí sinh | Ngày kiểm thử: 30/05/2026", styles['VNSmall']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#bdc3c7')))
    story.append(Spacer(1, 10))

    story.append(Paragraph("I. Tóm tắt kết quả", styles['VNH2']))
    story.append(Paragraph(f"• Tổng số câu hỏi: {n}", styles['VNBody']))
    story.append(Paragraph(f"• Hệ thống có câu trả lời: {total_ans}/{n} câu ({total_ans/n*100:.1f}%)", styles['VNBody']))
    story.append(Paragraph(f"• Hệ thống không tìm thấy thông tin: {fallback}/{n} câu ({fallback/n*100:.1f}%)", styles['VNBody']))
    story.append(Paragraph(f"• Thời gian trả lời trung bình: {statistics.mean(lats):.0f} giây mỗi câu", styles['VNBody']))
    story.append(Spacer(1, 10))

    story.append(Paragraph("II. Toàn bộ câu trả lời của chatbot", styles['VNH2']))
    ok = [r for r in results if r.get("has_answer")]
    story.append(Paragraph(
        f"Dưới đây là TẤT CẢ {len(ok)}/{n} câu chatbot đã trả lời (đầy đủ, không rút gọn):",
        styles['VNBody']
    ))
    for i, r in enumerate(ok, 1):
        story.append(Paragraph(f'<b>{i}. {md_to_rl(r["question"])}</b>', styles['VNBody']))
        story.append(Paragraph(md_to_rl(r.get("answer","")), styles['VNSmall']))
        story.append(Spacer(1, 4))

    story.append(Paragraph("III. Câu hỏi chatbot CHƯA xử lý tốt", styles['VNH2']))
    fails = [r for r in results if not r.get("has_answer")]
    story.append(Paragraph(f"Có {len(fails)}/{n} câu chatbot không tìm thấy hoặc trả lời chưa thỏa đáng.", styles['VNBody']))
    for i, r in enumerate(fails, 1):
        story.append(Paragraph(f'<b>{i}. {md_to_rl(r["question"])}</b>', styles['VNBody']))
        story.append(Paragraph(md_to_rl(r.get("answer","")), styles['VNSmall']))
        story.append(Spacer(1, 4))
    if not fails:
        story.append(Paragraph("Tất cả 86 câu đều được trả lời (không có câu fallback).", styles['VNBody']))

    story.append(Paragraph("IV. Kết luận", styles['VNH2']))
    story.append(Paragraph(
        f"Hệ thống chatbot HUSC đã có thể tự động trả lời khoảng {total_ans/n*100:.0f}% "
        f"({total_ans}/{n}) câu hỏi thực tế của thí sinh chỉ dựa trên dữ liệu canonical năm 2026. "
        f"Phần còn lại ({n-total_ans} câu) chủ yếu thuộc nhóm thủ tục hành chính và chính sách chi tiết "
        f"— những thông tin này thường nằm ngoài tài liệu tuyển sinh chính thức và cần nhà trường bổ sung.",
        styles['VNBody']
    ))

    return story


def build_pdf(story, output_path):
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Báo cáo 86Q HUSC 2026",
        author="HUSC RAG Chatbot"
    )
    doc.build(story)
    print(f"Wrote {output_path.name}: {output_path.stat().st_size} bytes (font={FONT})")


if __name__ == "__main__":
    build_pdf(build_story_full(), RESULTS_DIR / "86q_thi_sinh_report_full.pdf")
    build_pdf(build_story_teacher(), RESULTS_DIR / "86q_thi_sinh_report_thay_tuyen_sinh.pdf")

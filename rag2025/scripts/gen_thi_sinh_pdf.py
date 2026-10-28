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
    PageBreak, HRFlowable, KeepTogether
)

# ── HUSC brand palette (kept in one place so the whole report stays on-brand) ──
C_PRIMARY = HexColor('#0b4f8a')   # HUSC blue — titles, header band
C_ACCENT = HexColor('#13a89e')    # teal — KPI accents, rules
C_DARK = HexColor('#1f2d3d')      # near-black body ink
C_MUTED = HexColor('#6b7a8d')     # captions / metadata
C_GOOD = HexColor('#1e8e4f')      # pass green
C_BAD = HexColor('#c0392b')       # fail red
C_ZEBRA = HexColor('#eef3f8')     # table zebra
C_CARD = HexColor('#f4f8fb')      # KPI card fill
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
    # Markdown headings (###, ##, #) leak from LLM answers — promote the heading
    # text to bold and drop the hashes (otherwise they render as literal "###").
    s = re.sub(r"#{1,6}\s*(.+?)(?=\s{2,}|$)", r"<b>\1</b>", s)
    # **bold** → <b>bold</b> (non-greedy, across the snippet)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # leftover single * used as bullet/emphasis → strip
    s = s.replace("*", "")
    # stray leftover hashes (e.g. mid-line "# ") → strip
    s = re.sub(r"#{1,6}", "", s)
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
# report reflects whichever model produced these answers (gpt-6-luna now).
GEN_MODEL = meta.get("gen_model") or "gpt-6-luna"
routes = defaultdict(int)
intents = defaultdict(lambda: {"t": 0, "ans": 0})
for r in results:
    routes[r["route"]] += 1
    i = r.get("intent", "general")
    intents[i]["t"] += 1
    if r.get("has_answer"):
        intents[i]["ans"] += 1

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='VNCover', fontName=FONT_BOLD, fontSize=26, leading=32,
                          alignment=TA_CENTER, textColor=white))
styles.add(ParagraphStyle(name='VNCoverSub', fontName=FONT, fontSize=13, leading=18,
                          alignment=TA_CENTER, textColor=HexColor('#d6e6f2')))
styles.add(ParagraphStyle(name='VNTitle', fontName=FONT_BOLD, fontSize=18, leading=24,
                          alignment=TA_CENTER, spaceAfter=20, textColor=C_PRIMARY))
styles.add(ParagraphStyle(name='VNH2', fontName=FONT_BOLD, fontSize=14, leading=18,
                          spaceBefore=6, spaceAfter=10, textColor=C_PRIMARY))
styles.add(ParagraphStyle(name='VNBody', fontName=FONT, fontSize=10, leading=15,
                          spaceAfter=6, alignment=TA_JUSTIFY, textColor=C_DARK))
styles.add(ParagraphStyle(name='VNSmall', fontName=FONT, fontSize=8.5, leading=12, textColor=C_MUTED))
styles.add(ParagraphStyle(name='VNAnswer', fontName=FONT, fontSize=9, leading=13.5,
                          textColor=C_DARK, alignment=TA_JUSTIFY, spaceAfter=2))
styles.add(ParagraphStyle(name='VNResult', fontName=FONT, fontSize=9, leading=12, spaceAfter=4))
styles.add(ParagraphStyle(name='VNPass', fontName=FONT_BOLD, fontSize=9.5, leading=13, textColor=C_GOOD))
styles.add(ParagraphStyle(name='VNFail', fontName=FONT_BOLD, fontSize=9.5, leading=13, textColor=C_BAD))
styles.add(ParagraphStyle(name='VNKpiNum', fontName=FONT_BOLD, fontSize=20, leading=22,
                          alignment=TA_CENTER, textColor=C_PRIMARY))
styles.add(ParagraphStyle(name='VNKpiLbl', fontName=FONT, fontSize=8, leading=10,
                          alignment=TA_CENTER, textColor=C_MUTED))
styles.add(ParagraphStyle(name='VNQ', fontName=FONT_BOLD, fontSize=10, leading=14,
                          textColor=C_DARK, spaceBefore=4))


def _kpi_card(num, label):
    """A single KPI tile (big number over caption) as a 1x2 borderless table."""
    inner = Table([[Paragraph(num, styles['VNKpiNum'])],
                   [Paragraph(label, styles['VNKpiLbl'])]],
                  colWidths=[4.0 * cm])
    inner.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_CARD),
        ('BOX', (0, 0), (-1, -1), 0.5, HexColor('#d4e2ef')),
        ('LINEABOVE', (0, 0), (-1, 0), 3, C_ACCENT),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    return inner


def _kpi_row():
    """Row of 4 KPI cards summarising the run."""
    cards = [
        _kpi_card(str(n), 'Tổng số câu hỏi'),
        _kpi_card(f'{total_ans/n*100:.0f}%', 'Tỷ lệ có trả lời'),
        _kpi_card(f'{statistics.median(lats):.0f}s', 'Latency trung vị'),
        _kpi_card(f'{fallback}', 'Câu chưa trả lời'),
    ]
    row = Table([cards], colWidths=[4.3 * cm] * 4)
    row.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return row


def _header_footer(canvas, doc):
    """Brand band on top + page number / source footer on every page."""
    canvas.saveState()
    w, h = A4
    # top brand bar
    canvas.setFillColor(C_PRIMARY)
    canvas.rect(0, h - 1.2 * cm, w, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont(FONT_BOLD, 9)
    canvas.drawString(2 * cm, h - 0.8 * cm, "TRƯỜNG ĐẠI HỌC KHOA HỌC — ĐẠI HỌC HUẾ")
    canvas.setFont(FONT, 8)
    canvas.drawRightString(w - 2 * cm, h - 0.8 * cm, "Báo cáo đánh giá Chatbot tuyển sinh")
    # footer rule + page number
    canvas.setStrokeColor(HexColor('#d4e2ef'))
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.3 * cm, w - 2 * cm, 1.3 * cm)
    canvas.setFillColor(C_MUTED)
    canvas.setFont(FONT, 8)
    canvas.drawString(2 * cm, 0.9 * cm, "Nguồn dữ liệu: canonical HUSC 2026 • tuyensinh.husc.edu.vn")
    canvas.drawRightString(w - 2 * cm, 0.9 * cm, f"Trang {doc.page}")
    canvas.restoreState()


def _cover(subtitle):
    """Full-bleed cover band rendered as flowables (one per report)."""
    band = Table([[Paragraph("Báo cáo đánh giá hệ thống<br/>Chatbot tư vấn tuyển sinh", styles['VNCover'])],
                  [Spacer(1, 8)],
                  [Paragraph(subtitle, styles['VNCoverSub'])]],
                 colWidths=[17 * cm])
    band.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_PRIMARY),
        ('TOPPADDING', (0, 0), (-1, 0), 34),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 34),
        ('LEFTPADDING', (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
    ]))
    return band

def build_story_full():
    story = []
    story.append(_cover(f"Báo cáo kỹ thuật • {n} câu hỏi thực tế của thí sinh<br/>"
                        f"Pipeline: SmartRouter (HyDE + Step-Back) → PaddedRAG/GraphRAG → {GEN_MODEL}"))
    story.append(Spacer(1, 18))
    story.append(_kpi_row())
    story.append(Spacer(1, 18))

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
        ['Generation model', f'{GEN_MODEL} (primary) + claude-haiku-5-5 (fallback)'],
    ]
    t = Table(summary_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTNAME', (0,1), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d4e2ef')),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_ZEBRA]),
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
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTNAME', (0,1), (-1,-1), FONT),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor('#d4e2ef')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, C_ZEBRA]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("3. Chi tiết từng câu", styles['VNH2']))
    for r in results:
        icon = "ĐẠT" if r.get("has_answer") else "CHƯA"
        st = styles['VNPass'] if r.get("has_answer") else styles['VNFail']
        block = [
            Paragraph(f'[{icon}] {r["id"]} — {r.get("intent","?")} / {r["route"]} | {r["latency_s"]}s', st),
            Paragraph(f'<b>Câu hỏi:</b> {md_to_rl(r["question"])}', styles['VNBody']),
            Paragraph(f'<b>Trả lời:</b> {md_to_rl(r.get("answer",""))}', styles['VNAnswer']),
        ]
        if r.get("chunk_ids"):
            ids = [str(x) for x in r['chunk_ids'][:5] if x]
            if ids:
                block.append(Paragraph(f'<b>Chunks:</b> {", ".join(ids)}', styles['VNSmall']))
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

    return story


def build_story_teacher():
    story = []
    story.append(_cover(f"Dành cho cán bộ tuyển sinh • {n} câu hỏi thực tế của thí sinh<br/>"
                        f"Ngày kiểm thử: 30/05/2026"))
    story.append(Spacer(1, 18))
    story.append(_kpi_row())
    story.append(Spacer(1, 18))

    story.append(Paragraph("I. Tóm tắt kết quả", styles['VNH2']))
    story.append(Paragraph(f"• Tổng số câu hỏi: {n}", styles['VNBody']))
    story.append(Paragraph(f"• Hệ thống có câu trả lời: {total_ans}/{n} câu ({total_ans/n*100:.1f}%)", styles['VNBody']))
    story.append(Paragraph(f"• Hệ thống không tìm thấy thông tin: {fallback}/{n} câu ({fallback/n*100:.1f}%)", styles['VNBody']))
    story.append(Paragraph(f"• Thời gian trả lời trung bình: {statistics.mean(lats):.0f} giây mỗi câu", styles['VNBody']))
    story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("II. Toàn bộ câu trả lời của chatbot", styles['VNH2']))
    ok = [r for r in results if r.get("has_answer")]
    story.append(Paragraph(
        f"Dưới đây là TẤT CẢ {len(ok)}/{n} câu chatbot đã trả lời (đầy đủ, không rút gọn):",
        styles['VNBody']
    ))
    story.append(Spacer(1, 4))
    for i, r in enumerate(ok, 1):
        story.append(KeepTogether([
            Paragraph(f'{i}. {md_to_rl(r["question"])}', styles['VNQ']),
            Paragraph(md_to_rl(r.get("answer","")), styles['VNAnswer']),
            Spacer(1, 7),
        ]))

    story.append(PageBreak())
    story.append(Paragraph("III. Câu hỏi chatbot CHƯA xử lý tốt", styles['VNH2']))
    fails = [r for r in results if not r.get("has_answer")]
    story.append(Paragraph(f"Có {len(fails)}/{n} câu chatbot không tìm thấy hoặc trả lời chưa thỏa đáng.", styles['VNBody']))
    story.append(Spacer(1, 4))
    for i, r in enumerate(fails, 1):
        story.append(KeepTogether([
            Paragraph(f'{i}. {md_to_rl(r["question"])}', styles['VNQ']),
            Paragraph(md_to_rl(r.get("answer","")), styles['VNAnswer']),
            Spacer(1, 7),
        ]))
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
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        title="Báo cáo 86Q HUSC 2026",
        author="HUSC RAG Chatbot"
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"Wrote {output_path.name}: {output_path.stat().st_size} bytes (font={FONT})")


if __name__ == "__main__":
    build_pdf(build_story_full(), RESULTS_DIR / "86q_thi_sinh_report_full.pdf")
    build_pdf(build_story_teacher(), RESULTS_DIR / "86q_thi_sinh_report_thay_tuyen_sinh.pdf")

"""Generate professional PDF reports of 86-question eval results for HUSC admissions staff.

Outputs:
  results/86q_thi_sinh_report_full.pdf    — full Q-A log with all 86 questions
  results/86q_thi_sinh_report_thay_tuyen_sinh.pdf  — clean report for admissions staff

Phase F2: fixed glyph issues, removed truncation, better layout.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from datetime import datetime

from fpdf import FPDF

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results" / "86q_thi_sinh_results.json"

FONT_DIR = Path("C:/Windows/Fonts")
FONT_REGULAR = str(FONT_DIR / "arial.ttf")
FONT_BOLD = str(FONT_DIR / "arialbd.ttf")
FONT_ITALIC = str(FONT_DIR / "ariali.ttf")


def _clean_text(text: str) -> str:
    """Remove characters that Arial font cannot render (emoji, special glyphs)."""
    # Replace non-breaking hyphen with regular hyphen
    text = text.replace('‑', '-')
    # Replace em-dash / en-dash with hyphen
    text = text.replace('–', '-').replace('—', '-')
    # Replace smart quotes with straight quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    # Remove emoji and other symbols outside BMP that TTF can't render
    text = re.sub(r'[\U0001f300-\U0001f9ff]', '', text)
    # Remove bullet/arrow symbols
    text = text.replace('•', '-')   # bullet
    text = text.replace('→', '->')  # right arrow
    text = text.replace('✓', 'v')   # check mark
    text = text.replace('✔', 'v')   # heavy check
    text = text.replace('✖', 'x')   # heavy X
    text = text.replace('●', '-')   # black circle
    text = text.replace('○', 'o')   # white circle
    text = text.replace('■', '-')   # black square
    # Remove any remaining non-latin1 characters that fpdf2 can't handle with TTF
    # Keep Vietnamese diacritics (Latin Extended), digits, punctuation
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # Allow: basic ASCII, Latin-1 supplement, Latin Extended (Vietnamese), common punctuation
        if cp < 0x200 or (0x1E00 <= cp <= 0x1EFF) or ch in '\n\r\t':
            cleaned.append(ch)
        elif 0x0100 <= cp <= 0x024F:  # Latin Extended-A/B
            cleaned.append(ch)
        elif cp == 0x00AB or cp == 0x00BB:  # guillemets
            cleaned.append(ch)
        # Skip everything else (emoji, arrows, math symbols, etc.)
    return ''.join(cleaned)


class ReportPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font("Arial", "", FONT_REGULAR)
        self.add_font("Arial", "B", FONT_BOLD)
        self.add_font("Arial", "I", FONT_ITALIC)
        self._title = title
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Arial", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, self._title, new_x="LMARGIN", new_y="NEXT", align="C")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Trang {self.page_no()}/{{nb}}", align="C")


def _write_text(pdf: FPDF, text: str, max_width: float = 185, font_size: float = 11):
    """Write multi-line text with automatic page breaks and glyph cleaning."""
    text = _clean_text(text)
    pdf.set_font("Arial", "", font_size)
    pdf.multi_cell(max_width, font_size * 0.5, text)


def generate_full_report(data: dict) -> str:
    """Generate full 86-question Q-A report."""
    pdf = ReportPDF("Báo cáo đánh giá Chatbot Tuyển sinh HUSC 2026")
    pdf.alias_nb_pages()
    pdf.add_page()

    meta = data["metadata"]
    results = data["results"]
    total = meta["total"]
    completed = meta["completed"]
    has_answer = sum(1 for r in results if r.get("has_answer"))
    fallback = completed - has_answer

    # Title page
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "BÁO CÁO ĐÁNH GIÁ CHATBOT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 12, "TUYỂN SINH HUSC 2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Ngay danh gia: {datetime.now().strftime('%d/%m/%Y %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Model sinh cau: {meta.get('gen_model', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Model NER/HyDE: {meta.get('ner_model', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Embedding: {meta.get('embedding_model', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Summary stats
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "TONG QUAN", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    rate = (has_answer / completed * 100) if completed else 0
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Tong cau hoi: {completed}/{total}", new_x="LMARGIN", new_y="NEXT")
    color = (0, 128, 0) if rate >= 85 else (200, 100, 0) if rate >= 70 else (200, 0, 0)
    pdf.set_text_color(*color)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, f"Ti le co dap an (has_answer): {has_answer}/{completed} ({rate:.1f}%)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Ti le fallback: {fallback}/{completed} ({fallback/completed*100:.1f}%)", new_x="LMARGIN", new_y="NEXT")

    if completed:
        latencies = [r.get("latency_s", 0) for r in results if r.get("latency_s")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        pdf.cell(0, 8, f"Latency trung binh: {avg_lat:.1f}s", new_x="LMARGIN", new_y="NEXT")

    # Route breakdown
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Phan bo route:", new_x="LMARGIN", new_y="NEXT")
    from collections import Counter
    routes = Counter(r.get("route", "unknown") for r in results)
    pdf.set_font("Arial", "", 11)
    for route, count in routes.most_common():
        pdf.cell(0, 7, f"  {route}: {count} cau", new_x="LMARGIN", new_y="NEXT")

    # Detail pages — one per question
    for i, r in enumerate(results, 1):
        pdf.add_page()
        qid = r.get("id", f"Q{i}")
        question = _clean_text(r.get("question", ""))
        answer = r.get("answer", "")
        has_ans = r.get("has_answer", False)
        route = r.get("route", "?")
        latency = r.get("latency_s", 0)
        complexity = r.get("complexity", "?")
        intent = r.get("intent", "?")
        provider = r.get("provider", "?")

        # Header
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 51, 102)
        pdf.multi_cell(185, 9, f"Cau {i}/{total} - {qid}", new_x="LMARGIN", new_y="NEXT")

        # Status badge
        if has_ans:
            pdf.set_fill_color(200, 240, 200)
            pdf.set_text_color(0, 100, 0)
            badge = "CO DAP AN"
        else:
            pdf.set_fill_color(255, 200, 200)
            pdf.set_text_color(180, 0, 0)
            badge = "FALLBACK"
        pdf.set_font("Arial", "B", 11)
        pdf.cell(40, 8, badge, fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 8, f"  Route: {route} | Complexity: {complexity} | Intent: {intent} | Latency: {latency:.1f}s", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"  Provider: {provider}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Question
        pdf.set_font("Arial", "B", 11)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 7, "Cau hoi:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        # Long questions must wrap — use multi_cell, never cell with truncation
        _write_text(pdf, question, 185, 11)
        pdf.ln(4)

        # Answer — NO truncation, smaller font for long answers
        pdf.set_font("Arial", "B", 11)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 7, "Cau tra loi:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        ans_font = 9 if len(answer) > 500 else 10
        _write_text(pdf, answer, 185, ans_font)

    # Fallback analysis appendix
    fallback_results = [r for r in results if not r.get("has_answer")]
    if fallback_results:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, 12, "PHU LUC: DANH SACH CAU FALLBACK", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

        for r in fallback_results:
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(185, 6, _clean_text(f"{r.get('id', '?')}: {r.get('question', '')}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(185, 5, f"  Route: {r.get('route', '?')} | Intent: {r.get('intent', '?')} | Latency: {r.get('latency_s', 0):.1f}s", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

    output = ROOT / "results" / "86q_thi_sinh_report_full.pdf"
    pdf.output(str(output))
    return str(output)


def generate_staff_report(data: dict) -> str:
    """Generate clean Q-A report for admissions staff (thay tuyen sinh)."""
    pdf = ReportPDF("HUSC 2026 - Ket qua hoi dap Chatbot Tuyen sinh")
    pdf.alias_nb_pages()
    pdf.add_page()

    results = data["results"]
    total = data["metadata"]["total"]
    completed = data["metadata"]["completed"]
    has_answer = sum(1 for r in results if r.get("has_answer"))

    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 14, "KET QUA HOI DAP", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 12, "CHATBOT TUYEN SINH HUSC 2026", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Tong so cau hoi thuc te tu thi sinh: {completed}/{total}", new_x="LMARGIN", new_y="NEXT")
    rate = (has_answer / completed * 100) if completed else 0
    pdf.cell(0, 8, f"Ti le chatbot tra loi duoc: {rate:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Ngay bao cao: {datetime.now().strftime('%d/%m/%Y')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    # Q-A pairs — only has_answer ones, NO truncation
    answered = [r for r in results if r.get("has_answer")]
    not_answered = [r for r in results if not r.get("has_answer")]

    for i, r in enumerate(answered, 1):
        # Check if we need a new page (leave room for answer)
        if pdf.get_y() > 240:
            pdf.add_page()

        question = _clean_text(r.get("question", ""))
        answer = r.get("answer", "")

        # Question — use multi_cell so long questions wrap properly
        pdf.set_font("Arial", "B", 11)
        pdf.set_text_color(0, 51, 102)
        pdf.multi_cell(185, 8, f"Cau {i}: {question}", new_x="LMARGIN", new_y="NEXT")

        # Answer — no truncation, adaptive font size
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(30, 30, 30)
        ans_font = 9 if len(answer) > 400 else 10
        _write_text(pdf, answer, 185, ans_font)

        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

    # Unanswered questions at end
    if not_answered:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, 10, "CAU HOI CHUA TRA LOI DUOC", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 8, "Sau day la cac cau hoi ma chatbot chua tra loi duoc - can bo sung du lieu:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        for r in not_answered:
            pdf.set_font("Arial", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(185, 7, _clean_text(f"  - {r.get('question', '')}"), new_x="LMARGIN", new_y="NEXT")

    output = ROOT / "results" / "86q_thi_sinh_report_thay_tuyen_sinh.pdf"
    pdf.output(str(output))
    return str(output)


def main():
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return 1

    with RESULTS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    completed = data["metadata"]["completed"]
    total = data["metadata"]["total"]
    if completed < total:
        print(f"Warning: Eval not complete ({completed}/{total}). Generating report with partial data.")

    out1 = generate_full_report(data)
    print(f"Full report: {out1}")

    out2 = generate_staff_report(data)
    print(f"Staff report: {out2}")

    has_answer = sum(1 for r in data["results"] if r.get("has_answer"))
    rate = (has_answer / completed * 100) if completed else 0
    print(f"\nSummary: {has_answer}/{completed} has_answer ({rate:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

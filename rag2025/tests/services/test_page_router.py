"""Page router (plan §PL-6.8): each page is text, scan or table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.page_router import route_pages  # noqa: E402

PROSE = "Điều 3. Đối tượng áp dụng\nQuy chế này áp dụng cho thí sinh dự tuyển đại học chính quy."
TABLE = (
    "7220201 Kỹ thuật phần mềm chỉ tiêu 120 25.5\n"
    "7340101 Quản trị kinh doanh chỉ tiêu 90 24.75"
)


def test_blank_page_is_scan() -> None:
    routing = route_pages(["", PROSE])
    assert list(routing.scan_pages) == [1]


def test_prose_page_is_text() -> None:
    routing = route_pages([PROSE])
    assert list(routing.table_pages) == []


def test_major_code_table_page_is_table() -> None:
    routing = route_pages([TABLE])
    assert list(routing.table_pages) == [1]


def test_ruling_operators_at_threshold_route_to_table() -> None:
    assert list(route_pages([PROSE], ruling_ops=[10]).table_pages) == [1]


def test_ruling_operators_below_threshold_stay_text() -> None:
    assert list(route_pages([PROSE], ruling_ops=[9]).table_pages) == []


def test_short_numeric_run_routes_to_table() -> None:
    text = "Bảng điểm sàn theo năm tuyển sinh\n" + "\n".join(["12", "13.5", "14", "15", "16"])
    assert list(route_pages([text]).table_pages) == [1]

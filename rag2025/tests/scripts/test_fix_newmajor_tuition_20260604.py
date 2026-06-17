"""Tests for fix_newmajor_tuition_20260604 corpus correction helpers."""
from __future__ import annotations
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fix_newmajor_tuition_20260604.py"
spec = importlib.util.spec_from_file_location("fix_nm", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_old_major_block_vii_525_to_490():
    line = "4. Văn học (mã 7229030) - chỉ tiêu 104, học phí Khối VII 525.000đ/TC, đào tạo 4 năm."
    out = mod._fix_block_line(line)
    assert "490.000đ/TC" in out
    assert "525.000" not in out


def test_old_major_block_iv_540_to_555():
    line = "11. Công nghệ sinh học (mã 7420201) - học phí Khối IV 540.000đ/TC, đào tạo 4 năm."
    out = mod._fix_block_line(line)
    assert "555.000đ/TC" in out
    assert "540.000" not in out


def test_block_v_600_unchanged():
    line = "17. Kỹ thuật phần mềm (mã 7480103) - học phí Khối V 600.000đ/TC."
    assert mod._fix_block_line(line) == line


def test_new_major_tuition_stripped_7440102SC():
    line = "13. Vật lý học - bán dẫn (mã 7440102SC) [NGÀNH MỚI 2026] - học phí Khối IV 540.000đ/TC, đào tạo 4 năm."
    out = mod._fix_block_line(line)
    assert "540.000" not in out and "555.000" not in out
    assert "chưa công bố" in out


def test_new_major_tuition_stripped_7460108_and_7510302IC():
    for code, blk, num in [("7460108", "V", "600.000"), ("7510302IC", "V", "600.000")]:
        line = f"X. Ngành mới (mã {code}) [NGÀNH MỚI 2026] - học phí Khối {blk} {num}đ/TC."
        out = mod._fix_block_line(line)
        assert num not in out
        assert "chưa công bố" in out


def test_hannom_code_fixed():
    assert mod.fix_hannom_text("| 13 VII | 7485104 | Hán – Nôm |") == "| 13 VII | 7485104 | Hán – Nôm |".replace("7485104", "7220104")


def test_idempotent_on_corrected_line():
    line = "4. Văn học (mã 7229030) - học phí Khối VII 490.000đ/TC."
    assert mod._fix_block_line(line) == line


def test_triet_hoc_mien_unchanged():
    line = "2. Triết học (mã 7229001) - MIỄN học phí, đào tạo 4 năm."
    assert mod._fix_block_line(line) == line

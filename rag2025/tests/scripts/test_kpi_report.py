"""Tests for kpi_report — đo KPI vận hành (T16).

Quy tắc (lead + spec 02 §9):
- Mọi số KPI phải kèm (a) đường dẫn artifact nguồn, (b) ngày đo.
- KPI chưa đo được (chưa có dữ liệu/chấm sự kiện) -> in đúng nhãn
  "chưa đo được: <lý do>", KHÔNG crash, KHÔNG bịa số, KHÔNG gán nhãn
  K2/K5 cho proxy.
- Đầu vào thiếu -> lỗi rõ ràng (exit != 0), không im lặng trả 0.

Test PIN hành vi THẬT của artifact baseline_T00 (chạy trực tiếp trên
results/baseline_T00/, không bịa số).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "kpi_report.py"


@pytest.fixture
def kpi():
    """Import kpi_report module (no side effects: only runs under __main__)."""
    spec = importlib.util.spec_from_file_location("kpi_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kpi_report"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_k2_not_measured_when_baseline_degraded(kpi, capsys):
    """K2 (86q) phải in 'chưa đo được' + lý do (provider LLM chết) —
    KHÔNG in 99.3% hay bất kỳ số K2 nào."""
    code = kpi.main(["k2"])
    assert code == 0
    out = capsys.readouterr().out
    # Nhãn 'chưa đo được' phải xuất hiện KÈM lý do degrade
    assert "K2 (86q eval): chưa đo được" in out
    assert "baseline degraded" in out
    # Không bịa số K2: KHÔNG có đường '86q (eval): 301/303' (tỷ lệ K2)
    # (Proxy truy xuất 301/303 được in riêng dưới nhãn 'Proxy', KHÔNG phải K2)
    assert "K2 (86q eval): 301/303" not in out
    assert "99.3%" not in out or "Proxy" in out  # nếu 99.3% xuất hiện, phải nằm trong proxy


def test_k2_heldout_not_measured_when_degraded(kpi, capsys):
    """K2 (heldout40) phải in 'chưa đo được' khi baseline degraded."""
    code = kpi.main(["k2"])
    assert code == 0
    out = capsys.readouterr().out
    assert "K2 (heldout40): chưa đo được" in out
    assert "fallback" in out


def test_k2_proxy_retrieval_separate_from_k2(kpi, capsys):
    """Proxy truy xuất (KHÔNG phải K2) được in dưới nhãn riêng, KHÔNG gán K2.
    Con số proxy phải bám artifact thật: 86q 301/303, heldout40 38/40."""
    code = kpi.main(["k2"])
    assert code == 0
    out = capsys.readouterr().out
    # Proxy line phải có mặt và tách riêng
    assert "Proxy truy xuất (KHÔNG phải K2)" in out
    # Proxy value bám artifact thật
    assert "301/303" in out  # 86q proxy
    assert "38/40" in out  # heldout40 proxy (38 câu có retrieved_chunks)


def test_k5_not_measured_when_no_scoring(kpi, capsys):
    """K5 phải in 'chưa đo được' (thiếu chấm sự kiện/judge_scores) —
    KHÔNG in 77.5% hay 0.0% như KPI."""
    code = kpi.main(["k5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "K5 (heldout40): chưa đo được" in out
    assert "thiếu chấm sự kiện" in out or "judge_scores = None" in out
    # Không gán K5 cho proxy: đường 'heldout40: 31/40 = 77.5%' (KPI thật)
    # KHÔNG xuất hiện — 31/40 chỉ nằm dưới nhãn Proxy.
    assert "K5 (heldout40): 31/40" not in out
    assert "77.5%" not in out or "Proxy" in out


def test_k5_proxy_route_match_is_separate(kpi, capsys):
    """Proxy route+expected_behavior (KHÔNG phải K5) phải in con số 31/40
    dưới nhãn Proxy riêng (bám heldout_metrics.json)."""
    code = kpi.main(["k5"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Proxy route+expected_behavior (KHÔNG phải K5)" in out
    assert "31/40" in out  # con số proxy, NOT K5


def _write_probe_log(data_dir, rows):
    """Ghi probe_log.jsonl giả lập cho test (đủ 2 mốc thời gian)."""
    probe_dir = data_dir / "kpi"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "probe_log.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")


def _write_queue(data_dir, rows):
    """Ghi unanswered/queue.jsonl giả lập cho test."""
    qdir = data_dir / "unanswered"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "queue.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")


def test_k3_not_measured_when_log_empty(kpi, capsys, tmp_path):
    """K3 log trống -> 'chưa có dữ liệu', KHÔNG in 0 phút (0 là số bịa)."""
    code = kpi.main(["k3", "--data-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "chưa đo được: chưa có dữ liệu" in out
    assert "K3 (median phút)" not in out


def test_k3_populated_reports_real_minutes(kpi, capsys, tmp_path):
    """K3 log đủ 2 mốc (upload 10:00 -> serve 10:30) -> median 30.0 phút."""
    _write_probe_log(tmp_path, [
        {"event": "published", "release_id": "r1",
         "upload_received_at": "2026-10-08T10:00:00+00:00",
         "published_at": "2026-10-08T10:05:00+00:00", "served_at": None},
        {"event": "served", "release_id": "r1",
         "served_at": "2026-10-08T10:30:00+00:00"},
    ])
    code = kpi.main(["k3", "--data-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "K3 (median phút)" in out
    assert "30.0" in out
    assert "chưa đo được" not in out


def test_k4_not_measured_when_queue_missing(kpi, capsys, tmp_path):
    """K4 thiếu queue.jsonl -> 'chưa đo được', KHÔNG in 0/0."""
    code = kpi.main(["k4", "--data-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "chưa đo được: chưa có hàng đợi unanswered" in out
    assert "48h" in out


def test_k4_counts_with_out_of_scope_exclusion(kpi, capsys, tmp_path):
    """K4: 2 mục (1 xử lý trong 48h, 1 ngoài phạm vi) -> tử 1/mẫu 1."""
    now = kpi.datetime.now(kpi.timezone.utc)
    asked = (now - kpi.timedelta(days=1)).isoformat()
    _write_queue(tmp_path, [
        {"question": "q1", "trace_id": "t1", "asked_at": asked,
         "route": "v2", "status_code": "X",
         "resolved_at": now.isoformat(), "resolved_by_record_id": "r",
         "out_of_scope": False},
        {"question": "spam", "trace_id": "t2", "asked_at": asked,
         "route": "v2", "status_code": "X",
         "resolved_at": None, "resolved_by_record_id": None,
         "out_of_scope": True},
    ])
    code = kpi.main(["k4", "--data-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "1/1" in out
    assert "chưa đo được" not in out


def test_missing_baseline_dir_exits_nonzero(kpi, capsys):
    """Đầu vào thiếu -> lỗi rõ ràng (exit != 0), không im lặng trả 0."""
    code = kpi.main(["k2", "--baseline-dir", "nosuchdir_xyz"])
    assert code != 0
    err = capsys.readouterr().err
    assert "không tìm thấy artifact baseline_T00" in err


def test_empty_artifact_exits_nonzero(kpi, capsys, tmp_path):
    """Artifact JSONL rỗng (tồn tại nhưng 0 dòng) -> exit != 0, không trả 0.
    (0/0 là số liệu giả tạo, KHÔNG phải 'chưa đo được')."""
    empty_dir = tmp_path / "empty_baseline"
    empty_dir.mkdir()
    (empty_dir / "husc86q_records_t00.jsonl").write_text("", encoding="utf-8")
    (empty_dir / "heldout40_records_t00.jsonl").write_text("", encoding="utf-8")
    (empty_dir / "heldout_metrics.json").write_text('{"per_question": []}', encoding="utf-8")
    code = kpi.main(["k2", "--baseline-dir", str(empty_dir)])
    assert code != 0
    err = capsys.readouterr().err
    assert "artifact rỗng" in err or "0 dòng" in err


def test_missing_baseline_file_exits_nonzero(kpi, capsys, tmp_path):
    """Baseline dir rỗng -> lỗi rõ ràng, không trả về 0."""
    empty = tmp_path / "empty_baseline"
    empty.mkdir()
    code = kpi.main(["k5", "--baseline-dir", str(empty)])
    assert code != 0
    err = capsys.readouterr().err
    assert "không tìm thấy" in err or "file không tồn tại" in err


def test_unified_table_has_kpi_column_headers(kpi, capsys):
    """'all' in bảng KPI tổng hợp nhất quán: KPI | Giá trị | Nguồn | Ngày đo."""
    code = kpi.main(["all"])
    assert code == 0
    out = capsys.readouterr().out
    assert "BẢNG KPI TỔNG HỢP" in out
    assert "KPI" in out and "Giá trị" in out and "Nguồn" in out and "Ngày đo" in out
    for k in ("K1", "K2", "K3", "K4", "K5"):
        assert k in out


def test_kpi_table_values_not_fabricated(kpi, capsys):
    """Bảng tổng hợp: K2/K5 phải in 'chưa đo được' (chưa phải 99.3%/77.5%);
    K3/K4 cũng 'chưa đo được'; KHÔNG in số KPI bịa."""
    code = kpi.main(["all"])
    assert code == 0
    out = capsys.readouterr().out
    # Trong BẢNG TỔNG HỢP (phân khúc phía sau), mỗi dòng KPI là một line
    table = out.split("BẢNG KPI TỔNG HỢP", 1)[1]
    # K2/K5 trong bảng phải là 'chưa đo được' (không gán số)
    k2_line = next(l for l in table.splitlines() if l.startswith("K2 "))
    k5_line = next(l for l in table.splitlines() if l.startswith("K5 "))
    assert "chưa đo được" in k2_line
    assert "chưa đo được" in k5_line
    # K3: probe log trống -> 'chưa đo được'; K4: tỷ lệ thật đọc
    # từ queue.jsonl (KHÔNG bịa) — khớp mẫu 'a/b = x%'
    k3_line = next(l for l in table.splitlines() if l.startswith("K3 "))
    assert "chưa đo được" in k3_line
    k4_line = next(l for l in table.splitlines() if l.startswith("K4 "))
    assert "K4 (trong 48h / tạo trong tuần)" in k4_line
    assert re.search(r"\d+/\d+ = ", k4_line)


def test_k1_prints_template_and_process(kpi, capsys):
    """K1 là quy trình tay -> in bảng template, không in giá trị số."""
    code = kpi.main(["k1"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Bảng ghi" in out
    assert "STT" in out
    assert "chiến_lược" in out and "hoc_phi" in out
    assert "%" not in out


def test_out_dir_writes_json(kpi, tmp_path):
    """--out-dir ghi bảng KPI dưới dạng JSON hợp lệ (5 dòng KPI)."""
    out = tmp_path / "out"
    code = kpi.main(["all", "--out-dir", str(out)])
    assert code == 0
    files = list(out.glob("kpi_report_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 5
    by_name = {d["kpi"]: d for d in data}
    # K4 là tỷ lệ thật từ queue.jsonl (KHÔNG bịa); K2/K5 'chưa đo được'
    assert re.search(r"\d+/\d+ = ", by_name["K4 Hàng đợi unanswered xử lý trong 48h"]["value"])
    assert "chưa đo được" in by_name["K2 Tỷ lệ bot trả lời có trích dẫn"]["value"]
    assert "chưa đo được" in by_name["K5 Đúng trên bộ câu vàng"]["value"]


def test_helpers_detect_fallback_and_measurable(kpi):
    """Pin hành vi helper: fallback/empty -> không đo; answer thật + label -> đo."""
    assert kpi._is_fallback_answer("dịch vụ tạo câu trả lời đang tạm thời không khả dụng")
    assert kpi._is_fallback_answer("")
    assert kpi._is_fallback_answer(None)
    assert not kpi._is_fallback_answer("HUSC có 2 cơ sở.")

    # degraded: toàn bộ fallback / None -> không đo
    degraded = [
        {"answer": "dịch vụ tạo câu trả lời đang tạm thời không khả dụng",
         "expected_behavior": None, "judge_scores": None, "retrieved_chunks": ["a"]},
        {"answer": "dịch vụ tạo câu trả lời đang tạm thời không khả dụng",
         "expected_behavior": None, "judge_scores": None, "retrieved_chunks": []},
    ]
    assert kpi._has_measurable_answers(degraded) is False

    # có 1 câu thật + expected_behavior -> đo
    measurable = degraded + [
        {"answer": "HUSC có 2 cơ sở.", "expected_behavior": "answer",
         "judge_scores": {"correct": True}, "retrieved_chunks": ["a"]},
    ]
    assert kpi._has_measurable_answers(measurable) is True

    # judge schema
    assert kpi._is_judged_correct({"correct": True}) is True
    assert kpi._is_judged_correct({"correct": False}) is False
    assert kpi._is_judged_correct({"score": 0.8}) is True
    assert kpi._is_judged_correct({"score": 0.2}) is False
    with pytest.raises(Exception):
        kpi._is_judged_correct(None)  # None -> không đoán (nâng lỗi rõ)

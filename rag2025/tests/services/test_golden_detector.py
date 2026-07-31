"""Gate G2/G2b/G2c — golden set 40 on supersession_detector.detect()."""
from __future__ import annotations

import json
from pathlib import Path

from src.services.supersession_detector import detect

_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "legal_vn" / "golden_40.jsonl"
)

_GOLDEN: list[dict] = []
with open(_FIXTURE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            _GOLDEN.append(json.loads(line))


def test_golden_40_gates():
    tp = fp = fn = tn = 0
    g2c_fp = 0

    for row in _GOLDEN:
        result = detect(row["text"], source_doc_id="golden")
        hits = len(result) > 0
        if row["expect_positive"] and hits:
            tp += 1
        elif not row["expect_positive"] and hits:
            fp += 1
            if "Căn cứ" in row["note"] or "can_cu" in row["note"]:
                g2c_fp += 1
        elif row["expect_positive"] and not hits:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"precision={precision:.3f} recall={recall:.3f}")

    assert precision >= 0.95
    assert recall >= 0.80
    assert g2c_fp == 0

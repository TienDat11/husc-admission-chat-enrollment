"""§PL-6.2 admission roadmap 5-step state machine tests (pure)."""
from __future__ import annotations

from services.admission_roadmap import build_admission_roadmap


def _statuses(steps):
    return [s.status for s in steps]


def test_all_done_when_complete():
    profile = {
        "raw_score": 24.0,
        "method": "thi_thpt",
        "to_hop": "A00",
        "target_majors": ["CNTT"],
    }
    steps = build_admission_roadmap(profile)
    assert _statuses(steps) == ["done", "done", "done", "done", "done"]
    assert steps[4].needs_input["field"] is None


def test_first_missing_is_needs_input_rest_blocked():
    profile = {"raw_score": 20.0}  # only step 1 done
    steps = build_admission_roadmap(profile)
    assert _statuses(steps) == ["done", "needs_input", "blocked", "blocked", "blocked"]
    assert steps[1].needs_input["field"] == "method"
    assert "phương thức" in steps[1].needs_input["hint"]


def test_empty_profile_all_needs_input_then_blocked():
    steps = build_admission_roadmap({})
    assert _statuses(steps) == [
        "needs_input", "blocked", "blocked", "blocked", "blocked",
    ]
    assert steps[0].needs_input["field"] == "raw_score"


def test_missing_target_majors_blocks_step4():
    profile = {"raw_score": 20.0, "method": "thi_thpt", "to_hop": "A00"}
    steps = build_admission_roadmap(profile)
    assert _statuses(steps) == ["done", "done", "done", "needs_input", "blocked"]
    assert steps[3].needs_input["field"] == "target_majors"


def test_step5_source_from_thoi_gian():
    profile = {
        "raw_score": 20.0, "method": "thi_thpt", "to_hop": "A00",
        "target_majors": ["CNTT"],
    }
    steps = build_admission_roadmap(
        profile, thoi_gian_source={"document_id": "snap:R1", "page": 2025}
    )
    assert steps[4].status == "done"
    assert steps[4].source == {"document_id": "snap:R1", "page": 2025}


def test_keys_and_fields():
    steps = build_admission_roadmap({"raw_score": 10.0})
    assert [s.n for s in steps] == [1, 2, 3, 4, 5]
    assert [s.key for s in steps] == [
        "hoso_diem", "chon_phuong_thuc", "chon_to_hop",
        "so_diem_chuan", "moc_thoi_gian",
    ]
    # every step carries needs_input {field, hint}
    for s in steps:
        assert set(s.needs_input.keys()) == {"field", "hint"}

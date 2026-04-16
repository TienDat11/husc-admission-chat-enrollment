"""Tests for eval_core module."""
import json
import pytest
from pathlib import Path

# Import from the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from notebooks.eval_core import load_test_questions, normalize_pipeline_output, should_abort_after_smoke, call_pipeline, PipelineError


# =============================================================================
# Tests for load_test_questions
# =============================================================================

def test_load_test_questions_uses_fallback_when_primary_missing(tmp_path):
    """When primary path does not exist, should fall back to fallback_path."""
    primary = tmp_path / "missing.json"
    fallback = tmp_path / "fallback.json"
    fallback.write_text(
        '[{"question":"q","ground_truth_answer":"a","category":"simple"}]',
        encoding="utf-8"
    )
    rows, used_path = load_test_questions(str(primary), str(fallback))
    assert len(rows) == 1
    assert rows[0]["question"] == "q"
    assert used_path == str(fallback)


def test_load_test_questions_uses_fallback_when_primary_invalid_json(tmp_path):
    """When primary path exists but has invalid JSON, should fall back to fallback_path."""
    primary = tmp_path / "invalid.json"
    fallback = tmp_path / "fallback.json"
    primary.write_text("not valid json {", encoding="utf-8")
    fallback.write_text(
        '[{"question":"fallback_q","ground_truth_answer":"fallback_a","category":"simple"}]',
        encoding="utf-8"
    )
    rows, used_path = load_test_questions(str(primary), str(fallback))
    assert len(rows) == 1
    assert rows[0]["question"] == "fallback_q"
    assert used_path == str(fallback)


def test_load_test_questions_uses_primary_when_valid(tmp_path):
    """When primary path exists and has valid JSON, should use primary."""
    primary = tmp_path / "primary.json"
    fallback = tmp_path / "fallback.json"
    primary.write_text(
        '[{"question":"primary_q","ground_truth_answer":"primary_a","category":"primary_cat"}]',
        encoding="utf-8"
    )
    fallback.write_text(
        '[{"question":"fallback_q","ground_truth_answer":"fallback_a","category":"fallback_cat"}]',
        encoding="utf-8"
    )
    rows, used_path = load_test_questions(str(primary), str(fallback))
    assert len(rows) == 1
    assert rows[0]["question"] == "primary_q"
    assert rows[0]["category"] == "primary_cat"
    assert used_path == str(primary)


def test_load_test_questions_raises_when_both_fail(tmp_path):
    """When both primary and fallback fail, should raise FileNotFoundError."""
    primary = tmp_path / "missing.json"
    fallback = tmp_path / "also_missing.json"
    # Both paths don't exist
    with pytest.raises(FileNotFoundError) as exc_info:
        load_test_questions(str(primary), str(fallback))
    assert "neither" in str(exc_info.value).lower() or "both" in str(exc_info.value).lower()


# =============================================================================
# Tests for normalize_pipeline_output
# =============================================================================

def test_normalize_pipeline_output_has_required_keys():
    """Output must contain all required keys."""
    raw = {"answer": "ok", "sources": ["s1"], "confidence": 0.9}
    out = normalize_pipeline_output(raw, mode="v2")
    required_keys = {
        "answer", "context_chunks", "source_ids",
        "confidence", "groundedness_score", "route", "raw"
    }
    assert required_keys.issubset(out.keys())


def test_normalize_pipeline_output_v2_sources_mapping():
    """For mode='v2', sources should map to source_ids."""
    raw = {
        "answer": "the answer is 42",
        "sources": ["src1", "src2", "src3"],
        "confidence": 0.95,
        "chunks": [
            {"text": "chunk one"},
            {"text": "chunk two"}
        ]
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["source_ids"] == ["src1", "src2", "src3"]
    assert out["context_chunks"] == [{"text": "chunk one"}, {"text": "chunk two"}]
    assert out["confidence"] == 0.95
    assert out["answer"] == "the answer is 42"


def test_normalize_pipeline_output_v1_chunks_mapping():
    """For mode='v1', chunks should map to context_chunks."""
    raw = {
        "answer": "v1 answer",
        "chunks": [
            {"text": "first chunk"},
            {"text": "second chunk"},
            {"text": "third chunk"}
        ],
        "confidence": 0.85
    }
    out = normalize_pipeline_output(raw, mode="v1")
    assert out["context_chunks"] == [{"text": "first chunk"}, {"text": "second chunk"}, {"text": "third chunk"}]
    # v1 mode should NOT have source_ids mapping from sources
    assert out["source_ids"] == []


def test_normalize_pipeline_output_v2_missing_sources():
    """When sources key is missing in v2 mode, source_ids should be empty list."""
    raw = {
        "answer": "no sources here",
        "confidence": 0.5
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["source_ids"] == []
    assert out["context_chunks"] == []


def test_normalize_pipeline_output_preserves_raw():
    """The 'raw' key should contain the original input data."""
    raw = {
        "answer": "test answer",
        "some_field": "original value",
        "confidence": 0.7
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["raw"] == raw
    assert out["raw"]["some_field"] == "original value"


def test_normalize_pipeline_output_has_groundedness_score():
    """groundedness_score should be present (extracted from raw or defaulted)."""
    raw = {
        "answer": "scored answer",
        "groundedness_score": 0.88,
        "sources": ["s1"]
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert "groundedness_score" in out
    assert out["groundedness_score"] == 0.88


def test_normalize_pipeline_output_has_route():
    """route should be present in output."""
    raw = {
        "answer": "routed answer",
        "route": "enrollment",
        "sources": ["s1"]
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert "route" in out
    assert out["route"] == "enrollment"


def test_normalize_pipeline_output_defaults_groundedness_and_route():
    """When raw has no groundedness_score or route, should use defaults."""
    raw = {
        "answer": "minimal input",
        "sources": ["s1"]
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert "groundedness_score" in out
    assert "route" in out
    # Defaults should be 0.0 and ""
    assert out["groundedness_score"] == 0.0
    assert out["route"] == ""


def test_normalize_pipeline_output_invalid_mode_treated_as_v1():
    """When mode is unknown, should fall back to v1 behavior (no source_ids mapping)."""
    raw = {
        "answer": "unknown mode",
        "sources": ["s1", "s2"],
        "chunks": [{"text": "chunk"}],
    }
    out = normalize_pipeline_output(raw, mode="unknown")
    # v1 fallback: source_ids should not map from sources
    assert out["source_ids"] == []
    # chunks should still map to context_chunks
    assert out["context_chunks"] == [{"text": "chunk"}]


def test_normalize_pipeline_output_sources_string_not_list():
    """When sources is a string (not a list), source_ids should be empty."""
    raw = {
        "answer": "string sources",
        "sources": "src1",  # string instead of list
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["source_ids"] == []


def test_normalize_pipeline_output_sources_dict_not_list():
    """When sources is a dict (not a list), source_ids should be empty."""
    raw = {
        "answer": "dict sources",
        "sources": {"id": "src1"},  # dict instead of list
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["source_ids"] == []


def test_normalize_pipeline_output_chunks_string_not_list():
    """When chunks is a string (not a list), context_chunks should be empty."""
    raw = {
        "answer": "string chunks",
        "chunks": "chunk1",  # string instead of list
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["context_chunks"] == []


def test_normalize_pipeline_output_chunks_dict_not_list():
    """When chunks is a dict (not a list), context_chunks should be empty."""
    raw = {
        "answer": "dict chunks",
        "chunks": {"text": "single chunk"},  # dict instead of list
    }
    out = normalize_pipeline_output(raw, mode="v2")
    assert out["context_chunks"] == []


# =============================================================================
# Tests for should_abort_after_smoke
# =============================================================================

def test_should_abort_after_smoke_when_failures_exceed_half():
    assert should_abort_after_smoke(total=10, failures=6) is True
    assert should_abort_after_smoke(total=10, failures=5) is False
    assert should_abort_after_smoke(total=0, failures=0) is True
    assert should_abort_after_smoke(total=5, failures=3) is True


def test_should_abort_after_smoke_edge_cases():
    """Edge cases: total <= 0 always aborts."""
    assert should_abort_after_smoke(total=0, failures=0) is True
    assert should_abort_after_smoke(total=-1, failures=0) is True


def test_should_abort_after_smoke_at_exactly_half():
    """At exactly 50% failure rate, should NOT abort."""
    assert should_abort_after_smoke(total=4, failures=2) is False


def test_should_abort_after_smoke_all_failures():
    """100% failure rate should abort."""
    assert should_abort_after_smoke(total=3, failures=3) is True


def test_should_abort_after_smoke_no_failures():
    """0% failure rate should not abort."""
    assert should_abort_after_smoke(total=10, failures=0) is False


# =============================================================================
# Tests for call_pipeline
# =============================================================================

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
    def json(self):
        return self._json
    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


class TestCallPipeline:

    def test_call_pipeline_v1_mode(self, monkeypatch):
        """mode='v1' should POST to /query with force_rag_only=False."""
        recorded = {}
        def mock_post(url, json, timeout):
            recorded["url"] = url
            recorded["json"] = json
            recorded["timeout"] = timeout
            return MockResponse({"answer": "v1 result"})
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        result = call_pipeline("http://localhost:8000", "test query", mode="v1")
        assert result == {"answer": "v1 result"}
        assert recorded["url"] == "http://localhost:8000/query"
        assert recorded["json"] == {"query": "test query", "force_rag_only": False}
        assert recorded["timeout"] == 120

    def test_call_pipeline_v2_mode(self, monkeypatch):
        """mode='v2' should POST to /v2/query with top_k."""
        recorded = {}
        def mock_post(url, json, timeout):
            recorded["url"] = url
            recorded["json"] = json
            return MockResponse({"answer": "v2 result"})
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        result = call_pipeline("http://localhost:8000", "test query", mode="v2", top_k=10)
        assert result == {"answer": "v2 result"}
        assert recorded["url"] == "http://localhost:8000/v2/query"
        assert recorded["json"] == {"query": "test query", "top_k": 10}

    def test_call_pipeline_v2_default_top_k(self, monkeypatch):
        """v2 mode should default top_k to 5."""
        recorded = {}
        def mock_post(url, json, timeout):
            recorded["json"] = json
            return MockResponse({})
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        call_pipeline("http://localhost:8000", "test query", mode="v2")
        assert recorded["json"]["top_k"] == 5

    def test_call_pipeline_raises_on_error(self, monkeypatch):
        """Should raise for HTTP error responses."""
        def mock_post(url, json, timeout):
            return MockResponse({}, status_code=500)
        import requests
        monkeypatch.setattr(requests, "post", mock_post)
        with pytest.raises(requests.HTTPError):
            call_pipeline("http://localhost:8000", "test query", mode="v2")


    def test_call_pipeline_raises_pipeline_error_on_connection_error(self, monkeypatch):
        """Should raise PipelineError on connection error."""
        import requests
        def mock_post(url, json, timeout):
            raise requests.ConnectionError('Connection refused')
        monkeypatch.setattr(requests, 'post', mock_post)
        with pytest.raises(PipelineError) as exc_info:
            call_pipeline('http://localhost:8000', 'test query', mode='v2')
        assert 'Failed to connect' in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_call_pipeline_raises_pipeline_error_on_non_json_response(self, monkeypatch):
        """Should raise PipelineError when response is not JSON."""
        class NonJsonResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                raise json.JSONDecodeError('Expecting value', '', 0)
        def mock_post(url, json, timeout):
            return NonJsonResponse()
        import requests
        monkeypatch.setattr(requests, 'post', mock_post)
        with pytest.raises(PipelineError) as exc_info:
            call_pipeline('http://localhost:8000', 'test query', mode='v2')
        assert 'non-JSON' in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

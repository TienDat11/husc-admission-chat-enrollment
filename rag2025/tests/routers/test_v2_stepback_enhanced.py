"""Tests for enhanced _parse_step_back_response + RouterResult new fields.

Pure unit tests — no live LLM. Validates the JSON parser + dataclass shape.
"""
import json

import pytest

from src.services.query_router import _parse_step_back_response
from src.services.query_router import RouterResult
from src.services.query_router import QueryRoute


# ----- JSON happy path (5) -----

class TestParseJsonHappyPath:
    def test_basic_json(self):
        raw = '{"step_back": "abstracted", "sub_queries": ["q1"], "guardrail_signals": {}}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "abstracted"
        assert subs == ["q1"]
        assert isinstance(gs, dict)

    def test_full_json(self):
        raw = json.dumps({
            "step_back": "abstracted Q",
            "sub_queries": ["q1", "q2", "q3"],
            "guardrail_signals": {
                "vague": True,
                "abusive": False,
                "off_topic": False,
                "abbreviation_detected": ["hp", "sv"],
                "rewritten_query": "học phí sinh viên",
            }
        })
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "abstracted Q"
        assert subs == ["q1", "q2", "q3"]
        assert gs["vague"] is True
        assert gs["abusive"] is False
        assert gs["abbreviation_detected"] == ["hp", "sv"]
        assert gs["rewritten_query"] == "học phí sinh viên"

    def test_empty_sub_queries(self):
        raw = '{"step_back": "x", "sub_queries": []}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "x"
        assert subs == []

    def test_only_step_back(self):
        raw = '{"step_back": "just one"}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "just one"
        assert subs == []
        assert gs["vague"] is False

    def test_markdown_code_fence_stripped(self):
        raw = '```json\n{"step_back": "x", "sub_queries": ["a"]}\n```'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "x"
        assert subs == ["a"]


# ----- JSON with weird whitespace (2) -----

class TestParseJsonWhitespace:
    def test_extra_spaces(self):
        raw = '   {"step_back": "trimmed", "sub_queries": ["a"]}   '
        sb, subs, gs = _parse_step_back_response(raw)
        # parser strips outer whitespace of the raw, not inner field values
        assert sb.strip() == "trimmed"
        assert subs == ["a"]

    def test_newlines_in_json(self):
        raw = '{\n  "step_back": "multiline",\n  "sub_queries": ["q1", "q2"]\n}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "multiline"
        assert subs == ["q1", "q2"]


# ----- Non-JSON fallback (3) -----

class TestParseNonJsonFallback:
    def test_plain_text(self):
        raw = "Just a plain Vietnamese sentence without JSON structure."
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == raw
        assert subs == []
        assert gs["vague"] is False
        assert gs["abusive"] is False
        assert gs["off_topic"] is False
        assert gs["abbreviation_detected"] == []
        assert gs["rewritten_query"] == ""

    def test_empty_string(self):
        sb, subs, gs = _parse_step_back_response("")
        assert sb == ""
        assert subs == []
        assert isinstance(gs, dict)

    def test_garbage(self):
        raw = "@@@!!! not json at all ###"
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == raw
        assert subs == []


# ----- Partial JSON (2) -----

class TestParsePartialJson:
    def test_missing_guardrail_signals(self):
        raw = '{"step_back": "x", "sub_queries": ["a"]}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == "x"
        assert subs == ["a"]
        # defaults filled
        assert gs["vague"] is False
        assert gs["abusive"] is False

    def test_missing_step_back(self):
        raw = '{"sub_queries": ["a"]}'
        sb, subs, gs = _parse_step_back_response(raw)
        # falls back to raw
        assert "sub_queries" in sb


# ----- Invalid JSON structures (3) -----

class TestParseInvalidStructures:
    def test_sub_queries_not_list(self):
        raw = '{"step_back": "x", "sub_queries": "not a list"}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert subs == []

    def test_guardrail_signals_not_dict(self):
        raw = '{"step_back": "x", "guardrail_signals": "not a dict"}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert isinstance(gs, dict)
        assert gs["vague"] is False

    def test_data_not_dict(self):
        raw = '[1, 2, 3]'
        sb, subs, gs = _parse_step_back_response(raw)
        assert sb == raw
        assert subs == []


# ----- RouterResult field round-trip (5) -----

class TestRouterResultNewFields:
    def _make(self, **overrides):
        base = dict(
            original_query="orig",
            step_back_query="step",
            hypothetical_doc="",
            hyde_variants=[],
            route=QueryRoute.PADDED_RAG,
            complexity=1,
            intent="general",
            reasoning="",
        )
        base.update(overrides)
        return RouterResult(**base)

    def test_sub_queries_field(self):
        r = self._make(sub_queries=["a", "b"])
        assert r.sub_queries == ["a", "b"]

    def test_guardrail_signals_field(self):
        r = self._make(guardrail_signals={"vague": True})
        assert r.guardrail_signals["vague"] is True

    def test_rewritten_query_field(self):
        r = self._make(rewritten_query="rewritten")
        assert r.rewritten_query == "rewritten"

    def test_all_three_fields(self):
        r = self._make(
            sub_queries=["q1", "q2"],
            guardrail_signals={"vague": True, "rewritten_query": "rw"},
            rewritten_query="rw",
        )
        assert r.sub_queries == ["q1", "q2"]
        assert r.guardrail_signals["vague"] is True
        assert r.rewritten_query == "rw"

    def test_defaults_none(self):
        r = self._make()
        assert r.sub_queries is None
        assert r.guardrail_signals is None
        assert r.rewritten_query is None or r.rewritten_query == ""


# ----- Coercion edge cases (3) -----

class TestParseCoercion:
    def test_sub_queries_with_non_strings_filtered(self):
        # parser doesn't filter — it just trusts the LLM
        raw = '{"step_back": "x", "sub_queries": ["a", 1, null, "b"]}'
        sb, subs, gs = _parse_step_back_response(raw)
        # what does the parser do? — accepts as-is
        assert subs == ["a", 1, None, "b"]

    def test_abbreviation_detected_with_non_list(self):
        raw = '{"step_back": "x", "guardrail_signals": {"abbreviation_detected": "hp"}}'
        sb, subs, gs = _parse_step_back_response(raw)
        # not list → default empty
        assert gs["abbreviation_detected"] == []

    def test_rewritten_query_non_string_coerced(self):
        raw = '{"step_back": "x", "guardrail_signals": {"rewritten_query": 42}}'
        sb, subs, gs = _parse_step_back_response(raw)
        # int → str
        assert gs["rewritten_query"] == "42"


# ----- guardrail_signals default values (4) -----

class TestGuardrailSignalsDefaults:
    def test_all_default_false(self):
        sb, subs, gs = _parse_step_back_response('{"step_back":"x"}')
        assert gs["vague"] is False
        assert gs["abusive"] is False
        assert gs["off_topic"] is False

    def test_abbreviation_default_empty(self):
        sb, subs, gs = _parse_step_back_response('{"step_back":"x"}')
        assert gs["abbreviation_detected"] == []

    def test_rewritten_query_default_empty(self):
        sb, subs, gs = _parse_step_back_response('{"step_back":"x"}')
        assert gs["rewritten_query"] == ""

    def test_partial_signals_default_rest(self):
        raw = '{"step_back":"x","guardrail_signals":{"vague":true}}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert gs["vague"] is True
        assert gs["abusive"] is False  # default


# ----- Abbreviation detection (3) -----

class TestAbbreviationDetection:
    def test_hp_detected(self):
        raw = '{"step_back":"x","guardrail_signals":{"abbreviation_detected":["hp"]}}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert "hp" in gs["abbreviation_detected"]

    def test_multiple_abbreviations(self):
        raw = '{"step_back":"x","guardrail_signals":{"abbreviation_detected":["hp","sv","nganh","dh"]}}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert set(gs["abbreviation_detected"]) == {"hp", "sv", "nganh", "dh"}

    def test_no_abbreviations(self):
        raw = '{"step_back":"x","guardrail_signals":{"abbreviation_detected":[]}}'
        sb, subs, gs = _parse_step_back_response(raw)
        assert gs["abbreviation_detected"] == []

"""Tests for chunker_3way (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "chunker_3way.py"


@pytest.fixture
def chunker3():
    spec = importlib.util.spec_from_file_location("chunker_3way", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_chunker_json_strips_fences(chunker3):
    text = "```json\n{\"chunks\": [{\"text\": \"a\"}]}\n```"
    chunks = chunker3._parse_chunker_json(text)
    assert chunks == [{"text": "a"}]


def test_parse_chunker_json_raises_on_invalid(chunker3):
    with pytest.raises(json.JSONDecodeError):
        chunker3._parse_chunker_json("not json")


def test_parse_chunker_json_raises_on_missing_chunks_key(chunker3):
    with pytest.raises(ValueError, match="chunks"):
        chunker3._parse_chunker_json('{"foo": "bar"}')


def test_system_v2_runs_in_process(chunker3):
    chunks = chunker3._run_system_v2_chunker("para1\n\npara2")
    assert len(chunks) == 2
    assert chunks[0]["text"] == "para1"
    assert chunks[0]["info_type"] == "unknown"


def test_haiku_call_uses_seed_temp_top_p(chunker3):
    """Verify the Haiku API call payload has temperature=0, top_p=1, model=haiku-4-5."""
    captured = {}

    def fake_post(self, url, headers=None, json=None, **_):
        captured.update(json)
        captured["__url__"] = url
        captured["__api_key__"] = (headers or {}).get("x-api-key")
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json = lambda: {"content": [{"text": '{"chunks":[{"text":"a"}]}'}]}
        return resp

    with patch.object(chunker3.httpx.Client, "post", new=fake_post):
        out = chunker3._run_haiku_chunker("<html/>", "PROMPT", api_key="k123")

    assert out == [{"text": "a"}]
    assert captured["temperature"] == 0
    assert captured["top_p"] == 1
    assert captured["model"] == chunker3.HAIKU_MODEL
    assert captured["__api_key__"] == "k123"
    assert "claude-haiku-4-5" in captured["model"]


def test_claude_cli_invocation_args_correct(chunker3):
    """Verify the Claude CLI subprocess args."""
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["input"] = kw.get("input", "")
        return MagicMock(returncode=0, stdout='{"result":"{\\"chunks\\":[{\\"text\\":\\"c\\"}]}"}', stderr="")

    with patch.object(chunker3.subprocess, "run", side_effect=fake_run):
        out = chunker3._run_claude_cli_chunker("<html/>", "PROMPT")

    assert out == [{"text": "c"}]
    cmd = captured["cmd"]
    assert "claude" in cmd[0] or cmd[0] == "claude"
    assert "--print" in cmd
    assert "--output-format" in cmd
    assert "json" in cmd
    assert "--model" in cmd
    assert chunker3.CLAUDE_CLI_MODEL in cmd
    assert "--permission-mode" in cmd
    assert "acceptEdits" in cmd


def test_run_3way_writes_three_jsonl_files(chunker3, tmp_path):
    fake_chunks = [{"text": "a", "info_type": "unknown"}]
    res = chunker3.run_3way(
        nid=74,
        html_text="dummy",
        output_dir=tmp_path,
        prompt_haiku="P_HAIKU",
        prompt_claude="P_CLAUDE",
        api_key="k",
        haiku_runner=lambda html, p: fake_chunks,
        claude_runner=lambda html, p: fake_chunks,
        system_runner=lambda html: fake_chunks,
    )
    assert (tmp_path / "74" / "system_v2.jsonl").exists()
    assert (tmp_path / "74" / "haiku_v1.jsonl").exists()
    assert (tmp_path / "74" / "claude_v1.jsonl").exists()
    assert (tmp_path / "74" / "manifest.sha256").exists()
    assert res["lanes"] == {"system_v2": 1, "haiku_v1": 1, "claude_v1": 1}
    assert res["errors"] == {}


def test_run_3way_manifest_has_three_lines(chunker3, tmp_path):
    fake_chunks = [{"text": "a"}]
    chunker3.run_3way(
        74, "x", tmp_path, "p1", "p2",
        api_key="k",
        haiku_runner=lambda h, p: fake_chunks,
        claude_runner=lambda h, p: fake_chunks,
        system_runner=lambda h: fake_chunks,
    )
    manifest = (tmp_path / "74" / "manifest.sha256").read_text(encoding="utf-8")
    lines = [l for l in manifest.splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        sha, name = line.split("  ")
        assert len(sha) == 64
        assert name in {"system_v2.jsonl", "haiku_v1.jsonl", "claude_v1.jsonl"}


def test_run_3way_continues_on_lane_failure(chunker3, tmp_path):
    def bad_haiku(html, prompt):
        raise RuntimeError("api down")
    res = chunker3.run_3way(
        74, "x", tmp_path, "p1", "p2",
        api_key="k",
        haiku_runner=bad_haiku,
        claude_runner=lambda h, p: [{"text": "a"}],
        system_runner=lambda h: [{"text": "b"}],
    )
    assert "haiku_v1" in res["errors"]
    assert "api down" in res["errors"]["haiku_v1"]
    assert (tmp_path / "74" / "haiku_v1.jsonl").exists()  # Error file written


def test_jsonl_outputs_sorted_keys_for_idempotency(chunker3, tmp_path):
    """Sorting keys ensures byte-equal output across runs (idempotency prerequisite)."""
    chunks = [{"info_type": "unknown", "text": "a", "suggested_title": "T"}]
    chunker3.run_3way(
        74, "x", tmp_path, "p1", "p2",
        api_key="k",
        haiku_runner=lambda h, p: chunks,
        claude_runner=lambda h, p: chunks,
        system_runner=lambda h: chunks,
    )
    text = (tmp_path / "74" / "haiku_v1.jsonl").read_text(encoding="utf-8")
    parsed = json.loads(text.strip())
    keys = list(parsed.keys())
    assert keys == sorted(keys)

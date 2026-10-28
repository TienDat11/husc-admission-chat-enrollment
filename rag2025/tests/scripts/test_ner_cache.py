"""
TDD: NER disk cache for src/services/ner_service.py.

Why: User is out of ramclouds quota. We must NOT re-NER 1575 chunks.
Two contracts pinned here:

  A) Disk cache (data/graph/ner_cache.jsonl) — keyed by (chunk_id, text_hash).
     - Cache hit (chunk_id+text_hash match) → no LLM call.
     - Cache miss → NER, then PERSIST result if entities non-empty.
     - Chunk fails (LLM error, empty entities) → NOT cached (retry next run).

    (Groq fallback contract removed: NER now uses the same-gateway ns24 fallback.)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- Load ner_service.py directly (avoid package-init side effects) ---------
NER_SVC_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "services" / "ner_service.py"
)


def _load_ner_module():
    spec = importlib.util.spec_from_file_location("ner_service_for_test", NER_SVC_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ner_service_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


NER = _load_ner_module()
NERService = NER.NERService
NERCache = NER.NERCache


def _chunk(cid: str, text: str):
    from src.domain.entities import Chunk
    return Chunk(chunk_id=cid, text=text)


# ─── A) DISK CACHE ──────────────────────────────────────────────────────────


def test_cache_hit_skips_llm_call(tmp_path):
    """Same chunk_id + same text → 2nd call must NOT invoke LLM."""
    from src.domain.entities import Entity, EntityType, ExtractionResult

    cache_path = tmp_path / "ner_cache.jsonl"
    # Pre-populate cache for chunk_X with sha256(text)[:16]
    import hashlib

    text = "Công nghệ thông tin 400 chỉ tiêu tổ hợp A00"
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    record = {
        "chunk_id": "chunk_X",
        "text_hash": text_hash,
        "entities": [
            {"text": "Công nghệ thông tin", "type": "NGANH", "normalized": "cong_nghe_thong_tin"},
        ],
        "triples": [],
        "model": "ns24/gemini-3.5-flash-lite",
    }
    cache_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    # Build NERService with cache_path; mock LLM so any call would explode
    fake_llm = MagicMock()
    fake_llm.chat_json = AsyncMock(side_effect=AssertionError("LLM called on cache hit!"))

    svc = NERService(llm=fake_llm, cache_path=cache_path)
    res = asyncio_run(svc.extract(_chunk("chunk_X", text)))

    assert res.is_success
    assert len(res.entities) == 1
    assert res.entities[0].normalized == "cong_nghe_thong_tin"
    assert fake_llm.chat_json.await_count == 0, "LLM must NOT be called on cache hit"


def test_cache_miss_when_text_changes(tmp_path):
    """Same chunk_id but text edited → cache miss → must call LLM."""
    from src.domain.entities import EntityType

    cache_path = tmp_path / "ner_cache.jsonl"
    # Cache has OLD text hash
    import hashlib

    old_text = "old text"
    old_hash = hashlib.sha256(old_text.encode("utf-8")).hexdigest()[:16]
    cache_path.write_text(
        json.dumps({"chunk_id": "chunk_Y", "text_hash": old_hash, "entities": [], "triples": []}) + "\n",
        encoding="utf-8",
    )

    new_text = "new text differs"
    fake_llm = MagicMock()
    fake_llm.chat_json = AsyncMock(
        return_value={"entities": [{"text": "CNTT", "type": "NGANH", "normalized": "cntt"}], "triples": []}
    )

    svc = NERService(llm=fake_llm, cache_path=cache_path)
    res = asyncio_run(svc.extract(_chunk("chunk_Y", new_text)))

    assert res.is_success
    assert fake_llm.chat_json.await_count == 1, "Text changed → LLM must be called"
    # Cache must now have the NEW hash (entity was non-empty)
    lines = [l for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()[:16]
    assert any(json.loads(l).get("text_hash") == new_hash for l in lines), "new entry must be appended"


def test_cache_NOT_persisted_on_empty_entities(tmp_path):
    """When NER returns empty entities, do NOT cache (allow retry next run)."""
    cache_path = tmp_path / "ner_cache.jsonl"

    fake_llm = MagicMock()
    fake_llm.chat_json = AsyncMock(
        return_value={"entities": [], "triples": []}  # empty
    )

    svc = NERService(llm=fake_llm, cache_path=cache_path)
    res = asyncio_run(svc.extract(_chunk("chunk_Z", "some text that NER botched")))

    # Result may not be success (no entities) but cache must be untouched
    assert not res.is_success or len(res.entities) == 0
    if cache_path.exists():
        content = cache_path.read_text(encoding="utf-8").strip()
        assert content == "" or "chunk_Z" not in content, "empty/failed NER must not be cached"


def test_cache_appended_after_each_success(tmp_path):
    """Cache is incrementally persisted after EACH successful NER."""
    cache_path = tmp_path / "ner_cache.jsonl"

    fake_llm = MagicMock()
    fake_llm.chat_json = AsyncMock(
        return_value={
            "entities": [{"text": "A00", "type": "TO_HOP", "normalized": "a00"}],
            "triples": [],
        }
    )

    svc = NERService(llm=fake_llm, cache_path=cache_path)
    asyncio_run(svc.extract(_chunk("c1", "text one")))
    asyncio_run(svc.extract(_chunk("c2", "text two")))

    lines = [l for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    ids = {json.loads(l)["chunk_id"] for l in lines}
    assert ids == {"c1", "c2"}


# ─── C) UNICODE NORMALIZATION ─────────────────────────────────────────────────


def test_text_hash_normalizes_nfc_before_hashing():
    """NFC and NFD forms of same Vietnamese text produce identical _text_hash."""
    import unicodedata

    text = "Công nghệ thông tin"
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    assert nfc != nfd, "test precondition: NFC ≠ NFD for Vietnamese diacritic text"
    assert NER._text_hash(nfc) == NER._text_hash(nfd)


# ─── D) CORRUPT PERSISTENCE RECOVERY ──────────────────────────────────────────


def test_cache_load_mixed_valid_and_corrupt_lines(tmp_path):
    """Corrupt lines between valid entries are skipped; both valid entries survive."""
    import hashlib

    cache_path = tmp_path / "ner_cache.jsonl"
    hash_a = hashlib.sha256(b"text A").hexdigest()[:16]
    hash_b = hashlib.sha256(b"text B").hexdigest()[:16]
    lines = [
        json.dumps({"chunk_id": "cA", "text_hash": hash_a, "entities": [], "triples": [], "model": "t"}),
        "}bad json{\n",
        json.dumps({"chunk_id": "cB", "text_hash": hash_b, "entities": [], "triples": [], "model": "t"}),
    ]
    cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cache = NERCache(path=cache_path)
    assert cache.get("cA", "text A") is not None
    assert cache.get("cB", "text B") is not None


# ─── E) CONCURRENT SAME-KEY WRITES ────────────────────────────────────────────


def test_cache_put_concurrent_same_key(tmp_path):
    """Concurrent puts to same chunk_id: all lines survive, no corruption, index populated."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_path = tmp_path / "ner_cache.jsonl"
    cache = NERCache(path=cache_path)

    def _put(i: int) -> int:
        cache.put(
            "same_key", f"text_{i}",
            entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
            triples=[], model="t",
        )
        return i

    n = 10
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(_put, i) for i in range(n)]):
            f.result()  # re-raises if put() threw

    lines = [l for l in cache_path.read_text("utf-8").splitlines() if l.strip()]
    assert len(lines) == n, f"expect {n} lines, got {len(lines)}"
    for l in lines:
        rec = json.loads(l)
        assert rec["chunk_id"] == "same_key"
    assert "same_key" in cache._index


# ─── F) NFD LEGACY BACKWARD COMPAT ──────────────────────────────────────────


def test_legacy_nfd_hash_still_readable(tmp_path):
    """Legacy cache entry with NFD raw SHA256 hash found by NFC query."""
    import hashlib, unicodedata
    cache_path = tmp_path / "ner_cache.jsonl"
    text = "Công nghệ thông tin"
    nfd = unicodedata.normalize("NFD", text)
    legacy_hash = hashlib.sha256(nfd.encode("utf-8")).hexdigest()[:16]
    record = {
        "chunk_id": "cX", "text_hash": legacy_hash,
        "entities": [{"text": "CNTT", "type": "NGANH", "normalized": "cntt"}],
        "triples": [], "model": "t",
    }
    cache_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    cache = NERCache(path=cache_path)
    assert cache.get("cX", text) is not None  # NFC query should find NFD-stored hash


# ─── G) NON-DICT JSON SKIP ──────────────────────────────────────────────────


def test_cache_load_skips_non_dict_json_lines(tmp_path):
    """Valid JSON lines that are not dicts (null, [], string) skip without crash."""
    import hashlib
    cache_path = tmp_path / "ner_cache.jsonl"
    h = hashlib.sha256(b"text").hexdigest()[:16]
    lines = [
        "null\n",
        "[]\n",
        '"just a string"\n',
        json.dumps({"chunk_id": "c1", "text_hash": h, "entities": [], "triples": [], "model": "t"}),
    ]
    cache_path.write_text("".join(lines) + "\n", encoding="utf-8")
    cache = NERCache(path=cache_path)
    assert cache.get("c1", "text") is not None
    assert len(cache._index) == 1, "only the valid dict line should be indexed"




def test_compact_os_replace_failure_cleans_unique_temp(tmp_path):
    """Failed os.replace cleans unique temp and preserves parseable canonical cache."""
    cache_path = tmp_path / "ner_cache.jsonl"
    original = {"chunk_id": "keep", "text_hash": "hash", "entities": [], "triples": [], "model": "t"}
    cache_path.write_text(json.dumps(original) + "\n", encoding="utf-8")
    cache = NERCache(path=cache_path, max_entries=1)
    candidate = {"new": {"chunk_id": "new", "text_hash": "hash2", "entities": [], "triples": [], "model": "t"}}

    with patch.object(NER.os, "replace", side_effect=OSError("replace failed")):
        assert cache._compact(candidate) is False

    # No leftover .tmp files at all (unique temp cleaned up)
    assert not any(f.suffix == ".tmp" for f in tmp_path.iterdir())
    assert cache_path.read_text(encoding="utf-8") == json.dumps(original) + "\n"
    assert json.loads(cache_path.read_text(encoding="utf-8")) == original
    assert cache._index["keep"] == original
    assert "new" not in cache._index


def test_compact_success_uses_unique_temp_in_canonical_dir(tmp_path):
    """Successful compact: unique temp in canonical dir, no leftover .tmp files."""
    cache_path = tmp_path / "ner_cache.jsonl"
    cache = NERCache(path=cache_path)
    candidate = {"keep": {"chunk_id": "keep", "text_hash": "hash", "entities": [], "triples": [], "model": "t"}}

    assert cache._compact(candidate) is True
    # No leftover .tmp files
    assert not any(f.suffix == ".tmp" for f in tmp_path.iterdir())
    # Canonical file written correctly
    assert json.loads(cache_path.read_text(encoding="utf-8")) == candidate["keep"]


# ─── L) TEMP-WRITE & CONCURRENT SAFETY ──────────────────────────────────


def test_compact_temp_write_failure_cleans_up_temp_file(tmp_path):
    """When writing to temp file raises OSError, unique temp cleaned up, canonical untouched."""
    from unittest.mock import MagicMock

    cache_path = tmp_path / "ner_cache.jsonl"
    original = {"chunk_id": "keep", "text_hash": "hash", "entities": [], "triples": [], "model": "t"}
    cache_path.write_text(json.dumps(original) + "\n", encoding="utf-8")
    cache = NERCache(path=cache_path, max_entries=1)
    candidate = {"new": {"chunk_id": "new", "text_hash": "h2", "entities": [], "triples": [], "model": "t"}}

    fake_tmp = tmp_path / ".ner_cache_collision_test.tmp"
    fake_tmp.write_text("")
    mock_file = MagicMock()
    mock_file.name = str(fake_tmp)
    mock_file.write.side_effect = OSError("write failed")
    mock_file.fileno.return_value = 12345
    mock_file.flush = MagicMock()
    mock_file.close = MagicMock()
    # Ensure with-statement returns self, preserving fileno config
    mock_file.__enter__.return_value = mock_file

    with patch("tempfile.NamedTemporaryFile", return_value=mock_file):
        result = cache._compact(candidate)

    assert result is False, "compaction must return False on write failure"
    # Temp file cleaned up in finally
    assert not fake_tmp.exists(), f"temp file must be cleaned up: {fake_tmp}"
    # Canonical file unchanged
    assert json.loads(cache_path.read_text(encoding="utf-8"))["chunk_id"] == "keep"
    # No .tmp artifacts left
    assert not any(f.suffix == ".tmp" for f in tmp_path.iterdir())
    # Index unchanged
    assert "new" not in cache._index
    assert cache._index["keep"] == original


def test_cross_instance_concurrent_writes_preserve_both(tmp_path):
    """Two NERCache instances, concurrent writes/compact: final canonical file
    retains committed records from both. Stale instance-local lock loses entries
    because last os.replace() overwrites other instance's compact. Path-scoped
    lock with pre-write reload prevents the lost-update race."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cache_path = tmp_path / "ner_cache.jsonl"
    max_entries = 2

    # Pre-populate to capacity so the NEXT put triggers compaction.
    preload = NERCache(path=cache_path, max_entries=max_entries)
    preload.put("old_0", "t0",
                entities=[{"text": "E0", "type": "NGANH", "normalized": "e0"}],
                triples=[], model="t")
    preload.put("old_1", "t1",
                entities=[{"text": "E1", "type": "NGANH", "normalized": "e1"}],
                triples=[], model="t")
    assert len(preload._index) == max_entries
    del preload

    # Both instances load the same file → both carry the same stale _index
    # snapshot that doesn't include the other's concurrent write.
    cache_a = NERCache(path=cache_path, max_entries=max_entries)
    cache_b = NERCache(path=cache_path, max_entries=max_entries)

    barrier = threading.Barrier(2, timeout=10)

    def _write_a():
        barrier.wait()
        cache_a.put("k_a", "text_a",
                    entities=[{"text": "A", "type": "NGANH", "normalized": "a"}],
                    triples=[], model="t")

    def _write_b():
        barrier.wait()
        cache_b.put("k_b", "text_b",
                    entities=[{"text": "B", "type": "NGANH", "normalized": "b"}],
                    triples=[], model="t")

    with ThreadPoolExecutor(max_workers=2) as pool:
        fa = pool.submit(_write_a)
        fb = pool.submit(_write_b)
        for f in as_completed([fa, fb]):
            f.result()

    # Reload fresh from disk — both k_a and k_b must survive
    # (they are newer than old_0/old_1 and fit within max_entries=2).
    fresh = NERCache(path=cache_path, max_entries=max_entries)
    ids = set(fresh._index.keys())
    assert "k_a" in ids, f"k_a lost (ids={ids})"
    assert "k_b" in ids, f"k_b lost (ids={ids})"
    assert len(ids) == max_entries, f"expected {max_entries} entries, got {len(ids)}"


# ─── H) DISK APPEND FAILURE SAFETY ─────────────────────────────────────────


def test_disk_append_failure_does_not_poison_index(tmp_path):
    """When append raises OSError, _index must NOT contain the entry."""
    from unittest.mock import patch
    cache_path = tmp_path / "ner_cache.jsonl"
    cache = NERCache(path=cache_path)
    with patch.object(Path, "open", side_effect=OSError("disk full")):
        cache.put("cX", "text",
                   entities=[{"text": "CNTT", "type": "NGANH", "normalized": "cntt"}],
                   triples=[], model="t")
    assert "cX" not in cache._index


# ─── I) SAME-KEY CONCURRENT CONSISTENCY ────────────────────────────────────


def test_cache_concurrent_same_key_final_consistency(tmp_path):
    """Concurrent same-key writes: all lines valid, reloaded index consistent."""
    from concurrent.futures import ThreadPoolExecutor
    cache_path = tmp_path / "ner_cache.jsonl"
    cache = NERCache(path=cache_path)
    n = 20

    def _put(i: int):
        cache.put("k", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_put, range(n)))

    lines = [l for l in cache_path.read_text("utf-8").splitlines() if l.strip()]
    assert len(lines) == n
    for l in lines:
        rec = json.loads(l)
        assert isinstance(rec, dict), f"non-dict line: {l}"
        assert rec["chunk_id"] == "k"

    cache2 = NERCache(path=cache_path)
    assert "k" in cache2._index
    assert len(cache2._index["k"]["text_hash"]) == 16


# ─── J) BOUNDED EVICTION ───────────────────────────────────────────────────


def test_bounded_eviction_max_entries(tmp_path):
    """max_entries is enforced: oldest evicted on overflow."""
    cache = NERCache(path=tmp_path / "cache.jsonl", max_entries=3)
    for i in range(5):
        cache.put(f"c{i}", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")
    assert len(cache._index) <= 3
    assert "c0" not in cache._index
    assert "c1" not in cache._index


def test_bounded_eviction_persists_across_reload(tmp_path):
    """Evicted entries stay gone after reload."""
    cache_path = tmp_path / "cache.jsonl"
    cache = NERCache(path=cache_path, max_entries=3)
    for i in range(5):
        cache.put(f"c{i}", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")
    # New NERCache instance reloads from disk
    cache2 = NERCache(path=cache_path, max_entries=3)
    assert len(cache2._index) <= 3
    assert "c0" not in cache2._index
    assert "c1" not in cache2._index


def test_bounded_eviction_file_bounded_after_compaction(tmp_path):
    """After compaction, file has at most max_entries lines (no stale entries)."""
    cache_path = tmp_path / "cache.jsonl"
    cache = NERCache(path=cache_path, max_entries=3)
    for i in range(5):
        cache.put(f"c{i}", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")
    lines = [l for l in cache_path.read_text("utf-8").splitlines() if l.strip()]
    assert len(lines) <= 3


def test_max_entries_validation(tmp_path):
    """max_entries must be >= 1."""
    with pytest.raises((AssertionError, ValueError)):
        NERCache(path=tmp_path / "c.jsonl", max_entries=0)


def test_bounded_eviction_idempotent_reload_file(tmp_path):
    """File after eviction+compaction is valid: all lines parse as valid records."""
    cache_path = tmp_path / "cache.jsonl"
    cache = NERCache(path=cache_path, max_entries=3)
    for i in range(5):
        cache.put(f"c{i}", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")
    raw = cache_path.read_text("utf-8")
    lines = [l for l in raw.splitlines() if l.strip()]
    assert len(lines) <= 3
    for l in lines:
        rec = json.loads(l)
        assert isinstance(rec, dict)
        assert "chunk_id" in rec
        assert "text_hash" in rec


# ─── K) COMPACTION FAILURE SAFETY ──────────────────────────────────────────


def test_compaction_failure_preserves_index_and_disk_semantics(tmp_path):
    """When compaction os.replace fails, _index + file must remain unchanged."""
    from unittest.mock import patch

    cache_path = tmp_path / "cache.jsonl"
    cache = NERCache(path=cache_path, max_entries=3)
    for i in range(3):
        cache.put(f"c{i}", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")
    assert len(cache._index) == 3
    assert "c0" in cache._index

    # Attempt 4th entry — triggers compact. Mock os.replace to fail.
    with patch("os.replace", side_effect=OSError("permission denied")):
        cache.put("c3", "text_3",
                   entities=[{"text": "E3", "type": "NGANH", "normalized": "e3"}],
                   triples=[], model="t")

    # _index must preserve c0,c1,c2 — no eviction, no c3
    assert len(cache._index) == 3, "index must NOT evict or grow on compact failure"
    assert "c0" in cache._index
    assert "c3" not in cache._index

    # File must still have c0,c1,c2 lines only
    lines = [l for l in cache_path.read_text("utf-8").splitlines() if l.strip()]
    ids = {json.loads(l)["chunk_id"] for l in lines}
    assert ids == {"c0", "c1", "c2"}, f"file should contain original 3, got {ids}"

    # Reload from disk must reconstruct original index
    cache2 = NERCache(path=cache_path, max_entries=3)
    assert len(cache2._index) == 3, "reload must see full original index"
    assert "c0" in cache2._index


def test_reload_enforces_max_entries_and_compacts(tmp_path):
    """Load oversized file with max_entries: index + file ≤ max_entries."""
    cache_path = tmp_path / "cache.jsonl"
    # Pre-create 5 entries (no max_entries so put appends)
    cache = NERCache(path=cache_path)  # no max_entries
    for i in range(5):
        cache.put(f"c{i}", f"text_{i}",
                   entities=[{"text": f"E{i}", "type": "NGANH", "normalized": f"e{i}"}],
                   triples=[], model="t")
    assert len(cache._index) == 5

    # Reload with max_entries=3 — must compact
    cache2 = NERCache(path=cache_path, max_entries=3)
    assert len(cache2._index) <= 3, f"expected ≤3 after load, got {len(cache2._index)}"
    assert "c0" not in cache2._index
    assert "c1" not in cache2._index

    # File must have ≤3 lines
    lines = [l for l in cache_path.read_text("utf-8").splitlines() if l.strip()]
    assert len(lines) <= 3
    ids = {json.loads(l)["chunk_id"] for l in lines}
    assert "c0" not in ids
    assert "c1" not in ids

    # Triple-check: reload again confirms persistence
    cache3 = NERCache(path=cache_path, max_entries=3)
    assert len(cache3._index) <= 3
    assert "c0" not in cache3._index


# ─── helpers ────────────────────────────────────────────────────────────────


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)

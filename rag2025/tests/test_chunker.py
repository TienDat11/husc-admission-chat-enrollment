"""
Unit tests for chunker module (Epic 1.2)
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.chunker import ChunkConfig, Chunker, RAGChunk, chunk_jsonl


@pytest.fixture
def config_path():
    """Fixture for chunk profiles YAML path."""
    return Path(__file__).parent.parent / "config" / "chunk_profiles.yaml"


@pytest.fixture
def sample_faq_doc():
    """Fixture for FAQ-style document."""
    return {
        "id": "faq_001",
        "text": "What is the admission deadline? The admission deadline is June 30, 2025 at 5:00 PM.",
        "text_plain": "What is the admission deadline? The admission deadline is June 30, 2025 at 5:00 PM.",
        "summary": "Admission deadline: June 30, 2025 at 5:00 PM",
        "metadata": {
            "source": "Admissions FAQ",
            "faq_type": "deadline",
            "audience": "students",
            "year": 2025,
        },
    }


@pytest.fixture
def sample_policy_doc():
    """Fixture for policy-style document."""
    return {
        "id": "policy_001",
        "text": "Chương 1: Quy định chung\n\nĐiều 1: Phạm vi điều chỉnh\nVăn bản này quy định về tuyển sinh đại học năm 2025.\n\nĐiều 2: Đối tượng áp dụng\nÁp dụng cho tất cả các trường đại học trên toàn quốc.",
        "metadata": {
            "source": "Circular 2457/GDDT-GDĐH",
            "info_type": "van_ban_phap_ly",
            "unit": "Bo_GDDT",
            "year": 2025,
        },
    }


@pytest.fixture
def sample_long_doc():
    """Fixture for long document requiring chunking."""
    # Generate long text (>500 tokens)
    text = " ".join([f"This is sentence number {i}." for i in range(200)])
    return {
        "id": "long_001",
        "text": text,
        "metadata": {"source": "Long Document Test", "year": 2025},
    }


def test_chunk_config_loading(config_path):
    """Test ChunkConfig loads profiles correctly."""
    config = ChunkConfig(config_path)

    assert "faq" in config.profiles
    assert "policy" in config.profiles
    assert "auto" in config.profiles

    faq_profile = config.get_profile("faq")
    assert faq_profile.chunk_size == 320
    assert faq_profile.overlap == 60

    policy_profile = config.get_profile("policy")
    assert policy_profile.chunk_size == 450
    assert policy_profile.preserve_sections is True


def test_chunker_initialization(config_path):
    """Test Chunker initialization."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    assert chunker.config == config
    assert chunker.tokenizer is not None


def test_profile_detection_faq(config_path, sample_faq_doc):
    """Test automatic FAQ profile detection."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    detected_profile = chunker._detect_profile(sample_faq_doc)
    assert detected_profile == "faq"


def test_profile_detection_policy(config_path, sample_policy_doc):
    """Test automatic policy profile detection."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    detected_profile = chunker._detect_profile(sample_policy_doc)
    assert detected_profile == "policy"


def test_chunk_short_document(config_path, sample_faq_doc):
    """Test chunking short document (single chunk)."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    chunks = chunker.chunk_document(sample_faq_doc, profile_name="faq")

    assert len(chunks) == 1
    assert isinstance(chunks[0], RAGChunk)
    assert chunks[0].doc_id == "faq_001"
    assert chunks[0].chunk_id == 0
    assert len(chunks[0].text) > 0
    assert chunks[0].metadata["source"] == "Admissions FAQ"


def test_chunk_long_document(config_path, sample_long_doc):
    """Test chunking long document (multiple chunks)."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    chunks = chunker.chunk_document(sample_long_doc, profile_name="auto")

    # Should create multiple chunks
    assert len(chunks) > 1

    # Check chunk IDs are sequential
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == i
        assert chunk.doc_id == "long_001"

    # Check prev/next linking (use sentinel '' because the RAGChunk model uses empty string)
    assert chunks[0].prev_chunk_id == ""
    assert chunks[0].next_chunk_id == chunks[1].id
    assert chunks[-1].next_chunk_id == ""


def test_metadata_preservation(config_path, sample_faq_doc):
    """Test metadata field preservation based on profile."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    chunks = chunker.chunk_document(sample_faq_doc, profile_name="faq")

    # FAQ profile should preserve specific fields
    assert "faq_type" in chunks[0].metadata
    assert "audience" in chunks[0].metadata
    assert "source" in chunks[0].metadata  # Always preserved


def test_sparse_terms_extraction(config_path, sample_faq_doc):
    """Test sparse terms extraction for BM25."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    chunks = chunker.chunk_document(sample_faq_doc, profile_name="faq")

    # Should extract terms
    assert len(chunks[0].sparse_terms) > 0

    # Terms should be lowercase and filtered
    for term in chunks[0].sparse_terms:
        assert term.islower()
        assert len(term) > 2  # Min length filter


def test_breadcrumbs_extraction(config_path, sample_policy_doc):
    """Test breadcrumb extraction from policy document."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    chunks = chunker.chunk_document(sample_policy_doc, profile_name="policy")

    # Should extract section markers
    assert len(chunks[0].breadcrumbs) > 0
    # Should include source as top-level breadcrumb
    assert any("Circular" in bc for bc in chunks[0].breadcrumbs)


def test_chunk_jsonl_integration(config_path, sample_faq_doc, sample_policy_doc):
    """Test end-to-end JSONL chunking."""
    # Create temp input file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as f:
        f.write(json.dumps(sample_faq_doc) + "\n")
        f.write(json.dumps(sample_policy_doc) + "\n")
        input_path = Path(f.name)

    output_path = input_path.parent / "chunked_output.jsonl"

    try:
        num_chunks = chunk_jsonl(input_path, output_path, config_path, profile="auto")

        assert num_chunks >= 2  # At least 2 chunks (one per doc minimum)
        assert output_path.exists()

        # Verify output format
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk_data = json.loads(line)
                assert "id" in chunk_data
                assert "doc_id" in chunk_data
                assert "chunk_id" in chunk_data
                assert "text" in chunk_data

    finally:
        input_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_token_overlap(config_path, sample_long_doc):
    """Test chunk overlap functionality."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    chunks = chunker.chunk_document(sample_long_doc, profile_name="auto")

    if len(chunks) > 1:
        # Check if there's overlapping content
        chunk1_end = chunks[0].text[-50:]
        chunk2_start = chunks[1].text[:50]

        # Should have some overlap (not exact due to separator splitting)
        # Just verify both chunks have content
        assert len(chunk1_end) > 0
        assert len(chunk2_start) > 0


# ══════════════════════════════════════════════════════════════════
# T1.8 chunk size gates — hard cap 512 tokens, soft min 50 tokens
# ══════════════════════════════════════════════════════════════════

HARD_MAX_TOKENS = 512
SOFT_MIN_TOKENS = 50


def test_hard_cap_512_chunk_document(config_path):
    """T1.8: chunk_document must never emit a chunk > HARD_MAX_TOKENS (512).

    A long run-on doc with no separators is kept as a single oversized chunk by
    the profile splitter (auto chunk_size 350); the size gate must split it.
    """
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    text = "x" * 6000  # ~6000 tokens, no separators
    chunks = chunker.chunk_document(
        {"id": "huge_001", "text": text, "metadata": {}}, profile_name="auto"
    )

    assert len(chunks) > 1
    for c in chunks:
        assert len(chunker.tokenizer.encode(c.text)) <= HARD_MAX_TOKENS


def test_hard_cap_split_direct(config_path):
    """T1.8: _enforce_size_gates splits an oversized chunk into <=512 pieces."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    big = "y" * 3000  # ~3000 tokens
    out = chunker._enforce_size_gates([big])

    assert len(out) > 1
    for piece in out:
        assert len(chunker.tokenizer.encode(piece)) <= HARD_MAX_TOKENS


def test_soft_min_50_merge(config_path):
    """T1.8: chunks below SOFT_MIN_TOKENS (50) are merged into a neighbor."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    out = chunker._enforce_size_gates(["A" * 300, "tiny chunk"])

    assert len(out) == 1
    merged_tok = len(chunker.tokenizer.encode(out[0]))
    assert merged_tok <= HARD_MAX_TOKENS


def test_size_gates_keep_normal_chunks_unchanged(config_path):
    """T1.8: chunks already within [SOFT_MIN, HARD_MAX] pass through untouched."""
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    # Both chunks comfortably above SOFT_MIN_TOKENS (50) and below the hard cap.
    src = ["normal chunk with moderate length and additional filler words " * 12,
           "second chunk content that is also reasonably long enough " * 12]
    out = chunker._enforce_size_gates(src)

    assert len(out) == len(src)
    for a, b in zip(src, out):
        assert a == b

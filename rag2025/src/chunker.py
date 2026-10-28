"""
Chunker Module with Profile Support (Epic 1.2)

Implements adaptive chunking strategies for FAQ vs Policy documents.
Supports token-based splitting, hierarchical breadcrumbs, and metadata preservation.
"""
import re
from pathlib import Path
from typing import Any, Literal

import tiktoken
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from services._metadata_helpers import get_source_label, get_source_breadcrumb

# T1.8 chunk size gates (hard cap + soft floor) are driven by config.settings so
# .env remains the single authoritative source for chunk-budget knobs.
from config.settings import settings

# ── ST-6 structure guards ───────────────────────────────────────────────────
# DECISION (2026-08): do NOT pull in underthesea / pyvi / VnCoreNLP for sentence
# splitting. A zero-dependency lookbehind regex over the Vietnamese sentence
# terminators (. ? ! ; :) followed by whitespace is sufficient to keep chunk
# edges off mid-sentence cuts. Revisit only if evals show weak-boundary failures.
_SENTENCE_TERMINATOR_RE = re.compile(r"(?<=[.?!;:])\s+")
# A markdown/pipe-table row is any line that starts with a pipe (optional
# leading whitespace). Rows are ATOMIC — a chunk boundary must never land
# inside one (fixes chunked_23 chunk_133 losing '407', chunked_24 chunk_100→101
# dropping a subject-combo table row).
_PIPE_TABLE_LINE_RE = re.compile(r"^\s*\|", re.MULTILINE)


class ChunkProfile(BaseModel):
    """Chunk profile configuration."""

    description: str
    chunk_size: int = Field(ge=100, le=600)
    overlap: int = Field(ge=0, le=200)
    separator_priority: list[str] = Field(default_factory=lambda: ["\n\n", "\n", " "])
    min_tokens: int = Field(ge=10, le=300)
    compression: Literal["full_text", "summary", "adaptive"] = "full_text"
    preserve_sections: bool = False
    section_regex: str | None = None
    preserve_metadata_fields: list[str] = Field(default_factory=list)


class ChunkConfig:
    """Chunk profile loader and manager."""

    def __init__(self, config_path: Path):
        """
        Load chunk profiles from YAML.

        Args:
            config_path: Path to chunk_profiles.yaml
        """
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.profiles: dict[str, ChunkProfile] = {}
        for name, config in data["profiles"].items():
            self.profiles[name] = ChunkProfile(**config)

        logger.info(f"Loaded {len(self.profiles)} chunk profiles from {config_path}")

    def get_profile(self, name: str) -> ChunkProfile:
        """Get profile by name, fallback to 'auto'."""
        if name not in self.profiles:
            logger.warning(f"Profile '{name}' not found, using 'auto'")
            return self.profiles["auto"]
        return self.profiles[name]


class RAGChunk(BaseModel):
    """RAG chunk model matching JSON Schema."""

    id: str
    doc_id: str
    chunk_id: int
    text: str
    text_plain: str | None = None
    summary: str | None = None
    metadata: dict[str, Any]
    breadcrumbs: list[str] = Field(default_factory=list)
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    sparse_terms: list[str] = Field(default_factory=list)


class Chunker:
    """
    Adaptive chunker with profile support.

    Supports:
    - Token-based splitting (tiktoken)
    - FAQ vs Policy profiles
    - Hierarchical breadcrumbs
    - Metadata preservation
    - Sparse term extraction
    """

    def __init__(self, config: ChunkConfig, encoding_name: str = "cl100k_base"):
        """
        Initialize chunker.

        Args:
            config: ChunkConfig instance
            encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4)
        """
        self.config = config
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        logger.info(f"Chunker initialized with encoding: {encoding_name}")

    def chunk_document(
        self, doc: dict[str, Any], profile_name: str = "auto"
    ) -> list[RAGChunk]:
        """
        Chunk a single document using specified profile.

        Args:
            doc: Document dict (must have 'id', 'text', 'metadata')
            profile_name: Profile name ('auto', 'faq', 'policy')

        Returns:
            List of RAGChunk objects
        """
        # Detect profile if 'auto'
        if profile_name == "auto":
            profile_name = self._detect_profile(doc)

        profile = self.config.get_profile(profile_name)
        logger.debug(f"Chunking doc {doc.get('id', 'unknown')} with profile: {profile_name}")

        # Extract text
        text = self._extract_text(doc, profile)

        # Split into chunks
        text_chunks = self._split_text(text, profile)

        # T1.8: enforce hard cap (512) + soft floor (50) after profile split
        text_chunks = self._enforce_size_gates(text_chunks)

        # Create RAGChunk objects
        chunks: list[RAGChunk] = []
        doc_id = doc.get("doc_id") or doc.get("id", "unknown")

        for i, chunk_text in enumerate(text_chunks):
            next_id = ""
            if i < len(text_chunks) - 1:
                next_id = f"{doc_id}_chunk_{i + 1}"
            chunk = RAGChunk(
                id=f"{doc_id}_chunk_{i}",
                doc_id=doc_id,
                chunk_id=i,
                text=chunk_text,
                text_plain=doc.get("text_plain"),
                summary=doc.get("summary"),
                metadata=self._filter_metadata(doc.get("metadata", {}), profile),
                breadcrumbs=self._extract_breadcrumbs(doc, chunk_text),
                sparse_terms=self._extract_sparse_terms(chunk_text),
                prev_chunk_id=f"{doc_id}_chunk_{i - 1}" if i > 0 else "",
                next_chunk_id=next_id,
            )
            chunks.append(chunk)

        logger.info(f"Doc {doc_id} chunked into {len(chunks)} chunks with profile {profile_name}")
        return chunks

    def _detect_profile(self, doc: dict[str, Any]) -> str:
        """
        Auto-detect profile based on metadata.

        Rules:
        - If metadata.faq_type exists -> 'faq'
        - If metadata.info_type contains 'van_ban_phap_ly' -> 'policy'
        - Otherwise -> 'auto'
        """
        metadata = doc.get("metadata", {})

        if "faq_type" in metadata:
            return "faq"

        info_type = metadata.get("info_type", "")
        if (
            "van_ban_phap_ly" in info_type
            or "policy" in info_type.lower()
            or metadata.get("legal_structure_detected") is True
        ):
            return "policy"

        text = str(doc.get("text") or "")
        if re.search(r"^(PHẦN|CHƯƠNG|MỤC|Điều|Khoản)\s+", text, re.IGNORECASE | re.MULTILINE):
            return "policy"

        return "auto"

    def _extract_text(self, doc: dict[str, Any], profile: ChunkProfile) -> str:
        """
        Extract text from document based on profile compression setting.

        Args:
            doc: Document dict
            profile: ChunkProfile

        Returns:
            Extracted text string
        """
        if profile.compression == "summary" and doc.get("summary"):
            return doc["summary"]
        elif profile.compression == "full_text" and doc.get("text"):
            return doc["text"]
        else:
            # Adaptive: prefer text_plain > text > summary
            return doc.get("text_plain") or doc.get("text") or doc.get("summary", "")

    def _split_text(self, text: str, profile: ChunkProfile) -> list[str]:
        """
        Split text into chunks based on token count and separators.

        Args:
            text: Text to split
            profile: ChunkProfile

        Returns:
            List of text chunks
        """
        # Tokenize
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= profile.chunk_size:
            return [text]

        # 1) Try structure-aware markdown/legal block split first
        parts = self._split_markdown_legal_blocks(text)
        separator_used = "\n\n"

        # 2) Fallback to separator split when structure split is weak
        if len(parts) <= 1:
            has_pipe_rows = bool(_PIPE_TABLE_LINE_RE.search(text))
            for sep in profile.separator_priority:
                # ST-6b: pipe-table rows are atomic — only newline-based
                # separators may split table text (a boundary between lines can
                # never land inside a row); skip ". " / " " and the like.
                if has_pipe_rows and "\n" not in sep:
                    continue
                # ST-6a: prefer Vietnamese sentence terminators (. ? ! ; : +
                # whitespace) over a plain-space split so chunk edges land at
                # sentence boundaries, not mid-sentence. Only relevant for
                # non-table text (rows are atomic, see above).
                if sep == " " and not has_pipe_rows:
                    if _SENTENCE_TERMINATOR_RE.search(text):
                        parts = _SENTENCE_TERMINATOR_RE.split(text)
                        separator_used = " "
                        break
                if sep in text:
                    parts = text.split(sep)
                    separator_used = sep
                    break
            else:
                parts = [text]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for part in parts:
            part = part.strip()
            if not part:
                continue

            part_tokens = len(self.tokenizer.encode(part))

            if current_tokens + part_tokens > profile.chunk_size:
                if current_chunk:
                    chunk_text = separator_used.join(current_chunk).strip()
                    if len(self.tokenizer.encode(chunk_text)) >= profile.min_tokens:
                        chunks.append(chunk_text)

                # ST-6d: `overlap_text` re-prefixes the tail of the previous
                # chunk onto the next one. This is INTENTIONAL
                # CHUNK_OVERLAP-driven duplication — adjacent chunks share
                # exactly the overlap window (the token span that appears in
                # both boundaries), not redundant data. It is the same
                # duplication produced by CHUNK_OVERLAP during normal
                # chunking, and it is required for continuity when a boundary
                # is split across two chunks.
                overlap_text = self._get_overlap_text(current_chunk, profile.overlap)
                current_chunk = [overlap_text, part] if overlap_text else [part]
                current_tokens = len(self.tokenizer.encode(separator_used.join(current_chunk)))
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        if current_chunk:
            chunk_text = separator_used.join(current_chunk).strip()
            if len(self.tokenizer.encode(chunk_text)) >= profile.min_tokens:
                chunks.append(chunk_text)

        logger.debug(f"Split {total_tokens} tokens into {len(chunks)} chunks")
        return chunks

    def _enforce_size_gates(
        self,
        text_chunks: list[str],
        hard_max_tokens: int | None = None,
        soft_min_tokens: int | None = None,
    ) -> list[str]:
        """T1.8: hard cap (split) + soft floor (merge) driven by config.settings.

        Split any chunk above the hard cap into balanced <=hard_max pieces, then
        merge chunks below the soft floor into a neighbor without exceeding the
        cap (best-effort — a lone tiny chunk is kept as-is). Knobs resolve from
        config.settings at call time unless explicit values are passed.
        """
        hard_max = hard_max_tokens if hard_max_tokens is not None else settings.HARD_MAX_TOKENS
        soft_min = soft_min_tokens if soft_min_tokens is not None else settings.SOFT_MIN_TOKENS

        if not text_chunks:
            return text_chunks

        # 1) Hard cap: split oversized chunks at balanced token boundaries,
        #    EXCEPT pipe-table text, which must split at row boundaries instead.
        pieces: list[str] = []
        for chunk in text_chunks:
            tokens = self.tokenizer.encode(chunk)
            if len(tokens) <= hard_max:
                pieces.append(chunk)
                continue
            if _PIPE_TABLE_LINE_RE.search(chunk):
                # ST-6b: pipe-table rows are atomic — never balanced-split a
                # table (that is what cut chunked_23 chunk_133's '407' and
                # dropped row 5 of the subject-combo table in chunked_24).
                pieces.extend(self._split_oversized_pipe_safe(chunk, hard_max))
                continue
            n_pieces = (len(tokens) + hard_max - 1) // hard_max
            base, rem = divmod(len(tokens), n_pieces)
            start = 0
            for i in range(n_pieces):
                end = start + base + (1 if i < rem else 0)
                piece = self.tokenizer.decode(tokens[start:end]).strip()
                if piece:
                    pieces.append(piece)
                start = end

        # 2) Soft floor: merge tiny chunks into previous neighbor (keep cap).
        merged: list[str] = []
        for chunk in pieces:
            if merged:
                cur_tok = len(self.tokenizer.encode(chunk))
                prev_tok = len(self.tokenizer.encode(merged[-1]))
                if cur_tok < soft_min and prev_tok + cur_tok <= hard_max:
                    merged[-1] = f"{merged[-1]}\n\n{chunk}"
                    continue
            merged.append(chunk)
        return merged

    def _split_oversized_pipe_safe(self, text: str, hard_max: int) -> list[str]:
        """ST-6b: split oversized pipe-table text at row boundaries.

        Every pipe-table line is atomic — a chunk boundary must never land
        inside one. Lines are packed whole into pieces of at most ``hard_max``
        tokens; a single line that still exceeds the cap is kept whole and
        flagged with a warning (never cut mid-row).
        """
        lines = text.split("\n")
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                pieces.append("\n".join(current))
                current = []
                current_tokens = 0

        for line in lines:
            if not line.strip():
                continue  # blank separators carry no row content
            line_tokens = len(self.tokenizer.encode(line))
            if line_tokens > hard_max:
                # Single row above the cap: keep whole + flag, never cut.
                flush()
                pieces.append(line)
                logger.warning(
                    "ST-6b: pipe-table row exceeds HARD_MAX_TOKENS ({} > {}) "
                    "and is kept whole: {}",
                    line_tokens,
                    hard_max,
                    line[:80],
                )
                continue
            if current_tokens + line_tokens > hard_max:
                flush()
            current.append(line)
            current_tokens += line_tokens
        flush()
        return pieces or [text]

    def _split_markdown_legal_blocks(self, text: str) -> list[str]:
        """Split text by markdown headings and legal heading markers."""
        lines = [ln.rstrip() for ln in text.splitlines()]
        if not lines:
            return [text]

        heading_re = re.compile(
            r"^(#{1,6}\s+.+|PHẦN\s+[IVXLC\d]+.*|CHƯƠNG\s+[IVXLC\d]+.*|MỤC\s+[IVXLC\d]+.*|Điều\s+\d+[\.:]?.*|Khoản\s+\d+[\.:]?.*|[a-zđ]\)\s+.+)",
            re.IGNORECASE,
        )

        blocks: list[list[str]] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current and current[-1] != "":
                    current.append("")
                continue

            if heading_re.match(stripped):
                if current:
                    blocks.append(current)
                current = [stripped]
            else:
                if not current:
                    current = [stripped]
                else:
                    current.append(stripped)

        if current:
            blocks.append(current)

        out = ["\n".join(b).strip() for b in blocks if any(x.strip() for x in b)]
        return out or [text]


    def _get_overlap_text(self, current_chunk: list[str], overlap_tokens: int) -> str:
        """
        Extract overlap text from current chunk.

        Args:
            current_chunk: List of text parts
            overlap_tokens: Number of tokens to overlap

        Returns:
            Overlap text string
        """
        if not current_chunk or overlap_tokens == 0:
            return ""

        full_text = "".join(current_chunk)
        tokens = self.tokenizer.encode(full_text)

        if len(tokens) <= overlap_tokens:
            return full_text

        # Take last N tokens
        overlap_tokens_list = tokens[-overlap_tokens:]
        overlap_text = self.tokenizer.decode(overlap_tokens_list)
        return overlap_text

    def _filter_metadata(
        self, metadata: dict[str, Any], profile: ChunkProfile
    ) -> dict[str, Any]:
        """
        Filter metadata fields based on profile.

        Args:
            metadata: Original metadata dict
            profile: ChunkProfile

        Returns:
            Filtered metadata dict
        """
        if not profile.preserve_metadata_fields:
            return metadata

        # Keep only specified fields
        filtered = {k: v for k, v in metadata.items() if k in profile.preserve_metadata_fields}

        # Always keep 'source' field
        if "source" not in filtered and "source" in metadata:
            filtered["source"] = get_source_label({"metadata": metadata})

        return filtered

    def _extract_breadcrumbs(self, doc: dict[str, Any], chunk_text: str) -> list[str]:
        """
        Extract hierarchical breadcrumbs from document/chunk.

        Args:
            doc: Document dict
            chunk_text: Chunk text

        Returns:
            List of breadcrumb strings
        """
        breadcrumbs: list[str] = []

        # Check for section markers in chunk text (e.g., "Chương 1", "Điều 2")
        section_pattern = r"^(Chương|Điều|Khoản|Mục)\s+(\d+|[IVX]+)"
        match = re.search(section_pattern, chunk_text, re.MULTILINE)
        if match:
            breadcrumbs.append(match.group(0))

        # Add document source as top-level breadcrumb
        metadata = doc.get("metadata", {})
        if "source" in metadata:
            breadcrumbs.insert(0, get_source_breadcrumb({"metadata": metadata}))

        return breadcrumbs

    def _extract_sparse_terms(self, text: str) -> list[str]:
        """
        Extract sparse terms for BM25 indexing.

        Simple tokenization: lowercase + split + filter stopwords.

        Args:
            text: Text to extract terms from

        Returns:
            List of terms
        """
        # Vietnamese stopwords (basic set)
        stopwords = {
            "và",
            "các",
            "của",
            "có",
            "được",
            "cho",
            "trong",
            "là",
            "một",
            "này",
            "để",
            "với",
            "theo",
            "từ",
            "đã",
            "sẽ",
            "không",
            "khi",
            "bằng",
        }

        # Tokenize
        tokens = re.findall(r"\b\w+\b", text.lower())

        # Filter stopwords, short tokens, and pure numeric tokens
        terms = [
            t for t in tokens
            if t not in stopwords and len(t) > 2 and any(c.isalpha() for c in t)
        ]

        return terms


def chunk_jsonl(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    profile: str = "auto",
) -> int:
    """
    Chunk all documents in a .jsonl file.

    Args:
        input_path: Input .jsonl file path
        output_path: Output .jsonl file path
        config_path: Path to chunk_profiles.yaml
        profile: Profile name ('auto', 'faq', 'policy')

    Returns:
        Number of chunks created
    """
    import json

    # Load config
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:

        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
                chunks = chunker.chunk_document(doc, profile_name=profile)

                for chunk in chunks:
                    outfile.write(chunk.model_dump_json() + "\n")

                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"Line {line_num}: Failed to chunk - {e}")

    logger.info(f"Chunked {total_chunks} total chunks to {output_path}")
    return total_chunks


def main():
    """CLI entry point for chunker."""
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python chunker.py <input.jsonl> <output.jsonl> "
            "<chunk_profiles.yaml> [profile]"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    config_path = Path(sys.argv[3])
    profile = sys.argv[4] if len(sys.argv) > 4 else "auto"

    num_chunks = chunk_jsonl(input_path, output_path, config_path, profile)
    print(f"\nChunked {num_chunks} total chunks.")
    sys.exit(0)


if __name__ == "__main__":
    main()

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
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
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

        # Create RAGChunk objects
        chunks: list[RAGChunk] = []
        doc_id = doc.get("id", "unknown")

        for i, chunk_text in enumerate(text_chunks):
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
            )
            chunks.append(chunk)

        # Link prev/next chunks
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].id

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
        if "van_ban_phap_ly" in info_type or "policy" in info_type.lower():
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

        # Split by separators (hierarchical)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        # Split by highest priority separator first
        for sep in profile.separator_priority:
            if sep in text:
                parts = text.split(sep)
                break
        else:
            parts = [text]  # No separator found, use whole text

        for part in parts:
            part_tokens = len(self.tokenizer.encode(part))

            if current_tokens + part_tokens > profile.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunk_text = "".join(current_chunk).strip()
                    if len(self.tokenizer.encode(chunk_text)) >= profile.min_tokens:
                        chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, profile.overlap)
                current_chunk = [overlap_text, part] if overlap_text else [part]
                current_tokens = len(self.tokenizer.encode("".join(current_chunk)))
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        # Flush final chunk
        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if len(self.tokenizer.encode(chunk_text)) >= profile.min_tokens:
                chunks.append(chunk_text)

        logger.debug(f"Split {total_tokens} tokens into {len(chunks)} chunks")
        return chunks

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
            filtered["source"] = metadata["source"]

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
            breadcrumbs.insert(0, metadata["source"])

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

        # Filter stopwords and short tokens
        terms = [t for t in tokens if t not in stopwords and len(t) > 2]

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

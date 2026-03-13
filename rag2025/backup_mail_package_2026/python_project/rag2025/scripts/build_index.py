"""
Build Index Script (Epic 2.2)

Reads chunked JSONL files, generates embeddings, and builds vector index.

Usage:
    python scripts/build_index.py \
        data/chunked_2.jsonl \
        index/vector_store.npz
"""
import sys
import os
# Add the project root and src directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import glob
import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from config.settings import RAGSettings
from services.embedding import EmbeddingService
from services.vector_store import NumpyVectorStore


def load_chunks(input_path: Path) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Load chunks from JSONL file.

    Args:
        input_path: Path to chunked JSONL file

    Returns:
        Tuple of (texts, ids, metadatas)
    """
    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    logger.info(f"Loading chunks from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
                
                # Handle both JSON objects and plain strings
                if isinstance(chunk, dict):
                    # Standard JSON object format
                    chunk_id = chunk.get("id", f"chunk_{line_num}")
                    text = chunk.get("text", "")
                    
                    if not text:
                        logger.warning(f"Line {line_num}: Empty text, skipping")
                        continue

                    # Build metadata
                    metadata = {
                        "id": chunk_id,
                        "doc_id": chunk.get("doc_id", "unknown"),
                        "chunk_id": chunk.get("chunk_id", 0),
                        "text": text,
                        "text_plain": chunk.get("text_plain", ""),
                        "summary": chunk.get("summary", ""),
                        "metadata": chunk.get("metadata", {}),
                        "breadcrumbs": chunk.get("breadcrumbs", []),
                        "sparse_terms": chunk.get("sparse_terms", []),
                    }
                elif isinstance(chunk, str):
                    # Plain string format - create chunk from string
                    text = chunk
                    if not text:
                        logger.warning(f"Line {line_num}: Empty string, skipping")
                        continue
                        
                    chunk_id = f"chunk_{line_num}"
                    metadata = {
                        "id": chunk_id,
                        "doc_id": "unknown",
                        "chunk_id": line_num - 1,
                        "text": text,
                        "text_plain": text,
                        "summary": "",
                        "metadata": {"source": "string_format"},
                        "breadcrumbs": [],
                        "sparse_terms": [],
                    }
                else:
                    logger.warning(f"Line {line_num}: Unsupported data type: {type(chunk)}, skipping")
                    continue

                texts.append(text)
                ids.append(chunk_id)
                metadatas.append(metadata)

            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON parse error - {e}")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Error processing chunk - {e}")
                continue

    logger.info(f"Loaded {len(texts)} chunks from {input_path}")
    return texts, ids, metadatas


def build_index(
    input_path: Path,
    output_path: Path,
    settings: RAGSettings | None = None,
) -> None:
    """
    Build vector index from chunked JSONL file.

    Args:
        input_path: Path to chunked JSONL file
        output_path: Path to save vector store (.npz)
        settings: RAGSettings instance (optional)
    """
    if settings is None:
        settings = RAGSettings()

    logger.info("=" * 60)
    logger.info("Building Vector Index")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Step 1: Load chunks
    texts, ids, metadatas = load_chunks(input_path)

    if not texts:
        logger.error("No chunks loaded, aborting")
        sys.exit(1)

    # Step 2: Initialize embedding service
    logger.info("Initializing embedding service...")
    embedding_service = EmbeddingService(settings)

    # Step 3: Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_service.encode_documents(texts)

    logger.info(f"Generated embeddings: shape={embeddings.shape}")

    # Step 4: Build vector store
    logger.info("Building vector store...")
    vector_store = NumpyVectorStore(dim=settings.EMBEDDING_DIM)
    vector_store.add_vectors(embeddings, ids, metadatas)

    # Step 5: Save vector store
    logger.info(f"Saving vector store to: {output_path}")
    vector_store.save(output_path)

    logger.info("=" * 60)
    logger.info("Index Build Complete!")
    logger.info("=" * 60)
    logger.info(f"Total chunks indexed: {vector_store.count()}")
    logger.info(f"Embedding dimension: {settings.EMBEDDING_DIM}")
    logger.info(f"Model: {settings.EMBEDDING_MODEL}")


def build_index_from_chunked_folder(
    chunked_folder: Path,
    output_path: Path,
    settings: RAGSettings | None = None,
) -> None:
    """
    Build vector index from all chunked JSONL files in a folder.

    Args:
        chunked_folder: Path to folder containing chunked JSONL files
        output_path: Path to save vector store (.npz)
        settings: RAGSettings instance (optional)
    """
    if settings is None:
        settings = RAGSettings()

    logger.info("=" * 60)
    logger.info("Building Vector Index from All Chunked Files")
    logger.info("=" * 60)
    logger.info(f"Input folder: {chunked_folder}")
    logger.info(f"Output: {output_path}")

    # Find all JSONL files in chunked folder
    chunked_files = sorted(chunked_folder.glob("*.jsonl"))
    if not chunked_files:
        logger.error(f"No JSONL files found in {chunked_folder}")
        raise ValueError(f"No JSONL files found in {chunked_folder}")

    logger.info(f"Found {len(chunked_files)} chunked files:")
    for file in chunked_files:
        logger.info(f"  - {file.name}")

    # Load all chunks from all files
    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for file_path in chunked_files:
        texts, ids, metadatas = load_chunks(file_path)
        all_texts.extend(texts)
        all_ids.extend(ids)
        all_metadatas.extend(metadatas)

    logger.info(f"Total chunks from all files: {len(all_texts)}")

    # Initialize embedding service
    logger.info("Initializing embedding service...")
    from services.embedding import EmbeddingService
    embedding_service = EmbeddingService(settings)
    logger.info(f"EmbeddingService initialized: model={settings.EMBEDDING_MODEL}, dim={settings.EMBEDDING_DIM}")

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
    embeddings = embedding_service.encode_batch(all_texts)
    logger.info(f"Generated embeddings: shape={embeddings.shape}")

    # Build vector store
    logger.info("Building vector store...")
    from services.vector_store import NumpyVectorStore
    
    vector_store = NumpyVectorStore(settings.EMBEDDING_DIM)
    vector_store.add_vectors(embeddings, all_ids, all_metadatas)

    # Save vector store
    logger.info(f"Saving vector store to: {output_path}")
    vector_store.save(output_path)

    logger.info("=" * 60)
    logger.info("Index Build Complete!")
    logger.info("=" * 60)
    logger.info(f"Total chunks indexed: {vector_store.count()}")
    logger.info(f"Files processed: {len(chunked_files)}")
    logger.info(f"Embedding dimension: {settings.EMBEDDING_DIM}")
    logger.info(f"Model: {settings.EMBEDDING_MODEL}")


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python scripts/build_index.py <input.jsonl|folder> <output.npz>")
        print("\nExamples:")
        print("  Single file:")
        print("    python scripts/build_index.py \\")
        print("        data/chunked_2.jsonl \\")
        print("        index/vector_store.npz")
        print("\n  All chunked files:")
        print("    python scripts/build_index.py \\")
        print("        data/chunked \\")
        print("        index/vector_store.npz")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        if input_path.is_file():
            # Single file mode
            build_index(input_path, output_path)
        elif input_path.is_dir():
            # Folder mode - process all chunked files
            build_index_from_chunked_folder(input_path, output_path)
        else:
            logger.error(f"Input not found: {input_path}")
            sys.exit(1)

        print("\n[OK] Index built successfully!")
        print(f"   Output: {output_path}")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

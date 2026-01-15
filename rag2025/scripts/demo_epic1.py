"""
Demo script for Epic 1 (Data Ingestion & Chunking)

Demonstrates end-to-end validation and chunking of 2.jsonl file.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from chunker import ChunkConfig, Chunker
from validate_jsonl import validate_jsonl


def demo_validation():
    """Demo: Validate 2.jsonl file."""
    logger.info("=" * 60)
    logger.info("DEMO: Epic 1.1 - Schema Validation")
    logger.info("=" * 60)

    # Paths
    root = Path(__file__).parent.parent.parent
    input_path = root / "2.jsonl"
    schema_path = Path(__file__).parent.parent / "config" / "rag_chunk_schema.json"
    output_path = Path(__file__).parent.parent / "data" / "validated_2.jsonl"

    # Validate
    logger.info(f"Validating: {input_path}")
    report = validate_jsonl(input_path, schema_path, output_path, strict=True)

    # Print summary
    print("\n" + report.summary())

    if report.errors:
        print("\nTop 5 Errors:")
        for err in report.errors[:5]:
            print(f"  Line {err['line']}: {err['error']}")

    if report.warnings:
        print(f"\nWarnings: {len(report.warnings)} total")
        for warn in report.warnings[:3]:
            print(f"  {warn}")

    logger.info(f"Validated data saved to: {output_path}")
    return output_path


def demo_chunking(validated_path: Path):
    """Demo: Chunk validated data with auto profile."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Epic 1.2 - Adaptive Chunking")
    logger.info("=" * 60)

    # Paths
    config_path = Path(__file__).parent.parent / "config" / "chunk_profiles.yaml"
    output_path = Path(__file__).parent.parent / "data" / "chunked_2.jsonl"

    # Load sample records
    import json

    logger.info(f"Loading first 5 records from: {validated_path}")

    with open(validated_path, "r", encoding="utf-8") as f:
        sample_docs = []
        for i, line in enumerate(f):
            if i >= 5:
                break
            sample_docs.append(json.loads(line))

    # Initialize chunker
    config = ChunkConfig(config_path)
    chunker = Chunker(config)

    # Chunk each document
    all_chunks = []
    for doc in sample_docs:
        logger.info(f"\nChunking doc: {doc['id']}")

        # Auto-detect profile
        profile_name = chunker._detect_profile(doc)
        logger.info(f"  Detected profile: {profile_name}")

        # Chunk
        chunks = chunker.chunk_document(doc, profile_name=profile_name)
        logger.info(f"  Generated {len(chunks)} chunks")

        # Display first chunk
        if chunks:
            chunk = chunks[0]
            logger.info(f"  Chunk 0 preview:")
            logger.info(f"    ID: {chunk.id}")
            logger.info(f"    Text length: {len(chunk.text)} chars")
            logger.info(f"    Sparse terms: {len(chunk.sparse_terms)} terms")
            logger.info(f"    Breadcrumbs: {chunk.breadcrumbs}")
            logger.info(f"    Text preview: {chunk.text[:100]}...")

        all_chunks.extend(chunks)

    # Save all chunks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    logger.info(f"\nAll {len(all_chunks)} chunks saved to: {output_path}")
    return output_path


def main():
    """Run full Epic 1 demo."""
    logger.info("Starting Epic 1 Demo: Data Ingestion & Chunking")
    logger.info("=" * 60)

    # Step 1: Validation
    try:
        validated_path = demo_validation()
    except FileNotFoundError:
        logger.error("Error: 2.jsonl not found in parent directory")
        logger.info("Please ensure D:/chunking/2.jsonl exists")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Step 2: Chunking
    try:
        chunked_path = demo_chunking(validated_path)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("Epic 1 Demo Complete!")
    logger.info("=" * 60)
    logger.info(f"Validated data: {validated_path}")
    logger.info(f"Chunked data: {chunked_path}")

    print("\n✅ NEXT ACTION: Run tests to verify implementation")
    print("   Command: pytest tests/ -v")


if __name__ == "__main__":
    main()

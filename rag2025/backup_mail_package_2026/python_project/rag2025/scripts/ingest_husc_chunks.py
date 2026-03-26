"""
Ingest HUSC enhanced chunks to Qdrant

This script ingests only the chunked_10_enhanced.jsonl file to Qdrant
without deleting the existing collection.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag2025")
ENHANCED_FILE = Path(__file__).parent.parent / "data" / "chunked" / "chunked_10_enhanced.jsonl"

# For Qdrant Cloud, remove port specification from URL
if ":6333" in QDRANT_URL:
    QDRANT_URL = QDRANT_URL.replace(":6333", "")
    logger.info(f"Using cleaned Qdrant URL: {QDRANT_URL}")

def ingest_enhanced_chunks():
    """Ingest enhanced chunks to Qdrant"""

    if not QDRANT_URL:
        logger.error("QDRANT_URL not set in .env file")
        logger.info("Please set QDRANT_URL in your .env file (e.g., http://localhost:6333)")
        return False

    # Initialize client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, check_compatibility=False)

    logger.info(f"=== Ingesting Enhanced HUSC Chunks ===")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Collection: {COLLECTION_NAME}")

    # Check if collection exists
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if COLLECTION_NAME not in collection_names:
            # Create collection
            logger.info(f"Creating new collection: {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,  # BGE-M3 dimension
                    distance=Distance.COSINE
                )
            )
        else:
            # Check existing collection info
            info = client.get_collection(COLLECTION_NAME)
            logger.info(f"Collection exists with {info.points_count} points")
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        return False

    # Load enhanced chunks
    if not ENHANCED_FILE.exists():
        logger.error(f"Enhanced chunks file not found: {ENHANCED_FILE}")
        return False

    chunks = []
    with open(ENHANCED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {ENHANCED_FILE.name}")

    if not chunks:
        logger.error("No chunks to ingest!")
        return False

    # Check for duplicates in Qdrant
    existing_ids = set()
    try:
        # Get points to check for existing chunk IDs
        all_points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100000,
            with_payload=["chunk_id"],
        )
        existing_ids = {point.payload.get("chunk_id") for point in all_points[0] if point.payload.get("chunk_id")}
        logger.info(f"Found {len(existing_ids)} existing chunks in collection")
    except Exception as e:
        logger.warning(f"Could not check existing chunks: {e}")

    # Filter out existing chunks
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    logger.info(f"New chunks to ingest: {len(new_chunks)}")

    if not new_chunks:
        logger.info("No new chunks to ingest (all already exist)")
        return True

    # Load BGE-M3 model
    logger.info("Loading BGE-M3 model...")
    model = SentenceTransformer('BAAI/bge-m3')

    # Embed and upload
    batch_size = 32
    total_points = 0

    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding & Uploading"):
        batch = new_chunks[i:i + batch_size]

        # Extract texts
        texts = [chunk.get("text", "") for chunk in batch]

        # Embed
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Create points with full payload
        points = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=hash(chunk["id"]) % (2**63),  # Use hash for consistent ID
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.get("id"),
                    "text": chunk.get("text", ""),
                    "summary": chunk.get("summary", ""),
                    "text_plain": chunk.get("text_plain", ""),
                    "text_raw": chunk.get("text_raw", ""),
                    "doc_id": chunk.get("doc_id"),
                    "chunk_number": chunk.get("chunk_id"),
                    "breadcrumbs": chunk.get("breadcrumbs", []),
                    "sparse_terms": chunk.get("sparse_terms", []),
                    "faq_type": chunk.get("faq_type", ""),
                    "metadata": chunk.get("metadata", {}),
                    "source": chunk.get("metadata", {}).get("source", "HUSC 2025"),
                }
            )
            points.append(point)

        # Upload to Qdrant
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            total_points += len(points)
        except Exception as e:
            logger.error(f"Error uploading batch {i}-{i+len(batch)}: {e}")
            continue

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"=== Ingest Complete! ===")
    logger.info(f"Points uploaded this run: {total_points}")
    logger.info(f"Total points in collection: {info.points_count}")

    return True


if __name__ == "__main__":
    success = ingest_enhanced_chunks()
    exit(0 if success else 1)

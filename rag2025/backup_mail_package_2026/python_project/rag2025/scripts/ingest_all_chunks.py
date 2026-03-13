"""
Ingest ALL chunks to Qdrant (fresh database)

This script ingests all chunks from chunked_*.jsonl files to Qdrant.
First, it deletes the existing collection to start fresh.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger

# Load environment variables
load_dotenv()

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "hue_admissions_2025")
CHUNKS_DIR = Path(__file__).parent.parent / "data" / "chunked"

# For Qdrant Cloud, remove port specification from URL
if ":6333" in QDRANT_URL:
    QDRANT_URL = QDRANT_URL.replace(":6333", "")

def ingest_all_chunks():
    """Ingest all chunks from all jsonl files to Qdrant"""

    if not QDRANT_URL:
        logger.error("QDRANT_URL not set in .env file")
        return False

    # Initialize client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY, check_compatibility=False)

    logger.info(f"=== Ingesting ALL Chunks to Fresh Database ===")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Collection: {COLLECTION_NAME}")

    # 1. Delete old collection to start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted old collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.info(f"No old collection to delete: {e}")

    # 2. Create new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # BGE-M3 dimension
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created new collection: {COLLECTION_NAME}")

    # 3. Load all chunks from all jsonl files
    if not CHUNKS_DIR.exists():
        logger.error(f"Chunks directory not found: {CHUNKS_DIR}")
        return False

    all_chunks = []
    jsonl_files = sorted(CHUNKS_DIR.glob("chunked_*.jsonl"))

    # Exclude the enhanced file (use original chunked_10.jsonl)
    jsonl_files = [f for f in jsonl_files if "enhanced" not in f.name]

    logger.info(f"Found {len(jsonl_files)} jsonl files")

    for jsonl_file in tqdm(jsonl_files, desc="Reading files"):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                file_chunks = []
                for line in f:
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            # Handle both dict and string format
                            if isinstance(chunk, str):
                                # Convert string to dict format
                                file_chunks.append({
                                    "id": f"chunk_{len(all_chunks)+len(file_chunks)}",
                                    "text": chunk,
                                    "metadata": {"source": jsonl_file.name}
                                })
                            elif isinstance(chunk, dict):
                                file_chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
                all_chunks.extend(file_chunks)
                logger.debug(f"Loaded {len(file_chunks)} chunks from {jsonl_file.name}")
        except Exception as e:
            logger.error(f"Error reading {jsonl_file.name}: {e}")

    logger.info(f"Total chunks loaded: {len(all_chunks)}")

    if not all_chunks:
        logger.error("No chunks found to ingest!")
        return False

    # 4. Load BGE-M3 model
    logger.info("Loading BGE-M3 model...")
    model = SentenceTransformer('BAAI/bge-m3')

    # 5. Embed and upload
    batch_size = 32
    total_points = 0

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & Uploading"):
        batch = all_chunks[i:i + batch_size]

        # Extract texts
        texts = [chunk.get("text", "") for chunk in batch]

        # Embed
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Create points
        points = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            chunk_id = chunk.get("id", f"chunk_{i+j}")

            point = PointStruct(
                id=abs(hash(chunk_id)) % (2**63),  # Use hash for consistent ID
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk_id,
                    "text": chunk.get("text", ""),
                    "summary": chunk.get("summary", ""),
                    "text_plain": chunk.get("text_plain", ""),
                    "text_raw": chunk.get("text_raw", ""),
                    "doc_id": chunk.get("doc_id", chunk_id),
                    "chunk_number": chunk.get("chunk_id", i + j),
                    "breadcrumbs": chunk.get("breadcrumbs", []),
                    "sparse_terms": chunk.get("sparse_terms", []),
                    "faq_type": chunk.get("faq_type", ""),
                    "metadata": chunk.get("metadata", {}),
                    "source": chunk.get("metadata", {}).get("source", "unknown"),
                }
            )
            points.append(point)

        # Upload batch to Qdrant
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            total_points += len(points)
        except Exception as e:
            logger.error(f"Error uploading batch {i}-{i+len(batch)}: {e}")
            continue

    # 6. Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"=== Ingest Complete! ===")
    logger.info(f"Points uploaded: {total_points}")
    logger.info(f"Total points in collection: {info.points_count}")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info(f"Status: {info.status}")

    return True


if __name__ == "__main__":
    success = ingest_all_chunks()
    exit(0 if success else 1)

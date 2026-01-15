"""
Qdrant Migration Script

Migrate from NumPy vector store to Qdrant Cloud:
1. Delete old collection (if exists)
2. Create new collection with BGE-M3 config
3. Load all chunks from data/chunked/*.jsonl
4. Embed and upload to Qdrant
"""
import asyncio
import os
import sys
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from loguru import logger


async def migrate_to_qdrant():
    """
    Migrate from NumPy to Qdrant
    """

    # Connect to Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        logger.error("QDRANT_URL not set in environment")
        logger.info("Current environment variables:")
        for key in ["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"]:
            logger.info(f"  {key}: {os.getenv(key, 'NOT SET')}")
        return

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key
    )

    collection_name = os.getenv("QDRANT_COLLECTION", "hue_admissions_2025")

    logger.info(f"=== Qdrant Migration ===")
    logger.info(f"URL: {qdrant_url}")
    logger.info(f"Collection: {collection_name}")

    # 1. Delete old collection if exists
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted old collection: {collection_name}")
    except Exception as e:
        logger.info(f"No old collection to delete: {e}")

    # 2. Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # bge-m3 dimension
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created new collection: {collection_name}")

    # 3. Load all chunks
    chunks_dir = Path(__file__).parent.parent / "data" / "chunked"

    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return

    all_chunks = []

    for jsonl_file in sorted(chunks_dir.glob("chunked_*.jsonl")):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        # Handle both dict and string format
                        if isinstance(chunk, dict):
                            all_chunks.append(chunk)
                        elif isinstance(chunk, str):
                            # Convert string to dict format
                            all_chunks.append({
                                "id": f"chunk_{len(all_chunks)}",
                                "text": chunk,
                                "metadata": {"source": "string_format"}
                            })
                    except json.JSONDecodeError:
                        continue

    logger.info(f"Loaded {len(all_chunks)} chunks from {chunks_dir}")

    if not all_chunks:
        logger.error("No chunks found to migrate!")
        return

    # 4. Embed and upload
    logger.info("Loading BGE-M3 model...")
    model = SentenceTransformer('BAAI/bge-m3')
    batch_size = 32

    points = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding & Uploading"):
        batch = all_chunks[i:i + batch_size]

        # Extract texts for embedding
        texts = [chunk.get("text", "") for chunk in batch]

        # Embed batch
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Create points
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=i + j,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.get("id", f"chunk_{i+j}"),
                    "text": chunk.get("text", ""),
                    "summary": chunk.get("summary", ""),
                    "metadata": chunk.get("metadata", {}),
                    "faq_type": chunk.get("faq_type", ""),
                    "text_raw": chunk.get("text_raw", "")
                }
            )
            points.append(point)

        # Upload batch to Qdrant
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []

    # Upload remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    # 5. Verify
    collection_info = client.get_collection(collection_name)
    logger.info(f"=== Migration Complete! ===")
    logger.info(f"Total vectors: {collection_info.points_count}")
    logger.info(f"Collection: {collection_name}")


if __name__ == "__main__":
    asyncio.run(migrate_to_qdrant())

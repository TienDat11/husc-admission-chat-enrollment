"""
Check Qdrant collection info
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL").replace(":6333", ""),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False
)

collection_name = os.getenv("QDRANT_COLLECTION", "hue_admissions_2025")

print(f"=== Qdrant Collection Info ===")
print(f"URL: {os.getenv('QDRANT_URL')}")
print(f"Collection: {collection_name}")

try:
    # Check collections
    collections = client.get_collections()
    print(f"\nAll collections: {[c.name for c in collections.collections]}")

    if collection_name in [c.name for c in collections.collections]:
        info = client.get_collection(collection_name)
        print(f"\nCollection '{collection_name}':")
        print(f"  - Points count: {info.points_count}")
        print(f"  - Vector size: {info.config.params.vectors.size}")
        print(f"  - Distance: {info.config.params.vectors.distance}")
        print(f"  - Status: {info.status}")

        # Sample 3 points
        points = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
        )[0]
        print(f"\nSample points (first 3):")
        for p in points:
            print(f"  - ID: {p.id}")
            print(f"    chunk_id: {p.payload.get('chunk_id')}")
            print(f"    text: {p.payload.get('text', '')[:80]}...")
    else:
        print(f"\nCollection '{collection_name}' not found!")
except Exception as e:
    print(f"\nError: {e}")

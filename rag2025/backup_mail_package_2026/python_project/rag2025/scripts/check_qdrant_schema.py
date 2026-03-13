"""
Check Qdrant collection schema and sample point payload
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

print(f"=== Checking Qdrant Collection Schema ===")
print(f"Collection: {collection_name}")

try:
    collection_info = client.get_collection(collection_name)
    print(f"\nCollection Info:")
    print(f"  - Vectors count: {collection_info.points_count}")
    print(f"  - Vector size: {collection_info.config.params.vectors.size}")
    print(f"  - Distance: {collection_info.config.params.vectors.distance}")

    # Check payload index schema
    # Qdrant Cloud doesn't expose payload_schema directly
    # Let's check a sample point instead
    print(f"\nGetting sample point to check payload structure...")

    # Get a sample point to check actual payload structure
    points = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
    )[0]

    if points:
        point = points[0]
        print(f"\nSample Point (ID: {point.id}):")
        print(f"  All payload keys: {list(point.payload.keys())}")

        # Check if metadata.faq_type exists
        metadata = point.payload.get('metadata', {})
        print(f"  Has metadata? {bool(metadata)}")
        print(f"  metadata keys: {list(metadata.keys()) if metadata else 'N/A'}")
        print(f"  Has metadata.faq_type? {'faq_type' in metadata}")
        print(f"  metadata.faq_type value: {metadata.get('faq_type') if 'faq_type' in metadata else 'N/A'}")

        # Check root-level faq_type
        print(f"  Root faq_type: {point.payload.get('faq_type', 'N/A')}")
        print(f"  Text preview: {point.payload.get('text', '')[:100]}...")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

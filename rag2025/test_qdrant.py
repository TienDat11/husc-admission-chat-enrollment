"""Test Qdrant API"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

# Initialize
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION", "hue_admissions_2025")

print(f"Connecting to Qdrant: {qdrant_url}")

client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

# Check collection
try:
    info = client.get_collection(collection_name)
    print(f"Collection: {collection_name}, Points: {info.points_count}")
except Exception as e:
    print(f"Collection error: {e}")
    exit(1)

# Load model
print("Loading BGE-M3...")
model = SentenceTransformer('BAAI/bge-m3')

# Test query
query_text = "điều kiện xét tuyển"
query_vector = model.encode(query_text, normalize_embeddings=True)

print(f"Query: {query_text}")
print(f"Vector shape: {query_vector.shape}")

# Test query_points
print("\nTesting query_points API...")
try:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=5,
        with_payload=True,
        with_vector=False
    )
    print(f"Response type: {type(response)}")
    print(f"Has points attribute: {hasattr(response, 'points')}")
    if hasattr(response, 'points'):
        print(f"Points count: {len(response.points)}")
        for i, point in enumerate(response.points):
            print(f"  Point {i+1}: id={point.id}, score={point.score:.3f}")
            if hasattr(point, 'payload') and point.payload:
                text = point.payload.get('text', '')[:100]
                print(f"    Text: {text}...")
except Exception as e:
    import traceback
    print(f"Query points error: {e}")
    traceback.print_exc()

print("\nDone!")

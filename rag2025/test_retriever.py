"""Direct test of bge_retriever"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

print("Importing bge_retriever...")
try:
    from services.bge_retriever import bge_retriever
    print(f"bge_retriever imported: {bge_retriever}")
    print(f"qdrant_client type: {type(bge_retriever.qdrant_client)}")

    # Try a simple retrieve
    import asyncio

    async def test():
        result = await bge_retriever.retrieve("test query", "test query", top_k=3)
        print(f"Result: {result}")

    asyncio.run(test())

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()

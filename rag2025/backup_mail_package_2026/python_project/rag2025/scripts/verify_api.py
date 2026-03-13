"""
API Verification Script

Tests the running API with sample queries.

Usage:
    python scripts/verify_api.py
"""
import json
import sys
import time

import requests
from loguru import logger

API_BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    logger.info("Testing /health endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()

        data = response.json()
        logger.info(f"✅ Health check passed")
        logger.info(f"   Status: {data['status']}")
        logger.info(f"   Vector store count: {data['vector_store_count']}")
        logger.info(f"   Embedding model: {data['embedding_model']}")
        logger.info(f"   Reranker model: {data['reranker_model']}")
        logger.info(f"   BM25 index built: {data['bm25_index_built']}")

        return True

    except requests.RequestException as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


def test_query(query: str, top_k: int = 5, force_rag_only: bool = True):
    """Test query endpoint."""
    logger.info(f"\nTesting /query endpoint with: '{query}'")

    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "force_rag_only": force_rag_only,
        }

        response = requests.post(
            f"{API_BASE_URL}/query", json=payload, timeout=30
        )
        response.raise_for_status()

        data = response.json()
        logger.info(f"✅ Query successful")
        logger.info(f"   Confidence: {data['confidence']:.3f}")
        logger.info(f"   Threshold: {data['threshold']:.3f}")
        logger.info(f"   Routing: {data['routing_decision']}")
        logger.info(f"   Total results: {data['total_results']}")

        if data["total_results"] > 0:
            logger.info(f"\n   Top result:")
            top_result = data["results"][0]
            logger.info(f"      Doc ID: {top_result['doc_id']}")
            logger.info(f"      Chunk ID: {top_result['chunk_id']}")
            logger.info(f"      Score: {top_result['score']:.3f}")
            logger.info(f"      Text preview: {top_result['text'][:100]}...")

        return True

    except requests.RequestException as e:
        logger.error(f"❌ Query failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"   Response: {e.response.text}")
        return False


def main():
    """Run verification tests."""
    logger.info("=" * 60)
    logger.info("API Verification Script")
    logger.info("=" * 60)
    logger.info(f"Target: {API_BASE_URL}")

    # Wait for API to be ready
    logger.info("\nWaiting for API to be ready...")
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is ready")
                break
        except requests.RequestException:
            pass

        if i < max_retries - 1:
            logger.info(f"   Retry {i + 1}/{max_retries}...")
            time.sleep(2)
    else:
        logger.error("❌ API not ready after 10 retries")
        logger.error("   Please start the API with: python src/main.py")
        sys.exit(1)

    # Test health endpoint
    if not test_health():
        logger.error("\n❌ Health check failed, aborting")
        sys.exit(1)

    # Test query endpoint with sample queries
    test_queries = [
        "What is the admission deadline?",
        "Hạn chót đăng ký xét tuyển là khi nào?",
        "How to register for admission?",
    ]

    results = []
    for query in test_queries:
        result = test_query(query, top_k=3, force_rag_only=True)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    passed = sum(results) + (1 if test_health() else 0)
    total = len(results) + 1
    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

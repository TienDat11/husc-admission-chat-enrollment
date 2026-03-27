# Spec Review v2 — hybrid-retrieval-integration

**Phase:** spec_review (loop 2/3)

## Status: issues (10 critical architectural problems)

### Critical Issues:

1. VectorStore is abstract class - Spec shows VectorStore(settings) but VectorStore is ABC, not instantiable. Need concrete LanceDBVectorStore adapter (~100-150 LOC, not "thin wrapper").

2. Type mismatch - HybridRetriever returns SearchResult (doc_id, chunk_id, score, text, metadata) but main.py expects RetrievedDocument (text, source, chunk_id, metadata, score, point_id). Field names differ: doc_id vs source.

3. Corpus loading unsafe - Spec uses to_pandas() loading entire table into RAM. 10k chunks x 4096-dim is huge. Need scan-based API.

4. EmbeddingService wrapper unnecessary - HybridRetriever only needs encode_query(), main.py already has embedding_encoder.

5. Fusion weights wiring incomplete - No exact line numbers in retriever.py.

6. Graceful degradation underspecified - Missing empty corpus check, length mismatch handling.

7. Query endpoint oversimplified - No SearchResult to RetrievedDocument conversion code shown.

8. Settings validation missing - Weight validators needed.

9. Testing strategy lacks concrete cases.

10. Timeline unrealistic - Claims 3h, actual 6-8h.

### Required Revisions:
- Complete LanceDBVectorStore adapter spec
- Type conversion function SearchResult to RetrievedDocument
- Safe corpus loading without to_pandas()
- Exact line numbers for changes
- Settings validators
- Expanded graceful degradation
- Revised time estimates

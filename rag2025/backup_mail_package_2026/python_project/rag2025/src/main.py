"""
FastAPI Application - RAG 2025 (Updated)

Main API service with:
- HYDE query enhancement (Input Layer)
- Qwen3 embedding retrieval with LanceDB (Retrieval Layer)
- LLM answer generation (Generation Layer)
- CORS support for UI connection
"""
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import new services
from services.query_enhancer import query_enhancer
from services.lancedb_retrieval import get_retriever, LanceDBRetrieverConfig
from services.llm_generator import llm_generator
from services.reranker import RerankerService
from services.query_cache import QueryCache
from services.guardrail import GuardrailService

# Keep existing imports for backward compatibility
from config.settings import RAGSettings
from chunker import ChunkConfig, Chunker

# Embedding encoder for query embedding
from sentence_transformers import SentenceTransformer
import os

# Initialize settings
settings = RAGSettings()

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API 2025 - HYDE + Qwen3 + LanceDB",
    description="Advanced RAG with query enhancement and embedded vector retrieval",
    version="2.0.0"
)

# CORS middleware - Updated for UI support
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*",  # Fallback for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if "*" in allowed_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== API Models ==========


class SimpleQueryRequest(BaseModel):
    """Simple query request - user only needs to send a string"""
    query: str = Field(..., min_length=1, description="User's question")
    force_rag_only: Optional[bool] = Field(default=False, description="Force RAG only mode")


class QueryResponse(BaseModel):
    """Enhanced query response with all fields"""
    original_query: str
    enhanced_query: str
    query_type: str

    answer: str
    sources: List[str]
    confidence: float

    top_k_used: int
    chunks_used: int
    provider: str

    status_code: str = "SUCCESS"
    status_reason: str = ""
    data_gap_hints: List[str] = Field(default_factory=list)
    internal_status_code: Optional[str] = None

    chunks: Optional[List[Dict[str, Any]]] = None


# Legacy models for backward compatibility
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    lancedb_connected: bool = False
    vectors_count: int = 0
    collection: str = ""
    embedding_model: str = ""
    reranker_model: str = ""


class ChunkPreviewRequest(BaseModel):
    """Chunk preview request model"""
    text: str = Field(
        ...,
        min_length=10,
        description="Text to chunk (minimum 10 characters)",
        example="Trong công tác tuyển sinh năm 2025, Bộ GDĐT quy định tất cả thí sinh phải đăng ký xét tuyển trực tuyến."
    )
    profile: str = Field(
        default="auto",
        description="Chunking profile to use",
        pattern="^(auto|faq|policy)$",
        example="auto"
    )


class ChunkPreview(BaseModel):
    """Individual chunk preview"""
    chunk_id: int = Field(description="Chunk index")
    text: str = Field(description="Chunk text")
    token_count: int = Field(description="Token count for this chunk")
    char_count: int = Field(description="Character count")
    sparse_terms_count: int = Field(description="Number of BM25 sparse terms extracted")


class ChunkPreviewResponse(BaseModel):
    """Chunk preview response model"""
    text_length: int = Field(description="Original text length in characters")
    profile_used: str = Field(description="Chunking profile used")
    profile_config: Dict[str, Any] = Field(description="Profile configuration details")
    chunks: List[ChunkPreview] = Field(description="List of chunk previews")
    total_chunks: int = Field(description="Total number of chunks generated")
    total_tokens: int = Field(description="Sum of tokens across all chunks")
    avg_tokens_per_chunk: float = Field(description="Average tokens per chunk")


# ========== Global Services ==========

query_enhancer_service = None
lancedb_retriever_service = None
llm_generator_service = None
embedding_encoder: SentenceTransformer | None = None
reranker_service: RerankerService | None = None
query_cache: QueryCache | None = None
guardrail_service: GuardrailService | None = None

# GraphRAG unified pipeline
unified_pipeline = None

# Debug tools
chunk_config: ChunkConfig | None = None
chunker: Chunker | None = None


# ========== Startup/Shutdown ==========


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global query_enhancer_service, lancedb_retriever_service, llm_generator_service, embedding_encoder, unified_pipeline, reranker_service, query_cache, guardrail_service

    logger.info("=" * 60)
    logger.info("Starting RAG API 2025 v3.0 - Unified PaddedRAG + GraphRAG")
    logger.info("=" * 60)

    try:
        # Legacy HYDE enhancer
        logger.info("Initializing HYDE Query Enhancer...")
        query_enhancer_service = query_enhancer

        logger.info("Initializing Embedding Encoder...")
        embedding_encoder = SentenceTransformer(settings.QWEN_EMBEDDING_MODEL)

        logger.info("Initializing LanceDB Retriever...")
        import concurrent.futures

        def init_lancedb():
            try:
                return get_retriever()
            except Exception as e:
                logger.error(f"LanceDB initialization failed: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(init_lancedb)
            try:
                lancedb_retriever_service = future.result(timeout=10.0)
                if lancedb_retriever_service:
                    logger.info("LanceDB Retriever initialized successfully")
                else:
                    logger.warning("LanceDB not available – degraded mode")
            except concurrent.futures.TimeoutError:
                logger.warning("LanceDB initialization timeout – degraded mode")
                lancedb_retriever_service = None

        logger.info("Initializing LLM Generator...")
        llm_generator_service = llm_generator

        logger.info("Initializing Reranker Service...")
        reranker_service = RerankerService(settings)

        logger.info("Initializing Query Cache...")
        query_cache = QueryCache(ttl_seconds=settings.CACHE_TTL)

        logger.info("Initializing Guardrail Service...")
        guardrail_service = GuardrailService(settings)

        # Initialize Unified GraphRAG Pipeline
        logger.info("Initializing Unified RAG Pipeline (PaddedRAG + GraphRAG)...")
        try:
            from services.graphrag_retriever import UnifiedRAGPipeline
            unified_pipeline = UnifiedRAGPipeline.from_disk(
                alpha=settings.GRAPHRAG_ALPHA,
                beta=1.0 - settings.GRAPHRAG_ALPHA,
            )
            graph_stats = unified_pipeline._graphrag.graph_stats
            logger.info(
                f"GraphRAG Pipeline ready: "
                f"nodes={graph_stats['nodes']}, edges={graph_stats['edges']}"
            )
        except Exception as e:
            logger.warning(f"GraphRAG pipeline init failed: {e} – graph routing disabled")
            unified_pipeline = None

        logger.info("=" * 60)
        logger.info("API Ready! PaddedRAG + GraphRAG + SmartRouter")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG API 2025")


# ========== API Endpoints ==========


@app.get("/", response_model=Dict[str, Any])
async def root():
    """API info"""
    return {
        "name": "RAG API 2025",
        "version": "2.0.0",
        "features": [
            "HYDE query enhancement",
            "Qwen3 multilingual retrieval",
            "Score boosting for near-matches",
            "LanceDB embedded vector store",
            "Multi-LLM fallback (Gemini → GLM-4 → Groq)"
        ],
        "endpoints": {
            "POST /query": "Main RAG query endpoint (simple string input)",
            "GET /health": "Health check with LanceDB status",
            "GET /docs": "Swagger API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with LanceDB connection"""
    response = HealthResponse(status="unknown")

    if lancedb_retriever_service:
        try:
            collection_info = lancedb_retriever_service.check_collection()
            if collection_info.get("exists"):
                response.lancedb_connected = True
                response.vectors_count = collection_info.get("vectors_count", 0)
                response.collection = settings.LANCEDB_TABLE
                response.status = "healthy"
            else:
                response.status = f"table_not_found: {collection_info.get('error', 'unknown')}"
        except Exception as e:
            response.status = f"lancedb_error: {str(e)[:50]}"
    else:
        response.status = "lancedb_not_initialized"

    response.embedding_model = settings.EMBEDDING_MODEL
    response.reranker_model = settings.RERANKER_MODEL

    return response


@app.post("/query", response_model=QueryResponse)
async def query(request: SimpleQueryRequest):
    """
    Main RAG endpoint with HYDE multi-variant enhancement

    Flow:
    1. HYDE: user query → 3-5 query variants
    2. Embedding: encode ALL variants to vectors
    3. LanceDB retrieval: retrieve chunks for ALL variants
    4. Merge & Deduplicate: combine results from all variants
    5. LLM: generate answer from merged chunks
    """
    if query_enhancer_service is None or lancedb_retriever_service is None or llm_generator_service is None or embedding_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Please check server logs."
        )

    logger.info(f"Received query: {request.query[:100]}...")

    try:
        cache_key = f"query:{request.query.strip().lower()}|force:{request.force_rag_only}"
        if query_cache:
            cached_response = query_cache.get(cache_key)
            if cached_response is not None:
                logger.info("Query cache hit")
                return QueryResponse(**cached_response)

        precheck = await guardrail_service.precheck(request.query) if guardrail_service else None
        if precheck and not precheck.is_in_scope:
            public_code = guardrail_service.public_status(precheck.internal_code)
            internal_code = precheck.internal_code if guardrail_service.expose_internal() else None
            logger.info(
                f"guardrail_block query='{request.query[:80]}' internal_code={precheck.internal_code} reason={precheck.reason}"
            )
            response_payload = {
                "original_query": request.query,
                "enhanced_query": request.query,
                "query_type": "out_of_scope",
                "answer": precheck.short_answer,
                "sources": [],
                "confidence": 0.0,
                "top_k_used": 0,
                "chunks_used": 0,
                "provider": "Guardrail",
                "status_code": public_code,
                "status_reason": precheck.reason,
                "data_gap_hints": precheck.data_gap_hints,
                "internal_status_code": internal_code,
                "chunks": [],
            }
            if query_cache:
                query_cache.set(cache_key, response_payload)
            return QueryResponse(**response_payload)

        # Step 1: HYDE Enhancement - get query variants
        enhanced_request = await query_enhancer_service.enhance_query(
            user_query=request.query,
            force_rag_only=request.force_rag_only
        )

        variants = enhanced_request.get("variants", [])
        original_query = enhanced_request["original_query"]
        detected_intent = enhanced_request["detected_intent"]
        query_type = enhanced_request["query_type"]
        top_k = enhanced_request["top_k"]

        # ========== INTENT → FAQ_TYPE MAPPING ==========
        # Dynamic faq_type selection based on query intent
        INTENT_TO_FAQ_TYPES = {
            "overview": ["tong_hop_tuyen_sinh", "tong_hop_nhom_nganh", "thong_tin_nganh"],
            "specific_program": ["thong_tin_nganh"],
            "tuition": ["hoc_phi"],
        }

        # Keyword sets for intent detection
        OVERVIEW_KEYWORDS = [
            "bao nhiêu ngành", "tổng cộng", "có mấy ngành",
            "tổng số", "tất cả ngành", "các ngành", "tuyển sinh mấy ngành"
        ]
        TUITION_KEYWORDS = ["học phí", "tiền học", "đóng phí", "tín chỉ", "tín", "bao nhiêu tiền", "1 kỳ", "một kỳ", "học kỳ"]

        def get_faq_type_filter(query: str) -> dict | None:
            """Build metadata filter based on query intent."""
            query_lower = query.lower()

            # Overview queries - allow summary and program chunks
            if any(kw in query_lower for kw in OVERVIEW_KEYWORDS):
                return {"or_conditions": [{"faq_type": t} for t in INTENT_TO_FAQ_TYPES["overview"]]}

            # Tuition queries - only hoc_phi chunks
            if any(kw in query_lower for kw in TUITION_KEYWORDS):
                return {"or_conditions": [{"faq_type": t} for t in INTENT_TO_FAQ_TYPES["tuition"]]}

            # Specific program queries (list only, not overview)
            # Detect program list queries for metadata filtering
            is_program_list = any(
                keyword in query_lower
                for keyword in ["các ngành", "danh sách ngành", "có ngành nào", "ngành nào"]
            )
            if is_program_list:
                return {"faq_type": "thong_tin_nganh"}

            # Default: no filter, pure semantic retrieval
            return None

        # Build metadata filter for this query
        metadata_filter = get_faq_type_filter(original_query)

        logger.info(
            f"Query type: {query_type}, intent: {detected_intent}, "
            f"variants: {len(variants)}, top_k={top_k}, filter: {metadata_filter}"
        )
        logger.debug(f"Variants: {variants}")

        # DEBUG: Log variant details
        for i, variant in enumerate(variants):
            logger.info(f"  Variant {i+1}: {variant[:80]}...")

        # Step 2: Embedding Encoding - encode ALL variants to vectors
        all_chunks = []
        all_scores = []

        # For program/overview queries, use higher top_k to get more results
        is_program_query = metadata_filter is not None
        retrieval_top_k = top_k * 3 if is_program_query else top_k

        for i, variant in enumerate(variants):
            if "Qwen" in settings.EMBEDDING_MODEL:
                enriched_variant = (
                    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                    f"Query: {variant}"
                )
            else:
                enriched_variant = variant

            query_vector = embedding_encoder.encode(
                enriched_variant,
                normalize_embeddings=True,
            ).tolist()

            logger.debug(f"Encoded variant {i+1}/{len(variants)}: {variant[:50]}...")

            retrieval_result = lancedb_retriever_service.retrieve(
                query_vector=query_vector,
                top_k=retrieval_top_k,
                metadata_filter=metadata_filter,
            )

            if retrieval_result.error_type is None and len(retrieval_result.documents) == 0 and metadata_filter is not None:
                logger.warning(f"Variant {i+1}: filtered retrieval returned 0 docs, falling back to no-filter")
                retrieval_result = lancedb_retriever_service.retrieve(
                    query_vector=query_vector,
                    top_k=retrieval_top_k,
                    metadata_filter=None,
                )

            if retrieval_result.error_type is not None:
                logger.warning(f"Variant {i+1} retrieval error: {retrieval_result.error_message}")
            else:
                for doc in retrieval_result.documents:
                    all_chunks.append(doc.to_dict())
                    all_scores.append(doc.score)

        logger.info(f"Retrieved {len(all_chunks)} total chunks from {len(variants)} variants")

        # Step 4: Merge & Deduplicate chunks based on unique IDs
        if all_chunks:
            # Deduplicate by chunk ID, keeping highest score
            unique_chunks = {}
            for chunk, score in zip(all_chunks, all_scores):
                chunk_id = (
                    chunk.get("chunk_id")
                    or chunk.get("point_id")
                    or chunk.get("source")
                    or chunk.get("text", "")[:120]
                )
                if chunk_id not in unique_chunks or score > unique_chunks[chunk_id]["score"]:
                    chunk["score"] = score  # Add score for tracking
                    unique_chunks[chunk_id] = chunk

            # Sort by score and take top_k
            chunks = sorted(
                unique_chunks.values(),
                key=lambda x: x.get("score", 0),
                reverse=True
            )[:top_k]

            # Calculate average confidence
            confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0
        else:
            chunks = []
            confidence = 0.0

        if chunks and reranker_service:
            chunks = reranker_service.rerank(
                query=original_query,
                chunks=chunks,
                top_k=top_k,
            )

        status_code = "SUCCESS"
        status_reason = ""
        data_gap_hints: List[str] = []
        internal_status_code: Optional[str] = None

        if len(chunks) == 0 and guardrail_service:
            no_result = await guardrail_service.classify_no_result(original_query)
            status_code = guardrail_service.public_status(no_result.internal_code)
            status_reason = no_result.reason
            data_gap_hints = no_result.data_gap_hints
            internal_status_code = (
                no_result.internal_code if guardrail_service.expose_internal() else None
            )
            logger.info(
                f"no_result_classification query='{original_query[:80]}' internal_code={no_result.internal_code} reason={no_result.reason}"
            )

        logger.info(f"After deduplication: {len(chunks)} unique chunks, avg_confidence={confidence:.3f}")

        # Step 5: Limit context size for program/overview queries
        # For program/overview queries, use fewer chunks to avoid token limit
        max_chunks_for_generation = 15 if is_program_query else 30
        if len(chunks) > max_chunks_for_generation:
            logger.info(f"Limiting chunks for generation: {len(chunks)} -> {max_chunks_for_generation}")
            chunks = chunks[:max_chunks_for_generation]

        # Step 6: LLM Answer Generation
        result = await llm_generator_service.generate_answer(
            query=original_query,
            chunks=chunks,
            confidence=confidence,
            is_program_list_query=is_program_query,  # Pass for higher token limit on list/overview queries
        )

        logger.info(f"Generated answer using {result['provider']}")

        # Step 6: Build response
        response_payload = {
            "original_query": original_query,
            "enhanced_query": f"({len(variants)} variants) " + "; ".join([v[:30] for v in variants[:2]]),
            "query_type": query_type,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": confidence,
            "top_k_used": top_k,
            "chunks_used": result["chunks_used"],
            "provider": result["provider"],
            "status_code": status_code,
            "status_reason": status_reason,
            "data_gap_hints": data_gap_hints,
            "internal_status_code": internal_status_code,
            "chunks": chunks,
        }

        if query_cache:
            query_cache.set(cache_key, response_payload)

        return QueryResponse(**response_payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ========== Debug/Tools Endpoints ==========


@app.post("/debug/preview-chunks", response_model=ChunkPreviewResponse, tags=["Debug/Tools"])
async def preview_chunks(request: ChunkPreviewRequest):
    """
    Preview chunking results without saving to disk

    This endpoint is useful for UAT and testing chunking quality
    """
    global chunk_config, chunker

    try:
        # Lazy load chunker on first use
        if chunk_config is None or chunker is None:
            config_path = Path(__file__).parent / "config" / "chunk_profiles.yaml"
            if not config_path.exists():
                config_path = Path(__file__).parent.parent / "config" / "chunk_profiles.yaml"

            if not config_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Chunk profiles config not found"
                )

            logger.info(f"Loading chunk config from: {config_path}")
            chunk_config = ChunkConfig(config_path)
            chunker = Chunker(chunk_config)

        # Validate profile
        profile_name = request.profile.lower()
        if profile_name not in chunk_config.profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid profile '{profile_name}'. Must be one of: {list(chunk_config.profiles.keys())}"
            )

        # Get profile config
        profile_config = chunk_config.get_profile(profile_name)

        # Create mock document
        mock_doc = {
            "id": "preview_doc",
            "text": request.text,
            "metadata": {"source": "preview_endpoint"}
        }

        # Chunk the document
        logger.info(f"Chunking text ({len(request.text)} chars) with profile: {profile_name}")
        chunks = chunker.chunk_document(mock_doc, profile_name=profile_name)

        # Build preview response
        chunk_previews: List[ChunkPreview] = []
        total_tokens = 0

        for chunk in chunks:
            token_count = len(chunker.tokenizer.encode(chunk.text))
            total_tokens += token_count

            preview = ChunkPreview(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                token_count=token_count,
                char_count=len(chunk.text),
                sparse_terms_count=len(chunk.sparse_terms)
            )
            chunk_previews.append(preview)

        # Calculate average
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        return ChunkPreviewResponse(
            text_length=len(request.text),
            profile_used=profile_name,
            profile_config={
                "description": profile_config.description,
                "chunk_size": profile_config.chunk_size,
                "overlap": profile_config.overlap,
                "min_tokens": profile_config.min_tokens,
                "separator_priority": profile_config.separator_priority,
            },
            chunks=chunk_previews,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            avg_tokens_per_chunk=avg_tokens
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chunk preview failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chunk preview failed: {str(e)}")


# ========== Development Server ==========


# ========== GraphRAG v2 Endpoint ==========


class UnifiedQueryRequest(BaseModel):
    """Unified query request for PaddedRAG + GraphRAG pipeline."""
    query: str = Field(..., min_length=1, description="User question (Vietnamese)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    force_route: Optional[str] = Field(
        default=None,
        description="Force routing: 'padded_rag' or 'graph_rag' (overrides auto-router)"
    )


class UnifiedQueryResponse(BaseModel):
    """Response from unified pipeline with routing metadata."""
    query: str
    route: str
    answer: str
    sources: List[str]
    confidence: float
    router_info: Dict[str, Any]
    graph_stats: Optional[Dict[str, Any]] = None
    latency_ms: float


@app.post("/v2/query", response_model=UnifiedQueryResponse)
async def unified_query(request: UnifiedQueryRequest):
    """Unified RAG query: auto-routes to PaddedRAG or GraphRAG.

    - Simple 1-hop queries → PaddedRAG (faster, ~150ms)
    - Multi-hop / comparative → GraphRAG (+PPR, ~400ms)

    Routing is performed by HyDE + Step-Back classification using gemini-2.5-flash.
    """
    if not unified_pipeline:
        raise HTTPException(
            status_code=503,
            detail="Unified pipeline not initialized. Check RAMCLOUDS_API_KEY in .env"
        )

    # 1. Get PaddedRAG baseline documents first (from LanceDB)
    baseline_docs = []
    if lancedb_retriever_service and embedding_encoder:
        try:
            baseline_query = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                f"Query: {request.query}"
            )
            query_vec = embedding_encoder.encode(
                baseline_query,
                normalize_embeddings=True,
            ).tolist()
            result = lancedb_retriever_service.retrieve(
                query_vector=query_vec,
                top_k=request.top_k * 3,
            )
            if result.is_success:
                baseline_docs = result.documents
        except Exception as exc:
            logger.warning(f"LanceDB retrieval failed: {exc}")

    # 2. Run unified pipeline (router + optional graph fusion)
    try:
        rag_result = await unified_pipeline.query(
            user_query=request.query,
            baseline_docs=baseline_docs,
            top_k=request.top_k,
        )

        if request.force_route == "padded_rag":
            rag_result.route = "padded_rag"
            rag_result.documents = baseline_docs[:request.top_k]
            rag_result.ppr_scores = {}
            rag_result.confidence = rag_result.documents[0].score if rag_result.documents else 0.0
            logger.info("Route override applied: padded_rag")
        elif request.force_route == "graph_rag" and baseline_docs:
            docs, ppr_scores = await unified_pipeline._graphrag.retrieve(
                query=request.query,
                router_result=rag_result.router_result,
                baseline_docs=baseline_docs,
                top_k=request.top_k,
            )
            rag_result.route = "graph_rag"
            rag_result.documents = docs
            rag_result.ppr_scores = ppr_scores
            rag_result.confidence = docs[0].score if docs else 0.0
            logger.info("Route override applied: graph_rag")

        # 3. Generate answer from top documents
        context_texts = [d.text for d in rag_result.documents[:5]]
        context = "\n\n".join(context_texts)

        answer = "Không có thông tin phù hợp."
        if context:
            try:
                from services.llm_client import get_llm_client
                llm = get_llm_client()
                resp = await llm.chat(
                    user_message=f"Câu hỏi: {request.query}\n\nNgữ cảnh:\n{context}",
                    system_message=(
                        "Bạn là trợ lý tư vấn tuyển sinh Đại học Khoa học Huế (HUSC). "
                        "Trả lời câu hỏi dựa HOÀN TOÀN vào ngữ cảnh được cung cấp. "
                        "Nếu ngữ cảnh không đủ thông tin, nói rõ điều đó. "
                        "Trả lời bằng tiếng Việt, súc tích (50-150 từ)."
                    ),
                    temperature=0.1,
                    max_tokens=512,
                )
                answer = resp.content
            except Exception as exc:
                logger.warning(f"Answer generation failed: {exc}")
                answer = context[:500] if context else "Không có thông tin."

        sources = list({
            d.metadata.get("source", d.chunk_id or "unknown")
            for d in rag_result.documents
        })

        return UnifiedQueryResponse(
            query=request.query,
            route=rag_result.route,
            answer=answer,
            sources=sources,
            confidence=rag_result.confidence,
            router_info={
                "step_back": rag_result.router_result.step_back_query,
                "intent": rag_result.router_result.intent,
                "complexity": rag_result.router_result.complexity,
                "reasoning": rag_result.router_result.reasoning,
                "hyde_variants": rag_result.router_result.hyde_variants,
            },
            graph_stats=unified_pipeline._graphrag.graph_stats,
            latency_ms=rag_result.latency_ms,
        )

    except Exception as exc:
        logger.error(f"Unified query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v2/graph/update")
async def incremental_graph_update(request: Dict[str, Any]):
    """Incrementally add new chunks to the knowledge graph.

    Request body: {"chunks": [{"id": "...", "text": "...", "faq_type": "..."}]}

    Enables 10-20 year scalability: add new admission data without full rebuild.
    """
    if not unified_pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    raw_chunks = request.get("chunks", [])
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")

    from src.domain.entities import Chunk as DomainChunk
    new_chunks = [
        DomainChunk(
            chunk_id=c["id"],
            text=c.get("text", ""),
            faq_type=c.get("faq_type", ""),
            metadata=c.get("metadata", {}),
        )
        for c in raw_chunks
        if c.get("text")
    ]

    await unified_pipeline.incremental_update(new_chunks)
    return {
        "status": "ok",
        "chunks_added": len(new_chunks),
        "graph_stats": unified_pipeline._graphrag.graph_stats,
    }


@app.get("/v2/graph/stats")
async def graph_stats():
    """Return current knowledge graph statistics."""
    if not unified_pipeline:
        return {"error": "Pipeline not initialized"}
    return unified_pipeline._graphrag.graph_stats


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )

"""
FastAPI Application - RAG 2025 (Updated)

Main API service with:
- HYDE query enhancement (Input Layer)
- BGE multi-vector retrieval with Qdrant (Retrieval Layer)
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
# Phase 7: Use ONLY qdrant_retrieval with query_points() API
from services.qdrant_retrieval import get_retriever, QdrantRetrieverConfig
from services.llm_generator import llm_generator

# Keep existing imports for backward compatibility
from config.settings import RAGSettings
from chunker import ChunkConfig, Chunker

# BGE-M3 encoder for query embedding
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
    title="RAG API 2025 - HYDE + BGE + Qdrant",
    description="Advanced RAG with query enhancement and multi-vector retrieval",
    version="2.0.0"
)

# CORS middleware - Updated for UI support
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"  # Fallback for development
    ],
    allow_credentials=True,
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
    # Original query info
    original_query: str
    enhanced_query: str
    query_type: str

    # Answer
    answer: str
    sources: List[str]
    confidence: float

    # Metadata
    top_k_used: int
    chunks_used: int
    provider: str

    # Retrieved chunks (optional, for debugging)
    chunks: Optional[List[Dict[str, Any]]] = None


# Legacy models for backward compatibility
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    qdrant_connected: bool = False
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

# Phase 7: New services (qdrant_retrieval with query_points() API ONLY)
query_enhancer_service = None
qdrant_retriever_service = None
llm_generator_service = None
bge_encoder: SentenceTransformer | None = None

# Debug tools
chunk_config: ChunkConfig | None = None
chunker: Chunker | None = None


# ========== Startup/Shutdown ==========


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global query_enhancer_service, qdrant_retriever_service, llm_generator_service, bge_encoder

    logger.info("=" * 60)
    logger.info("Starting RAG API 2025 v2.0 - Phase 7 (query_points API)")
    logger.info("=" * 60)

    try:
        # Initialize new RAG 2.0 services
        logger.info("Initializing HYDE Query Enhancer...")
        query_enhancer_service = query_enhancer

        logger.info("Initializing BGE-M3 Encoder...")
        bge_encoder = SentenceTransformer(settings.BGE_MODEL)

        logger.info("Initializing Qdrant Retriever (query_points API ONLY)...")
        qdrant_retriever_service = get_retriever()

        logger.info("Initializing LLM Generator...")
        llm_generator_service = llm_generator

        logger.info("=" * 60)
        logger.info("API Ready!")
        logger.info("=" * 60)
        logger.info(f"Features: HYDE + BGE-M3 + Qdrant (query_points) + Multi-LLM fallback")
        logger.info(f"CORS enabled for UI connection")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - allow API to start in degraded mode


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
            "BGE multi-vector retrieval",
            "Score boosting for near-matches",
            "Qdrant vector store",
            "Multi-LLM fallback (Gemini → GLM-4 → Groq)"
        ],
        "endpoints": {
            "POST /query": "Main RAG query endpoint (simple string input)",
            "GET /health": "Health check with Qdrant status",
            "GET /docs": "Swagger API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with Qdrant connection"""
    response = HealthResponse(status="unknown")

    # Check Qdrant connection via qdrant_retriever_service
    if qdrant_retriever_service:
        try:
            collection_info = qdrant_retriever_service.check_collection()
            if collection_info.get("exists"):
                response.qdrant_connected = True
                response.vectors_count = collection_info.get("vectors_count", 0)
                response.collection = settings.QDRANT_COLLECTION
                response.status = "healthy"
            else:
                response.status = f"collection_not_found: {collection_info.get('error', 'unknown')}"
        except Exception as e:
            response.status = f"qdrant_error: {str(e)[:50]}"
    else:
        response.status = "qdrant_not_initialized"

    # Add model info
    response.embedding_model = settings.EMBEDDING_MODEL
    response.reranker_model = settings.RERANKER_MODEL

    return response


@app.post("/query", response_model=QueryResponse)
async def query(request: SimpleQueryRequest):
    """
    Main RAG endpoint with HYDE multi-variant enhancement

    Flow:
    1. HYDE: user query → 3-5 query variants
    2. BGE Encoding: encode ALL variants to vectors
    3. Qdrant Retrieval (query_points API): retrieve chunks for ALL variants
    4. Merge & Deduplicate: combine results from all variants
    5. LLM: generate answer from merged chunks
    """
    if query_enhancer_service is None or qdrant_retriever_service is None or llm_generator_service is None or bge_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized. Please check server logs."
        )

    logger.info(f"Received query: {request.query[:100]}...")

    try:
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

        # Step 2: BGE Encoding - encode ALL variants to vectors
        all_chunks = []
        all_scores = []

        # For program/overview queries, use higher top_k to get more results
        is_program_query = metadata_filter is not None
        retrieval_top_k = top_k * 3 if is_program_query else top_k

        for i, variant in enumerate(variants):
            query_vector = bge_encoder.encode(
                variant,
                normalize_embeddings=True
            ).tolist()

            logger.debug(f"Encoded variant {i+1}/{len(variants)}: {variant[:50]}...")

            # Step 3: Qdrant Retrieval via query_points() API for this variant
            # Note: faq_type is at ROOT level, NOT nested in metadata
            retrieval_result = qdrant_retriever_service.retrieve(
                query_vector=query_vector,
                top_k=retrieval_top_k,
                metadata_filter=metadata_filter,
            )

            # SAFE FALLBACK: If filter returns 0 documents, retry without filter
            if retrieval_result.error_type is None and len(retrieval_result.documents) == 0 and metadata_filter is not None:
                logger.warning(f"Variant {i+1}: filtered retrieval returned 0 docs, falling back to no-filter")
                retrieval_result = qdrant_retriever_service.retrieve(
                    query_vector=query_vector,
                    top_k=retrieval_top_k,
                    metadata_filter=None,
                )

            # Check for errors
            if retrieval_result.error_type is not None:
                logger.warning(f"Variant {i+1} retrieval error: {retrieval_result.error_message}")
            else:
                # Collect documents with scores
                for doc in retrieval_result.documents:
                    all_chunks.append(doc.to_dict())
                    all_scores.append(doc.score)

        logger.info(f"Retrieved {len(all_chunks)} total chunks from {len(variants)} variants")

        # Step 4: Merge & Deduplicate chunks based on unique IDs
        if all_chunks:
            # Deduplicate by chunk ID, keeping highest score
            unique_chunks = {}
            for chunk, score in zip(all_chunks, all_scores):
                chunk_id = chunk.get("id")
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
        return QueryResponse(
            original_query=original_query,
            enhanced_query=f"({len(variants)} variants) " + "; ".join([v[:30] for v in variants[:2]]),
            query_type=query_type,
            answer=result["answer"],
            sources=result["sources"],
            confidence=confidence,
            top_k_used=top_k,
            chunks_used=result["chunks_used"],
            provider=result["provider"],
            chunks=chunks  # Include for debugging
        )

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

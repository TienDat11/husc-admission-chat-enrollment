"""
Graph Builder Script – Offline Knowledge Graph Construction

Pipeline:
1. Load JSONL chunks from data/chunked/
2. Extract entities & triples via NER (RAMCLOUDS_MODEL via ramclouds.me/v1)
3. Build NetworkX MultiDiGraph (schema-versioned, edge-key deduplication)
4. Save to data/graph/knowledge_graph.graphml + entity_index.json

Incremental mode (--incremental):
  Loads existing graph and ONLY processes NEW chunks (not in entity_index).
  Safe to run multiple times – idempotent.

Usage:
    cd rag2025
    python scripts/build_graph.py                  # full build
    python scripts/build_graph.py --incremental    # append new chunks only
    python scripts/build_graph.py --dry-run        # load chunks only, skip NER
    python scripts/build_graph.py --limit 20       # process only 20 chunks
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before any service imports (env vars needed by llm_client.py)
from src.config import env_loader  # noqa: F401

from src.domain.entities import Chunk
from src.domain.graph import KnowledgeGraph, KnowledgeGraphBuilder
from src.services.llm_client import get_llm_client
from src.services.ner_service import NERService


CHUNKED_DIR = Path(__file__).parent.parent / "data" / "chunked"
GRAPH_DIR = Path(__file__).parent.parent / "data" / "graph"
GRAPH_PATH = GRAPH_DIR / "knowledge_graph.graphml"
ENTITY_INDEX_PATH = GRAPH_DIR / "entity_index.json"


def load_all_chunks(limit: int = 0) -> list[Chunk]:
    """Load all chunks from JSONL files in data/chunked/."""
    chunks: list[Chunk] = []
    jsonl_files = sorted(CHUNKED_DIR.glob("chunked_*.jsonl"))
    jsonl_files = [f for f in jsonl_files if "enhanced" not in f.name]

    for filepath in jsonl_files:
        with filepath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text_plain") or data.get("text", "")
                    if not text:
                        continue
                    chunks.append(Chunk(
                        chunk_id=data["id"],
                        text=text,
                        faq_type=data.get("faq_type", ""),
                        metadata=data.get("metadata", {}),
                    ))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(f"Skipping malformed chunk in {filepath.name}: {exc}")

        if limit and len(chunks) >= limit:
            chunks = chunks[:limit]
            break

    logger.info(f"Loaded {len(chunks)} chunks from {len(jsonl_files)} JSONL files")
    return chunks


def load_entity_index() -> dict:
    """Load existing entity index (for incremental mode)."""
    if ENTITY_INDEX_PATH.exists():
        with ENTITY_INDEX_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_entity_index(results: list) -> dict:
    """Build entity_index.json: node_id → {text, entity_type, chunk_ids}."""
    index: dict = {}
    for result in results:
        if not result.is_success:
            continue
        for entity in result.entities:
            node_id = entity.node_id
            if node_id not in index:
                index[node_id] = {
                    "text": entity.text,
                    "entity_type": entity.entity_type.value,
                    "normalized": entity.normalized,
                    "chunk_ids": [],
                }
            if entity.chunk_id not in index[node_id]["chunk_ids"]:
                index[node_id]["chunk_ids"].append(entity.chunk_id)
    return index


def merge_entity_indexes(old: dict, new: dict) -> dict:
    """Merge two entity indexes (for incremental mode)."""
    merged = dict(old)
    for node_id, attrs in new.items():
        if node_id in merged:
            merged[node_id]["chunk_ids"] = list(
                set(merged[node_id]["chunk_ids"]) | set(attrs["chunk_ids"])
            )
        else:
            merged[node_id] = attrs
    return merged


async def main(dry_run: bool, limit: int, incremental: bool) -> None:
    logger.info("=== Graph Builder: Starting ===")
    logger.info(f"  Mode: {'incremental' if incremental else 'full'} | dry_run={dry_run} | limit={limit or 'all'}")

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = load_all_chunks(limit=limit)
    if not all_chunks:
        logger.error("No chunks found. Check data/chunked/ directory.")
        return

    # Incremental: skip chunks already in entity index
    if incremental and ENTITY_INDEX_PATH.exists():
        existing_index = load_entity_index()
        processed_chunks = {
            cid
            for attrs in existing_index.values()
            for cid in attrs.get("chunk_ids", [])
        }
        chunks = [c for c in all_chunks if c.chunk_id not in processed_chunks]
        logger.info(
            f"Incremental: {len(processed_chunks)} already processed, "
            f"{len(chunks)} new chunks to process"
        )
        if not chunks:
            logger.info("No new chunks to process. Graph is up to date.")
            return
    else:
        existing_index = {}
        chunks = all_chunks

    if dry_run:
        logger.info(f"[DRY RUN] Would process {len(chunks)} chunks. Exiting.")
        return

    llm = get_llm_client()
    providers = getattr(llm, "_providers", [])
    if not providers:
        logger.error(
            "No LLM provider configured for graph build. "
            "Set RAMCLOUDS_API_KEY (or OPENAI_API_KEY), RAMCLOUDS_BASE_URL, and RAMCLOUDS_MODEL in .env"
        )
        raise RuntimeError("Graph build aborted: no LLM provider configured")

    primary = providers[0]
    logger.info(
        f"NER provider: {primary.name} | model: {primary.model} | base_url: {primary.base_url}"
    )

    ner = NERService(llm=llm)
    logger.info(f"Step 1: NER extraction ({len(chunks)} chunks via {primary.model})...")
    results = await ner.extract_batch(chunks)

    success = sum(1 for r in results if r.is_success)
    failed = len(results) - success
    logger.info(f"NER: {success} success, {failed} failed")

    if success == 0:
        raise RuntimeError(
            "NER extraction failed for all chunks. "
            "Check provider credentials/model access and rerun graph build."
        )

    logger.info("Step 2: Building knowledge graph...")
    if incremental and GRAPH_PATH.exists():
        existing_graph = KnowledgeGraph.load(GRAPH_PATH)
        builder = KnowledgeGraphBuilder.from_graph(existing_graph)
        logger.info(f"  Incremental: starting from {existing_graph.node_count} nodes")
    else:
        builder = KnowledgeGraphBuilder()

    total_entities = 0
    total_triples = 0
    for result in results:
        if not result.is_success:
            continue
        builder.add_entities(result.entities)
        builder.add_triples(result.triples)
        total_entities += len(result.entities)
        total_triples += len(result.triples)

    knowledge_graph = builder.build()
    logger.info(
        f"Graph: {knowledge_graph.node_count} nodes, {knowledge_graph.edge_count} edges "
        f"(added {total_entities} entities, {total_triples} triples)"
    )

    logger.info(f"Step 3: Saving graph → {GRAPH_PATH}")
    knowledge_graph.save(GRAPH_PATH)

    logger.info(f"Step 4: Saving entity index → {ENTITY_INDEX_PATH}")
    new_index = build_entity_index(results)
    merged_index = merge_entity_indexes(existing_index, new_index)
    ENTITY_INDEX_PATH.write_text(
        json.dumps(merged_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = knowledge_graph.stats()
    logger.info("=== Graph Builder: Complete ===")
    logger.info(f"  Nodes        : {stats['nodes']}")
    logger.info(f"  Edges        : {stats['edges']}")
    logger.info(f"  Avg degree   : {stats['avg_degree']:.2f}")
    logger.info(f"  Max degree   : {stats['max_degree']}")
    logger.info(f"  Schema ver   : {stats['schema_version']}")
    logger.info(f"  Entity index : {len(merged_index)} unique entities")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build / update GraphRAG knowledge graph")
    parser.add_argument("--dry-run", action="store_true", help="Skip NER and graph build")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks (0=all)")
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only process NEW chunks not in existing entity index"
    )
    args = parser.parse_args()

    asyncio.run(main(dry_run=args.dry_run, limit=args.limit, incremental=args.incremental))

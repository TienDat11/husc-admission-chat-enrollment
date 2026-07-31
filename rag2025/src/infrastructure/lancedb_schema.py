# @spec(S13) unified chunk schema for LanceDB — M3-B1
"""Canonical pyarrow schema for the unified husc_chunks table.

17 fields: v2 §M3 (14) + superseded_by (v3 §2) + is_reference + crawl_date. Flat columns for filter
pushdown — NO nested struct. effective_from/effective_to are int32 YYYYMMDD.
ponytail: ceiling is year 9999; upgrade to pa.date32() if SQL date arithmetic needed.
"""
import pyarrow as pa

EMBEDDING_DIM = 1024  # KHOÁ — khớp bảng production + .env

CHUNK_SCHEMA = pa.schema([
    pa.field("id",             pa.string(), nullable=False),
    pa.field("doc_id",         pa.string(), nullable=False),
    pa.field("release_id",     pa.string(), nullable=True),
    pa.field("text",           pa.string(), nullable=False),
    pa.field("summary",        pa.string()),
    pa.field("vector",         pa.list_(pa.float32(), EMBEDDING_DIM), nullable=False),
    # ---- Cột phẳng cho filter pushdown ----
    pa.field("data_year",      pa.int32(),  nullable=False),
    pa.field("scope",          pa.string(), nullable=True),
    pa.field("doc_type",       pa.string(), nullable=True),
    pa.field("status",         pa.string(), nullable=True),
    pa.field("effective_from", pa.int32(),  nullable=True),
    pa.field("effective_to",   pa.int32(),  nullable=True),
    pa.field("is_tombstone",   pa.bool_(),  nullable=True),
    pa.field("superseded_by",  pa.string()),
    pa.field("metadata_json",  pa.string()),
    pa.field("is_reference",   pa.bool_(),  nullable=True),
    pa.field("crawl_date",     pa.string(), nullable=True),
])

# Index targets (BTREE cho range/equality, BITMAP cho low-cardinality)
BTREE_COLUMNS = ["id", "doc_id", "data_year", "effective_from", "effective_to", "superseded_by"]
BITMAP_COLUMNS = ["scope", "doc_type", "status", "is_tombstone", "is_reference"]
ALL_INDEX_COLUMNS = BTREE_COLUMNS + BITMAP_COLUMNS  # 10 columns, ≥8 indices expected

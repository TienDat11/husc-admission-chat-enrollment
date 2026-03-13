#!/usr/bin/env python3
"""
Enhance chunked_10.jsonl with missing fields required for Qdrant ingestion
"""

import json
import re
from datetime import datetime

# Read original chunks
original_chunks = []
with open(r"D:\chunking\rag2025_2\rag2025\data\chunked\chunked_10.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            original_chunks.append(json.loads(line))

# Helper to extract sparse terms from text
def extract_sparse_terms(text: str, max_terms: int = 10) -> list:
    """Extract meaningful keywords from text for sparse search"""
    # Remove special characters, keep Vietnamese words
    words = re.findall(r'\b[a-zA-ZÀ-Ỹà-ỹ0-9]+\b', text.lower())
    # Filter out common stop words
    stop_words = {'la', 'va', 'cac', 'cua', 'co', 'voi', 'theo', 'cho', 'trong',
                  'nam', 'duoc', 'nghanh', 'tuong', 'hoc', 'tuyen', 'sinh',
                  'vien', 'chi', 'tieu', 'khoa', 'truc', 'dai', 'hue', 'hoc'}

    # Keep meaningful terms (2+ chars, not in stop words)
    terms = [w for w in words if len(w) >= 3 and w not in stop_words]

    # Return unique terms, prioritized by frequency
    from collections import Counter
    term_counts = Counter(terms)
    return [term for term, _ in term_counts.most_common(max_terms)]

# Map FAQ type to info_type
FAQ_TYPE_TO_INFO_TYPE = {
    "thong_tin_nganh": "thong_tin_nganh",
    "tong_hop_tuyen_sinh": "chi_tieu_tong",
    "tong_hop_nhom_nganh": "nhom_nganh",
    "tong_hop_to_hop": "danh_sach_to_hop",
    "hoc_phi": "hoc_phi"
}

# Map FAQ type to audience
FAQ_TYPE_TO_AUDIENCE = {
    "thong_tin_nganh": "thi_sinh",
    "tong_hop_tuyen_sinh": "thi_sinh",
    "tong_hop_nhom_nganh": "thi_sinh",
    "tong_hop_to_hop": "thi_sinh",
    "hoc_phi": "thi_sinh"
}

# Enhanced chunks list
enhanced_chunks = []

for idx, chunk in enumerate(original_chunks):
    chunk_id = chunk["id"]
    faq_type = chunk.get("faq_type", "")
    name = chunk.get("metadata", {}).get("name", "")

    # Determine doc_id (use a common doc for HUSC 2025 data)
    doc_id = "husc_tuyen_sinh_2025"

    # Generate sparse terms
    text_for_terms = chunk.get("text", "") + " " + chunk.get("summary", "")
    sparse_terms = extract_sparse_terms(text_for_terms)

    # Generate breadcrumbs
    breadcrumbs = ["Đại học Huế (HUSC) - Tuyển sinh 2025"]
    if name:
        breadcrumbs.append(name)
    if faq_type:
        breadcrumbs.append(faq_type.replace("_", " ").title())

    # Determine info_type and audience
    info_type = FAQ_TYPE_TO_INFO_TYPE.get(faq_type, "thong_tin_chung")
    audience = FAQ_TYPE_TO_AUDIENCE.get(faq_type, "thi_sinh")

    # Create enhanced chunk
    enhanced_chunk = {
        **chunk,  # Keep all existing fields
        "doc_id": doc_id,
        "chunk_id": idx,
        "breadcrumbs": breadcrumbs,
        "sparse_terms": sparse_terms,
        "prev_chunk_id": f"{doc_id}_chunk_{idx-1}" if idx > 0 else None,
        "next_chunk_id": f"{doc_id}_chunk_{idx+1}" if idx < len(original_chunks) - 1 else None,
        "metadata": {
            **chunk.get("metadata", {}),
            "audience": audience,
            "info_type": info_type,
            "effective_date": "2025-01-01",
            "expired": False
        }
    }

    enhanced_chunks.append(enhanced_chunk)

    print(f"Processed chunk {idx + 1}/{len(original_chunks)}: {chunk_id}")

# Write to new file
output_path = r"D:\chunking\rag2025_2\rag2025\data\chunked\chunked_10_enhanced.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in enhanced_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"\n✓ Enhanced chunks saved to: {output_path}")
print(f"✓ Total chunks: {len(enhanced_chunks)}")

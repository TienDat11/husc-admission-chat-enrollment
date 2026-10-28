"""Probe 3: Embedding identity — which model produced vector[1024] in LanceDB?

Steps:
1. Read 5 chunks from LANCEDB_TABLE (rag2025) with stored vectors.
2. Re-embed using the SAME EmbeddingService.encode_documents() function.
3. Compute cosine(stored, re-embedded) per chunk.
4. Verdict: >0.99 = MATCH, <0.90 = REEMBED REQUIRED, else AMBIGUOUS.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rag2025"))
sys.path.insert(0, PROJECT_ROOT)
db_path = os.path.join(PROJECT_ROOT, "data", "lancedb")

from config.settings import settings
from services.embedding import EmbeddingService
import lancedb


def cosine(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def main():
    db = lancedb.connect(db_path)
    table_name = settings.LANCEDB_TABLE
    print(f"TABLE: {table_name}")
    t = db.open_table(table_name)
    df = t.to_pandas()
    print(f"TOTAL_ROWS: {len(df)}")
    vc = df["vector"].iloc[0]
    vdim = len(vc) if len(df) and vc is not None else "N/A"
    print(f"VECTOR_COL: vector")
    print(f"VECTOR_DIM: {vdim}")

    svc = EmbeddingService(settings)
    model_name = settings.EMBEDDING_MODEL
    expected_dim = settings.EMBEDDING_DIM
    provider = settings.EMBEDDING_PROVIDER
    print(f"MODEL_CONFIG: {model_name}")
    print(f"DIM_CONFIG: {expected_dim}")
    print(f"PROVIDER: {provider}")

    valid = df[df["vector"].notna()].head(5)
    texts = valid["text"].tolist()
    ids = valid["id"].tolist()
    stored_vecs = [np.array(valid["vector"].iloc[i], dtype=np.float32) for i in range(len(valid))]

    print(f"\nRE_EMBEDDING {len(texts)} chunks...")
    re_embedded = svc.encode_documents(texts)
    print(f"RE_EMBEDDED_SHAPE: {re_embedded.shape}")

    results = []
    for idx in range(len(valid)):
        c = cosine(stored_vecs[idx], re_embedded[idx])
        results.append((ids[idx], c))
        print(f"CHUNK_{idx+1}: id={ids[idx]} cosine={c:.6f}")

    avg_cosine = np.mean([r[1] for r in results])
    min_cosine = np.min([r[1] for r in results])
    max_cosine = np.max([r[1] for r in results])
    print(f"\nAVG_COSINE: {avg_cosine:.6f}")
    print(f"MIN_COSINE: {min_cosine:.6f}")
    print(f"MAX_COSINE: {max_cosine:.6f}")

    if avg_cosine > 0.99:
        verdict = "MATCH - no re-embed needed"
    elif avg_cosine < 0.90:
        verdict = "MISMATCH - full re-embed required"
    else:
        verdict = "AMBIGUOUS - inconclusive"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()

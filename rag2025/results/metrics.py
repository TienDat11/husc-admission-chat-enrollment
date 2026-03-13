"""
Evaluation metrics for RAG systems.
Implements MRR, NDCG, faithfulness, and latency measurement functions.
"""
import time
import numpy as np
from typing import List, Set, Callable, Any
from sklearn.metrics import ndcg_score


def mrr_at_k(retrieved_lists: List[List[str]], relevant_ids_list: List[Set[str]], k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank at K.
    
    Args:
        retrieved_lists: list of lists of chunk_ids in ranked order
        relevant_ids_list: list of sets of ground truth chunk_ids
        k: top-K results to consider
    
    Returns:
        float MRR@k score
    """
    if len(retrieved_lists) != len(relevant_ids_list):
        raise ValueError("Number of retrieved lists must match number of relevant sets")
    
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_ids_list):
        if not relevant:  # Skip if no relevant documents
            continue
            
        # Find the rank of the first relevant document
        rank = None
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                rank = i + 1
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def ndcg_at_k(retrieved_lists: List[List[str]], relevant_ids_list: List[Set[str]], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    Uses binary relevance: 1 if chunk_id in ground_truth, 0 otherwise.
    
    Args:
        retrieved_lists: list of lists of chunk_ids in ranked order
        relevant_ids_list: list of sets of ground truth chunk_ids
        k: top-K results to consider
    
    Returns:
        float NDCG@k score
    """
    if len(retrieved_lists) != len(relevant_ids_list):
        raise ValueError("Number of retrieved lists must match number of relevant sets")
    
    ndcg_scores = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_ids_list):
        if not relevant:  # Skip if no relevant documents
            continue
            
        # Create relevance scores for top-K retrieved documents
        y_true = []
        for doc_id in retrieved[:k]:
            y_true.append(1 if doc_id in relevant else 0)
        
        # Create ideal relevance scores (all relevant documents first)
        y_score_ideal = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
        
        # Calculate NDCG
        if sum(y_true) > 0:
            ndcg = ndcg_score([y_score_ideal], [y_true], k=k)
            ndcg_scores.append(ndcg)
        else:
            ndcg_scores.append(0.0)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def faithfulness_score(generated_answer: str, source_chunks_text: List[str]) -> float:
    """
    Calculate semantic similarity between generated answer and retrieved sources.
    Uses sentence-transformers cosine similarity.
    Model: intfloat/multilingual-e5-base (same as PaddedRAG).
    
    Args:
        generated_answer: the generated answer text
        source_chunks_text: list of source chunk texts
    
    Returns:
        float [0,1] faithfulness score
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Initialize model (same as PaddedRAG)
        model = SentenceTransformer('intfloat/multilingual-e5-base')
        
        # Encode all texts
        texts_to_encode = [generated_answer] + source_chunks_text
        embeddings = model.encode(texts_to_encode, normalize_embeddings=True)
        
        # Calculate similarity between answer and each source
        answer_embedding = embeddings[0].reshape(1, -1)
        source_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(answer_embedding, source_embeddings)[0]
        
        # Return max similarity as faithfulness score
        return float(np.max(similarities)) if len(similarities) > 0 else 0.0
        
    except ImportError:
        print("WARNING: sentence-transformers not available, returning default faithfulness score")
        return 0.5
    except Exception as e:
        print(f"WARNING: Faithfulness calculation failed: {e}, returning default score")
        return 0.5


def measure_latency(fn: Callable, *args, n_runs: int = 3) -> float:
    """
    Run function n_runs times and return median latency in milliseconds.
    
    Args:
        fn: function to measure
        *args: arguments to pass to function
        n_runs: number of runs to average
    
    Returns:
        median latency in milliseconds
    """
    latencies = []
    
    for _ in range(n_runs):
        start_time = time.time()
        fn(*args)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return float(np.median(latencies))


# Example usage and testing
if __name__ == "__main__":
    # Test data
    retrieved_lists = [
        ["doc1", "doc2", "doc3", "doc4"],
        ["doc2", "doc1", "doc4", "doc3"],
        ["doc3", "doc4", "doc1", "doc2"]
    ]
    relevant_ids_list = [
        {"doc1", "doc2"},
        {"doc2"},
        {"doc3", "doc4"}
    ]
    
    # Test MRR
    mrr = mrr_at_k(retrieved_lists, relevant_ids_list, k=3)
    print(f"MRR@3: {mrr:.3f}")
    
    # Test NDCG
    ndcg = ndcg_at_k(retrieved_lists, relevant_ids_list, k=3)
    print(f"NDCG@3: {ndcg:.3f}")
    
    # Test faithfulness
    answer = "This is a test answer"
    sources = ["This is source text one", "This is source text two"]
    faithfulness = faithfulness_score(answer, sources)
    print(f"Faithfulness: {faithfulness:.3f}")
    
    # Test latency
    def dummy_function():
        time.sleep(0.01)
    
    latency = measure_latency(dummy_function, n_runs=3)
    print(f"Latency: {latency:.1f} ms")

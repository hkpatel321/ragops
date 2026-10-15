"""
Non-LLM Retrieval Evaluation Metrics for RAGOps.
Calculates deterministic ranking and retrieval accuracy metrics:
- Hit Rate @ k
- Mean Reciprocal Rank (MRR @ k)
- Recall @ k
"""

from typing import Any, Dict, List, Optional, Set, Union

import numpy as np


def hit_rate_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Union[str, List[str], Set[str]],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Hit Rate @ k.
    Returns 1.0 if at least one ground-truth chunk appears in the top-k retrieved chunks, else 0.0.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Single chunk ID or collection of relevant chunk IDs.
        k: Cutoff rank (defaults to all retrieved).

    Returns:
        1.0 for hit, 0.0 for miss.
    """
    top_k_ids = retrieved_chunk_ids[:k] if k is not None else retrieved_chunk_ids
    if isinstance(ground_truth_chunk_ids, str):
        target_set = {ground_truth_chunk_ids}
    else:
        target_set = set(ground_truth_chunk_ids)

    return 1.0 if bool(set(top_k_ids) & target_set) else 0.0


def reciprocal_rank_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Union[str, List[str], Set[str]],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Reciprocal Rank (RR @ k).
    Returns 1 / rank of the first relevant chunk in the top-k list (1-indexed). Returns 0.0 if not found.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Single chunk ID or collection of relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Reciprocal rank value between 0.0 and 1.0.
    """
    top_k_ids = retrieved_chunk_ids[:k] if k is not None else retrieved_chunk_ids
    if isinstance(ground_truth_chunk_ids, str):
        target_set = {ground_truth_chunk_ids}
    else:
        target_set = set(ground_truth_chunk_ids)

    for rank_idx, chunk_id in enumerate(top_k_ids, start=1):
        if chunk_id in target_set:
            return 1.0 / rank_idx

    return 0.0


def recall_at_k(
    retrieved_chunk_ids: List[str],
    ground_truth_chunk_ids: Union[str, List[str], Set[str]],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Recall @ k.
    Fraction of ground-truth chunks retrieved in top-k.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        ground_truth_chunk_ids: Single chunk ID or collection of relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        Recall value between 0.0 and 1.0.
    """
    top_k_ids = retrieved_chunk_ids[:k] if k is not None else retrieved_chunk_ids
    if isinstance(ground_truth_chunk_ids, str):
        target_set = {ground_truth_chunk_ids}
    else:
        target_set = set(ground_truth_chunk_ids)

    if not target_set:
        return 0.0

    intersection = set(top_k_ids) & target_set
    return len(intersection) / len(target_set)


def compute_retrieval_metrics(
    eval_records: List[Dict[str, Any]],
    k: int = 4,
) -> Dict[str, float]:
    """
    Batch evaluation of retrieval quality across a list of query results.

    Each record must contain:
    - 'retrieved_chunk_ids': List[str]
    - 'ground_truth_chunk_id' or 'ground_truth_chunk_ids': str | List[str]

    Args:
        eval_records: List of evaluation record dictionaries.
        k: Top-k cutoff.

    Returns:
        Dictionary of aggregated metrics: hit_rate@k, mrr@k, recall@k.
    """
    if not eval_records:
        return {f"hit_rate@{k}": 0.0, f"mrr@{k}": 0.0, f"recall@{k}": 0.0, "total_queries": 0.0}

    hits: List[float] = []
    rrs: List[float] = []
    recalls: List[float] = []

    for item in eval_records:
        retrieved = item.get("retrieved_chunk_ids", [])
        gt = item.get("ground_truth_chunk_id") or item.get("ground_truth_chunk_ids", [])

        hits.append(hit_rate_at_k(retrieved, gt, k=k))
        rrs.append(reciprocal_rank_at_k(retrieved, gt, k=k))
        recalls.append(recall_at_k(retrieved, gt, k=k))

    return {
        f"hit_rate@{k}": round(float(np.mean(hits)), 4),
        f"mrr@{k}": round(float(np.mean(rrs)), 4),
        f"recall@{k}": round(float(np.mean(recalls)), 4),
        "total_queries": float(len(eval_records)),
    }

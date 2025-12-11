import math
from typing import List, Dict, Set

def dcg_at_k(r: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    r = r[:k]
    if not r:
        return 0.0
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(r))

def ndcg_at_k(r: List[int], k: int, ground_truth_count: int) -> float:
    """Normalized DCG at K."""
    dcg = dcg_at_k(r, k)
    # Ideal DCG: perfect ordering of all relevant docs
    # We assume binary relevance (1 or 0) for now, so IDCG is just sum of discounts for first N relevant
    ideal_rels = [1] * min(k, ground_truth_count)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def precision_at_k(r: List[int], k: int) -> float:
    """Precision at K."""
    if k == 0: return 0.0
    r = r[:k]
    return sum(r) / k

def recall_at_k(r: List[int], k: int, ground_truth_count: int) -> float:
    """Recall at K."""
    if ground_truth_count == 0: return 0.0
    r = r[:k]
    return sum(r) / ground_truth_count

def average_precision(r: List[int], ground_truth_count: int) -> float:
    """Average Precision (AP)."""
    if ground_truth_count == 0: return 0.0
    s = 0.0
    hits = 0
    for i, rel in enumerate(r):
        if rel:
            hits += 1
            s += hits / (i + 1)
    return s / ground_truth_count


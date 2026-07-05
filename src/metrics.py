"""
IR evaluation metrics — Recall@K, Precision@K, MRR, MAP, NDCG@K, Avg Time.

All ranking metrics use binary relevance (a chunk is relevant or it isn't)
and take the same two inputs:
  - retrieved: ordered list of chunk IDs returned by the retriever
  - relevant:  set of ground-truth chunk IDs that answer the query

Average retrieval time is computed separately from per-query durations.

Citations:
  - _instructions.md L617 (Recall@K, Precision@K)
  - _instructions.md L618 (MRR)
  - _instructions.md L619 (MAP)
  - _instructions.md L620 (NDCG@K)
  - _instructions.md L621 (Avg retrieval time)
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top K results.

    recall@K = |relevant ∩ retrieved[:K]| / |relevant|
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top K results that are relevant.

    precision@K = |relevant ∩ retrieved[:K]| / K
    """
    if not retrieved or k == 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result.

    MRR = 1 / rank_of_first_hit (1-indexed). Returns 0 if no hit.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def mean_average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Mean of precision values at each relevant hit position.

    For each position where a relevant doc appears, compute precision@i.
    MAP = mean of those precision values.
    """
    if not relevant or not retrieved:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / (i + 1)

    if hits == 0:
        return 0.0

    return precision_sum / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance: gain = 1 if relevant, 0 otherwise.
    DCG@K  = Σ gain_i / log2(i+1)  for i in 1..K
    IDCG@K = DCG of ideal ranking (all relevant docs at top).
    NDCG   = DCG / IDCG
    """
    if not relevant or not retrieved:
        return 0.0

    # DCG of actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed, formula uses 1-indexed+1

    # IDCG: ideal ranking puts all relevant docs first
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_retrieval_time(durations: list[float]) -> float:
    """Mean retrieval time across queries (seconds).

    Args:
        durations: List of per-query retrieval times in seconds.

    Returns:
        Average time in seconds. Returns 0.0 for empty list.
    """
    if not durations:
        return 0.0
    return sum(durations) / len(durations)

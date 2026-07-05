"""
Hybrid retrieval — combine BM25 and vector scores via min-max normalization.

BM25 scores (higher=better, unbounded) and FAISS L2 distances (lower=better,
unbounded) are on incompatible scales.  Min-max normalization maps each to
[0, 1] within its result set, then a weighted sum produces the hybrid score.

Citations:
  - _instructions.md L615 (hybrid retrieval)
  - _instructions.md L616 (score normalization)
"""

from __future__ import annotations


def normalize_scores(
    results: list[tuple[str, float]],
    higher_is_better: bool = True,
) -> list[tuple[str, float]]:
    """Min-max normalize scores to [0, 1].

    Args:
        results: List of (chunk_id, raw_score) tuples.
        higher_is_better: If True, highest score maps to 1.0.
            If False (e.g. FAISS distances), lowest score maps to 1.0.

    Returns:
        List of (chunk_id, normalized_score) tuples.
    """
    if not results:
        return []

    scores = [s for _, s in results]
    min_s = min(scores)
    max_s = max(scores)
    spread = max_s - min_s

    if spread == 0:
        # All scores identical (or single result) — treat as equally "best"
        return [(cid, 1.0) for cid, _ in results]

    if higher_is_better:
        return [(cid, (s - min_s) / spread) for cid, s in results]
    else:
        # Invert: lowest raw score (best distance) → 1.0
        return [(cid, 1.0 - (s - min_s) / spread) for cid, s in results]


def combine_results(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """Combine normalized vector and BM25 scores with weighted sum.

    Args:
        vector_results: Normalized (chunk_id, score) tuples from vector retrieval.
        bm25_results: Normalized (chunk_id, score) tuples from BM25 retrieval.
        alpha: Weight for vector scores. BM25 weight is (1 - alpha).
            alpha=1.0 → vector only, alpha=0.0 → BM25 only.

    Returns:
        List of (chunk_id, hybrid_score) tuples, sorted by score descending.
    """
    vector_dict = dict(vector_results)
    bm25_dict = dict(bm25_results)

    # Union of all chunk IDs from both retrievers
    all_ids = set(vector_dict.keys()) | set(bm25_dict.keys())

    combined = []
    for cid in all_ids:
        v_score = vector_dict.get(cid, 0.0)
        b_score = bm25_dict.get(cid, 0.0)
        hybrid = alpha * v_score + (1 - alpha) * b_score
        combined.append((cid, hybrid))

    # Sort by hybrid score descending (best first)
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined

"""Tests for IR metrics — hand-computed expected values for each metric.

TDD: every expected value is computed by hand before implementation.

Example ranking used across tests:
  retrieved = ["a", "b", "c", "d", "e"]  (5 results)
  relevant  = {"a", "c", "e"}            (3 ground-truth chunks)

  Hits at positions: a(1), c(3), e(5)  — positions are 1-indexed.

Citations:
  - _instructions.md L617 (Recall@K, Precision@K)
  - _instructions.md L618 (MRR)
  - _instructions.md L619 (MAP)
  - _instructions.md L620 (NDCG@K)
  - _instructions.md L621 (Avg retrieval time)
"""

import math

import pytest

from src.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    mean_average_precision,
    ndcg_at_k,
    average_retrieval_time,
)


# -- Shared test data -------------------------------------------------------

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c", "e"}


# -- Recall@K ----------------------------------------------------------------

class TestRecallAtK:
    """Recall@K = |relevant ∩ retrieved[:K]| / |relevant|"""

    def test_recall_at_1(self):
        # retrieved[:1] = ["a"], hits = {"a"}, recall = 1/3
        assert recall_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(1 / 3)

    def test_recall_at_3(self):
        # retrieved[:3] = ["a","b","c"], hits = {"a","c"}, recall = 2/3
        assert recall_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(2 / 3)

    def test_recall_at_5(self):
        # All 3 relevant found in top 5: recall = 3/3 = 1.0
        assert recall_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(1.0)

    def test_no_relevant_found(self):
        assert recall_at_k(["x", "y", "z"], RELEVANT, k=3) == pytest.approx(0.0)

    def test_empty_relevant_set(self):
        # No ground truth — recall is 0.0 (nothing to find)
        assert recall_at_k(RETRIEVED, set(), k=5) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert recall_at_k([], RELEVANT, k=5) == pytest.approx(0.0)


# -- Precision@K -------------------------------------------------------------

class TestPrecisionAtK:
    """Precision@K = |relevant ∩ retrieved[:K]| / K"""

    def test_precision_at_1(self):
        # retrieved[:1] = ["a"], hits = {"a"}, precision = 1/1
        assert precision_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(1.0)

    def test_precision_at_3(self):
        # retrieved[:3] = ["a","b","c"], hits = {"a","c"}, precision = 2/3
        assert precision_at_k(RETRIEVED, RELEVANT, k=3) == pytest.approx(2 / 3)

    def test_precision_at_5(self):
        # hits = {"a","c","e"}, precision = 3/5
        assert precision_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(3 / 5)

    def test_no_relevant_found(self):
        assert precision_at_k(["x", "y", "z"], RELEVANT, k=3) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert precision_at_k([], RELEVANT, k=5) == pytest.approx(0.0)


# -- MRR (Mean Reciprocal Rank) ---------------------------------------------

class TestMRR:
    """MRR = 1 / rank_of_first_relevant_result (1-indexed)"""

    def test_first_result_is_relevant(self):
        # "a" is relevant and at rank 1 → 1/1
        assert mrr(RETRIEVED, RELEVANT) == pytest.approx(1.0)

    def test_first_relevant_at_rank_3(self):
        # ["x", "y", "c", ...] → first relevant "c" at rank 3 → 1/3
        assert mrr(["x", "y", "c", "d", "e"], RELEVANT) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        assert mrr(["x", "y", "z"], RELEVANT) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert mrr([], RELEVANT) == pytest.approx(0.0)


# -- MAP (Mean Average Precision) -------------------------------------------

class TestMAP:
    """MAP = mean of precision@i for each position i where a relevant doc appears.

    For RETRIEVED = ["a","b","c","d","e"], RELEVANT = {"a","c","e"}:
      - "a" at rank 1: precision@1 = 1/1 = 1.0
      - "c" at rank 3: precision@3 = 2/3
      - "e" at rank 5: precision@5 = 3/5
      MAP = (1.0 + 2/3 + 3/5) / 3 = (1.0 + 0.6667 + 0.6) / 3 ≈ 0.7556
    """

    def test_map_standard_case(self):
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert mean_average_precision(RETRIEVED, RELEVANT) == pytest.approx(expected)

    def test_perfect_ranking(self):
        # All relevant at the top: ["a","c","e","x","y"]
        # "a"@1: 1/1, "c"@2: 2/2, "e"@3: 3/3 → MAP = (1+1+1)/3 = 1.0
        assert mean_average_precision(
            ["a", "c", "e", "x", "y"], RELEVANT
        ) == pytest.approx(1.0)

    def test_no_relevant_found(self):
        assert mean_average_precision(["x", "y", "z"], RELEVANT) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert mean_average_precision([], RELEVANT) == pytest.approx(0.0)


# -- NDCG@K (Normalized Discounted Cumulative Gain) -------------------------

class TestNDCG:
    """NDCG@K = DCG@K / IDCG@K

    Binary relevance: gain = 1 if relevant, 0 otherwise.
    DCG@K  = Σ gain_i / log2(i+1) for i in 1..K
    IDCG@K = DCG of ideal ranking (all relevant docs first).

    For RETRIEVED = ["a","b","c","d","e"], RELEVANT = {"a","c","e"}, K=5:
      DCG  = 1/log2(2) + 0 + 1/log2(4) + 0 + 1/log2(6)
           = 1.0 + 0 + 0.5 + 0 + 0.3869
           = 1.8869
      IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)  (3 relevant, all at top)
           = 1.0 + 0.6309 + 0.5
           = 2.1309
      NDCG = 1.8869 / 2.1309 ≈ 0.8855
    """

    def test_ndcg_standard_case(self):
        dcg = 1 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(6)
        idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(expected)

    def test_perfect_ranking(self):
        # Ideal order: ["a","c","e","x","y"] → NDCG = 1.0
        assert ndcg_at_k(
            ["a", "c", "e", "x", "y"], RELEVANT, k=5
        ) == pytest.approx(1.0)

    def test_ndcg_at_1_with_hit(self):
        # Top result is relevant → DCG = IDCG = 1/log2(2) → NDCG = 1.0
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=1) == pytest.approx(1.0)

    def test_ndcg_at_1_with_miss(self):
        # Top result not relevant → DCG = 0 → NDCG = 0.0
        assert ndcg_at_k(["x", "a", "c"], RELEVANT, k=1) == pytest.approx(0.0)

    def test_no_relevant_found(self):
        assert ndcg_at_k(["x", "y", "z"], RELEVANT, k=3) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert ndcg_at_k([], RELEVANT, k=5) == pytest.approx(0.0)

    def test_empty_relevant(self):
        assert ndcg_at_k(RETRIEVED, set(), k=5) == pytest.approx(0.0)


# -- Average Retrieval Time -------------------------------------------------

class TestAverageRetrievalTime:
    """Simple mean of per-query retrieval durations (seconds)."""

    def test_average_of_three(self):
        times = [0.1, 0.2, 0.3]
        assert average_retrieval_time(times) == pytest.approx(0.2)

    def test_single_time(self):
        assert average_retrieval_time([0.05]) == pytest.approx(0.05)

    def test_empty_list(self):
        assert average_retrieval_time([]) == pytest.approx(0.0)

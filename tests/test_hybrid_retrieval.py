"""Tests for hybrid retrieval — normalize + combine BM25 and vector scores.

TDD: tests use hand-crafted score lists to verify normalization math
and combined ranking. No API calls needed.

Citations:
  - _instructions.md L615 (hybrid retrieval)
  - _instructions.md L616 (score normalization)
"""

import pytest

from src.hybrid_retrieval import normalize_scores, combine_results


class TestNormalizeScores:
    """Verify min-max normalization to [0, 1]."""

    def test_normalizes_to_zero_one_range(self):
        results = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
        normalized = normalize_scores(results, higher_is_better=True)
        scores = [s for _, s in normalized]
        assert min(scores) == 0.0
        assert max(scores) == 1.0

    def test_higher_is_better_preserves_order(self):
        """Highest raw score should get normalized score of 1.0."""
        results = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
        normalized = normalize_scores(results, higher_is_better=True)
        norm_dict = dict(normalized)
        assert norm_dict["a"] == 1.0
        assert norm_dict["c"] == 0.0

    def test_lower_is_better_inverts(self):
        """Lowest raw score (best distance) should get normalized score of 1.0."""
        results = [("a", 0.0), ("b", 5.0), ("c", 10.0)]
        normalized = normalize_scores(results, higher_is_better=False)
        norm_dict = dict(normalized)
        assert norm_dict["a"] == 1.0  # distance 0 = best
        assert norm_dict["c"] == 0.0  # distance 10 = worst

    def test_single_result_gets_one(self):
        """Single result: min == max, should return 1.0."""
        results = [("a", 5.0)]
        normalized = normalize_scores(results, higher_is_better=True)
        assert normalized[0][1] == 1.0

    def test_all_same_scores_get_one(self):
        """All identical scores: everyone is equally 'best'."""
        results = [("a", 3.0), ("b", 3.0), ("c", 3.0)]
        normalized = normalize_scores(results, higher_is_better=True)
        for _, score in normalized:
            assert score == 1.0

    def test_empty_input(self):
        normalized = normalize_scores([], higher_is_better=True)
        assert normalized == []

    def test_midpoint_value(self):
        """Middle value should normalize to 0.5."""
        results = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
        normalized = normalize_scores(results, higher_is_better=True)
        norm_dict = dict(normalized)
        assert norm_dict["b"] == pytest.approx(0.5)


class TestCombineResults:
    """Verify weighted combination of two normalized result sets."""

    def test_equal_weight_averages_scores(self):
        """alpha=0.5: hybrid score should be average of both."""
        vector_results = [("a", 1.0), ("b", 0.0)]
        bm25_results = [("a", 0.0), ("b", 1.0)]
        combined = combine_results(vector_results, bm25_results, alpha=0.5)
        scores = dict(combined)
        assert scores["a"] == pytest.approx(0.5)
        assert scores["b"] == pytest.approx(0.5)

    def test_alpha_one_ignores_bm25(self):
        """alpha=1.0: only vector scores matter."""
        vector_results = [("a", 1.0), ("b", 0.0)]
        bm25_results = [("a", 0.0), ("b", 1.0)]
        combined = combine_results(vector_results, bm25_results, alpha=1.0)
        scores = dict(combined)
        assert scores["a"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.0)

    def test_alpha_zero_ignores_vector(self):
        """alpha=0.0: only BM25 scores matter."""
        vector_results = [("a", 1.0), ("b", 0.0)]
        bm25_results = [("a", 0.0), ("b", 1.0)]
        combined = combine_results(vector_results, bm25_results, alpha=0.0)
        scores = dict(combined)
        assert scores["a"] == pytest.approx(0.0)
        assert scores["b"] == pytest.approx(1.0)

    def test_results_sorted_descending(self):
        """Combined results should be sorted by hybrid score, best first."""
        vector_results = [("a", 0.8), ("b", 0.2)]
        bm25_results = [("a", 0.6), ("b", 0.9)]
        combined = combine_results(vector_results, bm25_results, alpha=0.5)
        scores = [s for _, s in combined]
        assert scores == sorted(scores, reverse=True)

    def test_non_overlapping_chunks(self):
        """Chunks appearing in only one retriever get 0 for the other."""
        vector_results = [("a", 1.0)]
        bm25_results = [("b", 1.0)]
        combined = combine_results(vector_results, bm25_results, alpha=0.5)
        scores = dict(combined)
        assert scores["a"] == pytest.approx(0.5)  # 0.5*1.0 + 0.5*0.0
        assert scores["b"] == pytest.approx(0.5)  # 0.5*0.0 + 0.5*1.0

    def test_returns_all_unique_chunks(self):
        """Union of both result sets — no duplicates, no missing."""
        vector_results = [("a", 0.8), ("b", 0.5), ("c", 0.2)]
        bm25_results = [("b", 0.9), ("c", 0.3), ("d", 0.7)]
        combined = combine_results(vector_results, bm25_results, alpha=0.5)
        chunk_ids = [cid for cid, _ in combined]
        assert set(chunk_ids) == {"a", "b", "c", "d"}
        assert len(chunk_ids) == 4  # no duplicates

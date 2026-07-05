"""Tests for reranker — mock Cohere API, verify wiring.

TDD: tests verify that retrieved results get reordered based on
reranker scores, chunk IDs are preserved, and the interface matches
our retriever output format.

Citations:
  - _instructions.md L629 (reranking)
  - _instructions.md L630 (with/without comparison, deltas)
"""

from unittest.mock import patch, MagicMock

import pytest

from src.reranker import rerank


@pytest.fixture
def retrieved_results():
    """Simulated retriever output — chunk IDs with scores."""
    return [
        ("chunk_0", 0.9),
        ("chunk_1", 0.7),
        ("chunk_2", 0.5),
        ("chunk_3", 0.3),
        ("chunk_4", 0.1),
    ]


@pytest.fixture
def chunk_texts():
    """Map of chunk IDs to their text content."""
    return {
        "chunk_0": "Revenue grew 15% year-over-year to $4.2 billion.",
        "chunk_1": "The company expanded into three new markets.",
        "chunk_2": "Operating expenses decreased by 8%.",
        "chunk_3": "Net income increased to $800 million.",
        "chunk_4": "Employee headcount grew to 12000.",
    }


def _mock_rerank_response(results, chunk_texts):
    """Build a mock Cohere rerank response that reverses the order."""
    mock_results = []
    for new_rank, (chunk_id, _) in enumerate(reversed(results)):
        mock_item = MagicMock()
        mock_item.index = len(results) - 1 - new_rank
        mock_item.relevance_score = 1.0 - (new_rank * 0.2)
        mock_results.append(mock_item)
    return MagicMock(results=mock_results)


class TestRerank:
    """Verify reranker wiring and output format."""

    def test_returns_list_of_tuples(self, retrieved_results, chunk_texts):
        with patch("src.reranker._call_cohere_rerank") as mock_rerank:
            mock_rerank.return_value = _mock_rerank_response(
                retrieved_results, chunk_texts
            )
            results = rerank("What was the revenue?", retrieved_results, chunk_texts)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_results_have_id_and_score(self, retrieved_results, chunk_texts):
        with patch("src.reranker._call_cohere_rerank") as mock_rerank:
            mock_rerank.return_value = _mock_rerank_response(
                retrieved_results, chunk_texts
            )
            results = rerank("What was the revenue?", retrieved_results, chunk_texts)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_preserves_all_chunk_ids(self, retrieved_results, chunk_texts):
        with patch("src.reranker._call_cohere_rerank") as mock_rerank:
            mock_rerank.return_value = _mock_rerank_response(
                retrieved_results, chunk_texts
            )
            results = rerank("What was the revenue?", retrieved_results, chunk_texts)
        original_ids = {cid for cid, _ in retrieved_results}
        reranked_ids = {cid for cid, _ in results}
        assert reranked_ids == original_ids

    def test_scores_are_descending(self, retrieved_results, chunk_texts):
        with patch("src.reranker._call_cohere_rerank") as mock_rerank:
            mock_rerank.return_value = _mock_rerank_response(
                retrieved_results, chunk_texts
            )
            results = rerank("What was the revenue?", retrieved_results, chunk_texts)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_and_texts_passed_to_api(self, retrieved_results, chunk_texts):
        with patch("src.reranker._call_cohere_rerank") as mock_rerank:
            mock_rerank.return_value = _mock_rerank_response(
                retrieved_results, chunk_texts
            )
            rerank("What was the revenue?", retrieved_results, chunk_texts)
            call_args = mock_rerank.call_args
            assert call_args[0][0] == "What was the revenue?"
            # Documents should be the chunk texts in retrieval order
            assert len(call_args[0][1]) == len(retrieved_results)

    def test_respects_top_n(self, retrieved_results, chunk_texts):
        """If top_n specified, only return that many results."""
        mock_results = []
        for i in range(3):
            mock_item = MagicMock()
            mock_item.index = i
            mock_item.relevance_score = 1.0 - (i * 0.2)
            mock_results.append(mock_item)
        with patch("src.reranker._call_cohere_rerank") as mock_rerank:
            mock_rerank.return_value = MagicMock(results=mock_results)
            results = rerank(
                "What was the revenue?", retrieved_results, chunk_texts, top_n=3
            )
        assert len(results) == 3

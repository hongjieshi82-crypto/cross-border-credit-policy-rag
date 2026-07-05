"""Tests for vector retrieval — embed query + FAISS search in one step.

TDD: mocks the embedding call so no API needed. Tests verify the
retrieval interface wires embedding → FAISS correctly.

Citations:
  - _instructions.md L611 (FAISS stores and retrieves correctly)
  - _instructions.md L612 (top-K with configurable K)
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.vector_retrieval import build_vector_retriever, query_vector


@pytest.fixture
def vector_retriever():
    """Build a vector retriever with known synthetic embeddings."""
    np.random.seed(42)
    embeddings = np.random.rand(10, 8).astype(np.float32)
    chunk_ids = [f"chunk_{i}" for i in range(10)]
    retriever = build_vector_retriever(embeddings, chunk_ids)
    return retriever, embeddings, chunk_ids


class TestBuildVectorRetriever:
    """Verify the retriever object is built correctly."""

    def test_returns_retriever_dict(self, vector_retriever):
        retriever, _, _ = vector_retriever
        assert "index" in retriever
        assert "id_map" in retriever
        assert "embeddings" in retriever

    def test_index_has_correct_count(self, vector_retriever):
        retriever, embeddings, _ = vector_retriever
        assert retriever["index"].ntotal == len(embeddings)


class TestQueryVector:
    """Verify text-to-results retrieval pipeline."""

    def test_returns_k_results(self, vector_retriever):
        retriever, embeddings, _ = vector_retriever
        # Mock embed_texts to return the first vector (simulating a query)
        with patch("src.vector_retrieval.embed_texts") as mock_embed:
            mock_embed.return_value = embeddings[0:1]
            results = query_vector(retriever, "some query", k=3)
        assert len(results) == 3

    def test_top_result_matches_query_vector(self, vector_retriever):
        """When query embeds to vector[0], chunk_0 should be top result."""
        retriever, embeddings, chunk_ids = vector_retriever
        with patch("src.vector_retrieval.embed_texts") as mock_embed:
            mock_embed.return_value = embeddings[0:1]
            results = query_vector(retriever, "some query", k=1)
        assert results[0][0] == chunk_ids[0]

    def test_results_have_id_and_distance(self, vector_retriever):
        retriever, embeddings, _ = vector_retriever
        with patch("src.vector_retrieval.embed_texts") as mock_embed:
            mock_embed.return_value = embeddings[0:1]
            results = query_vector(retriever, "some query", k=3)
        for chunk_id, distance in results:
            assert isinstance(chunk_id, str)
            assert isinstance(distance, float)

    def test_distances_are_ascending(self, vector_retriever):
        retriever, embeddings, _ = vector_retriever
        with patch("src.vector_retrieval.embed_texts") as mock_embed:
            mock_embed.return_value = embeddings[0:1]
            results = query_vector(retriever, "some query", k=5)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_passes_model_to_embed(self, vector_retriever):
        """Verify the embedding model name is forwarded correctly."""
        retriever, embeddings, _ = vector_retriever
        with patch("src.vector_retrieval.embed_texts") as mock_embed:
            mock_embed.return_value = embeddings[0:1]
            query_vector(retriever, "test query", model="text-embedding-3-large", k=1)
            mock_embed.assert_called_once_with(
                ["test query"], model="text-embedding-3-large"
            )

    def test_k_larger_than_index(self, vector_retriever):
        retriever, embeddings, _ = vector_retriever
        with patch("src.vector_retrieval.embed_texts") as mock_embed:
            mock_embed.return_value = embeddings[0:1]
            results = query_vector(retriever, "some query", k=100)
        assert len(results) == 10

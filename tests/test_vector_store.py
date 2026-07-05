"""Tests for FAISS vector store — build index, top-K retrieval, ID mapping.

TDD: uses synthetic vectors, no API calls needed.

Citations:
  - _instructions.md L611 (FAISS stores and retrieves correctly)
  - _instructions.md L612 (top-K with configurable K)
  - _instructions.md L589 (Flat index sufficient for < 10K vectors)
"""

import numpy as np
import pytest

from src.vector_store import build_index, query_index


@pytest.fixture
def small_index():
    """Build a small FAISS index with 10 known vectors."""
    np.random.seed(42)
    embeddings = np.random.rand(10, 8).astype(np.float32)
    chunk_ids = [f"chunk_{i}" for i in range(10)]
    index, id_map = build_index(embeddings, chunk_ids)
    return index, id_map, embeddings, chunk_ids


class TestBuildIndex:
    """Verify FAISS index is built correctly from embeddings."""

    def test_index_has_correct_count(self, small_index):
        index, _, embeddings, _ = small_index
        assert index.ntotal == len(embeddings)

    def test_index_has_correct_dimension(self, small_index):
        index, _, embeddings, _ = small_index
        assert index.d == embeddings.shape[1]

    def test_id_map_has_correct_length(self, small_index):
        _, id_map, _, chunk_ids = small_index
        assert len(id_map) == len(chunk_ids)

    def test_id_map_maps_positions_to_chunk_ids(self, small_index):
        _, id_map, _, chunk_ids = small_index
        for i, cid in enumerate(chunk_ids):
            assert id_map[i] == cid


class TestQueryIndex:
    """Verify top-K retrieval returns correct results."""

    def test_returns_k_results(self, small_index):
        index, id_map, embeddings, _ = small_index
        query = embeddings[0:1]  # query with first vector
        results = query_index(index, id_map, query, k=3)
        assert len(results) == 3

    def test_nearest_neighbor_is_itself(self, small_index):
        """Querying with a vector in the index should return itself as rank 1."""
        index, id_map, embeddings, chunk_ids = small_index
        query = embeddings[0:1]
        results = query_index(index, id_map, query, k=1)
        assert results[0][0] == chunk_ids[0]  # top result is chunk_0

    def test_results_have_id_and_distance(self, small_index):
        index, id_map, embeddings, _ = small_index
        query = embeddings[0:1]
        results = query_index(index, id_map, query, k=3)
        for chunk_id, distance in results:
            assert isinstance(chunk_id, str)
            assert isinstance(distance, float)

    def test_distances_are_ascending(self, small_index):
        """Results should be sorted by distance (nearest first)."""
        index, id_map, embeddings, _ = small_index
        query = embeddings[0:1]
        results = query_index(index, id_map, query, k=5)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_self_distance_is_zero(self, small_index):
        """Distance to itself should be 0 (or very close)."""
        index, id_map, embeddings, _ = small_index
        query = embeddings[0:1]
        results = query_index(index, id_map, query, k=1)
        assert results[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_k_larger_than_index(self, small_index):
        """Requesting more results than vectors should return all vectors."""
        index, id_map, embeddings, _ = small_index
        query = embeddings[0:1]
        results = query_index(index, id_map, query, k=100)
        assert len(results) == 10  # only 10 vectors in index

    def test_metadata_filter_limits_results(self, small_index):
        index, id_map, embeddings, chunk_ids = small_index
        metadata_by_id = {
            chunk_id: {"country": "Mexico" if i % 2 == 0 else "India"}
            for i, chunk_id in enumerate(chunk_ids)
        }
        query = embeddings[0:1]
        results = query_index(
            index,
            id_map,
            query,
            k=10,
            metadata_by_id=metadata_by_id,
            metadata_filter={"country": "India"},
        )
        assert results
        assert all(metadata_by_id[chunk_id]["country"] == "India" for chunk_id, _ in results)
        assert chunk_ids[0] not in [chunk_id for chunk_id, _ in results]

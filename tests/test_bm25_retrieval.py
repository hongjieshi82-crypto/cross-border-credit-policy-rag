"""Tests for BM25 retrieval — build index, query, ranked results.

TDD: uses synthetic text chunks, no API calls needed.

Citations:
  - _instructions.md L613 (BM25 keyword retrieval)
  - _instructions.md L614 (ranked results with scores)
"""

import pytest

from src.bm25_retrieval import build_bm25_index, query_bm25


@pytest.fixture
def animal_chunks():
    """Chunks with distinct vocabulary for predictable BM25 ranking."""
    texts = [
        "the cat sat on the mat",              # chunk_0: cat-heavy
        "the dog chased the ball in the park",  # chunk_1: dog-heavy
        "fish swim in the ocean deep water",    # chunk_2: fish-heavy
        "the cat and dog played together",      # chunk_3: cat + dog
        "birds fly high in the blue sky",       # chunk_4: birds-heavy
    ]
    chunk_ids = [f"chunk_{i}" for i in range(len(texts))]
    return texts, chunk_ids


class TestBuildBM25Index:
    """Verify BM25 index is built from text chunks."""

    def test_returns_index_object(self, animal_chunks):
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        assert index is not None

    def test_id_map_has_correct_length(self, animal_chunks):
        texts, chunk_ids = animal_chunks
        _, id_map = build_bm25_index(texts, chunk_ids)
        assert len(id_map) == len(chunk_ids)

    def test_id_map_maps_positions_to_chunk_ids(self, animal_chunks):
        texts, chunk_ids = animal_chunks
        _, id_map = build_bm25_index(texts, chunk_ids)
        for i, cid in enumerate(chunk_ids):
            assert id_map[i] == cid


class TestQueryBM25:
    """Verify BM25 query returns ranked results."""

    def test_returns_k_results(self, animal_chunks):
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        results = query_bm25(index, id_map, "cat", k=3)
        assert len(results) == 3

    def test_top_result_matches_query_term(self, animal_chunks):
        """Querying 'cat' should rank cat-heavy chunks highest."""
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        results = query_bm25(index, id_map, "cat", k=2)
        top_ids = [cid for cid, _ in results]
        # chunk_0 ("the cat sat on the mat") should be top result
        assert top_ids[0] == "chunk_0"

    def test_results_have_id_and_score(self, animal_chunks):
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        results = query_bm25(index, id_map, "dog", k=3)
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)

    def test_scores_are_descending(self, animal_chunks):
        """BM25 scores should be highest first (most relevant)."""
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        results = query_bm25(index, id_map, "cat", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_corpus(self, animal_chunks):
        """Requesting more results than chunks should return all chunks."""
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        results = query_bm25(index, id_map, "the", k=100)
        assert len(results) == len(chunk_ids)

    def test_nonmatching_query_returns_zero_scores(self, animal_chunks):
        """A query with no matching terms should return zero scores."""
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        results = query_bm25(index, id_map, "xylophone", k=3)
        scores = [s for _, s in results]
        assert all(s == 0.0 for s in scores)

    def test_metadata_filter_limits_results(self, animal_chunks):
        texts, chunk_ids = animal_chunks
        index, id_map = build_bm25_index(texts, chunk_ids)
        metadata_by_id = {
            "chunk_0": {"country": "Mexico", "institution": "Buro de Credito"},
            "chunk_1": {"country": "India", "institution": "RBI"},
            "chunk_2": {"country": "Mexico", "institution": "Buro de Credito"},
            "chunk_3": {"country": "India", "institution": "CIBIL"},
            "chunk_4": {"country": "Uganda", "institution": "Central Bank"},
        }
        results = query_bm25(
            index,
            id_map,
            "cat dog",
            k=5,
            metadata_by_id=metadata_by_id,
            metadata_filter={"country": "India"},
        )
        assert [chunk_id for chunk_id, _ in results] == ["chunk_3", "chunk_1"]

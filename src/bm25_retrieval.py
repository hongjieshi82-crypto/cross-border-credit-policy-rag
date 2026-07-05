"""
BM25 keyword retrieval — build index from text chunks, query top-K.

BM25 (Best Matching 25) is a bag-of-words ranking function that scores
documents by term frequency with diminishing returns and length normalization.
Complements vector (semantic) retrieval by catching exact keyword matches.

Citations:
  - _instructions.md L613 (BM25 keyword retrieval)
  - _instructions.md L614 (ranked results with scores)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rank_bm25 import BM25Okapi

from src.metadata_filter import MetadataFilter, metadata_matches


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_bm25_index(
    texts: list[str],
    chunk_ids: list[str],
) -> tuple[BM25Okapi, dict[int, str]]:
    """Build a BM25 index from text chunks.

    Args:
        texts: List of chunk text strings.
        chunk_ids: List of chunk ID strings, same length as texts.

    Returns:
        Tuple of (BM25 index, position-to-chunk-ID mapping).
    """
    tokenized = [_tokenize(t) for t in texts]
    index = BM25Okapi(tokenized)
    id_map = {i: cid for i, cid in enumerate(chunk_ids)}
    return index, id_map


def query_bm25(
    index: BM25Okapi,
    id_map: dict[int, str],
    query: str,
    k: int = 5,
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[str, float]]:
    """Query the BM25 index for top-K results.

    Args:
        index: The BM25 index to search.
        id_map: Position-to-chunk-ID mapping from build_bm25_index.
        query: Query string.
        k: Number of results to return.
        metadata_by_id: Optional chunk-ID to metadata map.
        metadata_filter: Optional hard filter applied before final top-K.

    Returns:
        List of (chunk_id, score) tuples, sorted by score descending.
    """
    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    # Pair each score with its position, sort descending
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Clamp k to corpus size
    k = min(k, len(scored))

    results: list[tuple[str, float]] = []
    for idx, score in scored:
        chunk_id = id_map[idx]
        if metadata_filter:
            metadata = metadata_by_id.get(chunk_id, {}) if metadata_by_id else {}
            if not metadata_matches(metadata, metadata_filter):
                continue
        results.append((chunk_id, float(score)))
        if len(results) >= k:
            break

    return results

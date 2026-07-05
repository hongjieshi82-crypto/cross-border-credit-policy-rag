"""
FAISS vector store — build index from embeddings, query top-K.

Uses IndexFlatL2 (exhaustive search) since our dataset is < 10K vectors.
Returns chunk IDs rather than raw FAISS indices, so the caller doesn't
need to know about the position-to-ID mapping.

Citations:
  - _instructions.md L611 (FAISS stores and retrieves correctly)
  - _instructions.md L612 (top-K with configurable K)
  - _instructions.md L589 (Flat index sufficient for < 10K vectors)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import faiss
import numpy as np

from src.metadata_filter import MetadataFilter, metadata_matches


def build_index(
    embeddings: np.ndarray,
    chunk_ids: list[str],
) -> tuple[faiss.IndexFlatL2, dict[int, str]]:
    """Build a FAISS flat index from embeddings.

    Args:
        embeddings: numpy array of shape (n_vectors, dimension).
        chunk_ids: list of chunk ID strings, same length as embeddings.

    Returns:
        Tuple of (FAISS index, position-to-chunk-ID mapping).
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    id_map = {i: cid for i, cid in enumerate(chunk_ids)}
    return index, id_map


def query_index(
    index: faiss.IndexFlatL2,
    id_map: dict[int, str],
    query_vector: np.ndarray,
    k: int = 5,
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[str, float]]:
    """Query the FAISS index for top-K nearest neighbors.

    Args:
        index: The FAISS index to search.
        id_map: Position-to-chunk-ID mapping from build_index.
        query_vector: Query embedding of shape (1, dimension).
        k: Number of results to return.
        metadata_by_id: Optional chunk-ID to metadata map.
        metadata_filter: Optional hard filter applied before final top-K.

    Returns:
        List of (chunk_id, distance) tuples, sorted by distance ascending.
    """
    # When filtering with a flat index, scan all candidates so the filter is
    # exact rather than merely applied to the first unfiltered k neighbors.
    search_k = index.ntotal if metadata_filter else min(k, index.ntotal)
    k = min(k, index.ntotal)

    distances, indices = index.search(query_vector, search_k)

    results: list[tuple[str, float]] = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= 0:  # FAISS returns -1 for empty slots
            chunk_id = id_map[int(idx)]
            if metadata_filter:
                metadata = metadata_by_id.get(chunk_id, {}) if metadata_by_id else {}
                if not metadata_matches(metadata, metadata_filter):
                    continue
            results.append((chunk_id, float(dist)))
            if len(results) >= k:
                break

    return results

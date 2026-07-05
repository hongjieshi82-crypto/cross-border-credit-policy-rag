"""
Vector retrieval — text-in, ranked-results-out wrapper around FAISS.

Bridges embedding.py and vector_store.py: takes a text query, embeds it,
searches the FAISS index, and returns ranked (chunk_id, distance) tuples.
Same output interface as bm25_retrieval for downstream interchangeability.

Citations:
  - _instructions.md L611 (FAISS stores and retrieves correctly)
  - _instructions.md L612 (top-K with configurable K)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.embedding import embed_texts
from src.metadata_filter import MetadataFilter
from src.vector_store import build_index, query_index


def build_vector_retriever(
    embeddings: np.ndarray,
    chunk_ids: list[str],
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a vector retriever from pre-computed embeddings.

    Args:
        embeddings: numpy array of shape (n_chunks, dimension).
        chunk_ids: list of chunk ID strings.
        metadata_by_id: Optional metadata map for hard filtering.

    Returns:
        Retriever dict with 'index', 'id_map', and 'embeddings' keys.
    """
    index, id_map = build_index(embeddings, chunk_ids)
    return {
        "index": index,
        "id_map": id_map,
        "embeddings": embeddings,
        "metadata_by_id": dict(metadata_by_id or {}),
    }


def query_vector(
    retriever: dict[str, Any],
    query: str,
    model: str = "text-embedding-3-small",
    k: int = 5,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[str, float]]:
    """Query with text: embed the query, then search FAISS.

    Args:
        retriever: Retriever dict from build_vector_retriever.
        query: Raw text query string.
        model: Embedding model to use for the query.
        k: Number of results to return.
        metadata_filter: Optional hard filter over chunk metadata.

    Returns:
        List of (chunk_id, distance) tuples, sorted by distance ascending.
    """
    query_embedding = embed_texts([query], model=model)
    return query_index(
        retriever["index"],
        retriever["id_map"],
        query_embedding,
        k=k,
        metadata_by_id=retriever.get("metadata_by_id"),
        metadata_filter=metadata_filter,
    )

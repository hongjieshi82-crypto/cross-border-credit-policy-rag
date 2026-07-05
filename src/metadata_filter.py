"""Metadata hard-filtering helpers for compliance-policy retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.models import Chunk, ChunkMetadata

MetadataFilter = Mapping[str, Any]


def _as_metadata_dict(metadata: ChunkMetadata | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize supported metadata shapes into a plain dict."""
    if isinstance(metadata, ChunkMetadata):
        return metadata.vector_metadata()
    return dict(metadata)


def metadata_matches(
    metadata: ChunkMetadata | Mapping[str, Any],
    metadata_filter: MetadataFilter | None = None,
) -> bool:
    """Return True when metadata satisfies all requested filter clauses.

    Supported forms:
        {"country": "Mexico"}
        {"country": ["Mexico", "India"]}
        {"country": {"$eq": "Mexico"}}
        {"institution": {"$in": ["RBI", "CIBIL"]}}
    """
    if not metadata_filter:
        return True

    values = _as_metadata_dict(metadata)
    for key, expected in metadata_filter.items():
        if expected is None:
            continue

        actual = values.get(key)
        if isinstance(expected, Mapping):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
            continue

        if isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False

    return True


def metadata_by_chunk_id(chunks: list[Chunk]) -> dict[str, dict[str, Any]]:
    """Build a chunk ID to metadata dict for retriever filtering."""
    return {chunk.id: chunk.metadata.vector_metadata() for chunk in chunks}


def filter_chunks(
    chunks: list[Chunk],
    metadata_filter: MetadataFilter | None = None,
) -> list[Chunk]:
    """Filter Chunk objects by metadata."""
    if not metadata_filter:
        return chunks
    return [
        chunk for chunk in chunks
        if metadata_matches(chunk.metadata, metadata_filter)
    ]

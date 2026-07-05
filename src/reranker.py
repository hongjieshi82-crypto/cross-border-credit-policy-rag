"""
Reranker — second-pass reordering of retrieval results via Cohere.

Takes the top-K candidates from a retriever and re-scores them using
Cohere's rerank model, which evaluates query-document relevance more
precisely than initial retrieval signals.

Citations:
  - _instructions.md L629 (reranking)
  - _instructions.md L630 (with/without comparison, deltas)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.retry import retry_with_backoff

# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Return a Cohere client, initialising on first call."""
    global _client
    if _client is not None:
        return _client

    import cohere

    for candidate in [Path(".env.local"), Path("../.env.local")]:
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate))
            break

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not found in environment variables. "
            "Please add it to your .env.local file."
        )

    _client = cohere.Client(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Cohere API call (seam for mocking in tests)
# ---------------------------------------------------------------------------


def _call_cohere_rerank(
    query: str,
    documents: list[str],
    model: str = "rerank-v3.5",
    top_n: int | None = None,
):
    """Call Cohere's rerank API with retry.

    This is the seam that tests mock — keeps API interaction
    isolated from the wiring logic.
    """
    client = _get_client()
    kwargs = {
        "query": query,
        "documents": documents,
        "model": model,
    }
    if top_n is not None:
        kwargs["top_n"] = top_n

    import cohere

    return retry_with_backoff(
        client.rerank,
        kwargs=kwargs,
        max_retries=5,
        base_delay=7.0,
        retryable_exceptions=(cohere.errors.too_many_requests_error.TooManyRequestsError,),
    )


# ---------------------------------------------------------------------------
# Rerank function
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    retrieved_results: list[tuple[str, float]],
    chunk_texts: dict[str, str],
    model: str = "rerank-v3.5",
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Rerank retrieved results using Cohere.

    Args:
        query: The search query.
        retrieved_results: List of (chunk_id, score) from initial retrieval.
        chunk_texts: Map of chunk_id → text content.
        model: Cohere rerank model name.
        top_n: Number of results to return. None = return all.

    Returns:
        List of (chunk_id, relevance_score) tuples, sorted by score descending.
    """
    # Build ordered document list matching retrieval order
    ordered_ids = [cid for cid, _ in retrieved_results]
    documents = [chunk_texts[cid] for cid in ordered_ids]

    response = _call_cohere_rerank(query, documents, model=model, top_n=top_n)

    # Map reranker indices back to chunk IDs
    results = [
        (ordered_ids[item.index], float(item.relevance_score))
        for item in response.results
    ]

    # Sort by relevance score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results

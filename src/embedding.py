"""
Embedding pipeline — batch embed texts with OpenAI models, cache to disk.

Supports text-embedding-3-small (1536 dims) and text-embedding-3-large (3072 dims).
Embeddings are cached as .npy files keyed by chunking config + model name,
so changing retrieval method doesn't trigger re-embedding.

Citations:
  - _instructions.md L608 (2 embedding models)
  - _instructions.md L609 (batch, not one-by-one)
  - _instructions.md L610 (cache to disk)
  - _instructions.md L84  (OpenAI Embedding API)
  - _instructions.md L90  (LiteLLM or OpenAI for switching)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.retry import retry_with_backoff

# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------

_client = None


def _get_client() -> Any:
    """Return an OpenAI client, initialising on first call.

    python-dotenv loads environment variables from a .env.local file into
    os.environ. This keeps sensitive data (like API keys) out of source code
    and version control.
    """
    global _client
    if _client is not None:
        return _client

    from dotenv import load_dotenv
    from openai import OpenAI

    for candidate in [Path(".env.local"), Path("../.env.local")]:
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate))
            break

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env.local file with your API key."
        )

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client

# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------


def split_into_batches(texts: list[str], max_batch_size: int = 2048) -> list[list[str]]:
    """Split a list of texts into batches of at most max_batch_size.

    OpenAI's embedding API accepts up to 2048 texts per request.
    """
    if not texts:
        return []
    return [texts[i:i + max_batch_size] for i in range(0, len(texts), max_batch_size)]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def get_cache_path(cache_dir: str, config_id: str, model: str) -> str:
    """Generate a deterministic cache file path for a config + model combo.

    Args:
        cache_dir: Directory to store cache files.
        config_id: Chunking config identifier (e.g. "pdfplumber_fixed_size_size500_overlap50").
        model: Embedding model name (e.g. "text-embedding-3-small").

    Returns:
        Full path to the .npy cache file.
    """
    filename = f"{config_id}_{model}.npy"
    return os.path.join(cache_dir, filename)


def save_embeddings_to_cache(embeddings: np.ndarray, path: str) -> None:
    """Save embeddings array to disk as .npy."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)


def load_cached_embeddings(path: str) -> np.ndarray | None:
    """Load cached embeddings from disk. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    return np.load(path)


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    cache_dir: str = "data/embeddings",
    config_id: str = "",
    max_batch_size: int = 2048,
) -> np.ndarray:
    """Embed a list of texts using an OpenAI embedding model.

    Checks cache first. If cached embeddings exist and match the expected
    count, returns them without hitting the API.

    Args:
        texts: List of text strings to embed.
        model: OpenAI embedding model name.
        cache_dir: Directory for cached .npy files.
        config_id: Chunking config identifier for cache keying.
        max_batch_size: Max texts per API call (OpenAI limit: 2048).

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.array([])

    # Check cache
    if config_id:
        cache_path = get_cache_path(cache_dir, config_id, model)
        cached = load_cached_embeddings(cache_path)
        if cached is not None and len(cached) == len(texts):
            return cached

    # Batch embed via OpenAI API
    client = _get_client()
    all_embeddings: list[list[float]] = []

    for batch in split_into_batches(texts, max_batch_size):
        response = retry_with_backoff(
            client.embeddings.create,
            args=(),
            kwargs={"input": batch, "model": model},
        )
        # Response embeddings are ordered by index
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)

    result = np.array(all_embeddings, dtype=np.float32)

    # Save to cache
    if config_id:
        save_embeddings_to_cache(result, cache_path)

    return result

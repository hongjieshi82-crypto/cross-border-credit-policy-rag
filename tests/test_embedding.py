"""Tests for embedding pipeline — caching, batching, and interface.

TDD: tests cover the scaffolding (cache hit/miss, batch splitting)
without requiring live API calls.

Citations:
  - _instructions.md L608 (2 embedding models)
  - _instructions.md L609 (batch, not one-by-one)
  - _instructions.md L610 (cache to disk)
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from src.embedding import (
    get_cache_path,
    load_cached_embeddings,
    save_embeddings_to_cache,
    split_into_batches,
)


@pytest.fixture
def tmp_cache_dir():
    """Create a temporary cache directory, clean up after test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------

class TestBatchSplitting:
    """Verify texts are split into batches respecting max_batch_size."""

    def test_single_batch(self):
        texts = ["a", "b", "c"]
        batches = split_into_batches(texts, max_batch_size=10)
        assert len(batches) == 1
        assert batches[0] == texts

    def test_multiple_batches(self):
        texts = [f"text_{i}" for i in range(5)]
        batches = split_into_batches(texts, max_batch_size=2)
        assert len(batches) == 3  # [2, 2, 1]
        assert sum(len(b) for b in batches) == 5

    def test_exact_fit(self):
        texts = [f"text_{i}" for i in range(4)]
        batches = split_into_batches(texts, max_batch_size=2)
        assert len(batches) == 2
        assert all(len(b) == 2 for b in batches)

    def test_empty_list(self):
        batches = split_into_batches([], max_batch_size=10)
        assert batches == []


# ---------------------------------------------------------------------------
# Cache path generation
# ---------------------------------------------------------------------------

class TestCachePath:
    """Verify cache paths are deterministic and include config info."""

    def test_deterministic(self):
        p1 = get_cache_path("dir", "config_abc", "text-embedding-3-small")
        p2 = get_cache_path("dir", "config_abc", "text-embedding-3-small")
        assert p1 == p2

    def test_different_model_different_path(self):
        p1 = get_cache_path("dir", "config_abc", "text-embedding-3-small")
        p2 = get_cache_path("dir", "config_abc", "text-embedding-3-large")
        assert p1 != p2

    def test_different_config_different_path(self):
        p1 = get_cache_path("dir", "config_abc", "text-embedding-3-small")
        p2 = get_cache_path("dir", "config_xyz", "text-embedding-3-small")
        assert p1 != p2

    def test_ends_with_npy(self):
        p = get_cache_path("dir", "config_abc", "text-embedding-3-small")
        assert p.endswith(".npy")


# ---------------------------------------------------------------------------
# Cache save/load round-trip
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:
    """Verify embeddings survive a save/load cycle."""

    def test_save_and_load(self, tmp_cache_dir):
        embeddings = np.random.rand(10, 1536).astype(np.float32)
        path = os.path.join(tmp_cache_dir, "test_embeddings.npy")

        save_embeddings_to_cache(embeddings, path)
        loaded = load_cached_embeddings(path)

        assert loaded is not None
        assert np.array_equal(embeddings, loaded)

    def test_load_returns_none_if_missing(self, tmp_cache_dir):
        path = os.path.join(tmp_cache_dir, "nonexistent.npy")
        loaded = load_cached_embeddings(path)
        assert loaded is None

    def test_shape_preserved(self, tmp_cache_dir):
        embeddings = np.random.rand(5, 3072).astype(np.float32)
        path = os.path.join(tmp_cache_dir, "test_embeddings.npy")

        save_embeddings_to_cache(embeddings, path)
        loaded = load_cached_embeddings(path)

        assert loaded.shape == (5, 3072)

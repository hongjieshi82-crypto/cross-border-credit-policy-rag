"""Tests for Phase 2 grid search runner — orchestration wiring.

Tests verify the runner correctly iterates configs × models × methods,
wires up retrieval + metrics, and returns ExperimentResult objects.
BM25 tests run end-to-end; vector/hybrid tests mock embed_texts.

Citations:
  - _instructions.md L625 (grid search runner)
  - _instructions.md L608 (2 embedding models)
  - _instructions.md L613-615 (3 retrieval methods)
"""

from unittest.mock import patch

import numpy as np
import pytest

from src.models import (
    Chunk, ChunkMetadata, ChunkingConfig,
    ExperimentResult, QAExample,
)
from src.grid_runner import run_single_experiment, run_phase2_grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(text: str, page: int, index: int) -> Chunk:
    """Build a valid Chunk with minimal boilerplate."""
    return Chunk(
        text=text,
        page_number=page,
        chunk_index=index,
        start_char=0,
        end_char=len(text),
        method="fixed_size",
        metadata=ChunkMetadata(chunk_size=500, overlap=50, parser="pypdf"),
    )


@pytest.fixture
def chunks():
    """Chunks with distinct vocabulary for predictable BM25 results."""
    return [
        _make_chunk("Revenue grew 15% to $4.2 billion in fiscal 2022", page=1, index=0),
        _make_chunk("The company expanded into three new international markets", page=2, index=1),
        _make_chunk("Operating expenses decreased by 8% due to automation", page=3, index=2),
        _make_chunk("Net income increased to $800 million from $650 million", page=4, index=3),
        _make_chunk("Employee headcount grew to 12000 across all regions", page=5, index=4),
    ]


@pytest.fixture
def qa_examples(chunks):
    """QA pairs where each question targets vocabulary in a specific chunk."""
    return [
        QAExample(
            question="What was the revenue growth percentage?",
            relevant_chunk_ids=[chunks[0].id],
            source_page=1,
            chunk_method="fixed_size",
        ),
        QAExample(
            question="How many new international markets did the company enter?",
            relevant_chunk_ids=[chunks[1].id],
            source_page=2,
            chunk_method="fixed_size",
        ),
        QAExample(
            question="By what percentage did operating expenses decrease?",
            relevant_chunk_ids=[chunks[2].id],
            source_page=3,
            chunk_method="fixed_size",
        ),
    ]


@pytest.fixture
def chunking_config():
    return ChunkingConfig(
        parser="pypdf",
        chunker="fixed_size",
        chunk_size=500,
        overlap=50,
    )


# ---------------------------------------------------------------------------
# run_single_experiment
# ---------------------------------------------------------------------------

class TestRunSingleExperiment:
    """Verify a single experiment produces correct ExperimentResult."""

    def test_bm25_returns_experiment_result(self, chunks, qa_examples, chunking_config):
        """BM25 needs no embeddings — full end-to-end test."""
        result = run_single_experiment(
            chunks=chunks,
            qa_examples=qa_examples,
            config=chunking_config,
            embedding_model="text-embedding-3-small",
            retrieval_method="bm25",
        )
        assert isinstance(result, ExperimentResult)

    def test_bm25_metrics_populated(self, chunks, qa_examples, chunking_config):
        result = run_single_experiment(
            chunks=chunks,
            qa_examples=qa_examples,
            config=chunking_config,
            embedding_model="text-embedding-3-small",
            retrieval_method="bm25",
        )
        assert result.metrics.total_queries == len(qa_examples)
        assert result.metrics.mrr >= 0.0
        assert result.metrics.avg_retrieval_time >= 0.0

    def test_config_matches_input(self, chunks, qa_examples, chunking_config):
        result = run_single_experiment(
            chunks=chunks,
            qa_examples=qa_examples,
            config=chunking_config,
            embedding_model="text-embedding-3-small",
            retrieval_method="bm25",
        )
        assert result.config.embedding_model == "text-embedding-3-small"
        assert result.config.retrieval_method == "bm25"
        assert result.config.chunking == chunking_config

    def test_vector_returns_experiment_result(self, chunks, qa_examples, chunking_config):
        np.random.seed(42)
        fake_embeddings = np.random.rand(len(chunks), 8).astype(np.float32)

        with patch("src.grid_runner.embed_texts") as mock_embed:
            mock_embed.return_value = fake_embeddings
            result = run_single_experiment(
                chunks=chunks,
                qa_examples=qa_examples,
                config=chunking_config,
                embedding_model="text-embedding-3-small",
                retrieval_method="vector",
            )
        assert isinstance(result, ExperimentResult)
        assert result.metrics.total_queries == len(qa_examples)

    def test_hybrid_returns_experiment_result(self, chunks, qa_examples, chunking_config):
        np.random.seed(42)
        fake_embeddings = np.random.rand(len(chunks), 8).astype(np.float32)

        with patch("src.grid_runner.embed_texts") as mock_embed:
            mock_embed.return_value = fake_embeddings
            result = run_single_experiment(
                chunks=chunks,
                qa_examples=qa_examples,
                config=chunking_config,
                embedding_model="text-embedding-3-small",
                retrieval_method="hybrid",
            )
        assert isinstance(result, ExperimentResult)
        assert result.metrics.total_queries == len(qa_examples)

    def test_recall_and_precision_in_valid_range(self, chunks, qa_examples, chunking_config):
        result = run_single_experiment(
            chunks=chunks,
            qa_examples=qa_examples,
            config=chunking_config,
            embedding_model="text-embedding-3-small",
            retrieval_method="bm25",
        )
        for k, val in result.metrics.recall_at_k.items():
            assert 0.0 <= val <= 1.0
        for k, val in result.metrics.precision_at_k.items():
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# run_phase2_grid
# ---------------------------------------------------------------------------

class TestRunPhase2Grid:
    """Verify the full grid iterates all combinations."""

    def test_correct_number_of_results(self, chunks, qa_examples, chunking_config):
        """1 config × 1 model × 2 methods = 2 results."""
        np.random.seed(42)
        fake_embeddings = np.random.rand(len(chunks), 8).astype(np.float32)

        with patch("src.grid_runner.embed_texts") as mock_embed:
            mock_embed.return_value = fake_embeddings
            results = run_phase2_grid(
                configs_with_chunks={chunking_config.config_id: (chunking_config, chunks)},
                qa_by_config={chunking_config.config_id: qa_examples},
                embedding_models=["text-embedding-3-small"],
                retrieval_methods=["bm25", "vector"],
            )
        assert len(results) == 2

    def test_all_results_are_experiment_results(self, chunks, qa_examples, chunking_config):
        np.random.seed(42)
        fake_embeddings = np.random.rand(len(chunks), 8).astype(np.float32)

        with patch("src.grid_runner.embed_texts") as mock_embed:
            mock_embed.return_value = fake_embeddings
            results = run_phase2_grid(
                configs_with_chunks={chunking_config.config_id: (chunking_config, chunks)},
                qa_by_config={chunking_config.config_id: qa_examples},
                embedding_models=["text-embedding-3-small"],
                retrieval_methods=["vector"],
            )
        for r in results:
            assert isinstance(r, ExperimentResult)

    def test_each_result_has_unique_experiment_id(self, chunks, qa_examples, chunking_config):
        np.random.seed(42)
        fake_embeddings = np.random.rand(len(chunks), 8).astype(np.float32)

        with patch("src.grid_runner.embed_texts") as mock_embed:
            mock_embed.return_value = fake_embeddings
            results = run_phase2_grid(
                configs_with_chunks={chunking_config.config_id: (chunking_config, chunks)},
                qa_by_config={chunking_config.config_id: qa_examples},
                embedding_models=["text-embedding-3-small"],
                retrieval_methods=["bm25", "vector", "hybrid"],
            )
        ids = [r.config.experiment_id for r in results]
        assert len(set(ids)) == 3  # all unique

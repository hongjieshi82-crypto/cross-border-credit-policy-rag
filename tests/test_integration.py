"""Integration test — end-to-end single config through the pipeline.

Runs a real BM25 pipeline: parse PDF → chunk → build index → retrieve →
compute metrics → save results → generate charts. No API calls needed.

Citations:
  - _instructions.md L632 (end-to-end integration test)
"""

import json
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.models import (
    Chunk, ChunkMetadata, ChunkingConfig,
    ExperimentResult, QAExample,
)
from src.grid_runner import run_single_experiment
from src.results_io import save_results, load_results, select_best_config
from src.display import display_phase2_results
from src.visualizations import plot_mrr_bar, plot_time_vs_quality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(text: str, page: int, index: int) -> Chunk:
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
def pipeline_data():
    """Realistic-ish chunks and QA for integration testing."""
    chunks = [
        _make_chunk(
            "Total revenue for fiscal year 2022 was $4.2 billion, "
            "representing a 15% increase over the prior year.",
            page=1, index=0,
        ),
        _make_chunk(
            "The company expanded operations into Brazil, India, and "
            "Germany during the fiscal year.",
            page=2, index=1,
        ),
        _make_chunk(
            "Operating expenses decreased by 8% year-over-year, primarily "
            "driven by automation initiatives across manufacturing.",
            page=3, index=2,
        ),
        _make_chunk(
            "Net income for the year reached $800 million, up from "
            "$650 million in the previous fiscal year.",
            page=4, index=3,
        ),
        _make_chunk(
            "The board of directors approved a quarterly dividend of "
            "$0.50 per share, a 10% increase.",
            page=5, index=4,
        ),
    ]

    qa_examples = [
        QAExample(
            question="What was the total revenue in fiscal year 2022?",
            relevant_chunk_ids=[chunks[0].id],
            source_page=1,
            chunk_method="fixed_size",
        ),
        QAExample(
            question="Which countries did the company expand into?",
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

    config = ChunkingConfig(
        parser="pypdf",
        chunker="fixed_size",
        chunk_size=500,
        overlap=50,
    )

    return chunks, qa_examples, config


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline integration: retrieve → metrics → save → visualize."""

    def test_bm25_pipeline(self, pipeline_data, tmp_dir):
        """BM25 end-to-end — no API calls needed."""
        chunks, qa_examples, config = pipeline_data

        # Step 1: Run experiment
        result = run_single_experiment(
            chunks=chunks,
            qa_examples=qa_examples,
            config=config,
            embedding_model="text-embedding-3-small",
            retrieval_method="bm25",
        )

        # Step 2: Verify result structure
        assert isinstance(result, ExperimentResult)
        assert result.metrics.total_queries == 3
        assert result.metrics.mrr > 0.0
        assert result.metrics.avg_retrieval_time > 0.0
        assert 5 in result.metrics.recall_at_k

        # Step 3: Save and reload
        results_path = os.path.join(tmp_dir, "results.json")
        save_results([result], results_path)
        loaded = load_results(results_path)
        assert len(loaded) == 1
        assert loaded[0].metrics.mrr == result.metrics.mrr

        # Step 4: Verify JSON structure
        with open(results_path) as f:
            data = json.load(f)
        assert "best_by_accuracy" in data
        assert "best_by_speed" in data
        assert "metadata" in data
        assert data["metadata"]["total_experiments"] == 1

        # Step 5: Best config selection
        best = select_best_config([result])
        assert best.config.retrieval_method == "bm25"

        # Step 6: Visualizations don't crash
        fig1 = plot_mrr_bar([result])
        assert fig1 is not None
        fig2 = plot_time_vs_quality([result])
        assert fig2 is not None

    def test_multi_method_pipeline(self, pipeline_data, tmp_dir):
        """Multiple retrieval methods in one grid run."""
        chunks, qa_examples, config = pipeline_data
        np.random.seed(42)
        fake_embeddings = np.random.rand(len(chunks), 8).astype(np.float32)

        results = []
        for method in ["bm25", "vector", "hybrid"]:
            from unittest.mock import patch
            if method == "bm25":
                result = run_single_experiment(
                    chunks=chunks,
                    qa_examples=qa_examples,
                    config=config,
                    embedding_model="text-embedding-3-small",
                    retrieval_method=method,
                )
            else:
                with patch("src.grid_runner.embed_texts") as mock_embed:
                    mock_embed.return_value = fake_embeddings
                    result = run_single_experiment(
                        chunks=chunks,
                        qa_examples=qa_examples,
                        config=config,
                        embedding_model="text-embedding-3-small",
                        retrieval_method=method,
                        embeddings=fake_embeddings,
                    )
            results.append(result)

        # All three produced valid results
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ExperimentResult)
            assert r.metrics.total_queries == 3

        # Save, reload, verify
        results_path = os.path.join(tmp_dir, "results.json")
        save_results(results, results_path)
        loaded = load_results(results_path)
        assert len(loaded) == 3

        # Display doesn't crash
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        display_phase2_results(results, console=console)
        assert len(buf.getvalue()) > 0

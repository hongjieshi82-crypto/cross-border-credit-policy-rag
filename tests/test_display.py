"""Smoke tests for Rich console display — verify functions don't crash
and produce non-empty output.

Not testing visual formatting — just that valid data renders without errors.

Citations:
  - _instructions.md L627 (formatted console tables)
"""

from io import StringIO

import pytest
from rich.console import Console

from src.models import (
    ChunkingConfig, ExperimentConfig, ExperimentResult, MetricsResult,
)
from src.parser_eval import ChunkQualityStats
from src.display import display_phase1_results, display_phase2_results


@pytest.fixture
def phase1_results():
    """Sample Phase 1 results with rank scores."""
    return [
        (ChunkQualityStats(
            parser="pypdf", chunker="fixed_size", chunk_size=500, overlap=50,
            chunk_count=20, avg_chunk_size=480.0, min_chunk_size=200,
            max_chunk_size=500, size_std=50.0, coverage=0.95, total_chars=9600,
        ), 0.87),
        (ChunkQualityStats(
            parser="pdfplumber", chunker="sentence", chunk_size=300, overlap=30,
            chunk_count=35, avg_chunk_size=280.0, min_chunk_size=100,
            max_chunk_size=300, size_std=40.0, coverage=0.92, total_chars=9800,
        ), 0.82),
    ]


@pytest.fixture
def phase2_results():
    """Sample Phase 2 experiment results."""
    return [
        ExperimentResult(
            config=ExperimentConfig(
                chunking=ChunkingConfig(
                    parser="pypdf", chunker="fixed_size",
                    chunk_size=500, overlap=50,
                ),
                embedding_model="text-embedding-3-small",
                retrieval_method="vector",
            ),
            metrics=MetricsResult(
                recall_at_k={1: 0.6, 3: 0.8, 5: 0.9, 10: 1.0},
                precision_at_k={1: 0.6, 3: 0.4, 5: 0.3, 10: 0.2},
                mrr=0.85,
                map_score=0.78,
                ndcg_at_k={1: 0.6, 3: 0.75, 5: 0.82, 10: 0.88},
                total_queries=10,
                avg_retrieval_time=0.023,
            ),
        ),
        ExperimentResult(
            config=ExperimentConfig(
                chunking=ChunkingConfig(
                    parser="pypdf", chunker="fixed_size",
                    chunk_size=500, overlap=50,
                ),
                embedding_model="text-embedding-3-small",
                retrieval_method="bm25",
            ),
            metrics=MetricsResult(
                recall_at_k={1: 0.5, 3: 0.7, 5: 0.85, 10: 0.95},
                precision_at_k={1: 0.5, 3: 0.35, 5: 0.28, 10: 0.18},
                mrr=0.72,
                map_score=0.65,
                ndcg_at_k={1: 0.5, 3: 0.65, 5: 0.75, 10: 0.82},
                total_queries=10,
                avg_retrieval_time=0.005,
            ),
        ),
    ]


class TestDisplayPhase1:
    """Verify Phase 1 table renders without errors."""

    def test_produces_output(self, phase1_results):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        display_phase1_results(phase1_results, console=console)
        output = buf.getvalue()
        assert len(output) > 0

    def test_contains_parser_names(self, phase1_results):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        display_phase1_results(phase1_results, console=console)
        output = buf.getvalue()
        assert "pypdf" in output
        assert "pdfplumber" in output


class TestDisplayPhase2:
    """Verify Phase 2 table renders without errors."""

    def test_produces_output(self, phase2_results):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        display_phase2_results(phase2_results, console=console)
        output = buf.getvalue()
        assert len(output) > 0

    def test_contains_retrieval_methods(self, phase2_results):
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=200)
        display_phase2_results(phase2_results, console=console)
        output = buf.getvalue()
        assert "vector" in output
        assert "bm25" in output

"""Smoke tests for visualizations — verify charts render without errors.

Not testing visual output — just that valid data produces figure objects.

Citations:
  - _instructions.md L631 (6 visualization charts)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing

import pytest

from src.models import (
    ChunkingConfig, ExperimentConfig, ExperimentResult, MetricsResult,
)
from src.visualizations import (
    plot_mrr_bar,
    plot_recall_vs_precision,
    plot_metrics_heatmap,
    plot_retrieval_comparison,
    plot_correlation_matrix,
    plot_time_vs_quality,
)


def _make_result(parser, chunker, model, method, mrr, avg_time,
                 reranking=False) -> ExperimentResult:
    return ExperimentResult(
        config=ExperimentConfig(
            chunking=ChunkingConfig(
                parser=parser, chunker=chunker,
                chunk_size=500, overlap=50,
            ),
            embedding_model=model,
            retrieval_method=method,
        ),
        metrics=MetricsResult(
            recall_at_k={1: 0.5, 3: 0.7, 5: 0.85, 10: 0.95},
            precision_at_k={1: 0.5, 3: 0.35, 5: 0.28, 10: 0.18},
            mrr=mrr,
            map_score=mrr - 0.05,
            ndcg_at_k={1: 0.5, 3: 0.65, 5: 0.75, 10: 0.82},
            total_queries=10,
            avg_retrieval_time=avg_time,
        ),
        use_reranking=reranking,
    )


@pytest.fixture
def sample_results():
    return [
        _make_result("pypdf", "fixed_size", "text-embedding-3-small", "bm25", 0.82, 0.005),
        _make_result("pypdf", "fixed_size", "text-embedding-3-small", "vector", 0.88, 0.025),
        _make_result("pypdf", "fixed_size", "text-embedding-3-small", "hybrid", 0.90, 0.030),
        _make_result("pypdf", "sentence", "text-embedding-3-small", "bm25", 0.80, 0.004),
        _make_result("pypdf", "sentence", "text-embedding-3-small", "vector", 0.86, 0.022),
        _make_result("pypdf", "sentence", "text-embedding-3-small", "hybrid", 0.88, 0.028),
    ]


class TestVisualizations:
    """Verify each chart function returns a matplotlib Figure."""

    def test_mrr_bar(self, sample_results):
        fig = plot_mrr_bar(sample_results)
        assert fig is not None
        assert len(fig.axes) > 0

    def test_recall_vs_precision(self, sample_results):
        fig = plot_recall_vs_precision(sample_results)
        assert fig is not None
        assert len(fig.axes) > 0

    def test_metrics_heatmap(self, sample_results):
        fig = plot_metrics_heatmap(sample_results)
        assert fig is not None
        assert len(fig.axes) > 0

    def test_retrieval_comparison(self, sample_results):
        fig = plot_retrieval_comparison(sample_results)
        assert fig is not None
        assert len(fig.axes) > 0

    def test_correlation_matrix(self, sample_results):
        fig = plot_correlation_matrix(sample_results)
        assert fig is not None
        assert len(fig.axes) > 0

    def test_time_vs_quality(self, sample_results):
        fig = plot_time_vs_quality(sample_results)
        assert fig is not None
        assert len(fig.axes) > 0

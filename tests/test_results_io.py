"""Tests for JSON results output — save/load experiment results.

Verifies round-trip persistence and best-config selection.

Citations:
  - _instructions.md L628 (JSON results output)
"""

import json
import os
import shutil
import tempfile

import pytest

from src.models import (
    ChunkingConfig, ExperimentConfig, ExperimentResult, MetricsResult,
)
from src.results_io import (
    save_results, load_results,
    select_best_config, select_fastest_above_bar,
)


def _make_result(retrieval: str, mrr: float, avg_time: float) -> ExperimentResult:
    """Helper to build an ExperimentResult with minimal boilerplate."""
    return ExperimentResult(
        config=ExperimentConfig(
            chunking=ChunkingConfig(
                parser="pypdf", chunker="fixed_size",
                chunk_size=500, overlap=50,
            ),
            embedding_model="text-embedding-3-small",
            retrieval_method=retrieval,
        ),
        metrics=MetricsResult(
            recall_at_k={1: 0.6, 3: 0.8, 5: 0.9, 10: 1.0},
            precision_at_k={1: 0.6, 3: 0.4, 5: 0.3, 10: 0.2},
            mrr=mrr,
            map_score=mrr - 0.05,
            ndcg_at_k={1: 0.6, 3: 0.75, 5: 0.82, 10: 0.88},
            total_queries=10,
            avg_retrieval_time=avg_time,
        ),
    )


@pytest.fixture
def sample_results():
    return [
        _make_result("bm25", mrr=0.85, avg_time=0.005),
        _make_result("vector", mrr=0.90, avg_time=0.025),
        _make_result("hybrid", mrr=0.90, avg_time=0.030),
    ]


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestSaveAndLoadResults:
    """Verify results survive a save/load round-trip."""

    def test_save_creates_file(self, sample_results, tmp_dir):
        path = os.path.join(tmp_dir, "results.json")
        save_results(sample_results, path)
        assert os.path.exists(path)

    def test_round_trip_preserves_count(self, sample_results, tmp_dir):
        path = os.path.join(tmp_dir, "results.json")
        save_results(sample_results, path)
        loaded = load_results(path)
        assert len(loaded) == len(sample_results)

    def test_round_trip_preserves_metrics(self, sample_results, tmp_dir):
        path = os.path.join(tmp_dir, "results.json")
        save_results(sample_results, path)
        loaded = load_results(path)
        for original, restored in zip(sample_results, loaded):
            assert original.metrics.mrr == restored.metrics.mrr
            assert original.metrics.map_score == restored.metrics.map_score
            assert original.config.retrieval_method == restored.config.retrieval_method

    def test_file_is_valid_json(self, sample_results, tmp_dir):
        path = os.path.join(tmp_dir, "results.json")
        save_results(sample_results, path)
        with open(path) as f:
            data = json.load(f)
        assert "results" in data
        assert "best_by_accuracy" in data
        assert "best_by_speed" in data
        assert "metadata" in data

    def test_loaded_results_are_experiment_results(self, sample_results, tmp_dir):
        path = os.path.join(tmp_dir, "results.json")
        save_results(sample_results, path)
        loaded = load_results(path)
        for r in loaded:
            assert isinstance(r, ExperimentResult)


class TestSelectBestConfig:
    """Verify best config selection logic."""

    def test_highest_mrr_wins(self, sample_results):
        best = select_best_config(sample_results)
        assert best.metrics.mrr == 0.90

    def test_tiebreak_by_fastest(self, sample_results):
        """Among configs with same MRR, fastest avg_time wins."""
        best = select_best_config(sample_results)
        # vector (0.025s) beats hybrid (0.030s) — both have MRR 0.90
        assert best.config.retrieval_method == "vector"

    def test_single_result(self):
        results = [_make_result("bm25", mrr=0.70, avg_time=0.01)]
        best = select_best_config(results)
        assert best.config.retrieval_method == "bm25"


class TestSelectFastestAboveBar:
    """Verify speed-first selection among quality-bar configs."""

    def test_picks_fastest_qualifying(self, sample_results):
        # vector (MRR 0.90, 0.025s) and hybrid (MRR 0.90, 0.030s) clear bar
        # bm25 (MRR 0.85, 0.005s) also clears bar and is fastest
        best = select_fastest_above_bar(sample_results)
        assert best.config.retrieval_method == "bm25"

    def test_returns_none_when_none_qualify(self):
        results = [_make_result("bm25", mrr=0.50, avg_time=0.005)]
        best = select_fastest_above_bar(results)
        assert best is None

    def test_respects_mrr_threshold(self):
        results = [
            _make_result("bm25", mrr=0.84, avg_time=0.001),  # below bar
            _make_result("vector", mrr=0.90, avg_time=0.025),  # above bar
        ]
        best = select_fastest_above_bar(results)
        assert best.config.retrieval_method == "vector"

    def test_custom_thresholds(self):
        results = [
            _make_result("bm25", mrr=0.70, avg_time=0.005),
            _make_result("vector", mrr=0.75, avg_time=0.025),
        ]
        # Lower the bar — both qualify, bm25 is faster
        best = select_fastest_above_bar(results, min_mrr=0.60, min_recall_at_5=0.0)
        assert best.config.retrieval_method == "bm25"

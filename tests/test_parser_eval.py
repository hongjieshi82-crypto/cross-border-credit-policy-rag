"""Tests for chunk quality evaluation function.

TDD: the evaluation function is deterministic — given chunks, return stats.
The orchestration (run_phase1_pregrid) is not tested here; it needs a real PDF.

Citations:
  - _plan.md T5.5 (Phase 1 pre-grid)
  - _plan.md resolved ambiguities (chunk quality statistics)
"""

import pytest

from src.models import Chunk, ChunkMetadata
from src.parser_eval import evaluate_chunk_quality, ChunkQualityStats


def _make_chunk(text: str, chunk_index: int = 0, page_number: int = 0) -> Chunk:
    """Helper to create a Chunk with minimal boilerplate."""
    return Chunk(
        text=text,
        page_number=page_number,
        chunk_index=chunk_index,
        start_char=0,
        end_char=max(len(text), 1),
        method="fixed_size",
        metadata=ChunkMetadata(chunk_size=100, overlap=0, parser="pdfplumber"),
    )


class TestEvaluateChunkQuality:
    """Verify chunk quality stats are computed correctly."""

    def test_returns_stats_object(self):
        chunks = [_make_chunk("Hello world", 0), _make_chunk("Foo bar baz", 1)]
        stats = evaluate_chunk_quality(chunks, source_text_length=100)
        assert isinstance(stats, ChunkQualityStats)

    def test_chunk_count(self):
        chunks = [_make_chunk(f"chunk {i}", i) for i in range(5)]
        stats = evaluate_chunk_quality(chunks, source_text_length=100)
        assert stats.chunk_count == 5

    def test_avg_chunk_size(self):
        chunks = [
            _make_chunk("a" * 100, 0),
            _make_chunk("b" * 200, 1),
        ]
        stats = evaluate_chunk_quality(chunks, source_text_length=300)
        assert stats.avg_chunk_size == 150.0

    def test_coverage(self):
        """Coverage = total chars in chunks / source text length."""
        chunks = [_make_chunk("a" * 80, 0)]
        stats = evaluate_chunk_quality(chunks, source_text_length=100)
        assert stats.coverage == pytest.approx(0.80)

    def test_coverage_with_overlap_can_exceed_one(self):
        """With overlap, total chunk chars can exceed source length."""
        chunks = [_make_chunk("a" * 80, 0), _make_chunk("a" * 80, 1)]
        stats = evaluate_chunk_quality(chunks, source_text_length=100)
        assert stats.coverage > 1.0

    def test_size_variance(self):
        """Equal-sized chunks should have zero variance."""
        chunks = [_make_chunk("a" * 50, i) for i in range(3)]
        stats = evaluate_chunk_quality(chunks, source_text_length=150)
        assert stats.size_std == 0.0

    def test_size_variance_nonzero(self):
        """Different-sized chunks should have nonzero variance."""
        chunks = [_make_chunk("a" * 50, 0), _make_chunk("b" * 200, 1)]
        stats = evaluate_chunk_quality(chunks, source_text_length=250)
        assert stats.size_std > 0

    def test_empty_chunks_list(self):
        stats = evaluate_chunk_quality([], source_text_length=100)
        assert stats.chunk_count == 0
        assert stats.avg_chunk_size == 0.0
        assert stats.coverage == 0.0

    def test_min_and_max_chunk_size(self):
        chunks = [
            _make_chunk("a" * 50, 0),
            _make_chunk("b" * 200, 1),
            _make_chunk("c" * 100, 2),
        ]
        stats = evaluate_chunk_quality(chunks, source_text_length=350)
        assert stats.min_chunk_size == 50
        assert stats.max_chunk_size == 200

"""
Phase 1 pre-grid: evaluate parser × chunking configurations.

Runs 3 parsers × N chunk sizes × M overlaps, scores each on chunk quality
statistics, and returns the top K parser+chunking combos for Phase 2.

Citations:
  - _plan.md T5.5, resolved ambiguities (two-phase grid)
  - _instructions.md L26 (compare parser output quality)
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, Field

from src.chunking import chunk_fixed_size, chunk_sentence, chunk_semantic
from src.models import Chunk
from src.parsing import PARSERS


# ---------------------------------------------------------------------------
# Chunk quality stats
# ---------------------------------------------------------------------------

class ChunkQualityStats(BaseModel):
    """Quality statistics for a set of chunks from one configuration."""

    parser: str = ""
    chunker: str = ""
    chunk_size: int = 0
    overlap: int = 0
    chunk_count: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    size_std: float = 0.0
    coverage: float = 0.0
    total_chars: int = 0


def evaluate_chunk_quality(
    chunks: list[Chunk],
    source_text_length: int,
) -> ChunkQualityStats:
    """Compute quality statistics for a list of chunks.

    Args:
        chunks: The chunks to evaluate.
        source_text_length: Length of the original source text in characters.

    Returns:
        ChunkQualityStats with computed metrics.
    """
    if not chunks:
        return ChunkQualityStats()

    sizes = [len(c.text) for c in chunks]
    total_chars = sum(sizes)

    return ChunkQualityStats(
        chunk_count=len(chunks),
        avg_chunk_size=statistics.mean(sizes),
        min_chunk_size=min(sizes),
        max_chunk_size=max(sizes),
        size_std=statistics.pstdev(sizes),  # population std dev
        coverage=total_chars / source_text_length if source_text_length > 0 else 0.0,
        total_chars=total_chars,
    )


# ---------------------------------------------------------------------------
# Chunker registry (mirrors PARSERS pattern from parsing.py)
# ---------------------------------------------------------------------------

CHUNKERS: dict[str, callable] = {
    "fixed_size": chunk_fixed_size,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}


# ---------------------------------------------------------------------------
# Phase 1 pre-grid runner
# ---------------------------------------------------------------------------

class PreGridResult(BaseModel):
    """Result of one pre-grid configuration evaluation."""

    stats: ChunkQualityStats
    rank_score: float = Field(
        ..., description="Composite score for ranking (higher = better)"
    )


def compute_rank_score(stats: ChunkQualityStats, target_chunk_size: int) -> float:
    """Compute a composite ranking score for a chunk configuration.

    Favors configs that:
    - Have reasonable coverage (close to 1.0, penalize far above or below)
    - Have low size variance (consistent chunk sizes)
    - Have avg chunk size close to the target
    - Produce a reasonable number of chunks (not too few, not too many)

    Returns a score where higher is better.
    """
    if stats.chunk_count == 0:
        return 0.0

    # Coverage score: 1.0 is ideal, penalize deviation
    coverage_score = 1.0 - abs(stats.coverage - 1.0) * 0.5

    # Size consistency: lower std relative to mean is better
    consistency_score = (
        1.0 - (stats.size_std / stats.avg_chunk_size)
        if stats.avg_chunk_size > 0
        else 0.0
    )

    # Target adherence: how close is avg size to the requested chunk_size
    target_score = 1.0 - abs(stats.avg_chunk_size - target_chunk_size) / target_chunk_size

    # Clamp all scores to [0, 1]
    coverage_score = max(0.0, min(1.0, coverage_score))
    consistency_score = max(0.0, min(1.0, consistency_score))
    target_score = max(0.0, min(1.0, target_score))

    # Weighted composite
    return (
        0.4 * coverage_score
        + 0.3 * consistency_score
        + 0.3 * target_score
    )


def run_phase1_pregrid(
    pdf_path: str,
    chunk_sizes: list[int] | None = None,
    overlaps: list[int] | None = None,
    top_k: int = 3,
) -> list[PreGridResult]:
    """Run Phase 1 pre-grid: evaluate all parser × chunker × size × overlap combos.

    Args:
        pdf_path: Path to the PDF file.
        chunk_sizes: List of chunk sizes to try. Defaults to [500, 750, 1000, 2000].
        overlaps: List of overlap values to try. Defaults to [0, 50, 100, 200].
        top_k: Number of top configurations to return.

    Returns:
        Top K PreGridResult objects, sorted by rank_score descending.
    """
    if chunk_sizes is None:
        chunk_sizes = [500, 750, 1000, 2000]
    if overlaps is None:
        overlaps = [0, 50, 100, 200]

    results: list[PreGridResult] = []

    for parser_name, parse_fn in PARSERS.items():
        # Parse PDF once per parser
        pages = parse_fn(pdf_path)
        full_text = "\n".join(text for _, text in pages)
        source_len = len(full_text)

        for chunker_name, chunk_fn in CHUNKERS.items():
            for size in chunk_sizes:
                for overlap in overlaps:
                    # Skip invalid combos (overlap >= chunk_size)
                    if overlap >= size:
                        continue

                    # Chunk all pages
                    all_chunks: list[Chunk] = []
                    for page_num, page_text in pages:
                        if not page_text.strip():
                            continue
                        chunks = chunk_fn(
                            text=page_text,
                            page_number=page_num,
                            chunk_size=size,
                            overlap=overlap,
                            parser=parser_name,
                        )
                        all_chunks.extend(chunks)

                    # Evaluate
                    stats = evaluate_chunk_quality(all_chunks, source_len)
                    stats.parser = parser_name
                    stats.chunker = chunker_name
                    stats.chunk_size = size
                    stats.overlap = overlap

                    score = compute_rank_score(stats, target_chunk_size=size)

                    results.append(PreGridResult(stats=stats, rank_score=score))

    # Sort by rank_score descending, return top K
    results.sort(key=lambda r: r.rank_score, reverse=True)
    return results[:top_k]

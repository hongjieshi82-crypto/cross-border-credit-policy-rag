"""
Rich console display — formatted tables for pipeline results.

Phase 1 table shows chunking config quality stats and rank scores.
Phase 2 table shows experiment results with IR metrics, sorted by MRR.

Citations:
  - _instructions.md L627 (formatted console tables)
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from src.models import ExperimentResult
from src.parser_eval import ChunkQualityStats


def display_phase1_results(
    results: list[tuple[ChunkQualityStats, float]],
    console: Console | None = None,
) -> None:
    """Display Phase 1 pre-grid results as a ranked table.

    Args:
        results: List of (ChunkQualityStats, rank_score) tuples,
            sorted by rank_score descending.
        console: Rich Console instance (defaults to stdout).
    """
    if console is None:
        console = Console()

    table = Table(title="Phase 1 — Chunking Config Rankings")

    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Parser")
    table.add_column("Chunker")
    table.add_column("Size", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Avg Size", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Score", justify="right", style="bold cyan")

    for rank, (stats, score) in enumerate(results, 1):
        table.add_row(
            str(rank),
            stats.parser,
            stats.chunker,
            str(stats.chunk_size),
            str(stats.overlap),
            str(stats.chunk_count),
            f"{stats.avg_chunk_size:.0f}",
            f"{stats.coverage:.2%}",
            f"{stats.size_std:.1f}",
            f"{score:.4f}",
        )

    console.print(table)


def display_phase2_results(
    results: list[ExperimentResult],
    console: Console | None = None,
) -> None:
    """Display Phase 2 grid results as a table sorted by MRR.

    Args:
        results: List of ExperimentResult objects.
        console: Rich Console instance (defaults to stdout).
    """
    if console is None:
        console = Console()

    # Sort by MRR descending
    sorted_results = sorted(results, key=lambda r: r.metrics.mrr, reverse=True)

    table = Table(title="Phase 2 — Experiment Results")

    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Parser")
    table.add_column("Chunker")
    table.add_column("Embedding")
    table.add_column("Retrieval")
    table.add_column("MRR", justify="right", style="bold cyan")
    table.add_column("MAP", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("P@5", justify="right")
    table.add_column("NDCG@5", justify="right")
    table.add_column("Avg Time", justify="right")

    for rank, result in enumerate(sorted_results, 1):
        cfg = result.config
        m = result.metrics
        table.add_row(
            str(rank),
            cfg.chunking.parser,
            cfg.chunking.chunker,
            cfg.embedding_model.replace("text-embedding-3-", ""),  # shorten
            cfg.retrieval_method,
            f"{m.mrr:.4f}",
            f"{m.map_score:.4f}",
            f"{m.recall_at_k.get(5, 0.0):.4f}",
            f"{m.precision_at_k.get(5, 0.0):.4f}",
            f"{m.ndcg_at_k.get(5, 0.0):.4f}",
            f"{m.avg_retrieval_time:.4f}s",
        )

    console.print(table)

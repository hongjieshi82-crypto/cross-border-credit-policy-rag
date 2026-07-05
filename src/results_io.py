"""
JSON results output — persist experiment results and best config.

Saves all ExperimentResult objects plus the best config to a single
JSON file. Best config selection: highest MRR, tiebreak by fastest
avg retrieval time.

Citations:
  - _instructions.md L628 (JSON results output)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.models import ExperimentResult


def select_best_config(results: list[ExperimentResult]) -> ExperimentResult:
    """Select the best experiment result.

    Primary sort: highest MRR.
    Tiebreak: fastest avg_retrieval_time.

    Args:
        results: List of ExperimentResult objects.

    Returns:
        The best ExperimentResult.
    """
    return max(
        results,
        key=lambda r: (r.metrics.mrr, -r.metrics.avg_retrieval_time),
    )


def select_fastest_above_bar(
    results: list[ExperimentResult],
    min_mrr: float = 0.85,
    min_recall_at_5: float = 0.90,
) -> ExperimentResult | None:
    """Select the fastest config that clears the quality bar.

    Filters to configs with MRR >= min_mrr and Recall@5 >= min_recall_at_5,
    then picks the fastest by avg_retrieval_time.

    Args:
        results: List of ExperimentResult objects.
        min_mrr: Minimum MRR threshold.
        min_recall_at_5: Minimum Recall@5 threshold.

    Returns:
        The fastest qualifying ExperimentResult, or None if none qualify.
    """
    qualifying = [
        r for r in results
        if r.metrics.mrr >= min_mrr
        and r.metrics.recall_at_k.get(5, 0.0) >= min_recall_at_5
    ]
    if not qualifying:
        return None
    return min(qualifying, key=lambda r: r.metrics.avg_retrieval_time)


def save_results(results: list[ExperimentResult], path: str) -> None:
    """Save all experiment results and best config to JSON.

    Output structure:
    {
        "metadata": { "timestamp": ..., "total_experiments": ... },
        "best_config": { ... },
        "results": [ ... ]
    }

    Args:
        results: List of ExperimentResult objects.
        path: Output JSON file path.
    """
    best_accuracy = select_best_config(results)
    best_speed = select_fastest_above_bar(results)

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_experiments": len(results),
        },
        "best_by_accuracy": best_accuracy.model_dump(),
        "best_by_speed": best_speed.model_dump() if best_speed else None,
        "results": [r.model_dump() for r in results],
    }

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_results(path: str) -> list[ExperimentResult]:
    """Load experiment results from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of validated ExperimentResult objects.
    """
    with open(path) as f:
        data = json.load(f)

    return [ExperimentResult(**r) for r in data["results"]]

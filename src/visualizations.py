"""
Visualizations — 6 charts for RAG pipeline evaluation results.

All functions take a list of ExperimentResult and return a matplotlib Figure.
Charts are saved to disk by the pipeline runner; these functions only create them.

Citations:
  - _instructions.md L631 (6 visualization charts)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models import ExperimentResult


def _results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convert ExperimentResult list to a DataFrame for plotting."""
    rows = []
    for r in results:
        cfg = r.config
        m = r.metrics
        label = (
            f"{cfg.chunking.parser}_{cfg.chunking.chunker}_"
            f"{cfg.embedding_model.replace('text-embedding-3-', '')}_"
            f"{cfg.retrieval_method}"
        )
        if r.use_reranking:
            label += "_reranked"
        rows.append({
            "label": label,
            "parser": cfg.chunking.parser,
            "chunker": cfg.chunking.chunker,
            "embedding": cfg.embedding_model.replace("text-embedding-3-", ""),
            "retrieval": cfg.retrieval_method,
            "reranked": r.use_reranking,
            "mrr": m.mrr,
            "map": m.map_score,
            "recall@5": m.recall_at_k.get(5, 0.0),
            "precision@5": m.precision_at_k.get(5, 0.0),
            "ndcg@5": m.ndcg_at_k.get(5, 0.0),
            "avg_time": m.avg_retrieval_time,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. MRR bar chart
# ---------------------------------------------------------------------------

def plot_mrr_bar(results: list[ExperimentResult]) -> plt.Figure:
    """Bar chart of MRR for each config, grouped by retrieval method."""
    df = _results_to_dataframe(results)
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = df["retrieval"].unique()
    x = np.arange(len(df))
    colors = {"bm25": "#2196F3", "vector": "#4CAF50", "hybrid": "#FF9800"}

    bars = ax.bar(
        x, df["mrr"],
        color=[colors.get(m, "#999") for m in df["retrieval"]],
    )

    ax.set_xlabel("Configuration")
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors.get(m, "#999"), label=m) for m in methods]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Recall vs Precision scatter
# ---------------------------------------------------------------------------

def plot_recall_vs_precision(results: list[ExperimentResult]) -> plt.Figure:
    """Scatter plot of Recall@5 vs Precision@5."""
    df = _results_to_dataframe(results)
    fig, ax = plt.subplots(figsize=(8, 8))

    markers = {"bm25": "o", "vector": "s", "hybrid": "D"}
    for method in df["retrieval"].unique():
        subset = df[df["retrieval"] == method]
        ax.scatter(
            subset["recall@5"], subset["precision@5"],
            label=method, marker=markers.get(method, "o"), s=100, alpha=0.8,
        )

    ax.set_xlabel("Recall@5")
    ax.set_ylabel("Precision@5")
    ax.set_title("Recall vs Precision (K=5)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Metrics heatmap
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(results: list[ExperimentResult]) -> plt.Figure:
    """Heatmap of key metrics across all configs."""
    df = _results_to_dataframe(results)
    metrics_cols = ["mrr", "map", "recall@5", "precision@5", "ndcg@5"]
    heatmap_data = df.set_index("label")[metrics_cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd",
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
    )
    ax.set_title("Metrics Heatmap")
    ax.set_ylabel("")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Retrieval method comparison
# ---------------------------------------------------------------------------

def plot_retrieval_comparison(results: list[ExperimentResult]) -> plt.Figure:
    """Grouped bar chart comparing retrieval methods across key metrics."""
    df = _results_to_dataframe(results)
    metrics = ["mrr", "map", "recall@5", "ndcg@5"]

    method_means = df.groupby("retrieval")[metrics].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    method_means.plot(kind="bar", ax=ax, width=0.8)

    ax.set_xlabel("Retrieval Method")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Method Comparison (averaged across configs)")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Metric")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(results: list[ExperimentResult]) -> plt.Figure:
    """Correlation matrix of all metrics."""
    df = _results_to_dataframe(results)
    metrics_cols = ["mrr", "map", "recall@5", "precision@5", "ndcg@5", "avg_time"]
    corr = df[metrics_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, vmin=-1, vmax=1, center=0, linewidths=0.5,
        square=True,
    )
    ax.set_title("Metric Correlation Matrix")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Time vs quality (the dual-lens chart)
# ---------------------------------------------------------------------------

def plot_time_vs_quality(
    results: list[ExperimentResult],
    quality_bar: float = 0.85,
) -> plt.Figure:
    """Scatter plot of avg retrieval time vs MRR — the speed/quality tradeoff.

    Args:
        results: List of ExperimentResult objects.
        quality_bar: MRR threshold for the quality bar reference line.
            Configurable — calibrate after seeing actual results.
    """
    df = _results_to_dataframe(results)
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {"bm25": "o", "vector": "s", "hybrid": "D"}
    for method in df["retrieval"].unique():
        subset = df[df["retrieval"] == method]
        ax.scatter(
            subset["avg_time"] * 1000,  # convert to ms
            subset["mrr"],
            label=method, marker=markers.get(method, "o"), s=120, alpha=0.8,
        )

    # Quality bar reference line
    ax.axhline(y=quality_bar, color="gray", linestyle="--", alpha=0.5,
               label=f"Quality bar (MRR ≥ {quality_bar})")

    ax.set_xlabel("Avg Retrieval Time (ms)")
    ax.set_ylabel("MRR")
    ax.set_title("Speed vs Quality — Production Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

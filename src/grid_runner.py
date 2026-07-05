"""
Phase 2 grid search runner — iterate configs × models × methods.

Orchestrates embedding, retrieval, and metric computation for each
experiment in the grid. QA datasets are loaded from disk (pre-generated).
Embeddings are cached per config × model to avoid redundant API calls.

Citations:
  - _instructions.md L625 (grid search runner)
  - _instructions.md L608 (2 embedding models)
  - _instructions.md L613-615 (3 retrieval methods)
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

import numpy as np

from src.models import (
    Chunk, ChunkingConfig, ExperimentConfig,
    ExperimentResult, MetricsResult, QAExample,
)
from src.embedding import embed_texts
from src.vector_store import build_index, query_index
from src.bm25_retrieval import build_bm25_index, query_bm25
from src.hybrid_retrieval import normalize_scores, combine_results
from src.metadata_filter import MetadataFilter, metadata_by_chunk_id
from src.metrics import (
    recall_at_k, precision_at_k, mrr,
    mean_average_precision, ndcg_at_k, average_retrieval_time,
)

# K values to compute metrics at
K_VALUES = [1, 3, 5, 10]


def _retrieve_bm25(
    chunks: list[Chunk],
    query: str,
    k: int,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[str, float]]:
    """Run BM25 retrieval for a single query."""
    texts = [c.text for c in chunks]
    chunk_ids = [c.id for c in chunks]
    index, id_map = build_bm25_index(texts, chunk_ids)
    return query_bm25(
        index,
        id_map,
        query,
        k=k,
        metadata_by_id=metadata_by_chunk_id(chunks),
        metadata_filter=metadata_filter,
    )


def _retrieve_vector(
    faiss_index,
    id_map: dict[int, str],
    query: str,
    model: str,
    k: int,
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[str, float]]:
    """Run vector retrieval for a single query."""
    query_embedding = embed_texts([query], model=model)
    return query_index(
        faiss_index,
        id_map,
        query_embedding,
        k=k,
        metadata_by_id=metadata_by_id,
        metadata_filter=metadata_filter,
    )


def _retrieve_hybrid(
    chunks: list[Chunk],
    faiss_index,
    id_map: dict[int, str],
    query: str,
    model: str,
    k: int,
    alpha: float = 0.5,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[str, float]]:
    """Run hybrid retrieval: BM25 + vector, normalized and combined."""
    metadata_by_id = metadata_by_chunk_id(chunks)
    bm25_results = _retrieve_bm25(
        chunks,
        query,
        k=k,
        metadata_filter=metadata_filter,
    )
    vector_results = _retrieve_vector(
        faiss_index,
        id_map,
        query,
        model,
        k=k,
        metadata_by_id=metadata_by_id,
        metadata_filter=metadata_filter,
    )

    norm_bm25 = normalize_scores(bm25_results, higher_is_better=True)
    norm_vector = normalize_scores(vector_results, higher_is_better=False)

    combined = combine_results(norm_vector, norm_bm25, alpha=alpha)
    return combined[:k]


def run_single_experiment(
    chunks: list[Chunk],
    qa_examples: list[QAExample],
    config: ChunkingConfig,
    embedding_model: str,
    retrieval_method: str,
    k: int = 10,
    embeddings: np.ndarray | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> ExperimentResult:
    """Run one experiment: retrieve for each QA query, compute metrics.

    Args:
        chunks: Chunks for this chunking config.
        qa_examples: Pre-generated QA pairs with ground truth chunk IDs.
        config: The chunking config being evaluated.
        embedding_model: Embedding model name.
        retrieval_method: One of 'bm25', 'vector', 'hybrid'.
        k: Max results to retrieve per query.
        embeddings: Pre-computed chunk embeddings (required for vector/hybrid).
        metadata_filter: Optional hard filter over chunk metadata.

    Returns:
        ExperimentResult with all metrics populated.
    """
    # Build indexes as needed
    faiss_index = None
    id_map = None
    metadata_by_id = metadata_by_chunk_id(chunks)
    if retrieval_method in ("vector", "hybrid"):
        if embeddings is None:
            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts, model=embedding_model,
                                     config_id=config.config_id)
        chunk_ids = [c.id for c in chunks]
        faiss_index, id_map = build_index(embeddings, chunk_ids)

    # Run retrieval for each query, collect results and timing
    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    durations: list[float] = []

    for qa in qa_examples:
        relevant = set(qa.relevant_chunk_ids)

        start = time.perf_counter()

        if retrieval_method == "bm25":
            results = _retrieve_bm25(
                chunks,
                qa.question,
                k=k,
                metadata_filter=metadata_filter,
            )
        elif retrieval_method == "vector":
            results = _retrieve_vector(faiss_index, id_map, qa.question,
                                       embedding_model, k=k,
                                       metadata_by_id=metadata_by_id,
                                       metadata_filter=metadata_filter)
        elif retrieval_method == "hybrid":
            results = _retrieve_hybrid(chunks, faiss_index, id_map,
                                       qa.question, embedding_model, k=k,
                                       metadata_filter=metadata_filter)
        else:
            raise ValueError(f"Unknown retrieval method: {retrieval_method}")

        elapsed = time.perf_counter() - start
        durations.append(elapsed)

        retrieved_ids = [chunk_id for chunk_id, _ in results]
        all_retrieved.append(retrieved_ids)
        all_relevant.append(relevant)

    # Compute metrics across all queries
    max_k = max(K_VALUES)
    recall_scores = {kv: 0.0 for kv in K_VALUES}
    precision_scores = {kv: 0.0 for kv in K_VALUES}
    ndcg_scores = {kv: 0.0 for kv in K_VALUES}
    mrr_total = 0.0
    map_total = 0.0

    n = len(qa_examples)
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        for kv in K_VALUES:
            recall_scores[kv] += recall_at_k(retrieved, relevant, k=kv)
            precision_scores[kv] += precision_at_k(retrieved, relevant, k=kv)
            ndcg_scores[kv] += ndcg_at_k(retrieved, relevant, k=kv)
        mrr_total += mrr(retrieved, relevant)
        map_total += mean_average_precision(retrieved, relevant)

    # Average across queries
    metrics = MetricsResult(
        recall_at_k={kv: recall_scores[kv] / n for kv in K_VALUES},
        precision_at_k={kv: precision_scores[kv] / n for kv in K_VALUES},
        mrr=mrr_total / n,
        map_score=map_total / n,
        ndcg_at_k={kv: ndcg_scores[kv] / n for kv in K_VALUES},
        total_queries=n,
        avg_retrieval_time=average_retrieval_time(durations),
    )

    experiment_config = ExperimentConfig(
        chunking=config,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
    )

    return ExperimentResult(config=experiment_config, metrics=metrics)


def run_phase2_grid(
    configs_with_chunks: dict[str, tuple[ChunkingConfig, list[Chunk]]],
    qa_by_config: dict[str, list[QAExample]],
    embedding_models: list[str] | None = None,
    retrieval_methods: list[str] | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[ExperimentResult]:
    """Run the full Phase 2 grid search.

    Args:
        configs_with_chunks: Map of config_id → (ChunkingConfig, chunks).
        qa_by_config: Map of config_id → list of QAExample.
        embedding_models: Models to test. Defaults to both OpenAI models.
        retrieval_methods: Methods to test. Defaults to all three.
        metadata_filter: Optional hard filter used by all retrieval methods.

    Returns:
        List of ExperimentResult for every grid cell.
    """
    if embedding_models is None:
        embedding_models = ["text-embedding-3-small", "text-embedding-3-large"]
    if retrieval_methods is None:
        retrieval_methods = ["bm25", "vector", "hybrid"]

    results: list[ExperimentResult] = []

    for config_id, (config, chunks) in configs_with_chunks.items():
        qa_examples = qa_by_config[config_id]

        for model in embedding_models:
            # Embed once per config × model (cached by embed_texts)
            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts, model=model,
                                     config_id=config.config_id)

            for method in retrieval_methods:
                result = run_single_experiment(
                    chunks=chunks,
                    qa_examples=qa_examples,
                    config=config,
                    embedding_model=model,
                    retrieval_method=method,
                    embeddings=embeddings,
                    metadata_filter=metadata_filter,
                )
                results.append(result)

    return results

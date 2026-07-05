"""
pytest configuration and fixtures for rag-pipeline tests.

Shared fixtures provide valid model instances that individual tests can
override to check specific constraints.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import (  # noqa: E402
    Chunk,
    ChunkMetadata,
    ChunkingConfig,
    QAExample,
    ExperimentConfig,
    ExperimentResult,
    MetricsResult,
)


# ---------------------------------------------------------------------------
# Valid data payloads — used as baselines for happy-path and rejection tests
# ---------------------------------------------------------------------------

VALID_CHUNK_METADATA: dict = {
    "chunk_size": 256,
    "overlap": 50,
    "parser": "pdfplumber",
}

VALID_CHUNK_DATA: dict = {
    "text": "The annual report shows revenue growth of 15% year-over-year.",
    "page_number": 5,
    "chunk_index": 0,
    "start_char": 1200,
    "end_char": 1456,
    "method": "fixed_size",
    "metadata": VALID_CHUNK_METADATA,
}

VALID_CHUNKING_CONFIG: dict = {
    "parser": "pdfplumber",
    "chunker": "fixed_size",
    "chunk_size": 256,
    "overlap": 50,
}

VALID_QA_EXAMPLE: dict = {
    "question": "What was the year-over-year revenue growth?",
    "relevant_chunk_ids": ["chunk-uuid-123"],
    "source_page": 5,
    "chunk_method": "fixed_size",
}

VALID_EXPERIMENT_CONFIG: dict = {
    "chunking": VALID_CHUNKING_CONFIG,
    "embedding_model": "text-embedding-3-small",
    "retrieval_method": "vector",
}

VALID_METRICS_RESULT: dict = {
    "recall_at_k": {1: 0.85, 3: 0.95, 5: 1.0},
    "precision_at_k": {1: 0.85, 3: 0.317, 5: 0.20},
    "mrr": 0.963,
    "map_score": 0.963,
    "ndcg_at_k": {1: 0.85, 3: 0.933, 5: 0.975},
    "total_queries": 20,
    "avg_retrieval_time": 0.045,
}

VALID_EXPERIMENT_RESULT: dict = {
    "config": VALID_EXPERIMENT_CONFIG,
    "metrics": VALID_METRICS_RESULT,
    "use_reranking": False,
}


# ---------------------------------------------------------------------------
# Fixtures — return mutable copies so tests can modify without side effects
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_chunk_data() -> dict:
    """Return a mutable copy of valid Chunk payload."""
    import copy
    return copy.deepcopy(VALID_CHUNK_DATA)


@pytest.fixture
def valid_chunking_config_data() -> dict:
    return {**VALID_CHUNKING_CONFIG}


@pytest.fixture
def valid_qa_example_data() -> dict:
    return {**VALID_QA_EXAMPLE, "relevant_chunk_ids": list(VALID_QA_EXAMPLE["relevant_chunk_ids"])}


@pytest.fixture
def valid_experiment_config_data() -> dict:
    import copy
    return copy.deepcopy(VALID_EXPERIMENT_CONFIG)


@pytest.fixture
def valid_experiment_result_data() -> dict:
    import copy
    return copy.deepcopy(VALID_EXPERIMENT_RESULT)

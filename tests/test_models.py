"""Tests for Pydantic models — Chunk, ChunkingConfig, QAExample, ExperimentConfig, ExperimentResult.

TDD: these tests define the contracts before the models are implemented.

Citations:
  - _instructions.md L170-188  (Chunk JSON schema)
  - _instructions.md L192-205  (QAExample JSON schema)
  - _instructions.md L208-226  (ExperimentResult / MetricsResult JSON schema)
  - _instructions.md L229-237  (ChunkingConfig JSON schema)
  - _plan.md rows 6, 47       (Pydantic models, config management)
"""

import pytest
from pydantic import ValidationError

from src.models import (
    Chunk,
    ChunkMetadata,
    ChunkingConfig,
    QAExample,
    ExperimentConfig,
    ExperimentResult,
    MetricsResult,
)


# ============================================================================
# Chunk
# ============================================================================

class TestChunk:
    """Verify Chunk enforces the schema from _instructions.md L170-188."""

    def test_valid_chunk(self, valid_chunk_data):
        """Happy path: valid data produces a valid Chunk with auto-generated ID."""
        chunk = Chunk(**valid_chunk_data)
        assert chunk.id is not None  # UUID auto-generated
        assert chunk.text == valid_chunk_data["text"]
        assert chunk.page_number == 5
        assert chunk.method == "fixed_size"
        assert chunk.metadata.chunk_size == 256

    def test_id_auto_generated_when_omitted(self, valid_chunk_data):
        """ID should be auto-populated as a UUID string if not provided."""
        chunk = Chunk(**valid_chunk_data)
        assert isinstance(chunk.id, str)
        assert len(chunk.id) > 0

    def test_explicit_id_preserved(self, valid_chunk_data):
        """If caller provides an ID, it should be kept."""
        valid_chunk_data["id"] = "my-custom-id"
        chunk = Chunk(**valid_chunk_data)
        assert chunk.id == "my-custom-id"

    def test_text_required(self, valid_chunk_data):
        del valid_chunk_data["text"]
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)

    def test_text_rejects_empty(self, valid_chunk_data):
        valid_chunk_data["text"] = ""
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)

    def test_page_number_required(self, valid_chunk_data):
        del valid_chunk_data["page_number"]
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)

    def test_page_number_rejects_negative(self, valid_chunk_data):
        valid_chunk_data["page_number"] = -1
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)

    def test_method_required(self, valid_chunk_data):
        del valid_chunk_data["method"]
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)

    @pytest.mark.parametrize("method", ["fixed_size", "sentence", "semantic"])
    def test_valid_methods(self, valid_chunk_data, method):
        valid_chunk_data["method"] = method
        chunk = Chunk(**valid_chunk_data)
        assert chunk.method == method

    def test_metadata_required(self, valid_chunk_data):
        del valid_chunk_data["metadata"]
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)

    def test_start_char_before_end_char(self, valid_chunk_data):
        """start_char must be < end_char."""
        valid_chunk_data["start_char"] = 1500
        valid_chunk_data["end_char"] = 1200
        with pytest.raises(ValidationError):
            Chunk(**valid_chunk_data)


# ============================================================================
# ChunkMetadata
# ============================================================================

class TestChunkMetadata:
    """Verify ChunkMetadata nested model."""

    def test_valid_metadata(self):
        meta = ChunkMetadata(chunk_size=256, overlap=50, parser="pdfplumber")
        assert meta.chunk_size == 256
        assert meta.overlap == 50

    def test_compliance_metadata_fields_are_optional(self):
        meta = ChunkMetadata(
            chunk_size=500,
            overlap=0,
            parser="layout_markdown",
            country="Mexico",
            institution="Buro de Credito",
            doc_type="Compliance_Law",
            effective_date="2026-03-01",
            legal_hierarchy="Chapter 3 > Article 12 > Section 3",
        )
        assert meta.country == "Mexico"
        assert meta.vector_metadata()["legal_hierarchy"] == (
            "Chapter 3 > Article 12 > Section 3"
        )

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(chunk_size=0, overlap=50, parser="pdfplumber")

    def test_overlap_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(chunk_size=256, overlap=-1, parser="pdfplumber")

    def test_parser_required(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(chunk_size=256, overlap=50)


# ============================================================================
# ChunkingConfig
# ============================================================================

class TestChunkingConfig:
    """Verify ChunkingConfig from _instructions.md L229-237."""

    def test_valid_config(self, valid_chunking_config_data):
        config = ChunkingConfig(**valid_chunking_config_data)
        assert config.parser == "pdfplumber"
        assert config.chunker == "fixed_size"
        assert config.chunk_size == 256
        assert config.overlap == 50

    def test_overlap_must_be_less_than_chunk_size(self, valid_chunking_config_data):
        """Overlap >= chunk_size makes no sense — chunks would fully overlap."""
        valid_chunking_config_data["overlap"] = 256
        with pytest.raises(ValidationError):
            ChunkingConfig(**valid_chunking_config_data)

    def test_overlap_can_be_zero(self, valid_chunking_config_data):
        valid_chunking_config_data["overlap"] = 0
        config = ChunkingConfig(**valid_chunking_config_data)
        assert config.overlap == 0

    def test_chunk_size_must_be_positive(self, valid_chunking_config_data):
        valid_chunking_config_data["chunk_size"] = 0
        with pytest.raises(ValidationError):
            ChunkingConfig(**valid_chunking_config_data)

    def test_config_id_deterministic(self, valid_chunking_config_data):
        """Same config values should produce the same config_id."""
        c1 = ChunkingConfig(**valid_chunking_config_data)
        c2 = ChunkingConfig(**valid_chunking_config_data)
        assert c1.config_id == c2.config_id

    def test_config_id_changes_with_params(self, valid_chunking_config_data):
        """Different params should produce different config_ids."""
        c1 = ChunkingConfig(**valid_chunking_config_data)
        valid_chunking_config_data["chunk_size"] = 512
        c2 = ChunkingConfig(**valid_chunking_config_data)
        assert c1.config_id != c2.config_id


# ============================================================================
# QAExample
# ============================================================================

class TestQAExample:
    """Verify QAExample from _instructions.md L192-205."""

    def test_valid_qa(self, valid_qa_example_data):
        qa = QAExample(**valid_qa_example_data)
        assert qa.question.startswith("What")
        assert len(qa.relevant_chunk_ids) >= 1

    def test_question_required(self, valid_qa_example_data):
        del valid_qa_example_data["question"]
        with pytest.raises(ValidationError):
            QAExample(**valid_qa_example_data)

    def test_question_rejects_empty(self, valid_qa_example_data):
        valid_qa_example_data["question"] = ""
        with pytest.raises(ValidationError):
            QAExample(**valid_qa_example_data)

    def test_chunk_ids_must_be_non_empty(self, valid_qa_example_data):
        valid_qa_example_data["relevant_chunk_ids"] = []
        with pytest.raises(ValidationError):
            QAExample(**valid_qa_example_data)

    def test_chunk_method_required(self, valid_qa_example_data):
        del valid_qa_example_data["chunk_method"]
        with pytest.raises(ValidationError):
            QAExample(**valid_qa_example_data)


# ============================================================================
# MetricsResult
# ============================================================================

class TestMetricsResult:
    """Verify MetricsResult from _instructions.md L208-226."""

    def test_valid_metrics(self):
        metrics = MetricsResult(
            recall_at_k={1: 0.85, 5: 1.0},
            precision_at_k={1: 0.85, 5: 0.20},
            mrr=0.963,
            map_score=0.963,
            ndcg_at_k={5: 0.975},
            total_queries=20,
            avg_retrieval_time=0.045,
        )
        assert metrics.mrr == 0.963
        assert metrics.total_queries == 20

    def test_mrr_rejects_above_one(self):
        with pytest.raises(ValidationError):
            MetricsResult(
                recall_at_k={5: 1.0}, precision_at_k={5: 0.2},
                mrr=1.5, map_score=0.9, ndcg_at_k={5: 0.9},
                total_queries=20, avg_retrieval_time=0.1,
            )

    def test_mrr_rejects_negative(self):
        with pytest.raises(ValidationError):
            MetricsResult(
                recall_at_k={5: 1.0}, precision_at_k={5: 0.2},
                mrr=-0.1, map_score=0.9, ndcg_at_k={5: 0.9},
                total_queries=20, avg_retrieval_time=0.1,
            )

    def test_total_queries_must_be_positive(self):
        with pytest.raises(ValidationError):
            MetricsResult(
                recall_at_k={5: 1.0}, precision_at_k={5: 0.2},
                mrr=0.9, map_score=0.9, ndcg_at_k={5: 0.9},
                total_queries=0, avg_retrieval_time=0.1,
            )


# ============================================================================
# ExperimentConfig
# ============================================================================

class TestExperimentConfig:
    """Verify ExperimentConfig — a full grid cell."""

    def test_valid_config(self, valid_experiment_config_data):
        config = ExperimentConfig(**valid_experiment_config_data)
        assert config.embedding_model == "text-embedding-3-small"
        assert config.retrieval_method == "vector"
        assert config.chunking.chunker == "fixed_size"

    @pytest.mark.parametrize("method", ["bm25", "vector", "hybrid"])
    def test_valid_retrieval_methods(self, valid_experiment_config_data, method):
        valid_experiment_config_data["retrieval_method"] = method
        config = ExperimentConfig(**valid_experiment_config_data)
        assert config.retrieval_method == method

    def test_experiment_id_deterministic(self, valid_experiment_config_data):
        """Same config should produce the same experiment_id."""
        c1 = ExperimentConfig(**valid_experiment_config_data)
        c2 = ExperimentConfig(**valid_experiment_config_data)
        assert c1.experiment_id == c2.experiment_id

    def test_experiment_id_changes_with_retrieval(self, valid_experiment_config_data):
        c1 = ExperimentConfig(**valid_experiment_config_data)
        valid_experiment_config_data["retrieval_method"] = "bm25"
        c2 = ExperimentConfig(**valid_experiment_config_data)
        assert c1.experiment_id != c2.experiment_id


# ============================================================================
# ExperimentResult
# ============================================================================

class TestExperimentResult:
    """Verify ExperimentResult — full result for one grid cell."""

    def test_valid_result(self, valid_experiment_result_data):
        result = ExperimentResult(**valid_experiment_result_data)
        assert result.use_reranking is False
        assert result.metrics.mrr == 0.963
        assert result.config.embedding_model == "text-embedding-3-small"

    def test_use_reranking_defaults_false(self, valid_experiment_result_data):
        del valid_experiment_result_data["use_reranking"]
        result = ExperimentResult(**valid_experiment_result_data)
        assert result.use_reranking is False

    def test_metrics_required(self, valid_experiment_result_data):
        del valid_experiment_result_data["metrics"]
        with pytest.raises(ValidationError):
            ExperimentResult(**valid_experiment_result_data)

    def test_config_required(self, valid_experiment_result_data):
        del valid_experiment_result_data["config"]
        with pytest.raises(ValidationError):
            ExperimentResult(**valid_experiment_result_data)

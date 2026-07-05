"""
Pydantic models for the RAG evaluation pipeline.

Defines the data contracts for chunks, configurations, QA examples,
and experiment results. All pipeline modules import from here.

Schemas follow _instructions.md L170-237 (JSON examples) with nested
Pydantic models for stronger validation.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Chunk models (_instructions.md L170-188)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Parser and compliance metadata attached to each chunk.

    The first three fields preserve the original evaluation contract.
    The optional compliance fields are filterable metadata for regulated
    credit-policy retrieval.
    """

    model_config = ConfigDict(extra="allow")

    chunk_size: int = Field(..., gt=0, description="Target chunk size in characters")
    overlap: int = Field(..., ge=0, description="Overlap between consecutive chunks")
    parser: str = Field(..., description="PDF parser used (e.g. 'pdfplumber')")
    country: str | None = Field(
        default=None,
        description="Jurisdiction/country, e.g. India, Mexico, Uganda",
    )
    institution: str | None = Field(
        default=None,
        description="Issuing institution or credit bureau, e.g. RBI or Buro de Credito",
    )
    doc_type: str | None = Field(
        default=None,
        description="Policy document type, e.g. Compliance_Law or Credit_Reporting_Rule",
    )
    effective_date: str | None = Field(
        default=None,
        description="Publication or effective date in ISO format when available",
    )
    legal_hierarchy: str | None = Field(
        default=None,
        description="Tree location such as Chapter 3 > Article 12 > Section 3",
    )

    def vector_metadata(self) -> dict:
        """Return metadata in a vector-store friendly dict shape."""
        return self.model_dump(exclude_none=True)


class PolicyDocumentMetadata(BaseModel):
    """Document-level metadata injected into every compliance-policy chunk."""

    model_config = ConfigDict(extra="allow")

    country: str | None = Field(default=None, description="Jurisdiction/country")
    institution: str | None = Field(default=None, description="Issuing institution")
    doc_type: str | None = Field(default=None, description="Policy document type")
    effective_date: str | None = Field(default=None, description="Effective/publication date")

    def chunk_metadata_fields(self) -> dict:
        """Return non-empty fields suitable for ChunkMetadata construction."""
        return self.model_dump(exclude_none=True)


class Chunk(BaseModel):
    """A piece of text extracted and split from the PDF."""

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique chunk identifier (auto-generated UUID if omitted)",
    )
    text: str = Field(..., min_length=1, description="The extracted chunk content")
    page_number: int = Field(..., ge=0, description="Source page in the PDF")
    chunk_index: int = Field(..., ge=0, description="Position in the chunk sequence")
    start_char: int = Field(..., ge=0, description="Start character offset in page text")
    end_char: int = Field(..., gt=0, description="End character offset in page text")
    method: str = Field(..., description="Chunking method: fixed_size, sentence, or semantic")
    metadata: ChunkMetadata

    @model_validator(mode="after")
    def start_before_end(self) -> Chunk:
        if self.start_char >= self.end_char:
            raise ValueError(f"start_char ({self.start_char}) must be < end_char ({self.end_char})")
        return self


# ---------------------------------------------------------------------------
# Configuration models (_instructions.md L229-237)
# ---------------------------------------------------------------------------

class ChunkingConfig(BaseModel):
    """Defines one chunking configuration for the grid search."""

    parser: str = Field(..., description="PDF parser to use")
    chunker: str = Field(..., description="Chunking method: fixed_size, sentence, semantic")
    chunk_size: int = Field(..., gt=0, description="Target chunk size in characters")
    overlap: int = Field(..., ge=0, description="Overlap between chunks")

    @model_validator(mode="after")
    def overlap_less_than_chunk_size(self) -> ChunkingConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self

    @property
    def config_id(self) -> str:
        """Deterministic identifier for this config combination."""
        return f"{self.parser}_{self.chunker}_size{self.chunk_size}_overlap{self.overlap}"


class ExperimentConfig(BaseModel):
    """A full grid cell: chunking + embedding + retrieval."""

    chunking: ChunkingConfig
    embedding_model: str = Field(..., description="e.g. text-embedding-3-small")
    retrieval_method: str = Field(..., description="bm25, vector, or hybrid")

    @property
    def experiment_id(self) -> str:
        """Deterministic identifier for this experiment."""
        return f"{self.embedding_model}_{self.chunking.config_id}_{self.retrieval_method}"


# ---------------------------------------------------------------------------
# QA models (_instructions.md L192-205)
# ---------------------------------------------------------------------------

class QAExample(BaseModel):
    """A synthetic question mapped to its source chunk(s) for evaluation."""

    question: str = Field(..., min_length=1, description="The evaluation question")
    relevant_chunk_ids: list[str] = Field(
        ..., min_length=1, description="Ground-truth chunk IDs this question was generated from"
    )
    source_page: int = Field(..., ge=0, description="Page the source chunk came from")
    chunk_method: str = Field(..., description="Chunking method that produced the source chunk")


# ---------------------------------------------------------------------------
# Results models (_instructions.md L208-226)
# ---------------------------------------------------------------------------

class MetricsResult(BaseModel):
    """IR metrics for one experiment run."""

    recall_at_k: dict[int, float] = Field(..., description="Recall at various K values")
    precision_at_k: dict[int, float] = Field(..., description="Precision at various K values")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    map_score: float = Field(..., ge=0.0, le=1.0, description="Mean Average Precision")
    ndcg_at_k: dict[int, float] = Field(..., description="NDCG at various K values")
    total_queries: int = Field(..., gt=0, description="Number of queries evaluated")
    avg_retrieval_time: float = Field(..., ge=0.0, description="Average retrieval time in seconds")


class ExperimentResult(BaseModel):
    """Full result for one grid search experiment."""

    config: ExperimentConfig
    metrics: MetricsResult
    use_reranking: bool = Field(default=False, description="Whether reranking was applied")

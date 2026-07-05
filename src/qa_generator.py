"""
Synthetic QA generator — create evaluation questions from chunk text.

For each chunk, asks an LLM to generate a question that the chunk answers.
The chunk ID becomes the ground truth for retrieval evaluation. Uses
Instructor + Pydantic for structured output, same pattern as the
synthetic-data-generator project.

Citations:
  - _instructions.md L622 (synthetic QA generation)
  - _instructions.md L623 (per-config QA, chunk ID validation)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from src.models import Chunk, QAExample
from src.retry import retry_with_backoff

# ---------------------------------------------------------------------------
# Lazy client initialization
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Return an Instructor-wrapped OpenAI client, initialising on first call.

    python-dotenv loads environment variables from a .env.local file into
    os.environ. This keeps sensitive data (like API keys) out of source code
    and version control.
    """
    global _client
    if _client is not None:
        return _client

    import instructor
    from openai import OpenAI

    for candidate in [Path(".env.local"), Path("../.env.local")]:
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate))
            break

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env.local file with your API key."
        )

    _client = instructor.from_openai(
        OpenAI(api_key=api_key, base_url=base_url)
    )
    return _client


# ---------------------------------------------------------------------------
# LLM call (seam for mocking in tests)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are given a passage from a corporate annual report.
Generate a specific, factual question that can be answered using ONLY the information in this passage.

The question should:
- Be answerable from the passage alone
- Ask about a specific fact, number, or detail
- Be the kind of question a financial analyst or investor might ask

Passage:
{chunk_text}

Respond with ONLY the question, nothing else."""


def _call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call the LLM and return a question string.

    This is the seam that tests mock — keeps LLM interaction
    isolated from the wiring logic.
    """
    client = _get_client()
    response = retry_with_backoff(
        client.chat.completions.create,
        kwargs={
            "model": model,
            "response_model": None,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 150,
        },
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# QA generation
# ---------------------------------------------------------------------------


def generate_qa_for_chunk(
    chunk: Chunk,
    model: str = "gpt-4o-mini",
) -> QAExample:
    """Generate a QA pair from a single chunk.

    Args:
        chunk: The source chunk to generate a question from.
        model: LLM model to use for generation.

    Returns:
        A QAExample with the generated question and the chunk's ID as ground truth.
    """
    prompt = _PROMPT_TEMPLATE.format(chunk_text=chunk.text)
    question = _call_llm(prompt, model=model)

    return QAExample(
        question=question,
        relevant_chunk_ids=[chunk.id],
        source_page=chunk.page_number,
        chunk_method=chunk.method,
    )


def generate_qa_dataset(
    chunks: list[Chunk],
    n_samples: int | None = None,
    model: str = "gpt-4o-mini",
) -> list[QAExample]:
    """Generate QA pairs from a list of chunks.

    Args:
        chunks: All chunks from a chunking config.
        n_samples: Number of chunks to sample. None = use all.
        model: LLM model to use for generation.

    Returns:
        List of QAExample objects (may be fewer than n_samples if errors occur).
    """
    # Sample chunks if requested
    if n_samples is not None:
        n_samples = min(n_samples, len(chunks))
        selected = random.sample(chunks, n_samples)
    else:
        selected = chunks

    results: list[QAExample] = []
    for chunk in selected:
        try:
            qa = generate_qa_for_chunk(chunk, model=model)
            results.append(qa)
        except Exception:
            # Skip failed generations, continue with remaining chunks
            continue

    return results


# ---------------------------------------------------------------------------
# Persistence (JSONL)
# ---------------------------------------------------------------------------

CHUNKS_DIR = "data/chunks"


def get_chunks_path(config_id: str, chunks_dir: str = CHUNKS_DIR) -> str:
    """Deterministic path for a config's persisted chunks.

    Convention: data/chunks/{config_id}.jsonl

    Args:
        config_id: Chunking config identifier (from ChunkingConfig.config_id).
        chunks_dir: Base directory for chunk files.

    Returns:
        Full path to the JSONL file.
    """
    return os.path.join(chunks_dir, f"{config_id}.jsonl")


def save_chunks(chunks: list[Chunk], path: str) -> None:
    """Save chunks to a JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


def load_chunks(path: str) -> list[Chunk]:
    """Load chunks from a JSONL file, preserving original IDs."""
    chunks: list[Chunk] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            chunks.append(Chunk(**data))
    return chunks


QA_DIR = "data/qa"


def get_qa_path(config_id: str, qa_dir: str = QA_DIR) -> str:
    """Deterministic path for a config's QA dataset.

    Convention: data/qa/{config_id}.jsonl

    Args:
        config_id: Chunking config identifier (from ChunkingConfig.config_id).
        qa_dir: Base directory for QA files.

    Returns:
        Full path to the JSONL file.
    """
    return os.path.join(qa_dir, f"{config_id}.jsonl")


def save_qa_dataset(qa_examples: list[QAExample], path: str) -> None:
    """Save QA examples to a JSONL file (one JSON object per line).

    Args:
        qa_examples: List of QAExample objects to persist.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for qa in qa_examples:
            f.write(json.dumps(qa.model_dump(), ensure_ascii=False) + "\n")


def load_qa_dataset(path: str) -> list[QAExample]:
    """Load QA examples from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of validated QAExample objects.
    """
    qa_examples: list[QAExample] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            qa_examples.append(QAExample(**data))
    return qa_examples

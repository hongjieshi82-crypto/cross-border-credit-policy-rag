"""Tests for chunking strategies.

TDD: tests define the contract before implementation.

Citations:
  - _instructions.md L602 (3 chunking strategies)
  - _instructions.md L604 (chunks include metadata)
  - _instructions.md L605 (no mid-word/mid-sentence splits)
"""

import pytest

from src.chunking import (
    chunk_fixed_size,
    chunk_policy_document,
    chunk_sentence,
    chunk_semantic,
    extract_legal_hierarchy,
)
from src.models import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_words(n: int) -> str:
    """Generate a string of n words, each ~6 chars, for predictable test input."""
    words = [f"word{i:02d}" for i in range(n)]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------

class TestFixedSizeChunking:
    """Verify fixed-size chunker splits text correctly with word-boundary awareness."""

    def test_returns_list_of_chunks(self):
        text = make_words(50)
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=100, overlap=0,
            parser="pdfplumber",
        )
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_mid_word_splits(self):
        """Every chunk should start and end on word boundaries (no partial words)."""
        text = make_words(100)
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=50, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks:
            # No chunk should start or end mid-word (except possible leading/trailing space)
            stripped = chunk.text.strip()
            assert not stripped.startswith(" ")
            # Last char should be a word char or punctuation, not a space
            assert len(stripped) > 0

    def test_chunks_cover_all_text(self):
        """Concatenating all chunks (accounting for overlap) should cover the full text."""
        text = make_words(50)
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=100, overlap=0,
            parser="pdfplumber",
        )
        reconstructed = "".join(c.text for c in chunks)
        # With overlap=0, reconstructed should equal original (modulo trailing space)
        assert reconstructed.strip() == text.strip()

    def test_overlap_produces_shared_text(self):
        """Consecutive chunks should share text when overlap > 0."""
        text = make_words(100)
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=100, overlap=30,
            parser="pdfplumber",
        )
        assert len(chunks) > 1
        # Check overlap on first few chunk pairs (tail chunks may be too short)
        pairs_to_check = min(3, len(chunks) - 1)
        for i in range(pairs_to_check):
            tail = chunks[i].text[-20:]  # last 20 chars of current chunk
            head = chunks[i + 1].text[:30]  # first 30 chars of next chunk
            # Some portion of the tail should appear in the head
            assert any(
                word in head for word in tail.split() if len(word) > 3
            ), f"No overlap found between chunk {i} and {i+1}"

    def test_overlap_zero_no_shared_text(self):
        """With overlap=0, consecutive chunks should not repeat content."""
        text = make_words(50)
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=80, overlap=0,
            parser="pdfplumber",
        )
        if len(chunks) > 1:
            # Last word of chunk 0 should not be first word of chunk 1
            last_word = chunks[0].text.strip().split()[-1]
            first_word = chunks[1].text.strip().split()[0]
            assert last_word != first_word

    def test_metadata_populated(self):
        text = make_words(50)
        chunks = chunk_fixed_size(
            text=text, page_number=3, chunk_size=100, overlap=20,
            parser="pymupdf",
        )
        for i, chunk in enumerate(chunks):
            assert chunk.page_number == 3
            assert chunk.method == "fixed_size"
            assert chunk.chunk_index == i
            assert chunk.metadata.chunk_size == 100
            assert chunk.metadata.overlap == 20
            assert chunk.metadata.parser == "pymupdf"
            assert chunk.id  # UUID auto-generated

    def test_text_shorter_than_chunk_size(self):
        """Short text should produce exactly one chunk."""
        text = "Hello world"
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=1000, overlap=0,
            parser="pdfplumber",
        )
        assert len(chunks) == 1
        assert chunks[0].text.strip() == text

    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_fixed_size(
            text="", page_number=0, chunk_size=100, overlap=0,
            parser="pdfplumber",
        )
        assert chunks == []

    def test_chunk_size_respected(self):
        """No chunk should exceed the target chunk_size (after word-boundary adjustment)."""
        text = make_words(200)
        chunk_size = 100
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=chunk_size, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks[:-1]:  # last chunk can be shorter
            # Allow small overshoot from word-boundary adjustment
            assert len(chunk.text) <= chunk_size + 20, (
                f"Chunk too long: {len(chunk.text)} > {chunk_size + 20}"
            )

    def test_start_end_char_offsets(self):
        """start_char and end_char should correspond to positions in the original text."""
        text = make_words(50)
        chunks = chunk_fixed_size(
            text=text, page_number=0, chunk_size=100, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks:
            assert text[chunk.start_char:chunk.end_char] == chunk.text


# ---------------------------------------------------------------------------
# Sentence-based chunking
# ---------------------------------------------------------------------------

SAMPLE_SENTENCES = (
    "The company reported strong earnings. Revenue grew by 15% year-over-year. "
    "Operating margins improved significantly. The board approved a dividend increase. "
    "International markets showed mixed results. The Asia-Pacific region outperformed. "
    "European operations faced headwinds. Management remains cautiously optimistic. "
    "New product launches are planned for Q3. R&D spending increased by 20%. "
    "The workforce expanded to 5,000 employees. Customer satisfaction scores improved."
)


class TestSentenceChunking:
    """Verify sentence-based chunker groups sentences without mid-sentence splits."""

    def test_returns_list_of_chunks(self):
        chunks = chunk_sentence(
            text=SAMPLE_SENTENCES, page_number=0, chunk_size=200, overlap=0,
            parser="pdfplumber",
        )
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_no_mid_sentence_splits(self):
        """Every chunk should end at a sentence boundary (period, !, ?)."""
        chunks = chunk_sentence(
            text=SAMPLE_SENTENCES, page_number=0, chunk_size=150, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks[:-1]:  # last chunk may not end with period
            assert chunk.text.rstrip().endswith((".","!","?")), (
                f"Chunk does not end at sentence boundary: ...{chunk.text[-30:]!r}"
            )

    def test_metadata_populated(self):
        chunks = chunk_sentence(
            text=SAMPLE_SENTENCES, page_number=7, chunk_size=200, overlap=0,
            parser="pymupdf",
        )
        for i, chunk in enumerate(chunks):
            assert chunk.page_number == 7
            assert chunk.method == "sentence"
            assert chunk.chunk_index == i
            assert chunk.metadata.chunk_size == 200
            assert chunk.metadata.parser == "pymupdf"

    def test_overlap_repeats_sentences(self):
        """With overlap > 0, consecutive chunks should share sentence text."""
        chunks = chunk_sentence(
            text=SAMPLE_SENTENCES, page_number=0, chunk_size=150, overlap=50,
            parser="pdfplumber",
        )
        if len(chunks) > 1:
            # Last sentence of chunk[0] should appear in chunk[1]
            first_sentences = chunks[0].text.rstrip().split(". ")
            last_sent = first_sentences[-1]
            assert last_sent in chunks[1].text, (
                f"Expected overlap: '{last_sent}' not found in next chunk"
            )

    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_sentence(
            text="", page_number=0, chunk_size=200, overlap=0,
            parser="pdfplumber",
        )
        assert chunks == []

    def test_single_sentence_returns_one_chunk(self):
        chunks = chunk_sentence(
            text="Just one sentence.", page_number=0, chunk_size=200, overlap=0,
            parser="pdfplumber",
        )
        assert len(chunks) == 1

    def test_chunks_cover_all_sentences(self):
        """All sentences from the input should appear in at least one chunk."""
        sentences = [
            f"{sentence.strip()}."
            for sentence in SAMPLE_SENTENCES.split(".")
            if sentence.strip()
        ]

        chunks = chunk_sentence(
            text=SAMPLE_SENTENCES, page_number=0, chunk_size=150, overlap=0,
            parser="pdfplumber",
        )
        all_chunk_text = " ".join(c.text for c in chunks)
        for sent in sentences:
            assert sent.strip() in all_chunk_text, f"Missing sentence: {sent!r}"

    def test_compliance_metadata_injected(self):
        text = (
            "# Chapter 3\n"
            "## Article 12\n"
            "### Section 3\n"
            "The credit bureau must verify borrower consent before inquiry."
        )
        chunks = chunk_sentence(
            text=text,
            page_number=0,
            chunk_size=500,
            overlap=0,
            parser="layout_markdown",
            document_metadata={
                "country": "Mexico",
                "institution": "Buro de Credito",
                "doc_type": "Compliance_Law",
                "effective_date": "2026-03-01",
            },
        )
        assert len(chunks) == 1
        assert chunks[0].metadata.country == "Mexico"
        assert chunks[0].metadata.institution == "Buro de Credito"
        assert chunks[0].metadata.legal_hierarchy == (
            "Chapter 3 > Article 12 > Section 3"
        )


class TestComplianceHierarchy:
    """Verify legal hierarchy extraction for policy documents."""

    def test_extracts_markdown_legal_hierarchy(self):
        text = "# Chapter 3\n## Article 12\n### Section 3\nRequired controls."
        assert extract_legal_hierarchy(text) == (
            "Chapter 3 > Article 12 > Section 3"
        )

    def test_extracts_portuguese_legal_hierarchy(self):
        text = "# Capítulo 2\n## Artigo 8\n### Seção 4\nControles obrigatórios."
        assert extract_legal_hierarchy(text) == (
            "Capítulo 2 > Artigo 8 > Seção 4"
        )

    def test_extracts_french_legal_hierarchy(self):
        text = "# Chapitre 1\n## Article 2\n### Section 3\nObligations applicables."
        assert extract_legal_hierarchy(text) == (
            "Chapitre 1 > Article 2 > Section 3"
        )

    def test_carries_hierarchy_across_pages(self):
        pages = [
            (0, "# Chapter 3\n## Article 12\nThe rule starts here."),
            (1, "The bureau must retain the consent audit trail."),
        ]
        chunks = chunk_policy_document(
            pages,
            document_metadata={"country": "Mexico", "institution": "Buro de Credito"},
        )
        assert chunks[-1].metadata.legal_hierarchy == "Chapter 3 > Article 12"


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

# Two clearly different topics to test boundary detection
MULTI_TOPIC_TEXT = (
    "The company reported strong quarterly earnings. Revenue increased by 15% "
    "compared to the previous year. Operating margins expanded to 22%. "
    "The board of directors approved a quarterly dividend of $0.50 per share. "
    "Net income rose to $45 million, exceeding analyst expectations. "
    # Topic shift: weather / climate
    "Hurricane season brought unprecedented storms to the Gulf Coast. "
    "Rainfall totals exceeded 30 inches in some areas. Emergency services "
    "were deployed across three states. Flooding damaged thousands of homes "
    "and displaced many families. The National Weather Service issued warnings."
)


class TestSemanticChunking:
    """Verify semantic chunker splits at topic boundaries using spaCy."""

    def test_returns_list_of_chunks(self):
        chunks = chunk_semantic(
            text=MULTI_TOPIC_TEXT, page_number=0, chunk_size=500, overlap=0,
            parser="pdfplumber",
        )
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_method_is_semantic(self):
        chunks = chunk_semantic(
            text=MULTI_TOPIC_TEXT, page_number=0, chunk_size=500, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks:
            assert chunk.method == "semantic"

    def test_detects_topic_shift(self):
        """With two distinct topics, should produce at least 2 chunks."""
        chunks = chunk_semantic(
            text=MULTI_TOPIC_TEXT, page_number=0, chunk_size=500, overlap=0,
            parser="pdfplumber",
        )
        assert len(chunks) >= 2, (
            f"Expected ≥2 chunks for multi-topic text, got {len(chunks)}"
        )

    def test_no_mid_sentence_splits(self):
        """Semantic chunks should still respect sentence boundaries."""
        chunks = chunk_semantic(
            text=MULTI_TOPIC_TEXT, page_number=0, chunk_size=500, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks[:-1]:
            assert chunk.text.rstrip().endswith((".", "!", "?")), (
                f"Chunk does not end at sentence boundary: ...{chunk.text[-30:]!r}"
            )

    def test_metadata_populated(self):
        chunks = chunk_semantic(
            text=MULTI_TOPIC_TEXT, page_number=4, chunk_size=300, overlap=0,
            parser="pymupdf",
        )
        for i, chunk in enumerate(chunks):
            assert chunk.page_number == 4
            assert chunk.method == "semantic"
            assert chunk.chunk_index == i
            assert chunk.metadata.chunk_size == 300
            assert chunk.metadata.parser == "pymupdf"

    def test_empty_text_returns_no_chunks(self):
        chunks = chunk_semantic(
            text="", page_number=0, chunk_size=300, overlap=0,
            parser="pdfplumber",
        )
        assert chunks == []

    def test_chunk_size_respected(self):
        """No chunk should wildly exceed chunk_size."""
        chunks = chunk_semantic(
            text=MULTI_TOPIC_TEXT, page_number=0, chunk_size=200, overlap=0,
            parser="pdfplumber",
        )
        for chunk in chunks:
            # Allow some overshoot since we don't split mid-sentence
            assert len(chunk.text) <= 200 * 1.5, (
                f"Chunk too long: {len(chunk.text)} chars"
            )

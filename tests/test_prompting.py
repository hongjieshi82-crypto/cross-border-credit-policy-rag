"""Tests for strict compliance answer prompt rendering."""

from src.models import Chunk, ChunkMetadata
from src.prompting import build_compliance_prompt, source_label_for_chunk


def _make_policy_chunk() -> Chunk:
    text = "A credit inquiry requires prior borrower consent."
    return Chunk(
        text=text,
        page_number=4,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        method="sentence",
        metadata=ChunkMetadata(
            chunk_size=500,
            overlap=0,
            parser="layout_markdown",
            country="Mexico",
            institution="Buro de Credito",
            doc_type="Compliance_Law",
            effective_date="2026-03-01",
            legal_hierarchy="Chapter 3 > Article 12 > Section 3",
        ),
    )


def test_source_label_uses_compliance_metadata():
    chunk = _make_policy_chunk()
    assert source_label_for_chunk(chunk) == (
        "【Mexico-Buro de Credito-Compliance_Law-"
        "Chapter 3 > Article 12 > Section 3】"
    )


def test_compliance_prompt_contains_guardrails_and_query():
    chunk = _make_policy_chunk()
    prompt = build_compliance_prompt([chunk], "When is consent required?")
    assert "当前合规政策库暂未收录此条规定" in prompt
    assert "绝对不允许凭空捏造" in prompt
    assert "When is consent required?" in prompt
    assert "A credit inquiry requires prior borrower consent." in prompt
    assert "出处标签" in prompt

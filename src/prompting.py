"""Prompt templates for compliance-policy RAG answers."""

from __future__ import annotations

from collections.abc import Sequence

from src.models import Chunk


COMPLIANCE_PROMPT_TEMPLATE = """你是精通跨国金融合规与征信政策的风控专家。请严格根据下方提供的【信贷政策库切块内容】回答用户的问题。

【硬性合规守则——违反将导致严重合规风险】：
1. 你的回答必须完全基于给定的【信贷政策库切块内容】。如果切块内容中没有提到用户问题的相关具体规定，请直接回答“当前合规政策库暂未收录此条规定”，绝对不允许凭空捏造、推理、演绎或借用你常识中的境外法律。
2. 在回答具体的合规限制、罚则或流程时，必须在每条结论后方明确注明出处标签（例如：摘自【墨西哥-央行令-第三章-Article 12】）。

【信贷政策库切块内容】
{retrieved_chunks}

【风控人员提问】
{user_query}
"""


def source_label_for_chunk(chunk: Chunk) -> str:
    """Build the citation label required by the compliance prompt."""
    metadata = chunk.metadata
    parts = [
        metadata.country,
        metadata.institution,
        metadata.doc_type,
        metadata.legal_hierarchy,
    ]
    non_empty = [part for part in parts if part]
    if not non_empty:
        non_empty = [f"page-{chunk.page_number}"]
    return "【" + "-".join(non_empty) + "】"


def format_retrieved_chunks(chunks: Sequence[Chunk]) -> str:
    """Format retrieved chunks with explicit citation labels."""
    formatted: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source_label = source_label_for_chunk(chunk)
        formatted.append(
            f"[Chunk {index}] 出处标签：{source_label}\n{chunk.text.strip()}"
        )
    return "\n\n".join(formatted)


def build_compliance_prompt(
    retrieved_chunks: str | Sequence[Chunk],
    user_query: str,
) -> str:
    """Render the strict anti-hallucination compliance prompt."""
    chunk_text = (
        format_retrieved_chunks(retrieved_chunks)
        if not isinstance(retrieved_chunks, str)
        else retrieved_chunks
    )
    return COMPLIANCE_PROMPT_TEMPLATE.format(
        retrieved_chunks=chunk_text,
        user_query=user_query,
    )

"""RAG prompt construction for grounded /ask responses."""

from __future__ import annotations

from typing import Any

DEFAULT_RETRIEVAL_K = 5

GROUNDING_PROMPT_TEMPLATE = """You are a grounded Q&A assistant. Answer the user's question using ONLY the retrieved context below.

Rules:
- Use ONLY information from the retrieved context. Do not use outside knowledge.
- When you use information from a chunk, cite its document_id in your answer (for example: [rag-intro-001]).
- If the retrieved context is insufficient to answer the question, say so clearly in your answer, use a low confidence score, and set sources_needed to true.

Retrieved context:
{context}

Question: {question}"""


def _format_chunk(chunk: dict[str, Any]) -> str:
    header_parts = [f"chunk_id={chunk.get('chunk_id')}"]
    if chunk.get("document_id") is not None:
        header_parts.append(f"document_id={chunk['document_id']}")
    if chunk.get("source"):
        header_parts.append(f"source={chunk['source']}")
    return f"[{' | '.join(header_parts)}]\n{chunk.get('text', '')}"


def build_grounding_prompt(*, question: str, chunks: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Build the grounded user message and the chunk IDs that were retrieved."""

    if chunks:
        context = "\n\n".join(_format_chunk(chunk) for chunk in chunks)
        chunk_ids = [chunk_id for chunk in chunks if (chunk_id := chunk.get("chunk_id"))]
    else:
        context = "(no chunks retrieved)"
        chunk_ids = []

    prompt = GROUNDING_PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip(),
    )
    return prompt, chunk_ids

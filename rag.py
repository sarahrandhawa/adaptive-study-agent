"""RAG prompt construction for grounded /ask responses."""

from __future__ import annotations

from typing import Any

DEFAULT_RETRIEVAL_K = 5

GROUNDING_PROMPT_TEMPLATE = """You are a grounded Q&A assistant. Answer the user's question using ONLY the retrieved context below.

Rules:
- Use ONLY information from the retrieved context. Do not use outside knowledge.
- If you cite where information came from, use only the source file name shown in the context header (for example: (from biology-notes.pdf)).
- Never include internal IDs, codes or bracketed identifiers in your answer.
- If the retrieved context is insufficient to answer the question, say so clearly in your answer, use a low confidence score, and set sources_needed to true.

Retrieved context:
{context}

Question: {question}"""


def _format_chunk(chunk: dict[str, Any]) -> str:
    """Context block for the model. Only the human-readable file name is shown;
    internal chunk and document IDs stay out of the prompt so they can't leak into answers."""

    source = chunk.get("source") or "uploaded course material"
    return f"[Source: {source}]\n{chunk.get('text', '')}"


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

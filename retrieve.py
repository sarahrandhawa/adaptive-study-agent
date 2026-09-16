"""Retrieval-only helpers for debug and future RAG wiring."""

from __future__ import annotations

from typing import Any

from vectorstore import VectorStoreConfigError, get_embedding_model, get_vector_store


class RetrieveValidationError(ValueError):
    """Raised when retrieval input fails validation."""


def retrieve_chunks(*, query: str, k: int = 5) -> dict[str, Any]:
    """Embed a question and return top-k Pinecone matches without calling an LLM."""

    normalized_query = query.strip()
    if not normalized_query:
        raise RetrieveValidationError("q must not be empty")

    if k < 1:
        raise RetrieveValidationError("k must be at least 1")

    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(normalized_query, k=k)
    except VectorStoreConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    chunks: list[dict[str, Any]] = []
    for document, score in results:
        metadata = document.metadata or {}
        chunk_index = metadata.get("chunk_index")
        if chunk_index is not None:
            chunk_index = int(chunk_index)
        document_id = metadata.get("document_id")
        chunk_id = (
            f"{document_id}:{chunk_index}"
            if document_id is not None and chunk_index is not None
            else None
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "score": round(float(score), 6),
                "document_id": document_id,
                "chunk_index": chunk_index,
                "source": metadata.get("source"),
                "text": document.page_content,
            }
        )

    return {
        "query": normalized_query,
        "embedding_model": get_embedding_model(),
        "k": k,
        "chunks": chunks,
    }

"""Retrieval-only helpers for debug and future RAG wiring."""

from __future__ import annotations

from typing import Any

from db import get_conn
from sources import live_chunk_ids
from vectorstore import VectorStoreConfigError, get_embedding_model, get_vector_store

# Pinecone can return vectors whose Postgres chunk is gone (delete not yet
# propagated, failed purge, rolled-back ingest). Fetch extra candidates so that
# dropping those still leaves k live chunks in the common case.
OVERFETCH_FACTOR = 2


class RetrieveValidationError(ValueError):
    """Raised when retrieval input fails validation."""


def retrieve_chunks(
    *,
    query: str,
    user_id,
    course_id,
    k: int = 5,
) -> dict[str, Any]:
    normalized_query = query.strip()

    if not normalized_query:
        raise RetrieveValidationError("query must not be empty")

    if k < 1:
        raise RetrieveValidationError("k must be at least 1")

    try:
        vector_store = get_vector_store(namespace=str(user_id))

        results = vector_store.similarity_search_with_score(
            normalized_query,
            k=k * OVERFETCH_FACTOR,
            filter={"course_id": str(course_id)},
        )
    except VectorStoreConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    candidate_ids = [(document.metadata or {}).get("chunk_id") for document, _ in results]

    live_ids: set[str] = set()
    if candidate_ids:
        with get_conn() as conn:
            with conn.cursor() as cur:
                live_ids = live_chunk_ids(cur, user_id=user_id, course_id=course_id, chunk_ids=candidate_ids)

    chunks: list[dict[str, Any]] = []

    for document, score in results:
        metadata = document.metadata or {}

        if str(metadata.get("chunk_id")) not in live_ids:
            continue
        if len(chunks) == k:
            break

        chunks.append(
            {
                "chunk_id": metadata.get("chunk_id"),
                "score": round(float(score), 6),
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
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
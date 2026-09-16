"""Document ingest pipeline: chunk, embed, upsert to Pinecone."""

from __future__ import annotations

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import VectorStoreConfigError, get_vector_store


class IngestValidationError(ValueError):
    """Raised when ingest input fails validation."""


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    chunk_size = int(os.getenv("INGEST_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("INGEST_CHUNK_OVERLAP", "100"))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def ingest_text(*, document_id: str, text: str, source: str | None = None) -> dict[str, str | int]:
    """Chunk text, embed with the shared vector store, and upsert into Pinecone."""

    normalized_id = document_id.strip()
    normalized_text = text.strip()
    normalized_source = source.strip() if source else None

    if not normalized_id:
        raise IngestValidationError("document_id must not be empty")
    if not normalized_text:
        raise IngestValidationError("text must not be empty")

    chunks = get_text_splitter().split_text(normalized_text)
    if not chunks:
        raise IngestValidationError("text produced no chunks after splitting")

    documents: list[Document] = []
    ids: list[str] = []

    for chunk_index, chunk_text in enumerate(chunks):
        metadata: dict[str, str | int] = {
            "document_id": normalized_id,
            "chunk_index": chunk_index,
        }
        if normalized_source:
            metadata["source"] = normalized_source

        documents.append(Document(page_content=chunk_text, metadata=metadata))
        ids.append(f"{normalized_id}:{chunk_index}")

    try:
        vector_store = get_vector_store()
        vector_store.add_documents(documents=documents, ids=ids)
    except VectorStoreConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    return {
        "document_id": normalized_id,
        "chunks_indexed": len(chunks),
        "status": "indexed",
    }

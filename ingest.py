"""Persistent document ingest pipeline: Postgres + Pinecone."""

from __future__ import annotations

import hashlib
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db import get_conn
from vectorstore import (
    VectorStoreConfigError,
    get_embedding_model,
    get_vector_store,
    delete_vectors
)


class IngestValidationError(ValueError):
    pass


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    chunk_size = int(os.getenv("INGEST_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("INGEST_CHUNK_OVERLAP", "100"))

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def ingest_document(*, user_id, course_id, filename: str, text: str) -> dict:
    filename = filename.strip()
    text = text.strip()

    if not filename:
        raise IngestValidationError("filename must not be empty")

    if not text:
        raise IngestValidationError("text must not be empty")

    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    splitter = get_text_splitter()
    split_documents = splitter.create_documents([text])

    if not split_documents:
        raise IngestValidationError("text produced no chunks")

    with get_conn() as conn:
        with conn.cursor() as cur:

            # Make sure this course actually belongs to this user.
            cur.execute(
                """
                SELECT id
                FROM courses
                WHERE id = %s
                  AND user_id = %s
                  AND archived_at IS NULL
                """,
                (course_id, user_id),
            )

            if cur.fetchone() is None:
                raise IngestValidationError("Course not found")

            # Don't ingest the exact same active file twice.
            cur.execute(
                """
                SELECT id, chunk_count
                FROM documents
                WHERE user_id = %s
                  AND course_id = %s
                  AND filename = %s
                  AND content_hash = %s
                  AND status = 'active'
                LIMIT 1
                """,
                (user_id, course_id, filename, content_hash),
            )

            existing = cur.fetchone()

            if existing is not None:
                return {
                    "document_id": existing[0],
                    "chunks_indexed": existing[1],
                    "status": "unchanged",
                }

            # Find an older active version with the same filename.
            cur.execute(
                """
                SELECT id
                FROM documents
                WHERE user_id = %s
                  AND course_id = %s
                  AND filename = %s
                  AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id, course_id, filename),
            )

            old_document = cur.fetchone()
            old_document_id = old_document[0] if old_document else None

            cur.execute(
                """
                INSERT INTO documents (
                    user_id,
                    course_id,
                    filename,
                    content_hash,
                    extracted_text,
                    char_count,
                    chunk_count,
                    embedding_model,
                    status,
                    replaces_document_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active', %s)
                RETURNING id
                """,
                (
                    user_id,
                    course_id,
                    filename,
                    content_hash,
                    text,
                    len(text),
                    len(split_documents),
                    get_embedding_model(),
                    old_document_id,
                ),
            )

            document_id = cur.fetchone()[0]

            pinecone_documents = []
            chunk_ids = []

            for chunk_index, split_document in enumerate(split_documents):
                chunk_text = split_document.page_content
                char_start = split_document.metadata.get("start_index", 0)
                char_end = char_start + len(chunk_text)

                cur.execute(
                    """
                    INSERT INTO chunks (
                        user_id,
                        course_id,
                        document_id,
                        chunk_index,
                        text,
                        char_start,
                        char_end
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        user_id,
                        course_id,
                        document_id,
                        


                        chunk_index,
                        chunk_text,
                        char_start,
                        char_end,
                    ),
                )

                chunk_id = cur.fetchone()[0]
                chunk_ids.append(str(chunk_id))

                pinecone_documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_id": str(chunk_id),
                            "document_id": str(document_id),
                            "course_id": str(course_id),
                            "chunk_index": chunk_index,
                            "source": filename,
                        },
                    )
                )

            # If this upload replaces an older document,
            # remember the old Pinecone vector IDs before changing anything.
            old_chunk_ids = []

            if old_document_id is not None:
                cur.execute(
                    """
                    SELECT id
                    FROM chunks
                    WHERE document_id = %s
                    AND deleted_at IS NULL
                    """,
                    (old_document_id,),
                )

                old_chunk_ids = [str(row[0]) for row in cur.fetchall()]


            # Upload the NEW chunks to Pinecone.
            try:
                vector_store = get_vector_store(namespace=str(user_id))

                vector_store.add_documents(
                    documents=pinecone_documents,
                    ids=chunk_ids,
                )
            except VectorStoreConfigError as exc:
                raise RuntimeError(str(exc)) from exc


            # New version succeeded, so retire the OLD version.
            if old_document_id is not None:
                delete_vectors(
                    namespace=str(user_id),
                    ids=old_chunk_ids,
                )

                cur.execute(
                    """
                    UPDATE documents
                    SET status = 'replaced',
                        deleted_at = now()
                    WHERE id = %s
                    """,
                    (old_document_id,),
                )

                cur.execute(
                    """
                    UPDATE chunks
                    SET deleted_at = now()
                    WHERE document_id = %s
                    """,
                    (old_document_id,),
                )

            return {
                "document_id": document_id,
                "chunks_indexed": len(chunk_ids),
                "status": "indexed",
            }
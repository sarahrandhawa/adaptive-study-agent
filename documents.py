import logging

from db import get_conn
from vectorstore import delete_vectors

logger = logging.getLogger("adaptive_study_agent")


def list_documents(user_id, course_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, char_count, chunk_count, created_at
                FROM documents
                WHERE user_id = %s
                  AND course_id = %s
                  AND status = 'active'
                  AND deleted_at IS NULL
                ORDER BY created_at DESC
                """,
                (user_id, course_id),
            )

            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "filename": row[1],
            "char_count": row[2],
            "chunk_count": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def delete_document(user_id, document_id):
    """Delete one of the user's documents. Safe to repeat.

    Postgres is marked first and committed, so the document's chunks immediately stop
    being retrievable, gradable or practicable (see sources.py) even if Pinecone is
    unavailable. Vector removal follows and is recorded in documents.vectors_deleted_at;
    if it fails, a repeated DELETE or scripts/purge_deleted_vectors.py retries it.

    Returns False only when the user has no such document (active or deleted).
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, vectors_deleted_at
                FROM documents
                WHERE id = %s
                  AND user_id = %s
                  AND status IN ('active', 'deleted')
                FOR UPDATE
                """,
                (document_id, user_id),
            )

            row = cur.fetchone()

            if row is None:
                return False

            status, vectors_deleted_at = row

            if status == "active":
                cur.execute(
                    """
                    UPDATE chunks
                    SET deleted_at = now()
                    WHERE document_id = %s
                      AND user_id = %s
                      AND deleted_at IS NULL
                    """,
                    (document_id, user_id),
                )

                cur.execute(
                    """
                    UPDATE documents
                    SET status = 'deleted',
                        deleted_at = now()
                    WHERE id = %s
                      AND user_id = %s
                    """,
                    (document_id, user_id),
                )

    if vectors_deleted_at is None:
        try_purge_document_vectors(user_id, document_id)

    return True


def purge_document_vectors(user_id, document_id) -> int:
    """Remove a deleted or replaced document's vectors from Pinecone.

    Only chunk ids belonging to this user and this non-active document are sent, and
    only to the user's namespace, so active vectors of other documents or users are
    never touched. Raises if Pinecone fails; the document stays pending.
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id
                FROM chunks c
                JOIN documents d
                    ON d.id = c.document_id
                   AND d.user_id = c.user_id
                WHERE c.document_id = %s
                  AND c.user_id = %s
                  AND d.status IN ('deleted', 'replaced')
                """,
                (document_id, user_id),
            )

            chunk_ids = [str(row[0]) for row in cur.fetchall()]

    delete_vectors(namespace=str(user_id), ids=chunk_ids)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET vectors_deleted_at = now()
                WHERE id = %s
                  AND user_id = %s
                  AND status IN ('deleted', 'replaced')
                """,
                (document_id, user_id),
            )

    return len(chunk_ids)


def try_purge_document_vectors(user_id, document_id) -> bool:
    """Best-effort purge: a failure is logged, not raised, because the document is
    already unusable and the purge stays pending for a retry."""

    try:
        purge_document_vectors(user_id, document_id)
        return True
    except Exception:  # noqa: BLE001 - Pinecone/DB errors must not undo a committed delete
        logger.warning("Vector purge pending for document %s", document_id, exc_info=True)
        return False


def pending_vector_purges(limit: int = 500):
    """Deleted or replaced documents whose vectors have not been purged yet (all users)."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, id
                FROM documents
                WHERE status IN ('deleted', 'replaced')
                  AND vectors_deleted_at IS NULL
                ORDER BY deleted_at NULLS FIRST
                LIMIT %s
                """,
                (limit,),
            )

            return [(row[0], row[1]) for row in cur.fetchall()]

from db import get_conn
from vectorstore import delete_vectors


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
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM documents
                WHERE id = %s
                  AND user_id = %s
                  AND status = 'active'
                  AND deleted_at IS NULL
                """,
                (document_id, user_id),
            )

            if cur.fetchone() is None:
                return False

            cur.execute(
                """
                SELECT id
                FROM chunks
                WHERE document_id = %s
                  AND user_id = %s
                  AND deleted_at IS NULL
                """,
                (document_id, user_id),
            )

            chunk_ids = [str(row[0]) for row in cur.fetchall()]

            delete_vectors(
                namespace=str(user_id),
                ids=chunk_ids,
            )

            cur.execute(
                """
                UPDATE chunks
                SET deleted_at = now()
                WHERE document_id = %s
                  AND user_id = %s
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

            return True
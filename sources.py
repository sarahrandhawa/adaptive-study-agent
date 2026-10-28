"""Source-material integrity: Postgres is the source of truth for which chunks are live.

Pinecone is treated as a cache that can be stale (eventual consistency after a
delete, a failed vector purge, or vectors orphaned by a rolled-back ingest). A
chunk may only be used for retrieval, question generation or grading when its
Postgres row is not deleted, its document is active, and its course is not archived.
"""

from __future__ import annotations

from uuid import UUID

# A chunk alias `c` is usable. Joins on documents `d` and courses `co` must be present.
LIVE_CHUNK_CONDITIONS = """
    c.deleted_at IS NULL
    AND d.status = 'active'
    AND d.deleted_at IS NULL
    AND co.archived_at IS NULL
"""

# A question alias `q` still has at least one live source chunk.
QUESTION_HAS_LIVE_SOURCE = f"""
    EXISTS (
        SELECT 1
        FROM question_sources src
        JOIN chunks c
            ON c.id = src.chunk_id
           AND c.user_id = src.user_id
        JOIN documents d
            ON d.id = c.document_id
           AND d.user_id = c.user_id
        JOIN courses co
            ON co.id = d.course_id
           AND co.user_id = d.user_id
        WHERE src.question_id = q.id
          AND src.user_id = q.user_id
          AND {LIVE_CHUNK_CONDITIONS}
    )
"""


def _valid_uuids(values) -> list[str]:
    valid = []
    for value in values:
        try:
            valid.append(str(UUID(str(value))))
        except (TypeError, ValueError):
            continue
    return valid


def live_chunk_ids(cur, *, user_id, course_id, chunk_ids) -> set[str]:
    """The subset of chunk_ids that are live for this user and course."""

    candidates = _valid_uuids(chunk_ids)
    if not candidates:
        return set()

    cur.execute(
        f"""
        SELECT c.id
        FROM chunks c
        JOIN documents d
            ON d.id = c.document_id
           AND d.user_id = c.user_id
        JOIN courses co
            ON co.id = d.course_id
           AND co.user_id = d.user_id
        WHERE c.id = ANY(%s::uuid[])
          AND c.user_id = %s
          AND c.course_id = %s
          AND {LIVE_CHUNK_CONDITIONS}
        """,
        (candidates, user_id, course_id),
    )

    return {str(row[0]) for row in cur.fetchall()}

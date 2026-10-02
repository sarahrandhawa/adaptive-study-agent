"""Read-only question history for the UI (no answers or key points exposed)."""

from db import get_conn


def list_questions(user_id, course_id, limit: int = 200):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    q.id,
                    q.prompt,
                    t.name,
                    q.concept_label,
                    q.created_at,
                    qs.mastery,
                    qs.attempt_count,
                    qs.last_score,
                    qs.last_attempt_at
                FROM questions q
                LEFT JOIN question_topics qt
                    ON qt.question_id = q.id
                   AND qt.is_primary = true
                   AND qt.user_id = q.user_id
                LEFT JOIN topics t
                    ON t.id = qt.topic_id
                   AND t.user_id = q.user_id
                LEFT JOIN question_state qs
                    ON qs.question_id = q.id
                   AND qs.user_id = q.user_id
                WHERE q.user_id = %s
                  AND q.course_id = %s
                  AND q.status = 'active'
                ORDER BY COALESCE(qs.last_attempt_at, q.created_at) DESC
                LIMIT %s
                """,
                (user_id, course_id, limit),
            )

            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "prompt": row[1],
            "topic": row[2],
            "concept_label": row[3],
            "created_at": row[4],
            "mastery": row[5],
            "attempt_count": row[6] or 0,
            "last_score": row[7],
            "last_attempt_at": row[8],
        }
        for row in rows
    ]

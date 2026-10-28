"""Read-only question listings for the UI.

list_questions: practice history summary; never exposes answers or key points.
list_question_bank: the study library; includes the stored reference answer.

Both only show questions that still have live source material (see sources.py).
"""

from db import get_conn
from sources import QUESTION_HAS_LIVE_SOURCE

QUESTION_BANK_LIMIT = 500


def list_questions(user_id, course_id, limit: int = 200):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
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
                  AND {QUESTION_HAS_LIVE_SOURCE}
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


def list_question_bank(user_id, course_id, limit: int = QUESTION_BANK_LIMIT):
    """Each stored question once, with its saved reference answer and key points.

    Reads persisted data only (no generation). One row per question: the primary
    topic link and question_state are both at most one row per question; duplicates
    are dropped defensively anyway.
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    q.id,
                    q.prompt,
                    q.reference_answer,
                    q.key_points,
                    q.concept_label,
                    q.created_at,
                    t.id,
                    t.name,
                    qs.attempt_count,
                    qs.mastery,
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
                  AND {QUESTION_HAS_LIVE_SOURCE}
                ORDER BY t.name NULLS LAST, q.created_at
                LIMIT %s
                """,
                (user_id, course_id, limit),
            )

            rows = cur.fetchall()

    bank = []
    seen = set()

    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])

        reference_answer = (row[2] or "").strip()
        key_points = [point for point in (row[3] or []) if isinstance(point, str) and point.strip()]

        bank.append(
            {
                "id": row[0],
                "prompt": row[1],
                "reference_answer": reference_answer or None,
                "key_points": key_points,
                "concept_label": row[4] or None,
                "created_at": row[5],
                "topic_id": row[6],
                "topic": row[7],
                "attempt_count": row[8] or 0,
                "mastery": row[9],
                "last_attempt_at": row[10],
            }
        )

    return bank

"""Learning-progress summaries for the dashboard. Read-only; every query is scoped to user_id.

Definitions (reusing the adaptive model, not a second scoring scheme):
- Question mastery: question_state.mastery, the moving average updated by record_attempt.
- Topic mastery: mean mastery of the topic's answered questions, the same aggregation as
  study.get_topic_mastery. Unlike that function, "no answered questions" is reported as
  None (not practiced) instead of the neutral 0.5 used to rank questions.
- Course mastery: the same mean over all answered questions in the course.
- Bands: adaptive.mastery_band, which uses the grading verdict thresholds.
- Average score / correct rate: computed from recorded attempts (history), including
  questions whose source document was later deleted. Question-bank counts only include
  questions that are still practicable (live source material).
"""

from __future__ import annotations

from typing import Any

from adaptive import mastery_band
from db import get_conn
from sources import QUESTION_HAS_LIVE_SOURCE

# Below this many graded answers, mastery is shown as an early estimate.
LOW_DATA_ATTEMPTS = 3
RECENT_ACTIVITY_LIMIT = 5
FOCUS_TOPIC_LIMIT = 3


def _stats(*, questions_in_bank=0, questions_answered=0, attempts=0, average_score=None,
           correct=0, mastery=None) -> dict[str, Any]:
    attempts = attempts or 0
    questions_answered = questions_answered or 0
    mastery = float(mastery) if mastery is not None and questions_answered else None
    return {
        "questions_in_bank": questions_in_bank or 0,
        "questions_answered": questions_answered,
        "attempts": attempts,
        "average_score": float(average_score) if attempts and average_score is not None else None,
        "correct_rate": (correct or 0) / attempts if attempts else None,
        "mastery": mastery,
        "mastery_band": mastery_band(mastery),
        "low_data": attempts < LOW_DATA_ATTEMPTS,
    }


def _course_stats(cur, user_id, course_id=None) -> dict[str, dict[str, Any]]:
    """Per-course raw aggregates for one course, or all of the user's courses."""

    stats: dict[str, dict[str, Any]] = {}

    def bucket(cid):
        return stats.setdefault(str(cid), {})

    cur.execute(
        f"""
        SELECT q.course_id, COUNT(*)
        FROM questions q
        WHERE q.user_id = %s
          AND (%s::uuid IS NULL OR q.course_id = %s::uuid)
          AND q.status = 'active'
          AND {QUESTION_HAS_LIVE_SOURCE}
        GROUP BY q.course_id
        """,
        (user_id, course_id, course_id),
    )
    for cid, count in cur.fetchall():
        bucket(cid)["questions_in_bank"] = count

    cur.execute(
        """
        SELECT q.course_id,
               COUNT(*) FILTER (WHERE qs.attempt_count > 0),
               AVG(qs.mastery) FILTER (WHERE qs.attempt_count > 0)
        FROM question_state qs
        JOIN questions q
            ON q.id = qs.question_id
           AND q.user_id = qs.user_id
        WHERE qs.user_id = %s
          AND (%s::uuid IS NULL OR q.course_id = %s::uuid)
        GROUP BY q.course_id
        """,
        (user_id, course_id, course_id),
    )
    for cid, answered, mastery in cur.fetchall():
        bucket(cid).update(questions_answered=answered, mastery=mastery)

    cur.execute(
        """
        SELECT q.course_id,
               COUNT(a.id),
               AVG(a.score),
               COUNT(*) FILTER (WHERE a.verdict = 'correct'),
               MAX(a.created_at),
               COUNT(*) FILTER (WHERE a.created_at >= now() - interval '7 days')
        FROM attempts a
        JOIN questions q
            ON q.id = a.question_id
           AND q.user_id = a.user_id
        WHERE a.user_id = %s
          AND (%s::uuid IS NULL OR q.course_id = %s::uuid)
        GROUP BY q.course_id
        """,
        (user_id, course_id, course_id),
    )
    for cid, attempts, average, correct, last_at, recent in cur.fetchall():
        bucket(cid).update(
            attempts=attempts,
            average_score=average,
            correct=correct,
            last_activity_at=last_at,
            attempts_last_7_days=recent,
        )

    return stats


def _course_summary(course: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "course_id": course["id"],
        "name": course["name"],
        **_stats(
            questions_in_bank=raw.get("questions_in_bank"),
            questions_answered=raw.get("questions_answered"),
            attempts=raw.get("attempts"),
            average_score=raw.get("average_score"),
            correct=raw.get("correct"),
            mastery=raw.get("mastery"),
        ),
        "last_activity_at": raw.get("last_activity_at"),
        "attempts_last_7_days": raw.get("attempts_last_7_days") or 0,
    }


def progress_overview(user_id, courses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summary for each of the user's active courses (pass list_courses(user_id))."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            stats = _course_stats(cur, user_id)

    return [_course_summary(course, stats.get(str(course["id"]), {})) for course in courses]


def _sort_key(topic):
    # Practised topics first, weakest first; then unpractised topics by name.
    if topic["mastery"] is None:
        return (1, 0.0, (topic["name"] or "").casefold())
    return (0, topic["mastery"], (topic["name"] or "").casefold())


def course_progress(user_id, course: dict[str, Any]) -> dict[str, Any]:
    """Course summary plus per-topic mastery, focus topics and recent activity."""

    course_id = course["id"]

    with get_conn() as conn:
        with conn.cursor() as cur:
            raw = _course_stats(cur, user_id, course_id).get(str(course_id), {})

            # Topics that have at least one question. Mastery uses the same
            # question_state/question_topics aggregation as study.get_topic_mastery.
            cur.execute(
                f"""
                WITH topic_questions AS (
                    SELECT qt.topic_id,
                           q.id AS question_id,
                           {QUESTION_HAS_LIVE_SOURCE} AS live
                    FROM questions q
                    JOIN question_topics qt
                        ON qt.question_id = q.id
                       AND qt.user_id = q.user_id
                    WHERE q.user_id = %s
                      AND q.course_id = %s
                      AND q.status = 'active'
                )
                SELECT t.id,
                       t.name,
                       COUNT(tq.question_id) FILTER (WHERE tq.live),
                       COUNT(qs.question_id) FILTER (WHERE qs.attempt_count > 0),
                       AVG(qs.mastery) FILTER (WHERE qs.attempt_count > 0),
                       MAX(qs.last_attempt_at)
                FROM topics t
                JOIN topic_questions tq
                    ON tq.topic_id = t.id
                LEFT JOIN question_state qs
                    ON qs.question_id = tq.question_id
                   AND qs.user_id = t.user_id
                WHERE t.user_id = %s
                  AND t.course_id = %s
                GROUP BY t.id, t.name
                """,
                (user_id, course_id, user_id, course_id),
            )
            topic_rows = cur.fetchall()

            cur.execute(
                """
                SELECT qt.topic_id,
                       COUNT(a.id),
                       AVG(a.score),
                       COUNT(*) FILTER (WHERE a.verdict = 'correct')
                FROM attempts a
                JOIN questions q
                    ON q.id = a.question_id
                   AND q.user_id = a.user_id
                JOIN question_topics qt
                    ON qt.question_id = q.id
                   AND qt.user_id = q.user_id
                WHERE a.user_id = %s
                  AND q.course_id = %s
                GROUP BY qt.topic_id
                """,
                (user_id, course_id),
            )
            topic_attempts = {str(row[0]): row[1:] for row in cur.fetchall()}

            cur.execute(
                f"""
                SELECT a.created_at,
                       a.verdict,
                       a.score,
                       q.prompt,
                       t.name,
                       {QUESTION_HAS_LIVE_SOURCE} AS live
                FROM attempts a
                JOIN questions q
                    ON q.id = a.question_id
                   AND q.user_id = a.user_id
                LEFT JOIN question_topics qt
                    ON qt.question_id = q.id
                   AND qt.is_primary = true
                   AND qt.user_id = q.user_id
                LEFT JOIN topics t
                    ON t.id = qt.topic_id
                   AND t.user_id = q.user_id
                WHERE a.user_id = %s
                  AND q.course_id = %s
                ORDER BY a.created_at DESC
                LIMIT %s
                """,
                (user_id, course_id, RECENT_ACTIVITY_LIMIT),
            )
            recent_rows = cur.fetchall()

    topics = []
    for topic_id, name, in_bank, answered, mastery, last_at in topic_rows:
        attempts, average, correct = topic_attempts.get(str(topic_id), (0, None, 0))
        topics.append(
            {
                "topic_id": topic_id,
                "name": name,
                **_stats(
                    questions_in_bank=in_bank,
                    questions_answered=answered,
                    attempts=attempts,
                    average_score=average,
                    correct=correct,
                    mastery=mastery,
                ),
                "last_attempt_at": last_at,
            }
        )
    topics.sort(key=_sort_key)

    return {
        **_course_summary(course, raw),
        "topics": topics,
        "focus_topics": [
            t["name"] for t in topics if t["mastery_band"] in ("needs_review", "developing")
        ][:FOCUS_TOPIC_LIMIT],
        "not_practiced_topics": [t["name"] for t in topics if t["mastery_band"] == "not_practiced"],
        "recent_activity": [
            {
                "attempted_at": created_at,
                "verdict": verdict,
                "score": score,
                # Don't resurface text derived from deleted material.
                "question": prompt if live else None,
                "topic": topic,
                "source_removed": not live,
            }
            for created_at, verdict, score, prompt, topic, live in recent_rows
        ],
    }

from datetime import datetime, timedelta, timezone

from adaptive import priority_score, recency_penalty, update_mastery


def test_update_mastery():
    assert update_mastery(0.5, 0.65) == 0.56


def test_unanswered_question_uses_topic_mastery():
    now = datetime.now(timezone.utc)

    priority = priority_score(
        question_mastery=None,
        topic_mastery=0.6,
        last_attempt_at=None,
        now=now,
    )

    assert priority == 0.3


def test_recent_question_gets_penalty():
    now = datetime.now(timezone.utc)

    penalty = recency_penalty(
        now - timedelta(minutes=5),
        now,
    )

    assert penalty > 0.49


def test_recency_penalty_disappears_after_24_hours():
    now = datetime.now(timezone.utc)

    penalty = recency_penalty(
        now - timedelta(hours=24),
        now,
    )

    assert penalty == 0.0


def test_new_question_beats_recent_weak_question():
    now = datetime.now(timezone.utc)

    existing_priority = priority_score(
        question_mastery=0.56,
        topic_mastery=0.56,
        last_attempt_at=now - timedelta(minutes=1),
        now=now,
    )

    new_priority = 0.5 * 0.56

    assert new_priority < existing_priority


def test_old_weak_question_can_return():
    now = datetime.now(timezone.utc)

    old_weak_priority = priority_score(
        question_mastery=0.2,
        topic_mastery=0.6,
        last_attempt_at=now - timedelta(days=2),
        now=now,
    )

    new_priority = 0.5 * 0.6

    assert old_weak_priority < new_priority
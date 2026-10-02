from datetime import datetime, timedelta

def update_mastery(old_mastery, score):
    return (old_mastery * 0.6) + (score * 0.4)

def next_due(verdict, correct_streak, now):
    if verdict != "correct":
        return now + timedelta(minutes=10)

    if correct_streak == 1:
        return now + timedelta(days=1)
    elif correct_streak == 2:
        return now + timedelta(days=3)
    else:
        return now + timedelta(days=7)

def recency_penalty(last_attempt_at, now):
    if last_attempt_at is None:
        return 0.0

    hours_since = (now - last_attempt_at).total_seconds() / 3600

    # Maximum penalty immediately after answering.
    # It linearly disappears over 24 hours.
    return 0.5 * max(0.0, 1.0 - (hours_since / 24.0))


def priority_score(
    *,
    question_mastery,
    topic_mastery,
    last_attempt_at,
    now,
):
    # No question mastery means this question has never been answered.
    if question_mastery is None:
        return 0.5 * topic_mastery

    return question_mastery + recency_penalty(
        last_attempt_at,
        now,
    )
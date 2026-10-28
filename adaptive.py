from datetime import datetime, timedelta

# Score thresholds shared by grading verdicts and progress bands (both are on the 0-1 scale).
CORRECT_THRESHOLD = 0.85
PARTIAL_THRESHOLD = 0.35


def mastery_band(mastery):
    """Label a mastery value with the same cutoffs used for grading verdicts.
    None means no answered questions yet, which is not the same as low mastery."""

    if mastery is None:
        return "not_practiced"
    if mastery >= CORRECT_THRESHOLD:
        return "mastered"
    if mastery >= PARTIAL_THRESHOLD:
        return "developing"
    return "needs_review"


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
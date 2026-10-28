import pytest
import adaptive

from datetime import datetime, timedelta, timezone

@pytest.mark.parametrize(
        "old_mastery, score, expected",
        [
            (0.3, 1.0, 0.58),
            (0.8, 0.0, 0.48),
            (0.5, 0.5, 0.5),
            (0.0, 1.0, 0.4),
            (1.0, 0.0, 0.6),
        ],
        ids=["correct-answer", "wrong-answer", "partial-credit", "inital-zero", "score-zero"],
)

def test_mastery_increase_after_correct_answer(old_mastery, score, expected):
    result = adaptive.update_mastery(old_mastery, score)

    assert result == pytest.approx(expected)

def test_priority_score():
    now = datetime(2026, 10, 28, tzinfo=timezone.utc)
    one_hour_ago = now - timedelta(hours=1)
    twelve_hours_ago = now - timedelta(hours=12)

    recent_score = adaptive.priority_score(question_mastery = 0.5, topic_mastery = 0.5, last_attempt_at = one_hour_ago, now = now) 
    older_score = adaptive.priority_score(question_mastery = 0.5, topic_mastery = 0.5, last_attempt_at = twelve_hours_ago, now = now) 

    assert older_score < recent_score
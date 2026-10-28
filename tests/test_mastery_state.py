"""study.record_attempt state transitions against an in-memory FakeDB."""

import pytest

import study
from adaptive import update_mastery
from fakes import QUESTION_ID, USER_ID, FakeDB


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    monkeypatch.setattr(study, "get_conn", fake.connect)
    return fake


def record(verdict, score, **overrides):
    kwargs = dict(
        user_id=USER_ID,
        question_id=QUESTION_ID,
        answer_text="answer",
        verdict=verdict,
        score=score,
        missed_points=[],
        feedback="f",
        explanation="e",
        identified_gap=None,
    )
    kwargs.update(overrides)
    return study.record_attempt(**kwargs)


def test_first_attempt_starts_from_neutral_mastery(db):
    result = record("correct", 0.9)

    state = db.state_for(USER_ID, QUESTION_ID)
    assert result["mastery"] == pytest.approx(0.5 * 0.6 + 0.9 * 0.4)
    assert result["attempt_count"] == 1
    assert result["correct_streak"] == 1
    assert state["mastery"] == pytest.approx(result["mastery"])
    assert state["attempt_count"] == 1
    assert state["last_score"] == 0.9


def test_attempt_row_is_history_and_state_points_at_it(db):
    result = record("partially_correct", 0.5, missed_points=["precipitation"], identified_gap="gap")

    assert len(db.attempts) == 1
    attempt = db.attempts[0]
    assert attempt["id"] == result["attempt_id"]
    assert attempt["verdict"] == "partially_correct"
    assert attempt["score"] == 0.5
    assert attempt["missed_points"] == ["precipitation"]
    assert db.state_for(USER_ID, QUESTION_ID)["last_attempt_at"] == attempt["created_at"]


def test_existing_state_is_updated_not_reset(db):
    db.seed_state(USER_ID, QUESTION_ID, mastery=0.7, attempt_count=3, correct_streak=2)

    result = record("correct", 1.0)

    assert result["mastery"] == pytest.approx(0.7 * 0.6 + 1.0 * 0.4)
    assert result["attempt_count"] == 4
    assert result["correct_streak"] == 3


@pytest.mark.parametrize("verdict, score", [("partially_correct", 0.5), ("incorrect", 0.1)])
def test_non_correct_verdict_resets_streak(db, verdict, score):
    db.seed_state(USER_ID, QUESTION_ID, mastery=0.8, attempt_count=5, correct_streak=4)

    result = record(verdict, score)

    assert result["correct_streak"] == 0
    assert result["attempt_count"] == 6
    assert result["mastery"] < 0.8


def test_sequential_attempts_accumulate(db):
    scores = [0.9, 0.2, 1.0]
    expected = 0.5
    for score in scores:
        record(study.verdict_from_score(score), score)
        expected = update_mastery(expected, score)

    state = db.state_for(USER_ID, QUESTION_ID)
    assert len(db.attempts) == 3
    assert state["attempt_count"] == 3
    assert state["correct_streak"] == 1
    assert state["mastery"] == pytest.approx(expected)


def test_failed_write_leaves_no_partial_state(monkeypatch):
    """If the question_state upsert fails, the attempt insert must roll back too."""

    fake = FakeDB()
    real_connect = fake.connect

    def connect_that_fails_on_state_upsert():
        conn = real_connect()
        real_cursor = conn.cursor

        def cursor():
            cur = real_cursor()
            real_execute = cur.execute

            def execute(sql, params=()):
                if "INSERT INTO question_state" in sql:
                    raise RuntimeError("simulated DB failure")
                return real_execute(sql, params)

            cur.execute = execute
            return cur

        conn.cursor = cursor
        return conn

    monkeypatch.setattr(study, "get_conn", connect_that_fails_on_state_upsert)

    with pytest.raises(RuntimeError):
        record("correct", 0.9)

    assert fake.attempts == []
    assert fake.state_for(USER_ID, QUESTION_ID) is None


@pytest.mark.parametrize("old", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("score", [0.0, 0.35, 0.85, 1.0])
def test_update_mastery_stays_in_unit_interval(old, score):
    assert 0.0 <= update_mastery(old, score) <= 1.0

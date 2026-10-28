"""Duplicate and concurrent grade submissions.

These reproduce confirmed state bugs. They are marked `regression` and are
expected to FAIL until a fix is approved. See TESTING_REPORT.md.
"""

import threading

import pytest
from fastapi.testclient import TestClient

import main
import study
from adaptive import update_mastery
from fakes import QUESTION_ID, USER_ID, FakeDB, FakeGrader, fake_question


@pytest.fixture
def api(monkeypatch):
    db = FakeDB()
    grader = FakeGrader(score=0.9)
    monkeypatch.setattr(study, "get_conn", db.connect)
    monkeypatch.setattr(study, "_call_structured", grader)
    monkeypatch.setattr(study, "load_question_for_grading", lambda **_: fake_question())
    main.app.dependency_overrides[main.current_user] = lambda: USER_ID
    try:
        yield TestClient(main.app), db, grader
    finally:
        main.app.dependency_overrides.clear()


SUBMISSION_ID = "77777777-7777-7777-7777-777777777777"


@pytest.mark.regression
def test_retry_with_same_submission_id_is_not_double_counted(api):
    """BUG: a double-click / Streamlit rerun / client retry of the same submission
    records a second attempt and applies the mastery and streak update twice.
    Intended design: the client sends an explicit submission_id; a retry reuses it.
    (Today GradeRequest silently ignores the extra field.)"""

    client, db, grader = api
    body = {
        "question_id": QUESTION_ID,
        "answer": "Evaporation relative to precipitation",
        "submission_id": SUBMISSION_ID,
    }

    first = client.post("/grade", json=body)
    second = client.post("/grade", json=body)

    assert first.status_code == 200
    assert second.status_code == 200
    state = db.state_for(USER_ID, QUESTION_ID)
    assert len(db.attempts) == 1, f"{len(db.attempts)} attempts recorded for one logical submission"
    assert state["attempt_count"] == 1
    assert state["correct_streak"] == 1
    assert state["mastery"] == pytest.approx(update_mastery(0.5, 0.9))
    assert second.json()["attempt_id"] == first.json()["attempt_id"]
    assert grader.calls == 1, "a replayed submission should not be re-graded by the LLM"


def test_same_answer_text_with_new_submission_ids_counts_twice(api):
    """Guard for the fix: dedupe must key on submission_id, not answer text. A student
    who deliberately re-answers with identical wording makes a new attempt."""

    client, db, _ = api
    answer = "Evaporation relative to precipitation"

    client.post("/grade", json={"question_id": QUESTION_ID, "answer": answer, "submission_id": SUBMISSION_ID})
    client.post(
        "/grade",
        json={"question_id": QUESTION_ID, "answer": answer, "submission_id": "88888888-8888-8888-8888-888888888888"},
    )

    assert len(db.attempts) == 2
    assert db.state_for(USER_ID, QUESTION_ID)["attempt_count"] == 2


def test_different_answers_are_separate_attempts(api):
    """Guard for the fix: deduplication must not swallow genuinely new answers."""

    client, db, _ = api

    client.post("/grade", json={"question_id": QUESTION_ID, "answer": "first try"})
    client.post("/grade", json={"question_id": QUESTION_ID, "answer": "second, better try"})

    state = db.state_for(USER_ID, QUESTION_ID)
    assert len(db.attempts) == 2
    assert state["attempt_count"] == 2
    assert state["correct_streak"] == 2


def _run_concurrently(target, n):
    errors = []

    def wrapper():
        try:
            target()
        except Exception as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=wrapper) for _ in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "concurrent submissions hung"
    if errors:
        raise errors[0]


@pytest.mark.regression
@pytest.mark.parametrize("existing_state", [False, True], ids=["first_attempt", "existing_state"])
def test_concurrent_attempts_do_not_lose_updates(monkeypatch, existing_state):
    """BUG: record_attempt reads question_state without a lock, then writes it back.
    Two overlapping submissions both read the same starting state, so one
    update is lost: 2 attempt rows exist but attempt_count/streak/mastery
    reflect only 1."""

    db = FakeDB(interleave_reads=2)
    if existing_state:
        db.seed_state(USER_ID, QUESTION_ID, mastery=0.5, attempt_count=0, correct_streak=0)
    monkeypatch.setattr(study, "get_conn", db.connect)

    def submit():
        study.record_attempt(
            user_id=USER_ID,
            question_id=QUESTION_ID,
            answer_text="a",
            verdict="correct",
            score=0.9,
            missed_points=[],
            feedback="",
            explanation="",
            identified_gap=None,
        )

    _run_concurrently(submit, 2)

    state = db.state_for(USER_ID, QUESTION_ID)
    assert len(db.attempts) == 2
    assert state["attempt_count"] == 2, (
        f"lost update: {len(db.attempts)} attempts but attempt_count={state['attempt_count']}"
    )
    assert state["correct_streak"] == 2
    assert state["mastery"] == pytest.approx(update_mastery(update_mastery(0.5, 0.9), 0.9))


def test_replayed_submission_returns_stored_result_and_flag(api):
    client, db, _ = api
    body = {"question_id": QUESTION_ID, "answer": "an answer", "submission_id": SUBMISSION_ID}

    first = client.post("/grade", json=body).json()
    second = client.post("/grade", json=body).json()

    assert "replayed" not in first
    assert second["replayed"] is True
    for key in ("verdict", "score", "feedback", "explanation", "attempt_count", "correct_streak"):
        assert second[key] == first[key], key


def test_submission_id_reused_for_another_question_is_400(api, monkeypatch):
    client, db, grader = api
    client.post("/grade", json={"question_id": QUESTION_ID, "answer": "a", "submission_id": SUBMISSION_ID})
    other_question = "99999999-9999-9999-9999-999999999999"
    monkeypatch.setattr(study, "load_question_for_grading", lambda **_: fake_question(other_question))

    response = client.post("/grade", json={"question_id": other_question, "answer": "a", "submission_id": SUBMISSION_ID})

    assert response.status_code == 400
    assert len(db.attempts) == 1
    assert grader.calls == 1


def test_malformed_submission_id_is_422(api):
    client, db, grader = api

    response = client.post("/grade", json={"question_id": QUESTION_ID, "answer": "a", "submission_id": "retry-1"})

    assert response.status_code == 422
    assert grader.calls == 0


def test_concurrent_retries_of_one_submission_record_one_attempt(monkeypatch):
    """Two in-flight requests with the same submission_id both miss the pre-check;
    the per-question lock makes the second one replay instead of inserting."""

    db = FakeDB(interleave_reads=2)
    monkeypatch.setattr(study, "get_conn", db.connect)
    results = []

    def submit():
        results.append(
            study.record_attempt(
                user_id=USER_ID,
                question_id=QUESTION_ID,
                answer_text="a",
                verdict="correct",
                score=0.9,
                missed_points=[],
                feedback="",
                explanation="",
                identified_gap=None,
                submission_id=SUBMISSION_ID,
            )
        )

    _run_concurrently(submit, 2)

    assert len(db.attempts) == 1
    assert db.state_for(USER_ID, QUESTION_ID)["attempt_count"] == 1
    assert sorted(bool(r.get("replayed")) for r in results) == [False, True]
    assert results[0]["attempt_id"] == results[1]["attempt_id"]


def test_unique_violation_backstop_is_503(api, monkeypatch):
    """If two different questions race with one submission_id, the unique index
    rejects the second insert; the API must answer 503, not 500."""

    client, db, _ = api
    client.post("/grade", json={"question_id": QUESTION_ID, "answer": "a", "submission_id": SUBMISSION_ID})
    # Simulate a true race: both the pre-check and the in-lock check miss
    # (they run under different per-question locks), so only the index can stop it.
    monkeypatch.setattr(study, "_load_submission", lambda cur, **_: None)
    other_question = "99999999-9999-9999-9999-999999999999"
    monkeypatch.setattr(study, "load_question_for_grading", lambda **_: fake_question(other_question))
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post("/grade", json={"question_id": other_question, "answer": "a", "submission_id": SUBMISSION_ID})

    assert response.status_code == 503
    assert len(db.attempts) == 1


def test_submission_id_is_scoped_per_user(monkeypatch):
    """Another user sending the same submission_id must get their own attempt,
    never a replay of someone else's grade."""

    from fakes import OTHER_USER_ID

    db = FakeDB()
    monkeypatch.setattr(study, "get_conn", db.connect)
    kwargs = dict(
        question_id=QUESTION_ID, answer_text="a", verdict="correct", score=0.9,
        missed_points=[], feedback="", explanation="", identified_gap=None, submission_id=SUBMISSION_ID,
    )

    mine = study.record_attempt(user_id=USER_ID, **kwargs)
    theirs = study.record_attempt(user_id=OTHER_USER_ID, **kwargs)

    assert "replayed" not in theirs
    assert theirs["attempt_id"] != mine["attempt_id"]
    assert study.find_submission(user_id=OTHER_USER_ID, submission_id=SUBMISSION_ID)["attempt_id"] == theirs["attempt_id"]
    assert len(db.attempts) == 2

"""Progress dashboard and question bank: pure logic, data mapping and API behaviour.
Real-SQL behaviour is covered in tests/test_postgres_features.py (opt-in)."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import adaptive
import main
import progress
import questions
import study
from fakes import COURSE_ID, OTHER_USER_ID, QUESTION_ID, USER_ID, ScriptedDB

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)
TOPIC_ID = "66666666-6666-6666-6666-666666666666"


# ── mastery bands reuse the grading thresholds ──────────────────────

@pytest.mark.parametrize(
    "mastery, band",
    [(None, "not_practiced"), (0.0, "needs_review"), (0.3499, "needs_review"), (0.35, "developing"),
     (0.8499, "developing"), (0.85, "mastered"), (1.0, "mastered")],
)
def test_mastery_band_boundaries(mastery, band):
    assert adaptive.mastery_band(mastery) == band


@pytest.mark.parametrize("value", [0.0, 0.3499, 0.35, 0.8499, 0.85, 1.0])
def test_bands_and_verdicts_share_thresholds(value):
    expected = {"incorrect": "needs_review", "partially_correct": "developing", "correct": "mastered"}
    assert adaptive.mastery_band(value) == expected[study.verdict_from_score(value)]


def test_zero_mastery_is_low_not_unpractised():
    """A genuine 0.0 must never be confused with 'no data'."""

    stats = progress._stats(questions_answered=1, attempts=1, average_score=0.0, correct=0, mastery=0.0)

    assert stats["mastery"] == 0.0
    assert stats["mastery_band"] == "needs_review"
    assert stats["average_score"] == 0.0


def test_no_data_reports_none_instead_of_neutral_default():
    stats = progress._stats()

    assert stats == {
        "questions_in_bank": 0, "questions_answered": 0, "attempts": 0, "average_score": None,
        "correct_rate": None, "mastery": None, "mastery_band": "not_practiced", "low_data": True,
    }


def test_low_data_flag_until_three_attempts():
    assert progress._stats(questions_answered=1, attempts=2, mastery=0.7)["low_data"] is True
    assert progress._stats(questions_answered=1, attempts=3, mastery=0.7)["low_data"] is False


def test_progress_overview_scopes_every_query_to_user(monkeypatch):
    fake = ScriptedDB(
        [(COURSE_ID, 4)],
        [(COURSE_ID, 2, 0.6)],
        [(COURSE_ID, 5, 0.7, 3, NOW, 2)],
    )
    monkeypatch.setattr(progress, "get_conn", fake.connect)

    [summary] = progress.progress_overview(USER_ID, [{"id": COURSE_ID, "name": "Bio"}])

    assert summary["questions_in_bank"] == 4
    assert summary["mastery"] == 0.6 and summary["mastery_band"] == "developing"
    assert summary["correct_rate"] == pytest.approx(0.6)
    assert summary["attempts_last_7_days"] == 2
    assert all(params[0] == USER_ID for _, params in fake.executed)


def test_course_with_no_activity_is_not_practiced(monkeypatch):
    monkeypatch.setattr(progress, "get_conn", ScriptedDB([], [], []).connect)

    [summary] = progress.progress_overview(USER_ID, [{"id": COURSE_ID, "name": "Bio"}])

    assert summary["mastery"] is None
    assert summary["average_score"] is None
    assert summary["last_activity_at"] is None
    assert summary["mastery_band"] == "not_practiced"


# ── question bank mapping ───────────────────────────────────────────

def test_question_bank_dedupes_and_handles_missing_history(monkeypatch):
    row = (QUESTION_ID, "Q?", "  A  ", ["k1", "  ", 3], "", NOW, TOPIC_ID, "water", None, None, None)
    legacy = ("99999999-9999-9999-9999-999999999999", "Old?", None, None, None, NOW, None, None, 2, 0.4, NOW)
    fake = ScriptedDB([row, row, legacy])
    monkeypatch.setattr(questions, "get_conn", fake.connect)

    bank = questions.list_question_bank(USER_ID, COURSE_ID)

    assert [b["id"] for b in bank] == [QUESTION_ID, legacy[0]]
    assert bank[0]["reference_answer"] == "A"
    assert bank[0]["key_points"] == ["k1"]
    assert bank[0]["concept_label"] is None
    assert bank[0]["attempt_count"] == 0
    assert bank[1]["reference_answer"] is None and bank[1]["key_points"] == [] and bank[1]["topic"] is None
    assert fake.params(0) == (USER_ID, COURSE_ID, questions.QUESTION_BANK_LIMIT)
    assert "EXISTS" in fake.sql(0)  # live-source filter


def test_question_bank_never_calls_the_llm(monkeypatch):
    def no_llm(*args, **kwargs):
        raise AssertionError("question bank must not generate anything")

    monkeypatch.setattr(study, "_call_structured", no_llm)
    monkeypatch.setattr(questions, "get_conn", ScriptedDB([]).connect)

    assert questions.list_question_bank(USER_ID, COURSE_ID) == []


# ── API ─────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    owners = {COURSE_ID: USER_ID}
    current = {"user": USER_ID}
    monkeypatch.setattr(main, "user_owns_course", lambda u, c: owners.get(str(c)) == str(u))
    monkeypatch.setattr(
        main,
        "list_courses",
        lambda u: [{"id": COURSE_ID, "name": "Bio", "created_at": NOW}] if str(u) == USER_ID else [],
    )
    main.app.dependency_overrides[main.current_user] = lambda: current["user"]
    try:
        yield TestClient(main.app), current
    finally:
        main.app.dependency_overrides.clear()


def _detail(course):
    return {
        **progress._course_summary(course, {"questions_answered": 1, "attempts": 1, "mastery": 0.3, "average_score": 0.0}),
        "topics": [{"topic_id": TOPIC_ID, "name": "water", **progress._stats(questions_answered=1, attempts=1, mastery=0.3), "last_attempt_at": NOW}],
        "focus_topics": ["water"],
        "not_practiced_topics": [],
        "recent_activity": [{"attempted_at": NOW, "verdict": "incorrect", "score": 0.0, "question": None, "topic": "water", "source_removed": True}],
    }


def test_progress_overview_endpoint(client, monkeypatch):
    http, _ = client
    seen = {}

    def fake_overview(user_id, courses):
        seen["args"] = (user_id, courses)
        return [progress._course_summary(courses[0], {})]

    monkeypatch.setattr(main, "progress_overview", fake_overview)

    body = http.get("/progress").json()

    assert body[0]["name"] == "Bio" and body[0]["mastery"] is None
    assert seen["args"][0] == USER_ID


def test_course_progress_endpoint_returns_topics(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(main, "course_progress", lambda user_id, course: _detail(course))

    body = http.get(f"/courses/{COURSE_ID}/progress").json()

    assert body["topics"][0]["mastery_band"] == "needs_review"
    assert body["recent_activity"][0]["source_removed"] is True


def test_course_progress_for_another_users_course_is_404(client, monkeypatch):
    http, current = client
    current["user"] = OTHER_USER_ID

    def must_not_run(*args):
        raise AssertionError("progress computed for a course the user doesn't own")

    monkeypatch.setattr(main, "course_progress", must_not_run)

    assert http.get(f"/courses/{COURSE_ID}/progress").status_code == 404


def test_question_bank_endpoint(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(
        main,
        "list_question_bank",
        lambda u, c: [{"id": QUESTION_ID, "prompt": "Q?", "reference_answer": "A", "key_points": ["k"],
                       "concept_label": None, "created_at": NOW, "topic_id": TOPIC_ID, "topic": "water",
                       "attempt_count": 2, "mastery": 0.5, "last_attempt_at": NOW}],
    )

    body = http.get("/question-bank", params={"course_id": COURSE_ID}).json()

    assert body[0]["reference_answer"] == "A"
    assert body[0]["topic"] == "water"


def test_question_bank_for_another_users_course_is_404(client, monkeypatch):
    http, current = client
    current["user"] = OTHER_USER_ID
    monkeypatch.setattr(main, "list_question_bank", lambda u, c: pytest.fail("bank read for non-owner"))

    assert http.get("/question-bank", params={"course_id": COURSE_ID}).status_code == 404


@pytest.mark.parametrize(
    "path, params",
    [("/progress", None), (f"/courses/{COURSE_ID}/progress", None), ("/question-bank", {"course_id": COURSE_ID})],
)
def test_new_endpoints_require_auth(monkeypatch, path, params):
    monkeypatch.setattr(main, "API_SHARED_SECRET", "secret")

    assert TestClient(main.app).get(path, params=params).status_code == 401


def test_question_bank_requires_valid_course_id(client):
    http, _ = client

    assert http.get("/question-bank").status_code == 422
    assert http.get("/question-bank", params={"course_id": "nope"}).status_code == 422


def test_course_progress_assembles_topics_focus_and_recent_activity(monkeypatch):
    weak, strong, fresh = ("t-weak", "Weak"), ("t-strong", "Strong"), ("t-fresh", "Fresh")
    fake = ScriptedDB(
        [(COURSE_ID, 3)],                                   # questions in bank
        [(COURSE_ID, 2, 0.55)],                             # answered + mastery
        [(COURSE_ID, 4, 0.5, 2, NOW, 4)],                   # attempts
        [(*strong, 1, 1, 0.9, NOW), (*fresh, 1, 0, None, None), (*weak, 1, 1, 0.2, NOW)],
        [("t-strong", 3, 0.9, 3), ("t-weak", 1, 0.0, 0)],  # per-topic attempts
        [(NOW, "incorrect", 0.0, "Secret", "Weak", False), (NOW, "correct", 1.0, "Visible", "Strong", True)],
    )
    monkeypatch.setattr(progress, "get_conn", fake.connect)

    detail = progress.course_progress(USER_ID, {"id": COURSE_ID, "name": "Bio"})

    assert [t["name"] for t in detail["topics"]] == ["Weak", "Strong", "Fresh"]
    assert detail["focus_topics"] == ["Weak"]
    assert detail["not_practiced_topics"] == ["Fresh"]
    fresh_topic = detail["topics"][2]
    assert fresh_topic["mastery"] is None and fresh_topic["attempts"] == 0 and fresh_topic["average_score"] is None
    assert detail["topics"][1]["correct_rate"] == 1.0
    assert [(r["question"], r["source_removed"]) for r in detail["recent_activity"]] == [(None, True), ("Visible", False)]
    assert fake.params(3) == (USER_ID, COURSE_ID, USER_ID, COURSE_ID)
    assert fake.params(5) == (USER_ID, COURSE_ID, progress.RECENT_ACTIVITY_LIMIT)
    assert all(USER_ID in params for _, params in fake.executed)

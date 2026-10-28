"""LLM and database failure handling. External services are faked."""

from types import SimpleNamespace

import openai
import psycopg
import pytest
from fastapi.testclient import TestClient

import main
import study
from fakes import COURSE_ID, QUESTION_ID, USER_ID, fake_question

FAKE_DSN_SECRET = "SuperSecretDbPassword123"


class FakeCompletions:
    """Stands in for client.chat.completions; plays back a scripted list of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=outcome))], usage=None)


def fake_client(monkeypatch, outcomes):
    completions = FakeCompletions(outcomes)
    monkeypatch.setattr(study, "_client", SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return completions


def length_error():
    return openai.LengthFinishReasonError(completion=SimpleNamespace(usage=None))


GOOD_DRAFT = study.GradeDraft(sufficient_material=True, score=0.5)


# ── study._call_structured retry behaviour ──────────────────────────

def test_call_structured_returns_first_success(monkeypatch):
    completions = fake_client(monkeypatch, [GOOD_DRAFT])

    assert study._call_structured("p", study.GradeDraft) is GOOD_DRAFT
    assert completions.calls == 1


@pytest.mark.parametrize(
    "first_failure",
    [None, ValueError("bad json")],
    ids=["parsed_none", "value_error"],
)
def test_call_structured_retries_once_then_succeeds(monkeypatch, first_failure):
    completions = fake_client(monkeypatch, [first_failure, GOOD_DRAFT])

    assert study._call_structured("p", study.GradeDraft) is GOOD_DRAFT
    assert completions.calls == 2


def test_call_structured_gives_up_after_two_failures(monkeypatch):
    completions = fake_client(monkeypatch, [None, ValueError("still bad")])

    with pytest.raises(study.StudyModelError):
        study._call_structured("p", study.GradeDraft)
    assert completions.calls == 2


@pytest.mark.regression
def test_call_structured_handles_truncated_model_output(monkeypatch):
    """BUG: openai.LengthFinishReasonError subclasses OpenAIError, not APIError, so it
    escapes the retry loop (`except (ValueError, APIError)`) and is never converted
    into StudyModelError. The endpoints then return a raw 500 instead of 503."""

    fake_client(monkeypatch, [length_error(), length_error()])

    with pytest.raises(study.StudyModelError):
        study._call_structured("p", study.GradeDraft)


# ── Endpoint error mapping ──────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    main.app.dependency_overrides[main.current_user] = lambda: USER_ID
    monkeypatch.setattr(main, "user_owns_course", lambda user_id, course_id: True)
    try:
        yield TestClient(main.app, raise_server_exceptions=False)
    finally:
        main.app.dependency_overrides.clear()


def test_grade_model_failure_is_503_with_generic_message(monkeypatch, client):
    monkeypatch.setattr(study, "load_question_for_grading", lambda **_: fake_question())

    def model_down(prompt, schema):
        raise study.StudyModelError("upstream detail that must not leak")

    monkeypatch.setattr(study, "_call_structured", model_down)

    response = client.post("/grade", json={"question_id": QUESTION_ID, "answer": "x"})

    assert response.status_code == 503
    assert "upstream detail" not in response.text


def test_quiz_model_failure_is_503(monkeypatch, client):
    def model_down(**kwargs):
        raise study.StudyModelError("boom")

    monkeypatch.setattr(main, "generate_quiz_question", model_down)

    assert client.post("/quiz", json={"course_id": COURSE_ID, "topic": "t"}).status_code == 503


@pytest.mark.regression
def test_ask_truncated_model_output_is_503(monkeypatch, client):
    """BUG: /ask catches (ValidationError, ValueError, APIError); LengthFinishReasonError
    is none of these, so a truncated answer becomes a 500."""

    monkeypatch.setattr(main, "retrieve_chunks", lambda **_: {"chunks": []})

    def truncated(prompt, model):
        raise length_error()

    monkeypatch.setattr(main, "call_model_structured", truncated)

    response = client.post("/ask", json={"course_id": COURSE_ID, "question": "q"})

    assert response.status_code == 503


def _db_outage(*args, **kwargs):
    raise psycopg.OperationalError(
        f'connection to server at "db.example.com" failed: password={FAKE_DSN_SECRET} rejected'
    )


def _patch_db_outage(monkeypatch, endpoint):
    if endpoint == "list_courses":
        monkeypatch.setattr(main, "list_courses", _db_outage)
        return lambda c: c.get("/courses")
    monkeypatch.setattr(study, "load_question_for_grading", _db_outage)
    return lambda c: c.post("/grade", json={"question_id": QUESTION_ID, "answer": "x"})


@pytest.mark.parametrize("endpoint", ["list_courses", "grade"])
def test_database_error_does_not_leak_details(monkeypatch, client, endpoint):
    response = _patch_db_outage(monkeypatch, endpoint)(client)

    assert FAKE_DSN_SECRET not in response.text
    assert "db.example.com" not in response.text


@pytest.mark.regression
@pytest.mark.parametrize("endpoint", ["list_courses", "grade"])
def test_database_outage_is_503(monkeypatch, client, endpoint):
    """BUG: psycopg.Error is not caught anywhere (endpoints only catch RuntimeError and
    domain errors), so a database outage surfaces as an unhandled 500 instead of the
    API's 'temporarily unavailable' 503."""

    response = _patch_db_outage(monkeypatch, endpoint)(client)

    assert response.status_code == 503

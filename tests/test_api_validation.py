"""FastAPI request validation. Auth is overridden; downstream services are stubbed
so a test fails loudly if an invalid request reaches them."""

import pytest
from fastapi.testclient import TestClient

import main
import study
from fakes import COURSE_ID, QUESTION_ID, USER_ID, FakeGrader


class Spy:
    def __init__(self, name, result=None):
        self.name = name
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def client(monkeypatch):
    main.app.dependency_overrides[main.current_user] = lambda: USER_ID
    monkeypatch.setattr(main, "user_owns_course", lambda user_id, course_id: True)
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.clear()


@pytest.fixture
def grader(monkeypatch):
    fake = FakeGrader()
    monkeypatch.setattr(study, "_call_structured", fake)

    def no_load(**_):
        raise AssertionError("invalid request reached load_question_for_grading")

    monkeypatch.setattr(study, "load_question_for_grading", no_load)
    return fake


@pytest.fixture
def downstream(monkeypatch):
    spies = {
        "generate_quiz_question": Spy("quiz", {"status": "ok"}),
        "ingest_document": Spy("ingest"),
        "create_course": Spy("create_course"),
        "retrieve_chunks": Spy("retrieve"),
    }
    for name, spy in spies.items():
        monkeypatch.setattr(main, name, spy)
    return spies


# ── /grade ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "body",
    [
        {},
        {"answer": "x"},
        {"question_id": QUESTION_ID},
        {"question_id": "not-a-uuid", "answer": "x"},
        {"question_id": QUESTION_ID, "answer": ""},
        {"question_id": QUESTION_ID, "answer": None},
        {"question_id": QUESTION_ID, "answer": 123},
    ],
    ids=["empty", "no_question_id", "no_answer", "bad_uuid", "empty_answer", "null_answer", "number_answer"],
)
def test_grade_rejects_malformed_body(client, grader, body):
    assert client.post("/grade", json=body).status_code == 422
    assert grader.calls == 0


@pytest.mark.parametrize("answer", ["   ", "\n\t"], ids=["spaces", "whitespace"])
def test_grade_whitespace_answer_is_400_without_llm_call(client, grader, answer):
    response = client.post("/grade", json={"question_id": QUESTION_ID, "answer": answer})

    assert response.status_code == 400
    assert grader.calls == 0


def test_grade_rejects_non_json_body(client, grader):
    response = client.post("/grade", content=b"answer=hi", headers={"Content-Type": "text/plain"})

    assert response.status_code == 422
    assert grader.calls == 0


@pytest.mark.regression
def test_grade_rejects_oversized_answer(client, grader):
    """BUG: GradeRequest.answer has no max_length, so an arbitrarily large answer
    is sent to the LLM (unbounded cost). /ask already caps input at 1000 chars."""

    response = client.post("/grade", json={"question_id": QUESTION_ID, "answer": "a" * 100_000})

    assert response.status_code == 422
    assert grader.calls == 0


# ── /quiz ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "body",
    [
        {"course_id": COURSE_ID},
        {"topic": "water"},
        {"course_id": "nope", "topic": "water"},
        {"course_id": COURSE_ID, "topic": ""},
    ],
    ids=["no_topic", "no_course", "bad_uuid", "empty_topic"],
)
def test_quiz_rejects_malformed_body(client, downstream, body):
    assert client.post("/quiz", json=body).status_code == 422
    assert downstream["generate_quiz_question"].calls == []


def test_quiz_whitespace_topic_is_400(client, monkeypatch):
    def no_topic_write(*args, **kwargs):
        raise AssertionError("blank topic reached the database")

    monkeypatch.setattr(study, "get_or_create_topic", no_topic_write)

    response = client.post("/quiz", json={"course_id": COURSE_ID, "topic": "   "})

    assert response.status_code == 400


@pytest.mark.regression
def test_quiz_rejects_oversized_topic(client, downstream):
    """BUG: QuizRequest.topic has no max_length; it is stored as a topic row and
    sent to the embedding model and the LLM."""

    response = client.post("/quiz", json={"course_id": COURSE_ID, "topic": "t" * 10_000})

    assert response.status_code == 422
    assert downstream["generate_quiz_question"].calls == []


# ── /ask ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "body",
    [
        {"course_id": COURSE_ID, "question": ""},
        {"course_id": COURSE_ID, "question": "q" * 1001},
        {"course_id": "nope", "question": "q"},
        {"question": "q"},
    ],
    ids=["empty", "too_long", "bad_uuid", "no_course"],
)
def test_ask_rejects_malformed_body(client, downstream, body):
    assert client.post("/ask", json=body).status_code == 422
    assert downstream["retrieve_chunks"].calls == []


def test_ask_accepts_question_at_max_length(client, monkeypatch):
    def fake_retrieve(**kwargs):
        return {"chunks": []}

    def fake_model(prompt, model):
        return main.Answer(answer="ok", confidence=0.1, sources_needed=True), 10, 6, 4

    monkeypatch.setattr(main, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(main, "call_model_structured", fake_model)

    response = client.post("/ask", json={"course_id": COURSE_ID, "question": "q" * 1000})

    assert response.status_code == 200
    assert response.json()["retrieved_chunk_ids"] == []


# ── /courses and /ingest ────────────────────────────────────────────

def test_create_course_rejects_empty_name(client, downstream):
    assert client.post("/courses", json={"name": ""}).status_code == 422
    assert downstream["create_course"].calls == []


def test_create_course_whitespace_name_is_400(client, monkeypatch):
    import courses

    monkeypatch.setattr(main, "create_course", courses.create_course)

    assert client.post("/courses", json={"name": "   "}).status_code == 400


@pytest.mark.parametrize(
    "body",
    [
        {"course_id": COURSE_ID, "filename": "", "text": "t"},
        {"course_id": COURSE_ID, "filename": "a.txt", "text": ""},
        {"course_id": "nope", "filename": "a.txt", "text": "t"},
    ],
    ids=["empty_filename", "empty_text", "bad_uuid"],
)
def test_ingest_rejects_malformed_body(client, downstream, body):
    assert client.post("/ingest", json=body).status_code == 422
    assert downstream["ingest_document"].calls == []


@pytest.mark.parametrize(
    "path",
    [f"/courses/not-a-uuid", "/documents/not-a-uuid"],
)
def test_delete_with_bad_uuid_is_422(client, path):
    assert client.delete(path).status_code == 422


@pytest.mark.parametrize("path", ["/documents", "/questions"])
def test_list_endpoints_require_valid_course_id(client, path):
    assert client.get(path).status_code == 422
    assert client.get(path, params={"course_id": "nope"}).status_code == 422

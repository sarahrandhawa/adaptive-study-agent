"""FastAPI endpoint behaviour (success and error mapping) with data access stubbed."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from fakes import COURSE_ID, USER_ID

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)
DOC_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture
def client(monkeypatch):
    main.app.dependency_overrides[main.current_user] = lambda: USER_ID
    monkeypatch.setattr(main, "user_owns_course", lambda user_id, course_id: True)
    try:
        yield TestClient(main.app, raise_server_exceptions=False)
    finally:
        main.app.dependency_overrides.clear()


def test_create_course_returns_created_course(monkeypatch, client):
    monkeypatch.setattr(main, "create_course", lambda user_id, name: COURSE_ID)
    monkeypatch.setattr(main, "list_courses", lambda user_id: [{"id": COURSE_ID, "name": "Bio", "created_at": NOW}])

    response = client.post("/courses", json={"name": "Bio"})

    assert response.status_code == 200
    assert response.json()["id"] == COURSE_ID


def test_create_course_not_found_after_insert_is_500(monkeypatch, client):
    monkeypatch.setattr(main, "create_course", lambda user_id, name: COURSE_ID)
    monkeypatch.setattr(main, "list_courses", lambda user_id: [])

    assert client.post("/courses", json={"name": "Bio"}).status_code == 500


@pytest.mark.parametrize("archived, status", [(True, 200), (False, 404)])
def test_delete_course(monkeypatch, client, archived, status):
    monkeypatch.setattr(main, "archive_course", lambda user_id, course_id: archived)

    assert client.delete(f"/courses/{COURSE_ID}").status_code == status


def test_ingest_success(monkeypatch, client):
    monkeypatch.setattr(main, "ingest_document", lambda **kw: {"document_id": DOC_ID, "chunks_indexed": 2, "status": "indexed"})

    response = client.post("/ingest", json={"course_id": COURSE_ID, "filename": "a.txt", "text": "t"})

    assert response.status_code == 200
    assert response.json() == {"document_id": DOC_ID, "chunks_indexed": 2, "status": "indexed"}


@pytest.mark.parametrize(
    "error, status",
    [
        (main.IngestValidationError("Course not found"), 400),
        (main.VectorStoreConfigError("Missing PINECONE_API_KEY"), 503),
        (RuntimeError("pinecone down"), 503),
    ],
    ids=["validation", "config", "runtime"],
)
def test_ingest_error_mapping(monkeypatch, client, error, status):
    def fail(**kw):
        raise error

    monkeypatch.setattr(main, "ingest_document", fail)

    response = client.post("/ingest", json={"course_id": COURSE_ID, "filename": "a.txt", "text": "t"})

    assert response.status_code == status
    assert "PINECONE_API_KEY" not in response.text


def test_list_documents_and_questions(monkeypatch, client):
    monkeypatch.setattr(main, "list_documents", lambda u, c: [{"id": DOC_ID, "filename": "a.txt", "char_count": 1, "chunk_count": 1, "created_at": NOW}])
    monkeypatch.setattr(main, "list_questions", lambda u, c: [])

    assert client.get("/documents", params={"course_id": COURSE_ID}).json()[0]["filename"] == "a.txt"
    assert client.get("/questions", params={"course_id": COURSE_ID}).json() == []


@pytest.mark.parametrize(
    "outcome, status",
    [(True, 200), (False, 404), (RuntimeError("pinecone down"), 503)],
    ids=["deleted", "not_found", "vector_error"],
)
def test_delete_document(monkeypatch, client, outcome, status):
    def delete(user_id, document_id):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(main, "delete_document", delete)

    assert client.delete(f"/documents/{DOC_ID}").status_code == status


def test_ask_success_reports_cost_and_chunk_ids(monkeypatch, client):
    monkeypatch.setattr(main, "retrieve_chunks", lambda **kw: {"chunks": [{"chunk_id": "c1", "source": "a.txt", "text": "t"}]})
    monkeypatch.setattr(
        main,
        "call_model_structured",
        lambda prompt, model: (main.Answer(answer="ok", confidence=0.9, sources_needed=False), 3000, 2000, 1000),
    )

    body = client.post("/ask", json={"course_id": COURSE_ID, "question": "q"}).json()

    assert body["retrieved_chunk_ids"] == ["c1"]
    assert body["tokens_used"] == 3000
    assert body["cost_usd"] == pytest.approx(main.compute_cost_usd(main.CHAT_MODEL, 2000, 1000), abs=1e-6)


def test_ask_retrieval_validation_error_is_400(monkeypatch, client):
    def bad(**kw):
        raise main.RetrieveValidationError("query must not be empty")

    monkeypatch.setattr(main, "retrieve_chunks", bad)

    assert client.post("/ask", json={"course_id": COURSE_ID, "question": "q"}).status_code == 400


def test_compute_cost_uses_per_1k_prices():
    assert main.compute_cost_usd("gpt-4o", 1000, 1000) == pytest.approx(0.0025 + 0.01)


def test_call_model_structured_parses_usage(monkeypatch):
    answer = main.Answer(answer="a", confidence=0.5, sources_needed=False)
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=answer))],
        usage=SimpleNamespace(total_tokens=30, prompt_tokens=20, completion_tokens=10),
    )
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw: completion)))
    monkeypatch.setattr(main, "client", fake)

    assert main.call_model_structured("p", "gpt-5-mini") == (answer, 30, 20, 10)


def test_call_model_structured_rejects_unparsed_output(monkeypatch):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))], usage=None)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **kw: completion)))
    monkeypatch.setattr(main, "client", fake)

    with pytest.raises(ValueError):
        main.call_model_structured("p", "gpt-5-mini")

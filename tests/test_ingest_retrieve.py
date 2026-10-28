"""Ingest, retrieval and vector-store helpers with Pinecone/OpenAI replaced by fakes."""

from types import SimpleNamespace

import pytest

import ingest
import retrieve
import vectorstore
from fakes import COURSE_ID, USER_ID, ScriptedDB

DOC_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
OLD_DOC_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class FakeVectorStore:
    def __init__(self, results=None, fail=None):
        self.added = []
        self.queries = []
        self.results = results or []
        self.fail = fail

    def add_documents(self, *, documents, ids):
        if self.fail:
            raise self.fail
        self.added.append((documents, ids))

    def similarity_search_with_score(self, query, *, k, filter):
        self.queries.append({"query": query, "k": k, "filter": filter})
        return self.results


@pytest.fixture
def vectors(monkeypatch):
    store = FakeVectorStore()
    namespaces = []
    deleted = []

    def get_store(namespace=None):
        namespaces.append(namespace)
        return store

    monkeypatch.setattr(ingest, "get_vector_store", get_store)
    monkeypatch.setattr(ingest, "delete_vectors", lambda **kw: deleted.append(kw))
    return SimpleNamespace(store=store, namespaces=namespaces, deleted=deleted)


def run_ingest(text="Water evaporates when heated.", filename="notes.txt"):
    return ingest.ingest_document(user_id=USER_ID, course_id=COURSE_ID, filename=filename, text=text)


# ── ingest_document ─────────────────────────────────────────────────

@pytest.mark.parametrize("filename, text", [("  ", "text"), ("a.txt", "  \n ")])
def test_ingest_rejects_blank_input_before_db(monkeypatch, filename, text):
    fake = ScriptedDB()
    monkeypatch.setattr(ingest, "get_conn", fake.connect)

    with pytest.raises(ingest.IngestValidationError):
        run_ingest(text=text, filename=filename)
    assert fake.executed == []


def test_ingest_rejects_course_not_owned(monkeypatch, vectors):
    fake = ScriptedDB([])
    monkeypatch.setattr(ingest, "get_conn", fake.connect)

    with pytest.raises(ingest.IngestValidationError, match="Course not found"):
        run_ingest()
    assert fake.params(0) == (COURSE_ID, USER_ID)
    assert vectors.store.added == []


def test_ingest_identical_file_is_unchanged(monkeypatch, vectors):
    fake = ScriptedDB([(COURSE_ID,)], [(DOC_ID, 3)])
    monkeypatch.setattr(ingest, "get_conn", fake.connect)

    assert run_ingest() == {"document_id": DOC_ID, "chunks_indexed": 3, "status": "unchanged"}
    assert vectors.store.added == []


def test_ingest_new_file_indexes_chunks_in_user_namespace(monkeypatch, vectors):
    # course ok, no duplicate, no older version, insert document, insert 1 chunk
    fake = ScriptedDB([(COURSE_ID,)], [], [], [(DOC_ID,)], [("chunk-1",)])
    monkeypatch.setattr(ingest, "get_conn", fake.connect)

    result = run_ingest()

    assert result == {"document_id": DOC_ID, "chunks_indexed": 1, "status": "indexed"}
    assert vectors.namespaces == [USER_ID]
    documents, ids = vectors.store.added[0]
    assert ids == ["chunk-1"]
    assert documents[0].metadata == {
        "chunk_id": "chunk-1",
        "document_id": DOC_ID,
        "course_id": COURSE_ID,
        "chunk_index": 0,
        "source": "notes.txt",
    }
    assert vectors.deleted == []


def test_ingest_new_version_retires_old_vectors_after_upload(monkeypatch, vectors):
    fake = ScriptedDB(
        [(COURSE_ID,)], [], [(OLD_DOC_ID,)], [(DOC_ID,)], [("chunk-1",)],
        [("old-1",), ("old-2",)], [], [],
    )
    monkeypatch.setattr(ingest, "get_conn", fake.connect)

    assert run_ingest(text="Updated notes.")["status"] == "indexed"
    assert vectors.deleted == [{"namespace": USER_ID, "ids": ["old-1", "old-2"]}]
    assert fake.sql(6).startswith("UPDATE documents SET status = 'replaced'")
    assert fake.params(6) == (OLD_DOC_ID,)


def test_ingest_vector_config_error_becomes_runtime_error_and_rolls_back(monkeypatch, vectors):
    vectors.store.fail = vectorstore.VectorStoreConfigError("Missing PINECONE_API_KEY")
    fake = ScriptedDB([(COURSE_ID,)], [], [], [(DOC_ID,)], [("chunk-1",)])
    monkeypatch.setattr(ingest, "get_conn", fake.connect)

    with pytest.raises(RuntimeError):
        run_ingest()
    assert fake.committed is False


# ── retrieve_chunks ─────────────────────────────────────────────────

@pytest.mark.parametrize("query, k", [("  ", 5), ("q", 0)])
def test_retrieve_validates_input(query, k):
    with pytest.raises(retrieve.RetrieveValidationError):
        retrieve.retrieve_chunks(query=query, user_id=USER_ID, course_id=COURSE_ID, k=k)


def test_retrieve_scopes_to_user_namespace_and_course(monkeypatch):
    doc = SimpleNamespace(
        page_content="text",
        metadata={"chunk_id": "c1", "document_id": DOC_ID, "chunk_index": 2, "source": "notes.txt"},
    )
    store = FakeVectorStore(results=[(doc, 0.87654321)])
    namespaces = []
    monkeypatch.setattr(retrieve, "get_vector_store", lambda namespace=None: namespaces.append(namespace) or store)

    result = retrieve.retrieve_chunks(query="  evaporation ", user_id=USER_ID, course_id=COURSE_ID, k=3)

    assert namespaces == [USER_ID]
    assert store.queries == [{"query": "evaporation", "k": 3, "filter": {"course_id": COURSE_ID}}]
    assert result["chunks"] == [
        {"chunk_id": "c1", "score": 0.876543, "document_id": DOC_ID, "chunk_index": 2, "source": "notes.txt", "text": "text"}
    ]


def test_retrieve_config_error_becomes_runtime_error(monkeypatch):
    def broken(namespace=None):
        raise vectorstore.VectorStoreConfigError("Missing PINECONE_INDEX_NAME")

    monkeypatch.setattr(retrieve, "get_vector_store", broken)

    with pytest.raises(RuntimeError, match="PINECONE_INDEX_NAME"):
        retrieve.retrieve_chunks(query="q", user_id=USER_ID, course_id=COURSE_ID)


# ── vectorstore config helpers ──────────────────────────────────────

def test_require_env_missing_raises(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX_NAME", "  ")

    with pytest.raises(vectorstore.VectorStoreConfigError, match="PINECONE_INDEX_NAME"):
        vectorstore.get_index_name()


@pytest.mark.parametrize("value, expected", [("true", True), ("YES", True), ("1", True), ("false", False), ("", False)])
def test_auto_create_index_flag(monkeypatch, value, expected):
    monkeypatch.setenv("PINECONE_AUTO_CREATE_INDEX", value)

    assert vectorstore._auto_create_index_enabled() is expected


def test_embedding_model_default_and_override(monkeypatch):
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    assert vectorstore.get_embedding_model() == "text-embedding-3-small"

    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    assert vectorstore.get_embedding_model() == "text-embedding-3-large"


def test_jsonify_converts_sdk_objects():
    class WithToDict:
        def to_dict(self):
            return {"a": (1, 2)}

    assert vectorstore._jsonify({"x": WithToDict(), "y": None, 3: object}) == {
        "x": {"a": [1, 2]},
        "y": None,
        "3": str(object),
    }


def test_check_pinecone_reports_missing_config_without_connecting(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "")

    result = vectorstore.check_pinecone()

    assert result["ok"] is False
    assert "PINECONE_API_KEY" in result["error"]
    assert result["config"]["pinecone_api_key_set"] is False


def test_delete_vectors_with_no_ids_does_not_connect(monkeypatch):
    def no_index():
        raise AssertionError("Pinecone should not be contacted")

    monkeypatch.setattr(vectorstore, "get_pinecone_index", no_index)

    assert vectorstore.delete_vectors(namespace=USER_ID, ids=[]) is None

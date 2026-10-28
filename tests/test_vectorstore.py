"""vectorstore.py with the Pinecone SDK, embeddings and LangChain store replaced by fakes."""

import time
from types import SimpleNamespace

import pytest

import vectorstore


class FakeIndexList(list):
    def names(self):
        return [info["name"] for info in self]


class FakeIndex:
    def __init__(self):
        self.deleted = []

    def delete(self, *, ids, namespace):
        self.deleted.append({"ids": ids, "namespace": namespace})

    def describe_index_stats(self):
        return {"total_vector_count": 7, "namespaces": {"u1": {"vector_count": 7}}}


class FakeDescription:
    def __init__(self, ready):
        self.status = {"ready": ready}

    def to_dict(self):
        return {"status": self.status, "dimension": 1536, "metric": "cosine", "host": "idx.example"}


class FakePinecone:
    def __init__(self, existing=(), ready_after=0, fail=None):
        self.existing = list(existing)
        self.created = []
        self.ready_after = ready_after
        self.describe_calls = 0
        self.index = FakeIndex()
        self.fail = fail

    def list_indexes(self):
        if self.fail:
            raise self.fail
        return FakeIndexList({"name": name} for name in self.existing)

    def create_index(self, **kwargs):
        self.created.append(kwargs)
        self.existing.append(kwargs["name"])

    def describe_index(self, name):
        self.describe_calls += 1
        return FakeDescription(ready=self.describe_calls > self.ready_after)

    def Index(self, name):
        return self.index


@pytest.fixture
def pinecone(monkeypatch):
    """Configure env and install a fake Pinecone client; clears the module's lru caches."""

    monkeypatch.setenv("PINECONE_API_KEY", "pc-test-not-real")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "study-index")
    monkeypatch.setenv("PINECONE_AUTO_CREATE_INDEX", "true")
    fake = FakePinecone()
    clients = []

    def make_client(api_key):
        clients.append(api_key)
        return fake

    for cached in (vectorstore.get_pinecone_client, vectorstore.get_embeddings, vectorstore.get_vector_store):
        cached.cache_clear()
    monkeypatch.setattr(vectorstore, "Pinecone", make_client)
    # Module-local stand-in so the global time module is never patched.
    monkeypatch.setattr(vectorstore, "time", SimpleNamespace(time=time.time, sleep=lambda seconds: None))
    fake.client_api_keys = clients
    yield fake
    for cached in (vectorstore.get_pinecone_client, vectorstore.get_embeddings, vectorstore.get_vector_store):
        cached.cache_clear()


def test_client_is_built_once_from_env(pinecone):
    assert vectorstore.get_pinecone_client() is vectorstore.get_pinecone_client()
    assert pinecone.client_api_keys == ["pc-test-not-real"]


def test_missing_api_key_raises_config_error(monkeypatch):
    vectorstore.get_pinecone_client.cache_clear()
    monkeypatch.setenv("PINECONE_API_KEY", "")

    with pytest.raises(vectorstore.VectorStoreConfigError, match="PINECONE_API_KEY"):
        vectorstore.get_pinecone_client()


def test_ensure_index_noop_when_auto_create_disabled(pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_AUTO_CREATE_INDEX", "false")

    vectorstore.ensure_index_exists()

    assert pinecone.created == []
    assert pinecone.client_api_keys == []


def test_ensure_index_noop_when_index_exists(pinecone):
    pinecone.existing = ["study-index"]

    vectorstore.ensure_index_exists()

    assert pinecone.created == []


def test_ensure_index_creates_serverless_index_and_waits_until_ready(pinecone, monkeypatch):
    monkeypatch.setenv("PINECONE_REGION", "eu-west-1")
    pinecone.ready_after = 2

    vectorstore.ensure_index_exists()

    created = pinecone.created[0]
    assert created["name"] == "study-index"
    assert created["dimension"] == vectorstore.EMBEDDING_DIMENSION
    assert created["metric"] == "cosine"
    assert created["spec"].region == "eu-west-1"
    assert pinecone.describe_calls == 3


def test_ensure_index_times_out_if_never_ready(pinecone, monkeypatch):
    pinecone.ready_after = 10**9
    clock = iter(range(0, 1000, 30))
    vectorstore.time.time = lambda: next(clock)

    with pytest.raises(TimeoutError, match="study-index"):
        vectorstore.ensure_index_exists()


def test_get_vector_store_is_cached_per_namespace(pinecone, monkeypatch):
    pinecone.existing = ["study-index"]
    built = []

    class FakeStore:
        def __init__(self, *, index, embedding, namespace):
            built.append(namespace)
            self.index = index

    monkeypatch.setattr(vectorstore, "PineconeVectorStore", FakeStore)
    monkeypatch.setattr(vectorstore, "OpenAIEmbeddings", lambda model: SimpleNamespace(model=model))

    a1 = vectorstore.get_vector_store(namespace="user-a")
    a2 = vectorstore.get_vector_store(namespace="user-a")
    vectorstore.get_vector_store(namespace="user-b")

    assert a1 is a2
    assert built == ["user-a", "user-b"]
    assert a1.index is pinecone.index


def test_delete_vectors_targets_user_namespace(pinecone):
    pinecone.existing = ["study-index"]

    vectorstore.delete_vectors(namespace="user-a", ids=["c1", "c2"])

    assert pinecone.index.deleted == [{"ids": ["c1", "c2"], "namespace": "user-a"}]


# ── check_pinecone diagnostics ──────────────────────────────────────

def test_check_pinecone_ok_reports_index_and_stats(pinecone):
    pinecone.existing = ["study-index"]

    result = vectorstore.check_pinecone()

    assert result["ok"] is True
    assert result["index"] == {"name": "study-index", "ready": True, "dimension": 1536, "metric": "cosine", "host": "idx.example"}
    assert result["stats"]["total_vector_count"] == 7
    assert "pc-test-not-real" not in repr(result)


def test_check_pinecone_reports_missing_index(pinecone):
    pinecone.existing = ["other-index"]

    result = vectorstore.check_pinecone()

    assert result["ok"] is False
    assert "study-index" in result["error"]
    assert result["indexes"] == ["other-index"]


def test_check_pinecone_reports_connectivity_errors(pinecone):
    pinecone.fail = ConnectionError("dns failure")

    result = vectorstore.check_pinecone()

    assert result["ok"] is False
    assert result["error"] == "ConnectionError: dns failure"

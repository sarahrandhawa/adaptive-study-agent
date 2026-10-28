"""Deleted/stale content can't reach retrieval; deletion is DB-first, retryable and scoped.
Fast tests with scripted fakes; the same behaviour runs against real Postgres in
tests/test_postgres_features.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import documents
import retrieve
import sources
from fakes import COURSE_ID, USER_ID, ScriptedDB

DOC_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
LIVE = "11111111-0000-0000-0000-000000000001"
STALE = "11111111-0000-0000-0000-000000000002"
LIVE_2 = "11111111-0000-0000-0000-000000000003"


class Store:
    def __init__(self, ids):
        self.ids = ids
        self.k = None

    def similarity_search_with_score(self, query, *, k, filter):
        self.k = k
        return [(SimpleNamespace(page_content=i, metadata={"chunk_id": i}), 0.5) for i in self.ids][:k]


def retrieve_with(monkeypatch, pinecone_ids, live_rows, k=5):
    store = Store(pinecone_ids)
    db = ScriptedDB(live_rows)
    monkeypatch.setattr(retrieve, "get_vector_store", lambda namespace=None: store)
    monkeypatch.setattr(retrieve, "get_conn", db.connect)
    return retrieve.retrieve_chunks(query="q", user_id=USER_ID, course_id=COURSE_ID, k=k), db, store


def test_stale_vectors_are_dropped_and_order_preserved(monkeypatch):
    result, db, _ = retrieve_with(monkeypatch, [STALE, LIVE, LIVE_2], [(LIVE,), (LIVE_2,)])

    assert [c["chunk_id"] for c in result["chunks"]] == [LIVE, LIVE_2]
    assert "c.deleted_at IS NULL" in db.sql(0)
    assert "d.status = 'active'" in db.sql(0)
    assert "co.archived_at IS NULL" in db.sql(0)


def test_overfetch_still_returns_at_most_k(monkeypatch):
    ids = [f"11111111-0000-0000-0000-0000000000{n:02d}" for n in range(10)]
    result, _, store = retrieve_with(monkeypatch, ids, [(i,) for i in ids], k=3)

    assert store.k == 3 * retrieve.OVERFETCH_FACTOR
    assert len(result["chunks"]) == 3


def test_all_stale_means_no_material(monkeypatch):
    result, _, _ = retrieve_with(monkeypatch, [STALE], [])

    assert result["chunks"] == []


def test_no_pinecone_matches_skips_database(monkeypatch):
    result, db, _ = retrieve_with(monkeypatch, [], [])

    assert result["chunks"] == []
    assert db.executed == []


def test_live_chunk_ids_ignores_malformed_ids_without_querying():
    db = ScriptedDB()

    assert sources.live_chunk_ids(db, user_id=USER_ID, course_id=COURSE_ID, chunk_ids=[None, "c1", 7]) == set()
    assert db.executed == []


# ── deletion ────────────────────────────────────────────────────────

@pytest.fixture
def pinecone(monkeypatch):
    calls = SimpleNamespace(deleted=[], fail=None)

    def delete_vectors(*, namespace, ids):
        if calls.fail:
            raise calls.fail
        calls.deleted.append((namespace, ids))

    monkeypatch.setattr(documents, "delete_vectors", delete_vectors)
    return calls


def test_repeat_delete_of_purged_document_is_idempotent(monkeypatch, pinecone):
    fake = ScriptedDB([("deleted", "2026-10-28")])
    monkeypatch.setattr(documents, "get_conn", fake.connect)

    assert documents.delete_document(USER_ID, DOC_ID) is True
    assert len(fake.executed) == 1  # nothing re-marked
    assert pinecone.deleted == []


def test_repeat_delete_retries_pending_purge(monkeypatch, pinecone):
    fake = ScriptedDB([("deleted", None)], [("c1",)], [])
    monkeypatch.setattr(documents, "get_conn", fake.connect)

    assert documents.delete_document(USER_ID, DOC_ID) is True
    assert not any(sql.startswith("UPDATE chunks") for sql, _ in fake.executed)
    assert pinecone.deleted == [(USER_ID, ["c1"])]
    assert fake.sql(2).startswith("UPDATE documents SET vectors_deleted_at = now()")


def test_pinecone_failure_does_not_fail_or_undo_the_delete(monkeypatch, pinecone):
    pinecone.fail = ConnectionError("pinecone down")
    fake = ScriptedDB([("active", None)], [], [], [("c1",)])
    monkeypatch.setattr(documents, "get_conn", fake.connect)

    assert documents.delete_document(USER_ID, DOC_ID) is True
    assert any(sql.startswith("UPDATE documents SET status = 'deleted'") for sql, _ in fake.executed)
    assert not any("vectors_deleted_at = now()" in sql for sql, _ in fake.executed)  # stays pending


def test_purge_only_targets_non_active_documents_of_this_user(monkeypatch, pinecone):
    fake = ScriptedDB([("c1",), ("c2",)], [])
    monkeypatch.setattr(documents, "get_conn", fake.connect)

    assert documents.purge_document_vectors(USER_ID, DOC_ID) == 2
    assert "d.status IN ('deleted', 'replaced')" in fake.sql(0)
    assert fake.params(0) == (DOC_ID, USER_ID)
    assert "status IN ('deleted', 'replaced')" in fake.sql(1)
    assert pinecone.deleted == [(USER_ID, ["c1", "c2"])]


def test_pending_vector_purges(monkeypatch):
    fake = ScriptedDB([(USER_ID, DOC_ID)])
    monkeypatch.setattr(documents, "get_conn", fake.connect)

    assert documents.pending_vector_purges(limit=10) == [(USER_ID, DOC_ID)]
    assert "vectors_deleted_at IS NULL" in fake.sql(0)
    assert fake.params(0) == (10,)


# ── ops script ──────────────────────────────────────────────────────

def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "purge_deleted_vectors.py"
    spec = importlib.util.spec_from_file_location("purge_deleted_vectors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_purge_script_is_dry_run_by_default(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(documents, "pending_vector_purges", lambda limit: [(USER_ID, DOC_ID)])
    monkeypatch.setattr(documents, "try_purge_document_vectors", lambda u, d: pytest.fail("dry run must not purge"))
    lines = []

    summary = script.run(apply=False, limit=5, out=lines.append)

    assert summary == {"pending": 1, "purged": 0, "failed": 0}
    assert "dry run" in lines[-1]


def test_purge_script_apply_reports_failures(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(documents, "pending_vector_purges", lambda limit: [(USER_ID, "d1"), (USER_ID, "d2")])
    monkeypatch.setattr(documents, "try_purge_document_vectors", lambda u, d: d == "d1")

    summary = script.run(apply=True, limit=5, out=lambda line: None)

    assert summary == {"pending": 2, "purged": 1, "failed": 1}
    monkeypatch.setattr(script, "run", lambda **kw: summary)
    assert script.main(["--apply"]) == 1

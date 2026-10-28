"""Real-Postgres checks for chunk integrity, the question bank, progress and migration 003.

Opt-in (pytest -m postgres, needs TEST_DATABASE_URL to a LOCAL disposable Postgres).
Pinecone is faked; all SQL, transactions and constraints are real.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

import documents
import ingest
import progress
import questions
import retrieve
import study
from pg_support import ROOT, disposable_schema

pytestmark = pytest.mark.postgres


@pytest.fixture
def pg(monkeypatch):
    with disposable_schema(monkeypatch, [study, documents, ingest, retrieve, questions, progress]) as connect:
        yield connect


@pytest.fixture
def pinecone(monkeypatch):
    """Records vector deletes; serves whatever matches a test puts in `results`."""

    state = SimpleNamespace(deleted=[], results=[], fail_delete=None, queries=[])

    def delete_vectors(*, namespace, ids):
        if state.fail_delete:
            raise state.fail_delete
        state.deleted.append((namespace, sorted(ids)))

    class Store:
        def similarity_search_with_score(self, query, *, k, filter):
            state.queries.append(k)
            return state.results[:k]

        def add_documents(self, *, documents, ids):
            pass

    monkeypatch.setattr(documents, "delete_vectors", delete_vectors)
    monkeypatch.setattr(retrieve, "get_vector_store", lambda namespace=None: Store())
    monkeypatch.setattr(ingest, "get_vector_store", lambda namespace=None: Store())
    return state


# ── seeding helpers (raw SQL; answers go through the real record_attempt) ──

def one(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def make_user(conn, email="a@example.com"):
    return one(conn, "INSERT INTO users (email) VALUES (%s) RETURNING id", (email,))


def make_course(conn, user, name="Bio", archived=False):
    return one(
        conn,
        "INSERT INTO courses (user_id, name, archived_at) VALUES (%s, %s, CASE WHEN %s THEN now() END) RETURNING id",
        (user, name, archived),
    )


def make_doc(conn, user, course, filename="notes.txt", status="active"):
    return one(
        conn,
        """INSERT INTO documents (user_id, course_id, filename, status, deleted_at, chunk_count, char_count)
           VALUES (%s, %s, %s, %s, CASE WHEN %s <> 'active' THEN now() END, 1, 10) RETURNING id""",
        (user, course, filename, status, status),
    )


def make_chunk(conn, user, course, doc, text="Water evaporates.", deleted=False):
    return one(
        conn,
        """INSERT INTO chunks (user_id, course_id, document_id, chunk_index, text, deleted_at)
           VALUES (%s, %s, %s, 0, %s, CASE WHEN %s THEN now() END) RETURNING id""",
        (user, course, doc, text, deleted),
    )


def make_topic(conn, user, course, name):
    return one(
        conn,
        """INSERT INTO topics (user_id, course_id, name, normalized_key, created_by)
           VALUES (%s, %s, %s, lower(%s), 'user') RETURNING id""",
        (user, course, name, name),
    )


def make_question(conn, user, course, topic, chunk_ids, prompt="Why?", reference="Because.", key_points=("k",)):
    from psycopg.types.json import Jsonb

    qid = one(
        conn,
        """INSERT INTO questions (user_id, course_id, question_type, prompt, reference_answer, key_points, concept_label, status)
           VALUES (%s, %s, 'short_answer', %s, %s, %s, 'concept', 'active') RETURNING id""",
        (user, course, prompt, reference, Jsonb(list(key_points))),
    )
    conn.execute(
        "INSERT INTO question_topics (user_id, question_id, topic_id, is_primary) VALUES (%s, %s, %s, true)",
        (user, qid, topic),
    )
    for rank, chunk_id in enumerate(chunk_ids):
        conn.execute(
            "INSERT INTO question_sources (user_id, question_id, chunk_id, rank) VALUES (%s, %s, %s, %s)",
            (user, qid, chunk_id, rank),
        )
    return qid


def answer(user, question, score):
    return study.record_attempt(
        user_id=user, question_id=question, answer_text="a", verdict=study.verdict_from_score(score),
        score=score, missed_points=[], feedback="", explanation="", identified_gap=None,
    )


def match(chunk_id, course):
    return (SimpleNamespace(page_content="t", metadata={"chunk_id": str(chunk_id), "course_id": str(course)}), 0.9)


# ── Feature 3: deleted content never reaches retrieval, practice or grading ──

def test_retrieval_drops_deleted_replaced_orphaned_foreign_and_archived_chunks(pg, pinecone):
    with pg() as conn:
        a = make_user(conn, "a@x")
        b = make_user(conn, "b@x")
        course = make_course(conn, a)
        archived = make_course(conn, a, "Old", archived=True)
        live = make_chunk(conn, a, course, make_doc(conn, a, course))
        deleted = make_chunk(conn, a, course, make_doc(conn, a, course, "gone.txt", "deleted"), deleted=True)
        replaced = make_chunk(conn, a, course, make_doc(conn, a, course, "v1.txt", "replaced"), deleted=True)
        foreign = make_chunk(conn, b, make_course(conn, b), make_doc(conn, b, make_course(conn, b, "B2")))
        in_archived = make_chunk(conn, a, archived, make_doc(conn, a, archived))
    orphan = uuid4()  # vector whose chunk row never committed
    pinecone.results = [match(c, course) for c in (deleted, orphan, foreign, replaced, in_archived, live)]

    result = retrieve.retrieve_chunks(query="water", user_id=a, course_id=course, k=3)

    assert [c["chunk_id"] for c in result["chunks"]] == [str(live)]


def test_delete_document_marks_postgres_then_purges_only_its_vectors(pg, pinecone):
    with pg() as conn:
        a = make_user(conn, "a@x")
        b = make_user(conn, "b@x")
        course = make_course(conn, a)
        doc = make_doc(conn, a, course)
        c1 = make_chunk(conn, a, course, doc)
        c2 = make_chunk(conn, a, course, doc)
        keep = make_chunk(conn, a, course, make_doc(conn, a, course, "keep.txt"))

    assert documents.delete_document(b, doc) is False  # another user: untouched
    assert documents.delete_document(a, doc) is True

    with pg() as conn:
        status, purged = conn.execute("SELECT status, vectors_deleted_at FROM documents WHERE id = %s", (doc,)).fetchone()
        live_chunks = {r[0] for r in conn.execute("SELECT id FROM chunks WHERE deleted_at IS NULL").fetchall()}
    assert status == "deleted" and purged is not None
    assert live_chunks == {keep}
    assert pinecone.deleted == [(str(a), sorted([str(c1), str(c2)]))]

    # Repeating the request is safe: still 200-equivalent, no second Pinecone call.
    assert documents.delete_document(a, doc) is True
    assert len(pinecone.deleted) == 1


def test_failed_vector_purge_keeps_content_unusable_and_is_retried(pg, pinecone):
    with pg() as conn:
        a = make_user(conn)
        course = make_course(conn, a)
        doc = make_doc(conn, a, course)
        chunk = make_chunk(conn, a, course, doc)
    pinecone.fail_delete = ConnectionError("pinecone down")

    assert documents.delete_document(a, doc) is True
    assert documents.pending_vector_purges() == [(a, doc)]
    pinecone.results = [match(chunk, course)]  # Pinecone still serves the stale vector
    assert retrieve.retrieve_chunks(query="water", user_id=a, course_id=course)["chunks"] == []

    pinecone.fail_delete = None
    assert documents.delete_document(a, doc) is True  # retry completes the purge
    assert documents.pending_vector_purges() == []
    assert pinecone.deleted == [(str(a), [str(chunk)])]


def test_grading_and_practice_ignore_deleted_sources(pg, pinecone, monkeypatch):
    with pg() as conn:
        a = make_user(conn)
        course = make_course(conn, a)
        doc1, doc2 = make_doc(conn, a, course, "one.txt"), make_doc(conn, a, course, "two.txt")
        c1, c2 = make_chunk(conn, a, course, doc1), make_chunk(conn, a, course, doc2)
        topic = make_topic(conn, a, course, "water")
        only_doc1 = make_question(conn, a, course, topic, [c1], prompt="Only from doc1")
        mixed = make_question(conn, a, course, topic, [c1, c2], prompt="From both")

    documents.delete_document(a, doc1)

    def no_llm(prompt, schema):
        raise AssertionError("grading must not call the LLM without live source material")

    monkeypatch.setattr(study, "_call_structured", no_llm)
    with pytest.raises(study.StudyValidationError):
        study.grade_student_answer(user_id=a, question_id=only_doc1, student_answer="x")

    assert [c["chunk_id"] for c in study.load_question_for_grading(user_id=a, question_id=mixed)["chunks"]] == [c2]
    assert [q["question_id"] for q in study.get_question_candidates(user_id=a, topic_id=topic)] == [mixed]
    assert [q["id"] for q in questions.list_question_bank(a, course)] == [mixed]
    assert [q["id"] for q in questions.list_questions(a, course)] == [mixed]


def test_ingest_replacement_retires_old_version_with_real_sql(pg, pinecone):
    with pg() as conn:
        a = make_user(conn)
        course = make_course(conn, a)

    first = ingest.ingest_document(user_id=a, course_id=course, filename="n.txt", text="Version one text.")
    second = ingest.ingest_document(user_id=a, course_id=course, filename="n.txt", text="Version two text.")

    with pg() as conn:
        rows = dict(conn.execute("SELECT id, status FROM documents").fetchall())
        old_chunks = [r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE document_id = %s AND deleted_at IS NOT NULL", (first["document_id"],)
        ).fetchall()]
        purged = one(conn, "SELECT vectors_deleted_at IS NOT NULL FROM documents WHERE id = %s", (first["document_id"],))
    assert rows == {first["document_id"]: "replaced", second["document_id"]: "active"}
    assert purged is True
    assert pinecone.deleted == [(str(a), sorted(str(c) for c in old_chunks))]


# ── Feature 2: question bank ────────────────────────────────────────────

def test_question_bank_lists_each_question_once_with_stored_answer(pg, pinecone):
    with pg() as conn:
        a = make_user(conn, "a@x")
        b = make_user(conn, "b@x")
        course = make_course(conn, a)
        chunk = make_chunk(conn, a, course, make_doc(conn, a, course))
        topic = make_topic(conn, a, course, "water")
        qid = make_question(conn, a, course, topic, [chunk], reference="Heat.", key_points=("heat", " "))
        legacy = make_question(conn, a, course, topic, [chunk], prompt="Old", reference="", key_points=())
    for score in (0.2, 0.6, 0.9):
        answer(a, qid, score)

    bank = questions.list_question_bank(a, course)

    assert [item["id"] for item in bank] == [qid, legacy]
    assert bank[0]["reference_answer"] == "Heat."
    assert bank[0]["key_points"] == ["heat"]
    assert bank[0]["attempt_count"] == 3
    assert bank[0]["topic"] == "water"
    assert bank[1]["reference_answer"] is None  # missing historical answer handled
    assert questions.list_question_bank(b, course) == []


# ── Feature 1: progress uses the adaptive model's numbers ─────────────

def test_course_progress_matches_adaptive_model(pg, pinecone):
    with pg() as conn:
        a = make_user(conn)
        course = make_course(conn, a)
        empty_course = make_course(conn, a, "Empty")
        chunk = make_chunk(conn, a, course, make_doc(conn, a, course))
        strong = make_topic(conn, a, course, "Strong")
        weak = make_topic(conn, a, course, "Weak")
        fresh = make_topic(conn, a, course, "Fresh")
        make_topic(conn, a, course, "No questions")  # created by a failed quiz request
        s1 = make_question(conn, a, course, strong, [chunk])
        s2 = make_question(conn, a, course, strong, [chunk])
        w1 = make_question(conn, a, course, weak, [chunk])
        make_question(conn, a, course, fresh, [chunk])
    for score in (1.0, 1.0, 1.0, 1.0):
        answer(a, s1, score)
    answer(a, s2, 1.0)
    answer(a, w1, 0.0)

    detail = progress.course_progress(a, {"id": course, "name": "Bio"})
    topics = {t["name"]: t for t in detail["topics"]}

    assert set(topics) == {"Strong", "Weak", "Fresh"}
    for topic_id, name in ((strong, "Strong"), (weak, "Weak")):
        assert topics[name]["mastery"] == pytest.approx(study.get_topic_mastery(user_id=a, topic_id=topic_id))
    assert topics["Fresh"]["mastery"] is None and topics["Fresh"]["mastery_band"] == "not_practiced"
    assert topics["Weak"]["mastery_band"] == "needs_review"  # 0.6*0.5 = 0.3
    assert topics["Strong"]["attempts"] == 5 and topics["Strong"]["correct_rate"] == 1.0
    assert [t["name"] for t in detail["topics"]] == ["Weak", "Strong", "Fresh"]
    assert detail["focus_topics"] == ["Weak", "Strong"]
    assert detail["not_practiced_topics"] == ["Fresh"]
    assert detail["questions_in_bank"] == 4 and detail["questions_answered"] == 3
    assert detail["attempts"] == 6 and detail["attempts_last_7_days"] == 6
    with pg() as conn:
        expected = one(conn, "SELECT AVG(mastery) FROM question_state WHERE user_id = %s", (a,))
    assert detail["mastery"] == pytest.approx(expected)
    assert len(detail["recent_activity"]) == progress.RECENT_ACTIVITY_LIMIT

    overview = {o["name"]: o for o in progress.progress_overview(a, [{"id": course, "name": "Bio"}, {"id": empty_course, "name": "Empty"}])}
    assert overview["Bio"]["mastery"] == pytest.approx(expected)
    assert overview["Empty"]["mastery"] is None and overview["Empty"]["average_score"] is None
    assert overview["Empty"]["mastery_band"] == "not_practiced"


def test_recent_activity_hides_text_from_deleted_material(pg, pinecone):
    with pg() as conn:
        a = make_user(conn)
        course = make_course(conn, a)
        doc = make_doc(conn, a, course)
        chunk = make_chunk(conn, a, course, doc)
        qid = make_question(conn, a, course, make_topic(conn, a, course, "t"), [chunk], prompt="Secret text")
    answer(a, qid, 0.5)
    documents.delete_document(a, doc)

    detail = progress.course_progress(a, {"id": course, "name": "Bio"})

    assert detail["recent_activity"][0]["question"] is None
    assert detail["recent_activity"][0]["source_removed"] is True
    assert detail["questions_in_bank"] == 0
    assert detail["questions_answered"] == 1  # history is kept


# ── Migration 003 ───────────────────────────────────────────────────

def test_migration_003_backfills_and_rolls_back(pg):
    up = (ROOT / "migrations" / "003_document_vector_purge.sql").read_text()
    down = (ROOT / "migrations" / "003_document_vector_purge.down.sql").read_text()
    with pg() as conn:
        conn.execute(down)
        a = make_user(conn)
        course = make_course(conn, a)
        active = make_doc(conn, a, course, "a.txt")
        deleted = make_doc(conn, a, course, "d.txt", "deleted")
        conn.commit()
        conn.execute(up)
        conn.execute(up)  # idempotent
        rows = dict(conn.execute(
            "SELECT id, vectors_deleted_at IS NOT NULL AND vectors_deleted_at = deleted_at FROM documents"
        ).fetchall())
        indexes = conn.execute(
            "SELECT count(*) FROM pg_indexes WHERE schemaname = current_schema() AND indexname = 'documents_pending_vector_purge'"
        ).fetchone()[0]
    assert rows == {active: False, deleted: True}
    assert indexes == 1

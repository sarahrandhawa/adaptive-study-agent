"""Data-access functions against a scripted fake connection.

These check row mapping, control flow, and that every query is scoped to the
caller's user_id (the app's authorization boundary). They do not prove the SQL
is valid Postgres; see tests/test_postgres.py for that.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import courses
import db
import documents
import memory_store
import questions
import study
import users
from fakes import COURSE_ID, QUESTION_ID, USER_ID, ScriptedDB

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)
DOC_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def use(monkeypatch, module, fake):
    monkeypatch.setattr(module, "get_conn", fake.connect)
    return fake


def test_get_conn_requires_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "  ")

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.get_conn()


# ── users ───────────────────────────────────────────────────────────

def test_ensure_user_returns_existing_identity_without_insert(monkeypatch):
    fake = use(monkeypatch, users, ScriptedDB([("existing-uuid",)]))

    assert users.ensure_user("google", "sub-1", email="a@b.c") == "existing-uuid"
    assert len(fake.executed) == 1
    assert fake.params(0) == ("google", "sub-1")


def test_ensure_user_creates_user_and_identity(monkeypatch):
    fake = use(monkeypatch, users, ScriptedDB([], [("new-uuid",)], []))

    assert users.ensure_user("google", "sub-2", email="e@x.com", name="Ann") == "new-uuid"
    assert fake.sql(1).startswith("INSERT INTO users")
    assert fake.params(1) == ("e@x.com", "Ann")
    assert fake.sql(2).startswith("INSERT INTO user_identities")
    assert fake.params(2) == ("new-uuid", "google", "sub-2")
    assert fake.committed


# ── courses ─────────────────────────────────────────────────────────

def test_create_course_rejects_blank_name_without_db(monkeypatch):
    fake = use(monkeypatch, courses, ScriptedDB())

    with pytest.raises(ValueError):
        courses.create_course(USER_ID, "   ")
    assert fake.executed == []


def test_create_course_strips_name_and_scopes_to_user(monkeypatch):
    fake = use(monkeypatch, courses, ScriptedDB([(COURSE_ID,)]))

    assert courses.create_course(USER_ID, "  Biology  ") == COURSE_ID
    assert fake.params(0) == (USER_ID, "Biology")


def test_list_courses_maps_rows_and_filters_by_user(monkeypatch):
    fake = use(monkeypatch, courses, ScriptedDB([(COURSE_ID, "Bio", NOW)]))

    assert courses.list_courses(USER_ID) == [{"id": COURSE_ID, "name": "Bio", "created_at": NOW}]
    assert "WHERE user_id = %s AND archived_at IS NULL" in fake.sql(0)
    assert fake.params(0) == (USER_ID,)


@pytest.mark.parametrize("rows, expected", [([(1,)], True), ([], False)])
def test_user_owns_course(monkeypatch, rows, expected):
    fake = use(monkeypatch, courses, ScriptedDB(rows))

    assert courses.user_owns_course(USER_ID, COURSE_ID) is expected
    assert fake.params(0) == (COURSE_ID, USER_ID)
    assert "archived_at IS NULL" in fake.sql(0)


@pytest.mark.parametrize("rows, expected", [([(COURSE_ID,)], True), ([], False)])
def test_archive_course_is_soft_delete_scoped_to_user(monkeypatch, rows, expected):
    fake = use(monkeypatch, courses, ScriptedDB(rows))

    assert courses.archive_course(USER_ID, COURSE_ID) is expected
    assert fake.sql(0).startswith("UPDATE courses SET archived_at = now()")
    assert fake.params(0) == (COURSE_ID, USER_ID)


# ── documents ───────────────────────────────────────────────────────

def test_list_documents_maps_rows(monkeypatch):
    fake = use(monkeypatch, documents, ScriptedDB([(DOC_ID, "notes.txt", 100, 2, NOW)]))

    assert documents.list_documents(USER_ID, COURSE_ID) == [
        {"id": DOC_ID, "filename": "notes.txt", "char_count": 100, "chunk_count": 2, "created_at": NOW}
    ]
    assert fake.params(0) == (USER_ID, COURSE_ID)


def test_delete_document_not_owned_touches_nothing(monkeypatch):
    fake = use(monkeypatch, documents, ScriptedDB([]))
    deleted_vectors = []
    monkeypatch.setattr(documents, "delete_vectors", lambda **kw: deleted_vectors.append(kw))

    assert documents.delete_document(USER_ID, DOC_ID) is False
    assert len(fake.executed) == 1
    assert deleted_vectors == []


def test_delete_document_soft_deletes_first_then_purges_vectors(monkeypatch):
    # lock doc (active, not purged), mark chunks, mark doc | purge: chunk ids, mark purged
    fake = use(monkeypatch, documents, ScriptedDB([("active", None)], [], [], [("c1",), ("c2",)], []))
    deleted_vectors = []

    def record_delete(**kw):
        # Postgres must already say "deleted" before Pinecone is touched.
        assert any(sql.startswith("UPDATE documents SET status = 'deleted'") for sql, _ in fake.executed)
        deleted_vectors.append(kw)

    monkeypatch.setattr(documents, "delete_vectors", record_delete)

    assert documents.delete_document(USER_ID, DOC_ID) is True
    assert "FOR UPDATE" in fake.sql(0)
    assert fake.sql(1).startswith("UPDATE chunks SET deleted_at = now()")
    assert fake.sql(2).startswith("UPDATE documents SET status = 'deleted'")
    assert deleted_vectors == [{"namespace": USER_ID, "ids": ["c1", "c2"]}]
    assert fake.sql(4).startswith("UPDATE documents SET vectors_deleted_at = now()")
    assert all(USER_ID in params for _, params in fake.executed)


# ── topics / questions ──────────────────────────────────────────────

def test_normalize_topic_collapses_whitespace_and_case():
    assert memory_store.normalize_topic("  Water   CYCLE ") == "water cycle"


def test_get_or_create_topic_upserts_normalized_key(monkeypatch):
    fake = use(monkeypatch, memory_store, ScriptedDB([("topic-uuid",)]))

    assert memory_store.get_or_create_topic(USER_ID, COURSE_ID, "  Water  Cycle ") == "topic-uuid"
    assert "ON CONFLICT (course_id, normalized_key)" in fake.sql(0)
    assert fake.params(0) == (USER_ID, COURSE_ID, "Water  Cycle", "water cycle", "user")


def test_get_or_create_topic_rejects_blank(monkeypatch):
    fake = use(monkeypatch, memory_store, ScriptedDB())

    with pytest.raises(ValueError):
        memory_store.get_or_create_topic(USER_ID, COURSE_ID, "   ")
    assert fake.executed == []


def test_list_questions_never_selects_answers(monkeypatch):
    fake = use(monkeypatch, questions, ScriptedDB([(QUESTION_ID, "Q?", "water", "evap", NOW, None, None, None, None)]))

    rows = questions.list_questions(USER_ID, COURSE_ID)

    assert rows[0]["attempt_count"] == 0
    assert rows[0]["topic"] == "water"
    assert "reference_answer" not in fake.sql(0)
    assert "key_points" not in fake.sql(0)
    assert fake.params(0) == (USER_ID, COURSE_ID, 200)


# ── study SQL helpers ───────────────────────────────────────────────

def test_save_question_links_topic_and_ranked_sources(monkeypatch):
    fake = use(monkeypatch, study, ScriptedDB([(QUESTION_ID,)], [], [], []))
    draft = study.QuizDraft(
        sufficient_material=True, question=" Q? ", reference_answer=" A ", key_points=["k"], concept_label=" c "
    )
    chunks = [{"chunk_id": "c1"}, {"chunk_id": None}, {"chunk_id": "c3"}]

    assert study.save_question(user_id=USER_ID, course_id=COURSE_ID, topic_id="t1", draft=draft, chunks=chunks) == QUESTION_ID
    assert fake.params(0)[3:5] == ("Q?", "A")
    assert fake.params(1) == (USER_ID, QUESTION_ID, "t1")
    # chunk without an id is skipped; rank keeps the retrieval position
    assert [p for _, p in fake.executed[2:]] == [(USER_ID, QUESTION_ID, "c1", 0), (USER_ID, QUESTION_ID, "c3", 2)]


def test_load_question_for_grading_not_found(monkeypatch):
    use(monkeypatch, study, ScriptedDB([]))

    with pytest.raises(study.StudyValidationError, match="not found"):
        study.load_question_for_grading(user_id=USER_ID, question_id=QUESTION_ID)


def test_load_question_for_grading_maps_question_and_sources(monkeypatch):
    fake = use(
        monkeypatch,
        study,
        ScriptedDB(
            [(QUESTION_ID, "Q?", "A", None, "c", "water")],
            [("c1", DOC_ID, 0, "text")],
        ),
    )

    question = study.load_question_for_grading(user_id=USER_ID, question_id=QUESTION_ID)

    assert question["key_points"] == []
    assert question["topic"] == "water"
    assert question["chunks"] == [{"chunk_id": "c1", "document_id": DOC_ID, "chunk_index": 0, "text": "text"}]
    assert fake.params(0) == (QUESTION_ID, USER_ID)
    assert fake.params(1) == (QUESTION_ID, USER_ID)


def test_get_question_candidates_maps_rows(monkeypatch):
    fake = use(monkeypatch, study, ScriptedDB([(QUESTION_ID, "Q?", "c", 0.4, NOW)]))

    assert study.get_question_candidates(user_id=USER_ID, topic_id="t1") == [
        {"question_id": QUESTION_ID, "question": "Q?", "concept_label": "c", "mastery": 0.4, "last_attempt_at": NOW}
    ]
    assert fake.params(0) == (USER_ID, "t1")


@pytest.mark.parametrize("average, expected", [(None, 0.5), (0.72, 0.72)])
def test_get_topic_mastery_defaults_to_neutral(monkeypatch, average, expected):
    use(monkeypatch, study, ScriptedDB([(average,)]))

    assert study.get_topic_mastery(user_id=USER_ID, topic_id="t1") == pytest.approx(expected)


def test_find_submission_maps_stored_attempt(monkeypatch):
    fake = use(
        monkeypatch,
        study,
        ScriptedDB([("att-1", QUESTION_ID, "correct", 0.9, None, "f", "e", None, 0.66, 1, 1)]),
    )

    found = study.find_submission(user_id=USER_ID, submission_id="sub-1")

    assert found["attempt_id"] == "att-1"
    assert found["missed_points"] == []
    assert found["attempt_count"] == 1
    assert fake.params(0) == (USER_ID, "sub-1")


def test_find_submission_missing(monkeypatch):
    use(monkeypatch, study, ScriptedDB([]))

    assert study.find_submission(user_id=USER_ID, submission_id="sub-1") is None


def test_record_attempt_takes_lock_before_reading_state(monkeypatch):
    fake = use(monkeypatch, study, ScriptedDB([], [], [("att", NOW)], []))

    study.record_attempt(
        user_id=USER_ID, question_id=QUESTION_ID, answer_text="a", verdict="correct", score=1.0,
        missed_points=[], feedback="", explanation="", identified_gap=None, submission_id=None,
    )

    assert "pg_advisory_xact_lock" in fake.sql(0)
    assert fake.params(0) == (USER_ID, QUESTION_ID)
    assert fake.sql(1).startswith("SELECT mastery, attempt_count, correct_streak FROM question_state")

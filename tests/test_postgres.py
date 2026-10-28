"""Real-Postgres checks for the attempt-recording SQL and migration 001.

Opt-in and disposable only: runs when TEST_DATABASE_URL points at a LOCAL
Postgres (localhost/127.0.0.1); any other host is refused. Each run creates a
throwaway schema and drops it afterwards. Example:

    docker run --rm -d --name asa-test-pg -e POSTGRES_PASSWORD=test -p 55432:5432 postgres:16
    TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres \
        .venv/bin/python -m pytest -m postgres
"""

import os
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest

import study

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parent.parent
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()


@pytest.fixture
def pg(monkeypatch):
    import psycopg

    host = urlparse(TEST_DATABASE_URL).hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.fail(f"TEST_DATABASE_URL must point at a local disposable database, not {host!r}")

    schema = f"asa_test_{uuid.uuid4().hex[:12]}"

    def connect():
        return psycopg.connect(TEST_DATABASE_URL, options=f"-c search_path={schema}")

    with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    try:
        with connect() as conn:
            conn.execute((ROOT / "schema.sql").read_text())
        monkeypatch.setattr(study, "get_conn", connect)
        yield connect
    finally:
        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")


def seed(connect):
    with connect() as conn:
        user_id = conn.execute("INSERT INTO users (email) VALUES ('t@example.com') RETURNING id").fetchone()[0]
        course_id = conn.execute(
            "INSERT INTO courses (user_id, name) VALUES (%s, 'c') RETURNING id", (user_id,)
        ).fetchone()[0]
        question_id = conn.execute(
            "INSERT INTO questions (user_id, course_id, prompt, status) VALUES (%s, %s, 'q', 'active') RETURNING id",
            (user_id, course_id),
        ).fetchone()[0]
    return user_id, question_id


def record(user_id, question_id, submission_id=None):
    return study.record_attempt(
        user_id=user_id, question_id=question_id, answer_text="a", verdict="correct", score=0.9,
        missed_points=[], feedback="f", explanation="e", identified_gap=None, submission_id=submission_id,
    )


def _submission_indexes(conn):
    return conn.execute(
        """
        SELECT indexname FROM pg_indexes
        WHERE schemaname = current_schema() AND tablename = 'attempts'
          AND indexdef ILIKE '%%submission_id%%'
        """
    ).fetchall()


def test_migration_is_idempotent_and_matches_fresh_schema(pg):
    migration = (ROOT / "migrations" / "001_attempts_submission_id.sql").read_text()
    with pg() as conn:
        conn.execute(migration)
        conn.execute(migration)
        assert _submission_indexes(conn) == [("attempts_user_submission_key",)]


def test_migration_rollback_and_reapply(pg):
    up = (ROOT / "migrations" / "001_attempts_submission_id.sql").read_text()
    down = (ROOT / "migrations" / "001_attempts_submission_id.down.sql").read_text()
    with pg() as conn:
        conn.execute(down)
        assert _submission_indexes(conn) == []
        conn.execute(up)
        assert _submission_indexes(conn) == [("attempts_user_submission_key",)]


def test_concurrent_attempts_are_all_counted(pg):
    user_id, question_id = seed(pg)
    errors = []

    def submit():
        try:
            record(user_id, question_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    with pg() as conn:
        attempts = conn.execute("SELECT count(*) FROM attempts").fetchone()[0]
        count, streak = conn.execute("SELECT attempt_count, correct_streak FROM question_state").fetchone()
    assert attempts == 8
    assert (count, streak) == (8, 8)


def test_submission_id_replays_and_is_unique(pg):
    import psycopg

    user_id, question_id = seed(pg)
    submission_id = uuid.uuid4()

    first = record(user_id, question_id, submission_id)
    second = record(user_id, question_id, submission_id)

    assert second["replayed"] is True
    assert second["attempt_id"] == first["attempt_id"]
    assert study.find_submission(user_id=user_id, submission_id=submission_id)["attempt_count"] == 1

    with pytest.raises(psycopg.errors.UniqueViolation):
        with pg() as conn:
            conn.execute(
                "INSERT INTO attempts (user_id, question_id, submission_id) VALUES (%s, %s, %s)",
                (user_id, question_id, submission_id),
            )

"""Duplicate course names via POST /courses.

Expected rule: a user cannot have two active courses whose names differ only in
capitalization or whitespace; the API answers 400. Other users and genuinely
different names are unaffected.
"""

import re
import uuid
from datetime import datetime, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient

import courses
import main
from fakes import OTHER_USER_ID, USER_ID, _normalize, unique_violation


def index_key(user_id, name):
    """Mirror of courses_user_normalized_name_unique (migrations/002):
    (user_id, LOWER(REGEXP_REPLACE(BTRIM(name), '[[:space:]]+', ' ', 'g')))."""

    return str(user_id), re.sub(r"[ \t\n\r\f\v]+", " ", name.strip(" ")).lower()


class FakeCoursesDB:
    """Stateful stand-in for the courses table, including the partial unique index
    from migrations/002 (active courses only). Understands today's create, list and
    archive SQL and fails loudly on anything else, so a fix must extend it deliberately."""

    def __init__(self):
        self.rows = []  # dicts: id, user_id, name, created_at, archived_at
        self._result = []

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        text = _normalize(sql)
        if text.startswith("INSERT INTO courses (user_id, name)"):
            user_id, name = params
            key = index_key(user_id, name)
            if any(index_key(r["user_id"], r["name"]) == key and r["archived_at"] is None for r in self.rows):
                raise unique_violation("courses_user_normalized_name_unique", "courses")
            row = {
                "id": uuid.uuid4(),
                "user_id": str(user_id),
                "name": name,
                "created_at": datetime.now(timezone.utc),
                "archived_at": None,
            }
            self.rows.append(row)
            self._result = [(row["id"],)]
        elif text.startswith("SELECT id, name, created_at FROM courses WHERE user_id = %s AND archived_at IS NULL"):
            (user_id,) = params
            self._result = [
                (r["id"], r["name"], r["created_at"])
                for r in reversed(self.rows)
                if r["user_id"] == str(user_id) and r["archived_at"] is None
            ]
        elif text.startswith("UPDATE courses SET archived_at = now()"):
            course_id, user_id = (str(p) for p in params)
            match = [r for r in self.rows if str(r["id"]) == course_id and r["user_id"] == user_id and r["archived_at"] is None]
            for r in match:
                r["archived_at"] = datetime.now(timezone.utc)
            self._result = [(r["id"],) for r in match]
        else:
            raise AssertionError(f"FakeCoursesDB does not understand this SQL yet: {text[:120]}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)

    def names_for(self, user_id):
        return [r["name"] for r in self.rows if r["user_id"] == str(user_id)]


@pytest.fixture
def api(monkeypatch):
    fake = FakeCoursesDB()
    monkeypatch.setattr(courses, "get_conn", fake.connect)
    current = {"user": USER_ID}
    main.app.dependency_overrides[main.current_user] = lambda: current["user"]
    try:
        yield TestClient(main.app), fake, current
    finally:
        main.app.dependency_overrides.clear()


@pytest.mark.regression
@pytest.mark.parametrize(
    "duplicate",
    ["Biology 101", "biology 101", "BIOLOGY 101", "  Biology 101  ", "Biology   101", "\tbiology\n101 "],
    ids=["exact", "lowercase", "uppercase", "outer_spaces", "inner_spaces", "mixed_case_and_whitespace"],
)
def test_duplicate_course_name_is_400(api, duplicate):
    """BUG: create_course inserts unconditionally. Without migration 002 the duplicate
    is stored (200). With migration 002's unique index (modelled by the fake), the
    insert raises UniqueViolation, which the app-wide database handler turns into a
    misleading 503 "database unavailable" instead of a 400 validation error."""

    client, fake, _ = api

    assert client.post("/courses", json={"name": "Biology 101"}).status_code == 200

    response = client.post("/courses", json={"name": duplicate})

    assert response.status_code == 400, f"duplicate {duplicate!r} returned {response.status_code}, expected 400"
    assert fake.names_for(USER_ID) == ["Biology 101"]


def test_different_course_name_is_allowed(api):
    client, fake, _ = api

    assert client.post("/courses", json={"name": "Biology 101"}).status_code == 200
    assert client.post("/courses", json={"name": "Biology 102"}).status_code == 200
    assert fake.names_for(USER_ID) == ["Biology 101", "Biology 102"]


def test_archived_course_name_can_be_reused(api):
    """The index is partial (WHERE archived_at IS NULL), so archiving frees the name."""

    client, fake, _ = api

    course_id = client.post("/courses", json={"name": "Biology 101"}).json()["id"]
    assert client.delete(f"/courses/{course_id}").status_code == 200
    assert client.post("/courses", json={"name": "biology 101"}).status_code == 200


def test_same_course_name_for_another_user_is_allowed(api):
    client, fake, current = api

    assert client.post("/courses", json={"name": "Biology 101"}).status_code == 200
    current["user"] = OTHER_USER_ID
    assert client.post("/courses", json={"name": "biology 101"}).status_code == 200
    assert fake.names_for(OTHER_USER_ID) == ["biology 101"]


def test_fake_violation_reports_constraint_like_postgres():
    """Guards the fake itself: production code may rely on these diag fields."""

    fake = FakeCoursesDB()
    fake.execute("INSERT INTO courses (user_id, name) VALUES (%s, %s) RETURNING id", (USER_ID, "Bio"))

    with pytest.raises(psycopg.errors.UniqueViolation) as caught:
        fake.execute("INSERT INTO courses (user_id, name) VALUES (%s, %s) RETURNING id", (USER_ID, " BIO "))

    assert isinstance(caught.value, psycopg.IntegrityError)
    assert caught.value.sqlstate == "23505"
    assert caught.value.diag.constraint_name == "courses_user_normalized_name_unique"
    assert caught.value.diag.table_name == "courses"

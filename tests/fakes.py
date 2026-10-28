"""In-memory stand-ins for Postgres and the grading LLM.

FakeDB understands exactly the SQL that study.record_attempt issues today and
fails loudly on anything else, so a future fix that adds new SQL must extend
this fake deliberately rather than silently passing.

Transaction model (approximates Postgres READ COMMITTED):
- Writes are buffered and applied when the connection context exits cleanly
  (commit); an exception discards them (rollback).
- Reads see only committed data.
- `SELECT ... FOR UPDATE` locks the row only if it already exists (as in
  Postgres); `pg_advisory_xact_lock(...)` always takes a lock. Both locks are
  held until commit/rollback.
- `interleave_reads=N` makes N concurrent transactions pause right after
  reading question_state until all N have read, which reproduces the
  read-modify-write race deterministically. If locking prevents the others
  from reaching that point, the pause times out and execution continues.
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import psycopg

from study import GradeDraft

LOCK_TIMEOUT_SECONDS = 5.0
BARRIER_TIMEOUT_SECONDS = 0.5


def _normalize(sql: str) -> str:
    return " ".join(sql.split())


class FakeDB:
    def __init__(self, *, interleave_reads: int = 0):
        self.question_state: dict[tuple, dict] = {}
        self.attempts: list[dict] = []
        self.connections_opened = 0
        self._mutex = threading.Lock()
        self._row_locks: dict[tuple, threading.Lock] = defaultdict(threading.Lock)
        self._advisory_locks: dict[tuple, threading.Lock] = defaultdict(threading.Lock)
        self._barrier = threading.Barrier(interleave_reads) if interleave_reads else None
        self._clock = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def connect(self):
        with self._mutex:
            self.connections_opened += 1
        return FakeConnection(self)

    def seed_state(self, user_id, question_id, *, mastery, attempt_count, correct_streak):
        self.question_state[(str(user_id), str(question_id))] = {
            "mastery": mastery,
            "attempt_count": attempt_count,
            "correct_streak": correct_streak,
            "last_score": None,
            "last_attempt_at": None,
        }

    def state_for(self, user_id, question_id):
        return self.question_state.get((str(user_id), str(question_id)))

    def _next_timestamp(self):
        with self._mutex:
            self._clock += timedelta(seconds=1)
            return self._clock


class FakeConnection:
    def __init__(self, db: FakeDB):
        self.db = db
        self.pending: list[tuple[str, dict]] = []
        self.held_locks: list[threading.Lock] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                with self.db._mutex:
                    for kind, payload in self.pending:
                        if kind == "attempt":
                            self.db.attempts.append(payload)
                        elif kind == "state":
                            key = (payload["user_id"], payload["question_id"])
                            self.db.question_state[key] = {
                                k: payload[k]
                                for k in (
                                    "mastery",
                                    "attempt_count",
                                    "correct_streak",
                                    "last_score",
                                    "last_attempt_at",
                                )
                            }
        finally:
            for lock in reversed(self.held_locks):
                lock.release()
            self.held_locks.clear()
        return False

    def cursor(self):
        return FakeCursor(self)


class FakeCursor:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self.db = conn.db
        self._rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _acquire(self, lock: threading.Lock):
        if not lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            raise TimeoutError("FakeDB lock wait timed out (possible deadlock)")
        self.conn.held_locks.append(lock)

    def execute(self, sql, params=()):
        text = _normalize(sql)

        if "pg_advisory_xact_lock" in text:
            self._acquire(self.db._advisory_locks[tuple(str(p) for p in params)])
            self._rows = [(None,)]
            return

        if text.startswith("SELECT mastery, attempt_count, correct_streak FROM question_state"):
            key = (str(params[0]), str(params[1]))
            if "FOR UPDATE" in text and key in self.db.question_state:
                self._acquire(self.db._row_locks[key])
            with self.db._mutex:
                state = self.db.question_state.get(key)
                self._rows = (
                    [(state["mastery"], state["attempt_count"], state["correct_streak"])]
                    if state
                    else []
                )
            if self.db._barrier is not None:
                try:
                    self.db._barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)
                except threading.BrokenBarrierError:
                    pass
            return

        if text.startswith("SELECT a.id, a.question_id") and "FROM attempts a" in text:
            user_id, submission_id = (str(p) for p in params)
            with self.db._mutex:
                match = next(
                    (a for a in self.db.attempts
                     if a["user_id"] == user_id and a["submission_id"] == submission_id),
                    None,
                )
                state = self.db.question_state.get((user_id, match["question_id"])) if match else None
            self._rows = (
                [(
                    match["id"], match["question_id"], match["verdict"], match["score"],
                    match["missed_points"], match["feedback"], match["explanation"],
                    match["identified_gap"],
                    state["mastery"] if state else None,
                    state["attempt_count"] if state else None,
                    state["correct_streak"] if state else None,
                )]
                if match
                else []
            )
            return

        if text.startswith("INSERT INTO attempts"):
            attempt_id = uuid.uuid4()
            created_at = self.db._next_timestamp()
            (user_id, question_id, answer_text, verdict, score, missed_points,
             feedback, explanation, identified_gap, grader_model, grader_version,
             submission_id) = params
            submission_id = str(submission_id) if submission_id is not None else None
            if submission_id is not None:
                with self.db._mutex:
                    taken = any(
                        a["user_id"] == str(user_id) and a["submission_id"] == submission_id
                        for a in self.db.attempts
                    )
                if taken:
                    raise psycopg.errors.UniqueViolation(
                        'duplicate key value violates unique constraint "attempts_user_submission_key"'
                    )
            self.conn.pending.append(
                (
                    "attempt",
                    {
                        "id": attempt_id,
                        "user_id": str(user_id),
                        "question_id": str(question_id),
                        "answer_text": answer_text,
                        "verdict": verdict,
                        "score": score,
                        "missed_points": getattr(missed_points, "obj", missed_points),
                        "feedback": feedback,
                        "explanation": explanation,
                        "identified_gap": identified_gap,
                        "submission_id": submission_id,
                        "created_at": created_at,
                    },
                )
            )
            self._rows = [(attempt_id, created_at)]
            return

        if text.startswith("INSERT INTO question_state") and "ON CONFLICT (user_id, question_id)" in text:
            user_id, question_id, mastery, attempt_count, correct_streak, last_score, last_attempt_at = params
            self.conn.pending.append(
                (
                    "state",
                    {
                        "user_id": str(user_id),
                        "question_id": str(question_id),
                        "mastery": mastery,
                        "attempt_count": attempt_count,
                        "correct_streak": correct_streak,
                        "last_score": last_score,
                        "last_attempt_at": last_attempt_at,
                    },
                )
            )
            self._rows = []
            return

        raise AssertionError(f"FakeDB does not understand this SQL yet: {text[:120]}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


USER_ID = "11111111-1111-1111-1111-111111111111"
OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"
QUESTION_ID = "33333333-3333-3333-3333-333333333333"
COURSE_ID = "44444444-4444-4444-4444-444444444444"
TOPIC_ID = "5555555-5555-5555-55555-55555555555"

def fake_question(question_id=QUESTION_ID, *, chunks=True):
    return {
        "id": question_id,
        "question": "What does the Harlow ratio measure?",
        "reference_answer": "Evaporation relative to precipitation.",
        "key_points": ["evaporation", "precipitation"],
        "concept_label": "Harlow ratio",
        "topic": "water cycle",
        "chunks": (
            [{"chunk_id": "c1", "document_id": "d1", "chunk_index": 0, "text": "The Harlow ratio..."}]
            if chunks
            else []
        ),
    }


class FakeGrader:
    """Replaces study._call_structured; returns a fixed GradeDraft and counts calls."""

    def __init__(self, **draft_fields):
        fields = {
            "sufficient_material": True,
            "score": 0.9,
            "missed_points": [],
            "feedback": " Nice work. ",
            "explanation": " It compares evaporation to precipitation. ",
            "identified_gap": None,
        }
        fields.update(draft_fields)
        self.draft = GradeDraft(**fields)
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt, schema):
        assert schema is GradeDraft
        self.calls += 1
        self.prompts.append(prompt)
        return self.draft


class ScriptedDB:
    """Generic fake connection for data-access tests: records every (sql, params) and
    answers each execute() with the next scripted result (a list of row tuples)."""

    def __init__(self, *results):
        self.results = list(results)
        self.executed: list[tuple[str, tuple]] = []
        self.committed = False
        self._rows: list[tuple] = []

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.committed = exc_type is None
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        self.executed.append((_normalize(sql), tuple(params)))
        self._rows = self.results.pop(0) if self.results else []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def sql(self, index):
        return self.executed[index][0]

    def params(self, index):
        return self.executed[index][1]

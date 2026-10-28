"""Shared helper for the opt-in real-Postgres tests (pytest -m postgres).

Only a LOCAL database is accepted; each use creates a throwaway schema, loads
schema.sql into it, points the given modules' get_conn at it, and drops it after.
"""

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()


@contextmanager
def disposable_schema(monkeypatch, modules):
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
        for module in modules:
            monkeypatch.setattr(module, "get_conn", connect)
        yield connect
    finally:
        with psycopg.connect(TEST_DATABASE_URL, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")

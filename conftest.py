"""Test isolation: no real Postgres, Pinecone, OpenAI, or .env secrets.

Runs before any application module is imported. Unless RUN_INTEGRATION=1:
- python-dotenv's load_dotenv becomes a no-op, so .env is never read.
- Required env vars are set to obviously fake values so module-level
  clients (OpenAI(), production guards in main.py) can be constructed.
Every non-integration test also gets guards that fail loudly if code tries to
open a TCP connection, a psycopg connection, or call an OpenAI client.
"""

import os
import socket

import pytest

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION", "").strip() == "1"

if not RUN_INTEGRATION:
    import dotenv

    dotenv.load_dotenv = lambda *args, **kwargs: False

    _FAKE_ENV = {
        "OPENAI_API_KEY": "sk-test-not-a-real-key",
        "DATABASE_URL": "",
        "PINECONE_API_KEY": "",
        "PINECONE_INDEX_NAME": "",
        "API_SHARED_SECRET": "",
        "ALLOW_DEV_USER": "false",
        "RENDER": "",
        "APP_ENV": "test",
        "CHAT_MODEL": "gpt-5-mini",
        "STUDY_MODEL": "gpt-4o-mini",
    }
    for key, value in _FAKE_ENV.items():
        os.environ[key] = value


class LiveServiceBlocked(RuntimeError):
    """Raised when an isolated test tries to reach a real external service."""


class _BlockedClient:
    def __getattr__(self, name):
        raise LiveServiceBlocked(f"OpenAI client access blocked in isolated tests ({name})")


def pytest_collection_modifyitems(config, items):
    skip_integration = pytest.mark.skip(reason="integration test: needs live services; set RUN_INTEGRATION=1")
    skip_postgres = pytest.mark.skip(reason="needs a local disposable Postgres; set TEST_DATABASE_URL")
    for item in items:
        if item.get_closest_marker("integration") and not RUN_INTEGRATION:
            item.add_marker(skip_integration)
        if item.get_closest_marker("postgres") and not os.getenv("TEST_DATABASE_URL", "").strip():
            item.add_marker(skip_postgres)


@pytest.fixture(autouse=True)
def _block_live_services(request, monkeypatch):
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("postgres"):
        yield
        return

    real_connect = socket.socket.connect

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise LiveServiceBlocked(f"Network connection blocked in isolated tests: {address!r}")
        return real_connect(sock, address)

    def blocked_create_connection(address, *args, **kwargs):
        raise LiveServiceBlocked(f"Network connection blocked in isolated tests: {address!r}")

    def blocked_psycopg_connect(*args, **kwargs):
        raise LiveServiceBlocked("Postgres connection blocked in isolated tests")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)

    import psycopg

    monkeypatch.setattr(psycopg, "connect", blocked_psycopg_connect)

    import study

    monkeypatch.setattr(study, "_client", _BlockedClient())

    import main

    monkeypatch.setattr(main, "client", _BlockedClient())

    yield

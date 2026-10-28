"""FastAPI authentication: main.current_user, ownership checks, production guards."""

import importlib.util
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
import study
from fakes import COURSE_ID, OTHER_USER_ID, QUESTION_ID, USER_ID, FakeGrader

SECRET = "test-shared-secret"
MAIN_PATH = Path(main.__file__)


@pytest.fixture
def identities(monkeypatch):
    """Fake ensure_user: deterministic UUID per (provider, subject); records calls."""

    calls = []

    def fake_ensure_user(provider, subject, email=None, name=None):
        calls.append({"provider": provider, "subject": subject, "email": email, "name": name})
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{provider}:{subject}"))

    monkeypatch.setattr(main, "ensure_user", fake_ensure_user)
    monkeypatch.setattr(main, "list_courses", lambda user_id: [])
    return calls


@pytest.fixture
def client():
    return TestClient(main.app)


def configure(monkeypatch, *, secret="", allow_dev=False):
    monkeypatch.setattr(main, "API_SHARED_SECRET", secret)
    monkeypatch.setattr(main, "ALLOW_DEV_USER", allow_dev)


def test_health_needs_no_auth(client):
    assert client.get("/health").status_code == 200


# ── Shared secret ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "headers",
    [
        {"X-Google-Sub": "abc"},
        {"X-Google-Sub": "abc", "X-API-Key": "wrong"},
        {"X-Google-Sub": "abc", "X-API-Key": ""},
        {"X-Google-Sub": "abc", "X-API-Key": SECRET + "x"},
    ],
    ids=["missing_key", "wrong_key", "empty_key", "key_with_suffix"],
)
def test_secret_required_when_configured(monkeypatch, client, identities, headers):
    configure(monkeypatch, secret=SECRET)

    response = client.get("/courses", headers=headers)

    assert response.status_code == 401
    assert identities == [], "user must not be created/looked up before the key is verified"


def test_secret_checked_even_when_dev_user_enabled(monkeypatch, client, identities):
    configure(monkeypatch, secret=SECRET, allow_dev=True)

    assert client.get("/courses").status_code == 401
    assert identities == []


def test_valid_secret_and_sub_authenticates(monkeypatch, client, identities):
    configure(monkeypatch, secret=SECRET)

    response = client.get("/courses", headers={"X-API-Key": SECRET, "X-Google-Sub": "google-123"})

    assert response.status_code == 200
    assert identities == [{"provider": "google", "subject": "google-123", "email": None, "name": None}]


def test_identity_headers_are_url_decoded(monkeypatch, client, identities):
    configure(monkeypatch, secret=SECRET)

    client.get(
        "/courses",
        headers={
            "X-API-Key": SECRET,
            "X-Google-Sub": "g-1",
            "X-User-Email": "a%2Bb%40example.com",
            "X-User-Name": "Zo%C3%AB%20Ng",
        },
    )

    assert identities[0]["email"] == "a+b@example.com"
    assert identities[0]["name"] == "Zoë Ng"


# ── Dev-user fallback ───────────────────────────────────────────────

@pytest.mark.parametrize("headers", [{}, {"X-Google-Sub": ""}], ids=["no_sub", "empty_sub"])
def test_no_identity_and_dev_disabled_is_401(monkeypatch, client, identities, headers):
    configure(monkeypatch, allow_dev=False)

    assert client.get("/courses", headers=headers).status_code == 401
    assert identities == []


def test_dev_user_used_only_when_enabled(monkeypatch, client, identities):
    configure(monkeypatch, allow_dev=True)

    assert client.get("/courses").status_code == 200
    assert identities[0]["provider"] == "dev"
    assert identities[0]["subject"] == "dev-local"


def test_google_sub_takes_precedence_over_dev_user(monkeypatch, client, identities):
    configure(monkeypatch, allow_dev=True)

    client.get("/courses", headers={"X-Google-Sub": "real-user"})

    assert [c["provider"] for c in identities] == ["google"]


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("get", "/courses", {}),
        ("post", "/courses", {"json": {"name": "Bio"}}),
        ("delete", f"/courses/{COURSE_ID}", {}),
        ("get", "/documents", {"params": {"course_id": COURSE_ID}}),
        ("delete", f"/documents/{uuid.uuid4()}", {}),
        ("get", "/questions", {"params": {"course_id": COURSE_ID}}),
        ("post", "/ingest", {"json": {"course_id": COURSE_ID, "filename": "a.txt", "text": "t"}}),
        ("post", "/quiz", {"json": {"course_id": COURSE_ID, "topic": "t"}}),
        ("post", "/grade", {"json": {"question_id": QUESTION_ID, "answer": "a"}}),
        ("post", "/ask", {"json": {"course_id": COURSE_ID, "question": "q"}}),
    ],
)
def test_every_data_endpoint_requires_auth(monkeypatch, client, identities, method, path, kwargs):
    configure(monkeypatch, secret=SECRET)

    response = getattr(client, method)(path, **kwargs)

    assert response.status_code == 401


# ── Cross-user isolation ────────────────────────────────────────────

@pytest.fixture
def as_other_user(monkeypatch):
    """Authenticate as OTHER_USER_ID; COURSE_ID belongs to USER_ID only."""

    owners = {COURSE_ID: USER_ID}
    monkeypatch.setattr(main, "user_owns_course", lambda user_id, course_id: owners.get(str(course_id)) == str(user_id))
    main.app.dependency_overrides[main.current_user] = lambda: OTHER_USER_ID
    try:
        yield
    finally:
        main.app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("get", "/documents", {"params": {"course_id": COURSE_ID}}),
        ("get", "/questions", {"params": {"course_id": COURSE_ID}}),
        ("post", "/quiz", {"json": {"course_id": COURSE_ID, "topic": "water"}}),
        ("post", "/ask", {"json": {"course_id": COURSE_ID, "question": "What is it?"}}),
    ],
)
def test_other_users_course_is_404(as_other_user, client, method, path, kwargs):
    response = getattr(client, method)(path, **kwargs)

    assert response.status_code == 404


def test_other_users_question_cannot_be_graded(monkeypatch, as_other_user, client):
    """load_question_for_grading filters by user_id; simulate 'no row for this user'."""

    grader = FakeGrader()
    monkeypatch.setattr(study, "_call_structured", grader)

    def not_found(*, user_id, question_id):
        assert str(user_id) == OTHER_USER_ID
        raise study.StudyValidationError("Question not found")

    monkeypatch.setattr(study, "load_question_for_grading", not_found)

    response = client.post("/grade", json={"question_id": QUESTION_ID, "answer": "x"})

    assert 400 <= response.status_code < 500
    assert grader.calls == 0


# ── Production startup guards (main.py import-time checks) ───────────

def _import_fresh_main():
    spec = importlib.util.spec_from_file_location("main_under_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "env, message",
    [
        ({"RENDER": "true", "ALLOW_DEV_USER": "true", "API_SHARED_SECRET": SECRET}, "ALLOW_DEV_USER"),
        ({"RENDER": "true", "ALLOW_DEV_USER": "false", "API_SHARED_SECRET": ""}, "API_SHARED_SECRET"),
        ({"APP_ENV": "production", "ALLOW_DEV_USER": "false", "API_SHARED_SECRET": ""}, "API_SHARED_SECRET"),
    ],
    ids=["render_dev_user", "render_no_secret", "app_env_no_secret"],
)
def test_production_refuses_unsafe_auth_config(monkeypatch, env, message):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(RuntimeError, match=message):
        _import_fresh_main()


def test_production_starts_with_safe_auth_config(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("ALLOW_DEV_USER", "false")
    monkeypatch.setenv("API_SHARED_SECRET", SECRET)

    module = _import_fresh_main()

    assert module.API_SHARED_SECRET == SECRET
    assert module.ALLOW_DEV_USER is False

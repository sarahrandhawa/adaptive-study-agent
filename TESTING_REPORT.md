# Testing Report — Adaptive Study Agent

Date: 2026-10-28. All results below come from runs in this repository's `.venv` (Python 3.12.13, pytest 9.1.1, pytest-cov 7.1.0). No production database, Pinecone index or paid OpenAI endpoint was contacted.

## Results

| Run | Passed | Failed | Skipped |
|---|---|---|---|
| Original baseline, `pytest -v` | collection error (duplicate `test_adaptive` module) | | |
| Before fixes, importlib mode | 143 | 13 | 1 |
| After your 3 fixes, before mine | 148 | 8 | 1 |
| After the first round of fixes (229 collected) | 225 | 0 | 4 |
| **Final, `pytest` (242 collected)** | **237** | **0** | **5** |
| Final, `-m regression` | 13 | 0 | — |
| Final, handwritten (`tests/test_adaptive.py`, `tests/test_question_generation.py`, `test_adaptive.py`) | 16 | 0 | — |

The 4 skipped tests are all opt-in:

- 1 `integration` test (live database, `RUN_INTEGRATION=1`).
- 4 `postgres` tests (local disposable database, `TEST_DATABASE_URL`). **These have not been executed yet.** No local Postgres was available because the Docker daemon was not running.

## Coverage (application modules; tests, `course/` and `evals/` excluded)

| Scope | Before | Final |
|---|---|---|
| **Backend** (excludes `rag_ui.py` and `check_db_connection.py`) | 53% | **97%** (736 stmts, 21 missed) |
| All app modules, including the Streamlit UI | 35% | 65% (1102 stmts) |

| Module | Final | Remaining gap |
|---|---|---|
| `courses.py`, `documents.py`, `memory_store.py`, `questions.py`, `rag.py`, `retrieve.py`, `users.py` | 100% | |
| `vectorstore.py` | 99% | One `_jsonify` branch (objects with `model_dump`) |
| `study.py` | 98% | Two context helpers and one fallback path |
| `ingest.py` | 98% | A text that splits into zero chunks |
| `main.py` | 97% | The "unparseable output" branch inside `/ask`, and a few lines with no realistic trigger |
| `db.py` | 90% | `psycopg.connect` itself (blocked by design) |
| `adaptive.py` | 60% | `next_due`, which is **unused by the app**. Deliberately not tested, so dead code doesn't inflate coverage. |

## Methodology

- **Isolation (`conftest.py`):** `.env` is never loaded, credentials are fake, and outbound TCP, `psycopg.connect` and the OpenAI clients are blocked. Any accidental real call fails the test.
- **`FakeDB` (`tests/fakes.py`):**
  - Models only the SQL used to record attempts.
  - Mimics Postgres transactions: writes apply on commit, are discarded on rollback, and reads see committed data only.
  - Mimics Postgres locking: `FOR UPDATE` locks only rows that exist, while advisory locks always lock.
  - A barrier forces concurrent transactions to interleave, so races reproduce deterministically.
  - The `(user_id, submission_id)` unique index is enforced.
  - Unknown SQL fails loudly.
- **`ScriptedDB`:** records every statement and its parameters for data-access tests. These check row mapping, control flow and **`user_id` scoping**, but not whether the SQL is valid Postgres.
- **Pinecone fake (`tests/test_vectorstore.py`):** covers index auto-creation, the readiness wait and its 60-second timeout, the per-namespace store cache, namespaced deletes, and the `check_pinecone` diagnostics (and that the API key never appears in the output).
- **Real-SQL suite (`tests/test_postgres.py`, opt-in):** runs against a throwaway schema and refuses non-local hosts. It checks:
  - the schema, plus migration 001 run twice, ending with exactly one submission index;
  - rollback followed by re-applying the migration;
  - 8 concurrent `record_attempt` calls;
  - submission replay and the unique index.
- **API tests:** FastAPI `TestClient` with dependency overrides. They cover the auth matrix, cross-user 404s, 422/400 validation, and error mapping (database errors → 503, LLM errors → 503, no leaked details).

## Bugs found and fixed

| # | Bug | Fix | Files | Guarded by |
|---|---|---|---|---|
| 1 | NaN or +inf grading score clamped to 1.0 and recorded as **correct** | Your fix: reject non-finite scores. **Follow-up:** it dropped the `score is None` check, so `math.isfinite(None)` raised `TypeError` (caught by `test_missing_score_records_no_attempt`); the None check is restored. | `study.py` | `test_grading.py::test_non_finite_score_is_not_recorded[nan,inf]`, `::test_missing_score_records_no_attempt` |
| 2 | Retried `/grade` counted twice (2 attempts, double mastery and streak update, 2 LLM calls) | Optional `submission_id` on `GradeRequest` (UUID). A replay returns the stored result with `replayed: true` and makes no LLM call. Reusing an ID on another question returns 400. A unique index is the final backstop. The UI keeps one ID per (question, answer text). | `main.py`, `study.py`, `rag_ui.py`, `schema.sql`, `migrations/001_attempts_submission_id.sql` | `test_duplicate_submissions.py` (6 tests, including a guard that identical text with new IDs counts twice) |
| 3 | Concurrent submissions lost updates (2 attempts stored, `attempt_count = 1`) | `pg_advisory_xact_lock(hashtext(user_id), hashtext(question_id))` at the start of `record_attempt`. Unlike `FOR UPDATE`, it also covers the first attempt, before any state row exists. | `study.py` | `test_concurrent_attempts_do_not_lose_updates[first_attempt,existing_state]`, `test_concurrent_retries_of_one_submission_record_one_attempt` |
| 4 | No length limit on grade answers | Your fix: `max_length=10000` | `main.py` | `test_api_validation.py::test_grade_rejects_oversized_answer` |
| 5 | No length limit on quiz topics | Your fix: `max_length=200` | `main.py` | `test_api_validation.py::test_quiz_rejects_oversized_topic` |
| 6 | Quiz drafts without a reference answer or key points were saved | Your fix: reject them in `_question_from_prompt` | `study.py` | `test_quiz_flow.py::test_draft_without_grading_rubric_is_not_saved[…]` |
| 7 | Truncated or filtered LLM output (`LengthFinishReasonError`) escaped the retry loop and returned 500 | Catch `OpenAIError` (the base class) in `_call_structured` and `/ask` | `study.py`, `main.py` | `test_failure_handling.py::test_call_structured_handles_truncated_model_output`, `::test_ask_truncated_model_output_is_503` |
| 8 | Database errors returned unhandled 500s | App-wide `psycopg.Error` handler: logs server-side, returns a generic 503 | `main.py` | `test_failure_handling.py::test_database_outage_is_503[…]`, `test_unique_violation_backstop_is_503` |

## Migration and idempotency review

| Item | Finding | Action |
|---|---|---|
| Index naming | **Defect fixed.** `schema.sql` had an inline `unique (user_id, submission_id)`, auto-named `attempts_user_id_submission_id_key`, while migration 001 creates `attempts_user_submission_key`. A fresh install that also ran the migration would have had two identical indexes. | `schema.sql` now creates the same named index. The Postgres test asserts exactly one. |
| Rollback | There was no down migration. | Added `migrations/001_attempts_submission_id.down.sql`. |
| Cross-user isolation | Lookups and the unique index are scoped by `user_id`, but no test proved it. | Added `test_submission_id_is_scoped_per_user`. |
| Lock correctness | `pg_advisory_xact_lock(int4, int4)` with `hashtext()` is a valid signature. It is transaction-scoped, so it is released at commit or rollback and is compatible with Supabase's transaction-mode pooler. Hash collisions only cause extra serialisation. | None (documented in the README) |
| Backward compatibility | The old code works on the migrated schema (column `NULL`). The new code fails on an **unmigrated** schema (missing column → 503), so the migration must run first. | Covered in the deployment order |
| Replay semantics | A replay returns the stored grade with the *current* `question_state`. `score` is a `real` column, so a replayed 0.9 reads back as roughly 0.89999998. | Documented; no change |
| In-flight retry | A retry made while the first request is still in the LLM call misses the pre-check and grades again (an extra LLM cost). The in-lock check makes it replay, so the state stays correct. | Documented as a limitation |

## Smoke tests (real processes, isolated copy of the app with no `.env`, fake config)

| Check | Result |
|---|---|
| `uvicorn main:app` starts | Pass: "Application startup complete" |
| `GET /health`, `GET /docs` | 200, 200 |
| OpenAPI schema includes `submission_id` | Pass |
| `/courses` with no key, wrong key, or a valid key but no identity | 401, 401, 401 |
| `/courses` with the database unreachable (closed local port) | 503 with a generic message; the fake DB password appears in neither the response nor the server log |
| Startup with `RENDER=true` and no secret, or with `ALLOW_DEV_USER=true` | Refuses to start with the expected `RuntimeError` |
| `streamlit run rag_ui.py` | Server starts: `/_stcore/health` returns `ok`, page returns 200 |

Not verified:

- **422 with the database down:** with the database unreachable, oversized and malformed `/grade` bodies return 503, not 422, because authentication looks up the user first. The 422 path is verified in-process.
- **Streamlit render:** the script render in Streamlit's `AppTest` fails with `st.user has no attribute "is_logged_in"`. The harness has no OIDC context, so the UI render and login flow were **not** verified.
- **Live services:** real Google sign-in, Pinecone, OpenAI and Supabase.

## Remaining risks

- **Real-SQL tests not executed.** `pg_advisory_xact_lock(hashtext…)`, the submission query, migration 001 and its rollback are validated only by fakes until `-m postgres` runs against a local Postgres. **This is the main blocker before deploying.**
- **Spoofable identity.** Header identity is trusted behind the shared secret, and the production guard applies only on Render or with `APP_ENV=production`.
- **Deleted documents.** `load_question_for_grading` doesn't filter deleted chunks.
- **Prompt injection.** It is untested and needs live-model evals.
- **Weak key-point check.** A list of blank strings, e.g. `[" "]`, still passes `_question_from_prompt`.
- **No UI or integration tests.** There are no automated tests for the Streamlit UI or the live Pinecone/OpenAI integration.
- **Brittle live test.** `test_attempt_persistence.py` depends on specific live data.

## Commands

```bash
.venv/bin/python -m pytest                                 # 237 passed, 5 skipped
.venv/bin/python -m pytest -m regression                   # 13 passed
.venv/bin/python -m pytest --cov --cov-report=term-missing
.venv/bin/python -m coverage report --omit='rag_ui.py,check_db_connection.py'   # backend: 97%

# Real-SQL checks against a local, disposable Postgres
docker run --rm -d --name asa-test-pg -e POSTGRES_PASSWORD=test -p 55432:5432 postgres:16
TEST_DATABASE_URL=postgresql://postgres:test@127.0.0.1:55432/postgres .venv/bin/python -m pytest -m postgres
docker stop asa-test-pg
```

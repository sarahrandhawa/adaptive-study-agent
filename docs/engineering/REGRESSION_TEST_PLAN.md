# Regression Test Plan — B1–B8

**Status:** plan only. No test code has been written.
**Code reference:** commit `d7f0bfd`.
**Companion docs:** `BUG_ROOT_CAUSE_ANALYSIS.md`, `REMEDIATION_ROADMAP.md`.

---

## 1. Existing test infrastructure

| Item | Finding |
|---|---|
| Framework | Plain `pytest`-style functions (`def test_...`, bare `assert`). **`pytest` is not in `requirements.txt`** and there is no `pytest.ini`, `pyproject.toml`, `conftest.py` or `tests/` package. |
| `test_adaptive.py` (6 tests) | Pure unit tests of `adaptive.py`. Deterministic, no I/O. **All 6 pass** (run on Oct 26 with plain Python, since pytest isn't installed). Covers `update_mastery`, `priority_score`, `recency_penalty`. `test_update_mastery` uses float `==` on `0.56`; it passes but is fragile — use `pytest.approx`. |
| `test_attempt_persistence.py` (1 test) | Connects to whatever `DATABASE_URL` points at and asserts on a **hard-coded question UUID** for the `dev` user. Not isolated, not repeatable, fails on any other database, and would read production if `.env` points there. **Not run** for that reason. |
| `evals/golden/` | Synthetic notes + 10 quiz topics + 20 grade items + 8 ask items. Includes an injected-instruction grade item (G1-2) — good adversarial coverage. `label` is `null` on all 20 grade items and the referenced runner `evals/run_golden.py` does not exist. |
| `course/test_all_stages.py`, `course/evals/*` | Archived course material; call the real OpenAI API; not related to this app. |
| CI | None (no `.github/`). |
| Mocks | None anywhere. |
| UI tests | None. |
| API contract tests | None. |
| Auth / authorization tests | None. |
| Regression tests for fixed bugs | None. |

### Coverage by module

| Module | Coverage | Critical untested paths |
|---|---|---|
| `adaptive.py` | Good (pure) | `next_due` (unused) |
| `study.py` | None | generation, validation, grading, `record_attempt`, `select_next_question` |
| `main.py` | None | auth dependency, every endpoint, error mapping |
| `courses.py` / `documents.py` / `ingest.py` | None | uniqueness, ownership, soft delete, vector consistency |
| `rag_ui.py` | None | course picker, quiz state, submit guards, rendering |
| `retrieve.py` / `rag.py` | None | prompt construction, empty retrieval |

---

## 2. Strategy

### Layers

| Layer | Purpose | Speed | Runs in CI |
|---|---|---|---|
| **Unit** | Pure logic: scoring math, verdict thresholds, normalisation, markdown normaliser, prompt builders | ms | Yes |
| **Service (mocked I/O)** | `study.py` functions with the model call and DB helpers patched | ms | Yes |
| **API** | FastAPI `TestClient` with `current_user` overridden and data/model functions patched | ms | Yes |
| **DB integration** | Constraints, transactions, idempotency, row locks against a throwaway Postgres | seconds | Yes, once a disposable DB is chosen (see open question) |
| **UI** | `streamlit.testing.v1.AppTest` running `rag_ui.py` with `api_request` patched | seconds | Yes |
| **Prompt contract** | Assert the prompts and schemas contain the rules the fixes rely on | ms | Yes |
| **Live-model evals** | Golden sets against the real model; report metrics, never gate CI | minutes, $ | Manual / nightly, opt-in |

### Tooling — no new runtime dependencies

- **pytest** — the only addition, as a dev-only dependency (`requirements-dev.txt`). Needs approval.
- **FastAPI `TestClient`** — already available (`fastapi[standard]`, `httpx`).
- **`streamlit.testing.v1.AppTest`** — ships with Streamlit ≥ 1.28; already installed.
- **`unittest.mock` / pytest `monkeypatch`** — standard library / pytest.
- **Postgres for DB tests** — either a local Postgres (Docker) or a Supabase branch. Both are decisions for Sarah (cost/tooling). Until then, DB-level cases are written but marked `@pytest.mark.db` and skipped when `TEST_DATABASE_URL` is unset. **Never** read `DATABASE_URL` in tests.

### Mocking strategy

1. **Import safety.** `main.py:57` and `study.py:26` create `OpenAI()` at import, and `main.py:64-65` rejects unknown `CHAT_MODEL`s. A `conftest.py` sets `OPENAI_API_KEY=test`, `CHAT_MODEL=gpt-5-mini`, `ALLOW_DEV_USER=false`, `API_SHARED_SECRET=test-secret`, clears `RENDER`/`APP_ENV`, and unsets `DATABASE_URL` before importing app modules.
2. **Model calls.** Patch `study._call_structured` (quiz/grade) and `main.call_model_structured` (ask) to return fixture `QuizDraft` / `GradeDraft` / `Answer` objects, or raise `StudyModelError` / `APIError` / timeouts. Assert on the prompt string passed in for contract tests.
3. **Patch where used.** `main.py` imports functions by name (`from courses import create_course`, …), so API tests patch `main.create_course`, not `courses.create_course`.
4. **Auth.** `app.dependency_overrides[main.current_user] = lambda: USER_A`. Separate API tests exercise the real `current_user` with headers.
5. **Pinecone.** Patch `retrieve.get_vector_store` / `vectorstore.delete_vectors` with an in-memory fake keyed by namespace and `course_id`.
6. **Streamlit.** In `AppTest`, patch `rag_ui.api_request` with a scripted fake and set `st.user` via the `AppTest` user-info hooks (or patch `st.user`) so the app runs as logged in.

### Fixture data

Reuse the audit's synthetic documents — they make wrong answers easy to detect:

| Fixture | Content | Used for |
|---|---|---|
| `velmora_chunks` | Founding 1142; Saltwort Crisis 1388–1391; Ostrel = shipbuilding; Varn = Council of Tides, 21 seats, 3 per island | B5, B6, cross-section |
| `kestrel_chunks` | PERCH/LAND handshake; MOLT = retransmission on wingbeat gap; hover rule | B4, B5 |
| `lumenite_chunks` | Lm(OH)₂ over 340 °C → LmO; 3-2-1 argon rule | B7, ask grounding |
| `contradictory_chunks` | Two chunks that disagree on a number | B4 boundary |
| `users` | `USER_A`, `USER_B` UUIDs | isolation |
| `courses` | A1, A2 for user A; B1 for user B with the same name as A1 | B1 |
| `question_rows` | Stored question + reference + key points + linked chunks | grading |
| Model fixtures | Canned `QuizDraft`/`GradeDraft` JSON per case, including malformed ones | all LLM tests |

---

## 3. Test cases

Columns: **ID · Bug · Level · Preconditions · Input · Expected · Assertions · Mocks/fixtures · Source files · CI**.
"Post-fix" means the test describes the corrected behaviour and should fail on `d7f0bfd` (write it first, watch it fail, then fix).

### 3.1 Course management (B1)

| ID | Level | Preconditions | Input | Expected | Assertions | Mocks | Files | CI |
|---|---|---|---|---|---|---|---|---|
| CM-01 | API | User A has no courses | `POST /courses {"name":"Biology"}` | 200, course returned | status 200; name `"Biology"`; one row | data layer fake | `main.py:209`, `courses.py:4` | Yes |
| CM-02 | API (post-fix) | A has "Biology" | `POST /courses {"name":"Biology"}` | 409 with friendly message | status 409; no new row | fake | same | Yes |
| CM-03 | Unit + API (post-fix) | A has "Biology" | `"biology"`, `"BIOLOGY"` | 409 | normaliser maps all to one key | — | `courses.py` | Yes |
| CM-04 | Unit + API (post-fix) | A has "Biology 101" | `"  Biology   101 "` | 409 | internal whitespace collapsed, trimmed | — | `courses.py` | Yes |
| CM-05 | API | B has "Biology" | A posts `"Biology"` | 200 | per-user uniqueness only | fake | `courses.py` | Yes |
| CM-06 | DB (post-fix) | A archived "Biology" | A creates "Biology" | Succeeds | partial unique index ignores archived rows | Postgres | migration | db-marked |
| CM-07 | DB (post-fix) | — | Two concurrent creates of "Biology" | Exactly one succeeds | unique index, one 409 | Postgres | migration | db-marked |
| CM-08 | API | A owns A1 | `DELETE /courses/{A1}` | 200; only A1 archived | `archive_course` called with A1 | fake | `main.py:228`, `courses.py:69` | Yes |
| CM-09 | API | — | `DELETE /courses/{random uuid}` | 404 | no rows changed | fake | same | Yes |
| CM-10 | API | B owns B1 | A deletes B1 | 404 (not 403, no existence leak) | B1 untouched | fake | same | Yes |
| CM-11 | API | — | `DELETE /courses/not-a-uuid` | 422 | — | — | `main.py:229` | Yes |
| CM-12 | UI (post-fix) | `/courses` returns two courses with the same name (legacy data) | Render picker; select 2nd option | Both reachable; selected ID is the 2nd | `selected_course_id` equals 2nd ID after rerun; labels differ | `AppTest`, fake API | `rag_ui.py:352-367` | Yes |
| CM-13 | UI (post-fix) | As CM-12, 2nd selected | Click Delete → Cancel | Selection unchanged | `selected_course_id` still 2nd ID | `AppTest` | `rag_ui.py:406-433` | Yes |
| CM-14 | UI | Two distinct courses, 2nd selected | Delete → Yes | `DELETE` sent with 2nd ID | captured request path | `AppTest` | same | Yes |
| CM-15 | UI (post-fix) | — | Create course with existing name | Error shown, picker unchanged | warning text; no `pending_course_id` | `AppTest` | `rag_ui.py:388-404` | Yes |
| CM-16 | API | Missing/incorrect `X-API-Key` with secret set | any course endpoint | 401 | — | real `current_user` | `main.py:68-89` | Yes |
| CM-17 | Unit | `IS_PRODUCTION` and no secret | import `main` | `RuntimeError` | — | env | `main.py:46-49` | Yes |

### 3.2 Answer submission and idempotency (B2, B3)

| ID | Level | Preconditions | Input | Expected | Assertions | Mocks | Files | CI |
|---|---|---|---|---|---|---|---|---|
| AS-01 | Service | Unanswered question | Valid answer | One attempt; mastery 0.5→0.68 for score 0.95 | `record_attempt` called once; returned mastery ≈ 0.68 | grade fixture 0.95; DB fake | `study.py:564-659` | Yes |
| AS-02 | API (post-fix) | AS-01 done with `submission_id=S` | Same body + same `S` | 200, same `attempt_id`; no new attempt; model **not** called | grader mock call count = 1; mastery unchanged | fake | `main.py:298`, `study.py` | Yes |
| AS-03 | DB (post-fix) | Question exists | Two concurrent `/grade` with same `S` | One attempt row | `count(attempts)=1`; `attempt_count=1` | Postgres, threads | `study.py:434-562` | db-marked |
| AS-04 | DB (post-fix) | Question exists | Two concurrent different answers | Two attempts; `attempt_count=2`; mastery = sequential application | no lost update | Postgres | same | db-marked |
| AS-05 | API | — | `answer: ""` | 422 | grader not called | fake | `main.py:143` | Yes |
| AS-06 | API | — | `answer: "   \n\t"` | 400 (today) / 422 (post-fix, validator strips) | grader not called; no attempt | fake | `main.py:143`, `study.py:571` | Yes |
| AS-07 | Service | Answered once | Different answer, new `S` | New attempt; mastery moves from current value | 2 attempts | fake | `study.py` | Yes |
| AS-08 | UI (post-fix) | Q1 answered and graded | Click "Get a question" → Q2 shown → submit with empty box | No `/grade` call; "Write an answer…" notice | fake API saw zero grade calls | `AppTest` | `rag_ui.py:538-600` | Yes |
| AS-09 | UI (post-fix) | Q1 graded with answer X | Load Q2 | Answer widget empty **and** session value empty | `at.text_area(key=…).value == ""` | `AppTest` | `rag_ui.py:556,575-583` | Yes |
| AS-10 | UI (post-fix) | Q1 graded | Click Submit again with unchanged text | No `/grade` call (or button disabled) | call count unchanged | `AppTest` | `rag_ui.py:585` | Yes |
| AS-11 | UI | Answer = whitespace | Submit | Notice, no request | — | `AppTest` | `rag_ui.py:588-590` | Yes |
| AS-12 | Service | Grader raises `StudyModelError` | Valid answer | 503; **no** attempt; mastery unchanged | `record_attempt` not called | model raises | `study.py:596`, `main.py:308` | Yes |
| AS-13 | Service | Grader returns `sufficient_material=false` | Valid answer | `status: insufficient_material`; no attempt | — | fixture | `study.py:599-613` | Yes |
| AS-14 | API | Question belongs to user B | A grades it | 400 "Question not found"; no attempt | — | fake | `study.py:367-395` | Yes |
| AS-15 | API (post-fix) | Question's course archived | Grade | 404/400; no attempt | — | fake | `study.py:367-390` | Yes |

### 3.3 Grading (B5, B6)

All use mocked `GradeDraft` payloads for CI. The **prompt-contract** cases assert that the prompt sent to the model contains the rules; the **scoring** cases assert the deterministic code that turns per-point results into a score (post-fix design from `BUG_ROOT_CAUSE_ANALYSIS.md` §B6).

| ID | Level | Input | Expected | Assertions | Files | CI |
|---|---|---|---|---|---|---|
| GR-01 | Service | All key points `covered` | score 1.0, `correct` | verdict, score | `study.py` | Yes |
| GR-02 | Service | Correct paraphrase → model marks points `covered` | `correct` | — | same | Yes |
| GR-03 | Service | 1 of 2 points covered | 0.5, `partially_correct` | — | same | Yes |
| GR-04 | Service | 0 covered | 0.0, `incorrect` | — | same | Yes |
| GR-05 | Service (post-fix) | Reversed relationship (Varn↔Ostrel) → points `contradicted` | score 0; `incorrect`; feedback lists the swap | contradiction caps score | same | Yes |
| GR-06 | Service (post-fix) | Answer says "14 seats", source 21 → `factual_errors` entry | Feedback mentions 14 vs 21; verdict not `correct` | — | same | Yes |
| GR-07 | Service (post-fix) | Student correctly rejects a false premise (`premise_challenge_valid=true`) | Full credit; question flagged for review | score ≥ 0.85; question status updated | same | Yes |
| GR-08 | Unit | Score boundaries 0.35, 0.849, 0.85 | `partially_correct`, `partially_correct`, `correct` | `verdict_from_score` | `study.py:115-120` | Yes |
| GR-09 | Unit | Model score 1.7 / −0.2 | clamped to 1.0 / 0.0 | — | `study.py:616` | Yes |
| GR-10 | Service | `_call_structured` gets `parsed=None` twice | `StudyModelError`; no attempt | — | `study.py:226-242` | Yes |
| GR-11 | Service | `APIError` then success | One attempt (retry works) | call count 2 | same | Yes |
| GR-12 | Service | Timeout raised | 503; no attempt | — | same | Yes |
| GR-13 | Contract | Build grade prompt | Prompt contains: accept paraphrases; judge premise against sources; list contradicted claims; never invent student content | string asserts | `study.py:48-86` | Yes |
| GR-14 | Contract | Injection answer (golden G1-2) | Prompt places student answer in a clearly delimited block and instructs to treat it as data | string asserts | same | Yes |
| GR-15 | Live eval | Golden `grade_items.jsonl` (after labelling) | Agreement with labels ≥ agreed threshold; G1-2 never `correct` | reported, not gating | `evals/` | Nightly/manual |

### 3.4 Question generation (B4)

| ID | Level | Input | Expected | Assertions | Files | CI |
|---|---|---|---|---|---|---|
| QG-01 | Service | Draft whose `supporting_quotes` all appear in chunks; verifier says supported | Question saved; `generation.validated=true` | `save_question` called once | `study.py:316-365` | Yes |
| QG-02 | Service (post-fix) | Draft "MOLT in the handshake" with verifier `supported=false` | Regenerate once; second draft saved | generator called twice | same | Yes |
| QG-03 | Service (post-fix) | Both drafts unsupported | `insufficient_material` with friendly detail; nothing saved | `save_question` not called | same | Yes |
| QG-04 | Service | Retrieval returns 0 chunks | `insufficient_material`; model not called | — | `study.py:323-331` | Yes |
| QG-05 | Service | Model returns `sufficient_material=false` | `insufficient_material` with model reason | — | `study.py:256-266` | Yes |
| QG-06 | Service (post-fix) | `contradictory_chunks` | Question avoids the contested fact, or verifier rejects | verifier result stored | same | Yes |
| QG-07 | Service | Model returns `parsed=None` twice | 503 via `StudyModelError` | — | `study.py:226-242` | Yes |
| QG-08 | Unit (post-fix) | Quote check: quote not in chunk text (after whitespace normalisation) | rejected | — | new helper | Yes |
| QG-09 | Service | Existing unanswered question wins selection | `mode="existing"`; model not called | — | `study.py:298-310,722-780` | Yes |
| QG-10 | Contract | Build quiz prompt | Contains: one claim per source; do not combine facts from separate sections into a single premise; return supporting quotes | string asserts | `study.py:29-46` | Yes |
| QG-11 | Live eval | Golden `quiz_topics.json` | Premise-support rate; Q10 returns `insufficient_material` | reported | `evals/` | Manual |

### 3.5 Mastery and persistence

| ID | Level | Expected | Files | CI |
|---|---|---|---|---|
| MA-01 | Unit | Replay of audit sequence: 0.5→0.68→(dup ignored)→0.68; 0.5→0.40; 0.30→0.58 | `adaptive.py:3-4` | Yes |
| MA-02 | Service (post-fix) | Duplicate submission leaves `question_state` byte-for-byte unchanged | `study.py:434-562` | Yes |
| MA-03 | Service | Failed grading (exception or insufficient) leaves state unchanged | `study.py:596-613` | Yes |
| MA-04 | Unit (property) | For any old ∈ [0,1] and score ∈ [0,1], new ∈ [0,1] | `adaptive.py` | Yes |
| MA-05 | DB | Attempts and state readable after reconnect (new connection) | schema | db-marked |
| MA-06 | API | User B cannot list A's questions (`/questions?course_id=A1` as B → 404) | `main.py:274-277` | Yes |
| MA-07 | Service | `select_next_question` considers only the requesting user's questions for the topic | `study.py:661-697` | Yes |
| MA-08 | Unit | Replace `test_attempt_persistence.py` with a fixture-based version (seed → grade → assert summary) | — | db-marked |

### 3.6 UI (B7, B8 and general)

| ID | Level | Input | Expected | Files | CI |
|---|---|---|---|---|---|
| UI-01 | UI | Switch course A1 → A2 | Quiz question, grade, topic input, ask input cleared | `rag_ui.py:369-371,308-312` | Yes |
| UI-02 | UI (post-fix) | Switch course | Past questions shows A2 data only on first render after switch (no A1 items) | `rag_ui.py:279-305` | Yes |
| UI-03 | Unit (post-fix) | Markdown normaliser on `"1. a\n2. b\nWhy storage matters\n- c"` and on `"…\n### H"` | Blank line inserted before heading/label | new helper used by `rag_ui.py:190-201` | Yes |
| UI-04 | UI | Ask returns `sources_needed=true` with partial answer | Banner wording matches agreed product copy | `rag_ui.py:190-201` | Yes |
| UI-05 | UI | `/grade` returns 503 | Error notice; no result block; answer preserved | `rag_ui.py:611-615` | Yes |
| UI-06 | UI | `api_request` returns status 0 (unreachable) | "Can't reach the study service" shown | `rag_ui.py:132-133` | Yes |
| UI-07 | UI | Submit disabled while grading / after grading unchanged answer | Button state | `rag_ui.py:585` | Yes |
| UI-08 | UI | Empty course → Ask | No API call; empty-state message (post-fix short-circuit) | `rag_ui.py:501-515` | Yes |
| UI-09 | Manual | Loading indicators visible on Ask, Quiz, Grade, Upload | — | `rag_ui.py` spinners | No (manual checklist) |

---

## 4. CI recommendations

1. **GitHub Actions workflow** (`.github/workflows/test.yml`): Python 3.12 (matches `runtime.txt`), `pip install -r requirements.txt -r requirements-dev.txt`, `pytest -m "not db and not live"`. No secrets needed.
2. **DB job** (after a disposable database is chosen): service container `postgres:16`, apply `schema.sql` + migrations, run `pytest -m db` with `TEST_DATABASE_URL`.
3. **Live evals**: separate manual/nightly workflow with an OpenAI key in repo secrets, results uploaded as an artifact; never required for merge.
4. **Guard against production**: `conftest.py` aborts if `TEST_DATABASE_URL` contains the production host or if `DATABASE_URL` is set.
5. **Pin test-relevant versions** (at least `streamlit`) so `AppTest` behaviour and selectbox serialisation match the deployed app.

---

## 5. Implementation order for the tests

1. `conftest.py`, env isolation, model/API fakes, pytest dev dependency. Move `test_adaptive.py` under `tests/unit/` and switch to `pytest.approx`; retire or rewrite `test_attempt_persistence.py`.
2. **B3 + B2 UI/API tests** (AS-05…AS-11, MA-01…MA-03) — protect learning data first.
3. **B1 tests** (CM-01…CM-17).
4. **B2 DB tests** (AS-03, AS-04, CM-06, CM-07) once a test database exists.
5. **B4 tests** (QG-*), then **B5/B6** (GR-*), including prompt contracts.
6. **B7/B8/UI** (UI-*).
7. Golden-set runner and labels for GR-15 / QG-11.

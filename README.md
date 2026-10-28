# Adaptive Study Agent

An AI study assistant that turns a student's own course materials into grounded Q&A and adaptive practice. Students upload notes (PDF or TXT), ask questions answered only from those notes, and get short-answer quiz questions. Each answer is graded against a stored rubric, and the next question is chosen based on what the student is weakest at.

## The problem

Generic chatbots answer from general knowledge, not from the student's syllabus, and they don't remember what the student has already mastered. Re-reading notes is passive, and hand-writing practice questions is slow. This project grounds every answer and question in the student's uploaded material and keeps a per-question learning record, so practice concentrates on weak areas instead of repeating what's already known.

## Features

- **Course workspaces:** each user has private courses with their own documents and question bank.
- **Ingestion:** PDF/TXT text is chunked (800 characters, 100 overlap) and stored in Postgres, then embedded into Pinecone in a per-user namespace.
- **Grounded Q&A (`/ask`):** retrieval-augmented answers with a confidence score, a "sources needed" flag, and the token count, latency and cost of each call.
- **Quiz generation (`/quiz`):** generates one question per request using only the retrieved material, with a reference answer and key points (drafts missing either are rejected), or reuses a question from the existing bank when that's the better choice.
- **Rubric grading (`/grade`):** the LLM scores 0–1 against the stored reference answer and key points. The **verdict is decided in code** from that score, and the attempt is recorded once even if the client retries.
- **Progress tracking:** per-question mastery, attempt count, correct streak and recency.

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2 |
| UI | Streamlit (Google sign-in via `st.login`) |
| LLM | OpenAI chat completions with structured outputs (`gpt-5-mini` for `/ask`, `gpt-4o-mini` for quiz and grading, both configurable) |
| Embeddings / vectors | `text-embedding-3-small` (1536-d) in a Pinecone serverless index, via LangChain |
| Database | PostgreSQL (Supabase), accessed with psycopg 3 |
| Tests | pytest, pytest-cov, in-memory database and LLM fakes |

## Architecture

```text
 ┌──────────────────────────┐   HTTPS + identity headers     ┌──────────────────────────────┐
 │  Streamlit UI (rag_ui)   │  X-Google-Sub / X-User-Email   │  FastAPI (main.py)           │
 │  Google OIDC (st.login)  │ ─────────────────────────────► │  current_user(): shared      │
 │  PDF/TXT text extraction │  X-API-Key (shared secret)     │  secret + identity → user_id │
 └──────────────────────────┘                                └──────────────┬───────────────┘
                                                                            │
        ┌───────────────────────────────┬───────────────────────────────────┼───────────────────────┐
        ▼                               ▼                                   ▼                       ▼
  ingest.py                      retrieve.py + rag.py                study.py                 courses / documents /
  chunk → Postgres rows          Pinecone similarity search          quiz generation,         questions / users
  → Pinecone upsert              (namespace = user_id,               grading, attempt         (SQL data access)
  (namespace = user_id)          filter = course_id) → prompt        recording, selection
        │                               │                                   │
        ▼                               ▼                                   ▼
 ┌──────────────┐               ┌──────────────┐                    ┌──────────────┐
 │  Pinecone    │               │  OpenAI      │                    │  PostgreSQL  │
 │  (vectors)   │               │  (LLM/embed) │                    │  (Supabase)  │
 └──────────────┘               └──────────────┘                    └──────────────┘
```

**Data flow for one practice loop:**

1. `POST /quiz {course_id, topic}`: verify course ownership, then find or create the topic row and run `select_next_question`.
   - **Reuse:** return a question from the bank with no LLM call.
   - **Generate:** retrieve the top-5 chunks, have the LLM return a `QuizDraft` (question, reference answer, key points), validate it, and save it with links to its source chunks.
2. `POST /grade {question_id, answer, submission_id}`: load the question **scoped to the user** with its source chunks.
   - If the `submission_id` was already graded, the stored result is returned without calling the LLM.
   - Otherwise the LLM returns a `GradeDraft`. Non-finite or missing scores are rejected; the score is clamped to [0, 1] and mapped to a verdict.
   - `record_attempt` then takes a per-(user, question) advisory lock, inserts the attempt, and updates `question_state`.

**Isolation:** every SQL query filters by `user_id`. Pinecone vectors live in a namespace per user, and retrieval is filtered by `course_id`. Another user's course returns 404.

## How the adaptive algorithm works

This is a **deterministic heuristic, not a trained machine-learning model.** No model is trained or fine-tuned on student data. The LLM is used only to write questions and to score an answer against a rubric; all adaptation decisions are plain arithmetic in `adaptive.py` and `study.py`.

1. **Score to verdict:** the LLM returns a score in [0, 1], and code assigns the verdict.
   - `≥ 0.85` → correct
   - `≥ 0.35` → partially correct
   - otherwise → incorrect
2. **Question mastery** is an exponential moving average, starting from a neutral 0.5:
   `mastery_new = 0.6 × mastery_old + 0.4 × score`.
3. **Topic mastery** is the mean mastery of the topic's answered questions, or 0.5 if none have been answered.
4. **Priority** (lower means "practise this next"):
   - Unanswered question: `0.5 × topic_mastery`.
   - Answered question: `mastery + recency_penalty`, where `recency_penalty = 0.5 × max(0, 1 − hours_since_last_attempt / 24)`. A question just answered is pushed back, and the penalty fades to zero after 24 hours.
5. **Reuse vs generate:** the question with the lowest priority is compared with a hypothetical new question, whose priority is `0.5 × topic_mastery`.
   - A new question is generated only when it is strictly better.
   - An unanswered existing question wins ties, so the bank is used before new questions are paid for.
   - Effect: weak, stale questions come back; fresh mistakes wait a while; well-mastered topics review existing questions instead of generating more.

## API endpoints

All endpoints except `/health` require authentication. In production that means `X-API-Key` (the shared secret) plus `X-Google-Sub`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/courses` | Create a course `{name}` |
| GET | `/courses` | List your active courses |
| DELETE | `/courses/{course_id}` | Archive (soft-delete) a course |
| POST | `/ingest` | Add a document `{course_id, filename, text}`. An identical re-upload returns `unchanged`; a new version replaces the old one. |
| GET | `/documents?course_id=` | List documents in a course |
| DELETE | `/documents/{document_id}` | Delete a document and its vectors |
| GET | `/questions?course_id=` | Question history with mastery (never includes answers) |
| POST | `/quiz` | Next question `{course_id, topic ≤ 200 chars}` |
| POST | `/grade` | Grade `{question_id, answer ≤ 10,000 chars, submission_id?}` |
| POST | `/ask` | Grounded answer `{course_id, question ≤ 1,000 chars}` |

**Idempotent grading:**

- The client generates a UUID `submission_id` per submission and reuses it on retries.
- A repeat returns the stored grade with `"replayed": true`, and no LLM call or mastery update is made. The replayed `mastery` and `attempt_count` reflect the current state.
- Reusing an ID on a different question returns 400. IDs are scoped per user.
- Omitting `submission_id` keeps the old behaviour, where every request is a new attempt.

**Error handling:**

- **422:** invalid input.
- **400:** a domain validation error.
- **401:** unauthenticated.
- **404:** a resource not owned by you.
- **503:** the database, vector store or LLM is unavailable, with a generic message. Details are logged server-side only and never returned.

## Run locally

Requires Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt        # app + test dependencies
cp .env.example .env                       # then fill in your own values
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM and embeddings |
| `CHAT_MODEL` | `/ask` model (must have a price entry in `main.py`) |
| `STUDY_MODEL` | Quiz and grading model (default `gpt-4o-mini`) |
| `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` | Vector store (`PINECONE_AUTO_CREATE_INDEX=true` to create the index) |
| `DATABASE_URL` | Postgres connection string |
| `API_SHARED_SECRET` | Must match the UI's value; required in production |
| `ALLOW_DEV_USER` | `true` only for local development without Google sign-in |

To set up the database, apply `schema.sql` to a new database. For an existing one, apply the files in `migrations/` instead (see Deployment).

Start the API and the UI in two terminals:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
streamlit run rag_ui.py          # http://localhost:8501
```

The UI reads `RAG_API_URL`, `API_SHARED_SECRET` and an `[auth]` section (Google OIDC) from the environment or `.streamlit/secrets.toml`. See `.streamlit/secrets.toml.example`.

## Testing

```bash
python -m pytest                          # isolated suite: no network, DB, Pinecone or OpenAI
python -m pytest --cov --cov-report=term-missing
```

- **Isolation:** `conftest.py` stops the app from reading `.env`, injects fake credentials, and blocks outbound TCP, `psycopg.connect` and the OpenAI clients. A test that tries to reach a real service fails.
- **Fakes:** an in-memory Postgres stand-in models commit, rollback and lock semantics, so duplicate and concurrent submissions can be reproduced deterministically.
- **Opt-in suites:**
  - `-m postgres`: real-SQL checks against a **local disposable** Postgres (`TEST_DATABASE_URL`; non-local hosts are refused).
  - `-m integration` with `RUN_INTEGRATION=1`: live database checks. Run these deliberately.
- **Legacy material:** the course exercises in `course/` are archived and excluded from discovery.

Current results and coverage are in [TESTING_REPORT.md](TESTING_REPORT.md).

## Deployment

The code supports this layout:

- **FastAPI** on Render. `RENDER=true` (or `APP_ENV=production`) enables the startup guards.
- **Streamlit** with Google OIDC.
- **Supabase** Postgres.
- **Pinecone** serverless.

In production the API refuses to start if `ALLOW_DEV_USER=true` or `API_SHARED_SECRET` is missing.

**Schema changes** are versioned in `migrations/` (`NNN_name.sql`, with a matching `NNN_name.down.sql` rollback):

- **Order:** apply each migration *before* deploying the code that depends on it.
- **Migration 001** (`001_attempts_submission_id.sql`) adds `attempts.submission_id` plus the unique index `attempts_user_submission_key (user_id, submission_id)`. It is idempotent and backward compatible: the previous code leaves the column `NULL`, and NULLs never conflict.
- **Fresh databases:** `schema.sql` already includes the same column and index.
- **Connection pooling:** concurrency control uses a *transaction-scoped* advisory lock (`pg_advisory_xact_lock`), so it works through Supabase's transaction-mode pooler. Don't replace it with a session-level lock.

## Limitations and future work

- **Identity trust:** the API trusts identity headers from the UI, protected by the shared secret. A deployment that isn't on Render and doesn't set `APP_ENV=production` only logs a warning when the secret is missing. Verifying the Google ID token in the API would remove this trust.
- **Spaced repetition isn't wired in.** `adaptive.next_due` exists, but nothing calls it and `question_state.next_due_at` is never written. Scheduling is driven only by the recency penalty.
- **Deleted documents:** grading doesn't filter deleted source chunks, so questions from a deleted document can still be served and graded against the old text.
- **No question deduplication:** generated questions aren't checked against the existing bank, and topic rows are created even when no material is found.
- **No prompt-injection defence:** student answers are inserted into the grading prompt without delimiters. This needs adversarial evals against the golden set in `evals/golden/`.
- **No rate limiting:** the `usage_daily` table is unused.
- **Row-level security:** RLS is enabled, but no policies are defined. Access control is enforced in the API.
- **Untested areas:** the Streamlit UI has no automated tests. The real-SQL suite (`-m postgres`) has to be run against a local Postgres; until then, the locking and migration SQL are verified only by in-memory fakes.
- **In-flight retries:** a retry sent while the first request is still being graded costs a second LLM call. The lock still records only one attempt.

# Remediation Roadmap

**Status:** proposal for review. Nothing here has been implemented.
**Code reference:** commit `d7f0bfd`.
**Inputs:** `BUG_ROOT_CAUSE_ANALYSIS.md`, `REGRESSION_TEST_PLAN.md`.

Priority: **P0** data integrity, security or incorrect learning outcomes · **P1** correctness and reliability · **P2** UI and usability.
Complexity: **S** < half a day · **M** 1–2 days · **L** 3+ days.

---

## 1. Sequence at a glance

| Step | Item | Priority | Bugs | Migration | Complexity | Depends on |
|---|---|---|---|---|---|---|
| 0 | Test harness and fakes | enabler | all | No | S | — |
| 1 | Stop stale answers reaching the grader | P0 | B3 | No | S | 0 |
| 2 | Block repeat submissions + lock state row | P0 | B2 | No | S | 0 |
| 3 | Reject duplicate course names; unique picker labels; honest delete copy | P0 | B1 | No | S | 0 |
| 4 | Migration tooling decision + duplicate cleanup + partial unique index | P0 | B1 | **Yes** | M | 3 |
| 5 | Submission idempotency key | P0 | B2 | **Yes** | M | 2, 4 (tooling) |
| 6 | Question-generation grounding check | P0 | B4 | No (uses existing `generation` jsonb) | M | 0 |
| 7 | Grader v2: per-point rubric, contradictions, premise rule, deterministic score | P0 | B5, B6 | No | M–L | 6 |
| 8 | Decide what to do with history graded by v1 / bad questions | P0 (decision) | B4–B6 | Maybe | S | 6, 7 |
| 9 | Reliability: vector/DB ordering, timeouts, logging, pooling, archived-course checks | P1 | — | Maybe | M | 0 |
| 10 | Render fixes: markdown normaliser, course-switch placeholders, lazy tabs | P2 | B7, B8 | No | S | 0 |
| 11 | UX items from the audit | P2 | — | No | S–M | 10 |

### Why this order minimises risk

1. **Tests first (0).** There is no harness, and steps 1–3 change the code paths that write learning data. Every later step starts by adding its failing test.
2. **Stop corrupting data before changing how it's scored (1–2 before 7).** Grading and mastery are coupled: every graded submission moves mastery. If the grader is re-tuned while stale or duplicate submissions still reach it, the before/after comparison is polluted and more bad attempts accumulate. Steps 1–2 are small UI/service changes with no migration.
3. **Cheap guard before the constraint (3 before 4).** The API check stops new duplicates immediately; the unique index follows once existing duplicates are cleaned up, so the migration can't fail on legacy rows.
4. **Generation before grading (6 before 7).** The grader treats the stored reference answer and key points as ground truth. Fixing the grader first would evaluate it against references that may themselves be wrong. Validating generation also produces the data to judge whether grader v2 is better.
5. **UI polish last (10–11).** B7/B8 are display-only.

---

## 2. P0 items

### Step 0 — Test harness (enabler)

- **Root cause addressed:** no isolated tests, no CI (`REGRESSION_TEST_PLAN.md` §1).
- **Files:** new `tests/conftest.py`, `requirements-dev.txt` (pytest), move `test_adaptive.py` → `tests/unit/`; retire `test_attempt_persistence.py` (reads a live DB with a hard-coded ID).
- **Implementation:** env isolation before imports; fakes for `study._call_structured`, `main.call_model_structured`, `rag_ui.api_request`, vector store; `dependency_overrides` for `current_user`; production guard.
- **Risks:** none to the app. Needs approval for the pytest dev dependency.
- **Acceptance:** `pytest -m "not db and not live"` runs offline, no secrets, < 30 s; existing 6 adaptive tests pass.
- **Commit:** `test: add isolated pytest harness and fakes`.

### Step 1 — B3: stale answer submitted for a new question

- **Root cause:** `quiz_answer` is a form widget key that is also reset programmatically (`rag_ui.py:312,556,575-583`); the browser can resubmit the previous value although the box looks empty.
- **Implementation (recommended):**
  1. Give the answer widget a per-question key: `key=f"quiz_answer_{question_id}"`. A new question gets a brand-new widget with no browser-side history, so nothing needs resetting.
  2. Read the answer from that key on submit; delete old keys when the question changes.
  3. API: add a Pydantic validator that strips `GradeRequest.answer` and rejects whitespace-only with 422 (keeps `study.py:571-574` as a second guard).
  4. Optional: include `question_id` in the payload check — the UI already does; the server already validates ownership.
- **Alternative:** move the textarea out of `st.form` and use a plain button. Simpler state model, but loses Cmd/Ctrl+Enter submit and causes reruns per keystroke-commit.
- **Risks:** low; UI only plus a stricter (still compatible) validator.
- **Migration:** no.
- **Tests:** AS-05, AS-06, AS-08, AS-09, AS-11.
- **Acceptance:** after loading a new question, submitting an empty box never calls `/grade`; whitespace-only answers return 422; no attempt row is created for either.
- **Before closing:** run the read-only query in `BUG_ROOT_CAUSE_ANALYSIS.md` §B3 to confirm the stored `answer_text`. If it is **not** the previous answer, revisit the mechanism before shipping.
- **Complexity:** S. **Commit:** `fix(ui): per-question answer widget so stale text can't be graded (B3)`.

### Step 2 — B2 (part 1): repeat submissions and lost updates

- **Root cause:** Submit stays enabled with the same text (`rag_ui.py:575-615`); `record_attempt` reads state without a lock (`study.py:451-458`).
- **Implementation:**
  1. UI: remember the last graded `(question_id, normalised answer)`; if unchanged, don't call `/grade` (show "Already graded — change your answer or get a new question"). Disable the button while a request is in flight.
  2. Service: in `record_attempt`, `INSERT … ON CONFLICT DO NOTHING` a default `question_state` row, then `SELECT … FOR UPDATE` before computing mastery. Same transaction as today.
- **Alternative:** `pg_advisory_xact_lock(hashtext(user_id||question_id))` — avoids the pre-insert, slightly more opaque.
- **Risks:** low. Lock scope is one row for one user; contention is negligible.
- **Migration:** no.
- **Tests:** AS-01, AS-04 (db), AS-07, AS-10, MA-01…MA-04.
- **Acceptance:** resubmitting identical text from the UI makes no API call; concurrent different answers produce N attempts and `attempt_count = N`.
- **Complexity:** S. **Commit:** `fix: prevent duplicate grading from the UI and lock question_state (B2)`.

### Step 3 — B1 (part 1): duplicate names and wrong delete target

- **Root cause:** no name uniqueness (`courses.py:4-21`, `schema.sql:15-21`) + Streamlit resolves the selectbox by label, last duplicate wins.
- **Implementation:**
  1. `courses.py`: `normalize_course_name()` = trim, collapse internal whitespace, `casefold()` for comparison; keep the user's display casing. Before insert, check for an active course with the same key → raise a `DuplicateCourseError`; `main.py` maps it to **409** "You already have a course called …".
  2. `rag_ui.py`: build **unique labels** for the picker even for legacy duplicates, e.g. append ` (created 2026-10-26 15:13)` or ` (2)` when a name repeats. Labels must be unique for Streamlit to round-trip the right ID.
  3. Delete confirmation names the course with that unique label.
  4. Fix the delete copy (`rag_ui.py:415-416`) to match reality (archived, recoverable), or decide to hard-delete (see step 9).
- **Alternative:** case-sensitive uniqueness only — simpler but still lets "biology"/"Biology" coexist; not recommended.
- **Risks:** the API check alone is racy (two tabs); step 4 closes that.
- **Migration:** no.
- **Tests:** CM-01…CM-05, CM-08…CM-17.
- **Acceptance:** creating a duplicate (any case/whitespace variant) returns 409 and no row; with legacy duplicates, every course is selectable and Delete archives exactly the selected ID.
- **Complexity:** S. **Commits:** `fix(api): reject duplicate course names (B1)`, `fix(ui): unique course labels and accurate delete copy (B1)`.

### Step 4 — B1 (part 2): database guarantee

- **Prerequisite decision:** how migrations are applied (Supabase SQL editor with numbered files in `migrations/`, or Supabase CLI). Production schema should be diffed against `schema.sql` first — it may have drifted.
- **Migration `001_course_name_unique.sql`:**
  1. Report duplicates (read-only query in the root-cause doc).
  2. Rename later duplicates to `"<name> (2)"`, `"(3)"`… — **no deletes**, no archives.
  3. `create unique index courses_user_active_name_uidx on courses (user_id, lower(regexp_replace(btrim(name), '\s+', ' ', 'g'))) where archived_at is null;`
  4. Handle `unique_violation` in `create_course` → same 409.
- **Risks:** renaming is user-visible; tell affected users (currently known: Sarah's audit duplicate). Index creation on a small table is instant.
- **Tests:** CM-06, CM-07 (db).
- **Acceptance:** concurrent duplicate creates → exactly one row; archived names can be reused.
- **Complexity:** M (mostly process). **Commit:** `db: enforce unique active course names per user (B1)`.

### Step 5 — B2 (part 2): server-side idempotency

- **Root cause:** `/grade` has no request identity; retries and double clicks create new attempts.
- **Migration `002_attempt_submission_id.sql`:** `alter table attempts add column submission_id uuid; create unique index attempts_user_submission_uidx on attempts (user_id, submission_id) where submission_id is not null;`
- **Implementation:** `GradeRequest.submission_id: UUID | None = None` (backward compatible). UI generates one per (question, answer text). In `grade_student_answer`, look up `(user_id, submission_id)` first and return the stored result **without** calling the model; on insert conflict, return the existing attempt.
- **Alternative:** content-hash dedupe within a time window (no client change). Blocks a deliberate identical retry; product decision.
- **Risks:** must store enough of the result on `attempts` to rebuild the response (it already stores verdict/score/feedback/explanation/missed points).
- **Tests:** AS-02, AS-03.
- **Acceptance:** same `submission_id` twice (sequential or concurrent) → one attempt, one model call, identical response.
- **Complexity:** M. **Commits:** `db: add attempts.submission_id`, `feat(api): idempotent /grade (B2)`.

### Step 6 — B4: grounded question generation

- **Root cause:** one unchecked call generates question + reference + key points (`study.py:29-46,256-266,316-353`).
- **Implementation:**
  1. Extend `QuizDraft` with `supporting_quotes: list[str]`; prompt: one claim per quote, don't merge facts from different sections into one premise.
  2. Deterministic check: each quote must appear (whitespace-normalised) in a retrieved chunk.
  3. Verifier call (small model): list the question's premises and the reference answer's claims, each with `supported` + quote. Any unsupported → regenerate once with the reason; second failure → `insufficient_material` ("I couldn't write a reliable question on that topic yet").
  4. Write `questions.generation = {model, prompt_version, verifier: {...}}`.
  5. Link only the chunks that contain the supporting quotes in `question_sources` (tighter grading context).
- **Alternatives:** (a) deterministic quote check only — no extra cost, catches invented text but not wrong combinations like B4; (b) verifier only on low-similarity retrievals — cheaper, misses cases like B4 where retrieval was fine. Recommended: quote check always + verifier on every newly generated question (reused questions cost nothing).
- **Cost/latency:** +1 small-model call on the generate path (~1–3 s); regenerate path +2.
- **Tests:** QG-01…QG-10.
- **Acceptance:** on the golden Kestrel fixture, a draft asserting MOLT is part of the handshake is rejected; every saved question has `generation.verifier.all_supported = true`.
- **Complexity:** M. **Commits:** `feat(study): require supporting quotes in quiz drafts`, `feat(study): verify and regenerate unsupported questions (B4)`.

### Step 7 — B5/B6: grader v2

- **Root cause:** holistic score; literal key-point coverage; no premise or contradiction handling (`study.py:48-86,106-113`).
- **Implementation:**
  1. `GradeDraft` v2: `key_point_results: [{point, status: covered|partial|missing|contradicted, evidence}]`, `factual_errors: [{student_claim, source_says}]`, `premise_issue: {valid_challenge: bool, explanation}`.
  2. Prompt: accept paraphrases; judge facts against **sources first**, reference answer second; list wrong claims explicitly (numbers, names, swaps); if the student correctly identifies a premise the sources contradict, give full credit and set `valid_challenge`.
  3. Score computed in code: covered 1, partial 0.5, missing 0, contradicted 0; any contradiction caps verdict at `partially_correct`; a swapped core relationship → `incorrect`. Feedback lists `factual_errors`.
  4. `valid_challenge=true` → mark the question `status='needs_review'` so it isn't served again.
  5. Write `grader_version='v2'` (column already exists).
  6. Student-facing wording: second person throughout ("You confused…", fixing "The student…").
- **Alternatives:** keep holistic score but add instructions — cheaper but untestable; not recommended.
- **Risks:** score distribution shifts; mastery trajectories change for new attempts. Evaluate on the golden set (label it first) before switching.
- **Tests:** GR-01…GR-14; GR-15 as an offline gate for the rollout decision.
- **Acceptance:** MOLT answer from the audit → `correct`; swapped Varn/Ostrel answer → `incorrect` with the swap named; "14 seats" flagged against 21; golden-set agreement ≥ the threshold agreed in review.
- **Complexity:** M–L. **Commits:** `feat(study): grader v2 schema and deterministic scoring`, `feat(study): grader v2 prompt (B5, B6)`.

### Step 8 — Historical data decision

Questions created before step 6 may have false premises; attempts graded before step 7 used v1. Options: (a) leave history, apply new logic going forward (simplest); (b) run the verifier over existing active questions and set `status='needs_review'` on failures; (c) recompute mastery from attempts with v2 — **not recommended** without re-grading, which costs money and changes users' numbers. Recommended: (a) + (b).

---

## 3. P1 items (step 9)

| Item | Root cause | Implementation | Migration | Complexity |
|---|---|---|---|---|
| Vector/DB ordering on ingest | Pinecone upsert before DB commit (`ingest.py:216-224`) | Commit rows with `status='indexing'`, upsert vectors, then set `active`; on failure mark `failed` and delete vectors | No (status is text) | M |
| Vector/DB ordering on delete | Vectors deleted before soft-delete commits (`documents.py:66-90`) | Commit `status='deleting'` first, then delete vectors, then `deleted` | No | S |
| Course archive honesty | Archive keeps everything (`courses.py:69-86`) | Decide: keep soft delete (fix copy, step 3) **or** also delete course vectors and mark documents deleted | No | S–M |
| `/grade` on archived course | Ownership only (`study.py:367-390`) | Join `courses` and require `archived_at is null` | No | S |
| Timeouts | UI 120 s vs OpenAI default 600 s + retries | Set explicit OpenAI client timeout (e.g. 45 s) and `max_retries=1`; keep `_call_structured` loop at 2 | No | S |
| Observability | `_call_structured` swallows the first error | Log model, attempt number, latency and error class; log prompt version | No | S |
| DB connections | New connection per helper (`db.py`) | `psycopg_pool.ConnectionPool` (part of psycopg; may need the `pool` extra — approval) | No | S |
| Auth configuration | Secret only enforced with `RENDER`/`APP_ENV` (`main.py:41-49`) | Require the secret unless `ALLOW_DEV_USER=true` is explicitly set | No | S |

---

## 4. P2 items (steps 10–11)

| Item | Bug | Implementation | Complexity |
|---|---|---|---|
| Markdown normaliser | B7 | Helper inserting a blank line before headings/bold-only lines after list blocks; used in `render_ask_result` and grade explanation | S |
| Course switch placeholders | B8 | On course change, render course-scoped sections into `st.empty()` placeholders cleared first; reset `quiz_topic_input` and the Ask input | S |
| Lazy tabs | B8 / latency | Load `/documents` and `/questions` only for the active section (e.g. `st.radio`/segmented control instead of `st.tabs`, or fetch on expander open) | S–M |
| Audit UX items | — | Next-question button; explain Score vs Mastery; "your notes" wording; partial-answer banner; empty-course short-circuit; logged-out description | S each |

---

## 5. Suggested commit boundaries

One concern per commit, each with its tests:

1. `test: add isolated pytest harness and fakes`
2. `fix(ui): per-question answer widget so stale text can't be graded (B3)`
3. `fix(api): strip and reject whitespace-only answers (B3)`
4. `fix(ui): don't regrade an unchanged answer (B2)`
5. `fix(study): lock question_state row when recording attempts (B2)`
6. `fix(api): reject duplicate course names (B1)`
7. `fix(ui): unique course labels and accurate delete copy (B1)`
8. `db: enforce unique active course names per user (B1)` (+ cleanup script)
9. `db: add attempts.submission_id` → `feat(api): idempotent /grade (B2)`
10. `feat(study): supporting quotes + verifier for quiz drafts (B4)`
11. `feat(study): grader v2 (B5, B6)`
12. P1 commits, one per row in §3
13. P2 commits, one per row in §4

---

## 6. Verification and deployment

1. **Branching:** feature branch per step; PR includes the failing-then-passing tests.
2. **Staging:** a separate Streamlit app + API + Supabase project/branch + Pinecone index (or namespace prefix) so migrations and grading changes are tried before production. Creating these may cost money — decision needed.
3. **Migrations:** apply to staging, run DB tests, then production during low traffic; migrations are additive and compatible with the previous app version (old app ignores `submission_id`; the unique index only affects new duplicate inserts).
4. **Order of deploys:** API before UI for steps that add optional fields (5); UI-only steps (1, 2-UI, 3-UI) can ship independently.
5. **Grader rollout (7):** label the golden set, compare v1 vs v2 offline, then switch; keep v1 code path behind `GRADER_VERSION` env for one release to allow rollback.
6. **Post-deploy checks:** repeat the Oct 26 audit journey on staging/production with synthetic courses; confirm B1–B8 no longer reproduce; watch logs for 409s, 422s and verifier rejection rates.
7. **Rollback:** each step is independently revertible; indexes can be dropped; `submission_id` is nullable.

---

## 7. Decisions needed before implementation

1. Approve `pytest` as a dev dependency (and `psycopg[pool]` later if pooling is wanted).
2. Disposable database for DB tests: local Docker Postgres, or a Supabase branch (possible cost)?
3. How migrations will be applied, and whether production may be diffed against `schema.sql` (read-only).
4. Course deletion: keep soft delete (fix copy) or hard-delete materials and vectors?
5. Should an identical answer resubmitted to the same question ever count as a new attempt?
6. Mastery prior: keep 0.5 starting value (100% first try shows 70%) or change the model? Out of scope for the bugs, but affects how results are explained.
7. Acceptable extra latency/cost for the question verifier (step 6).
8. Golden-set labelling owner and the agreement threshold for grader v2.
9. Confirm deployed `STUDY_MODEL`, `CHAT_MODEL`, Streamlit version, API host, and that `API_SHARED_SECRET` is set there.

# Bug Root Cause Analysis — B1–B8

**Source of observations:** `USER_EXPERIENCE_AUDIT.md` (Oct 26, 2026) and its screenshots.
**Code reference:** commit `d7f0bfd`. Line numbers are exact for that commit.
**Method:** code reading; replaying the mastery formula against the audit's numbers; reading Streamlit's selectbox implementation (`streamlit/streamlit` main, `lib/streamlit/elements/lib/options_selector_utils.py:322-344`, `lib/streamlit/elements/widgets/selectbox.py:82-172`). No production data was read.

### Confidence scale

- **Confirmed** — the code path deterministically produces the observed behaviour, and the audit's observations match it.
- **Strong hypothesis** — the code makes the behaviour likely and the observations fit, but one link needs runtime evidence (listed).
- **Unverified possibility** — plausible, consistent with the code, no direct evidence.

### Key supporting evidence: the mastery numbers

`update_mastery(old, score) = 0.6·old + 0.4·score` (`adaptive.py:3-4`), starting at `0.5` for a new question (`study.py:462-463`). Every mastery value in the audit reproduces exactly:

| Audit event | old → score | Computed | Audit showed |
|---|---|---|---|
| First correct answer | 0.5 → 0.95 | 0.680 | 68% |
| Same answer resubmitted (B2) | 0.68 → 0.95 | 0.788 | 79% |
| Swapped answer (B6) | 0.5 → 0.25 | 0.400 | 40% |
| "Blank" answer (B3) | 0.5 → 0.00 | 0.300 | 30% |
| Correct after that | 0.30 → 1.00 | 0.580 | 58% |
| MOLT answer (B5) | 0.5 → 0.65 | 0.560 | 56% |

This shows every submission, including the "blank" one, went through `record_attempt` as a normal graded attempt.

---

## Summary table

| Bug | Severity | Root cause | Evidence | Files/functions | Confidence | Proposed fix |
|---|---|---|---|---|---|---|
| B1 | High | No course-name uniqueness + Streamlit selectbox resolves selection by display label, last duplicate wins | Code + Streamlit source + audit behaviour match exactly | `courses.py:4-21,24-47`; `main.py:209-220`; `rag_ui.py:352-371,406-433`; Streamlit `create_mappings` | Confirmed | Reject duplicates (API + partial unique index); make labels unique in the picker |
| B2 | High | `/grade` is not idempotent; UI leaves Submit enabled with the same answer; no row lock | Code + 68→79% replay | `rag_ui.py:575-615`; `main.py:298-311`; `study.py:434-562,564-659` | Confirmed | Submission key + unique constraint; lock `question_state` row; disable resubmit of unchanged answer |
| B3 | High | A non-empty stale answer was sent although the box looked empty; the backend can't receive an empty answer | Backend rejects empty after strip; attempt was recorded with score 0; feedback matches previous answer | `rag_ui.py:308-312,556,575-600`; `study.py:571-574` | Strong hypothesis (mechanism) / Confirmed (no empty answer reached the grader) | Per-question widget key; clear answer outside form widget state; server-side guard |
| B4 | High | Question, reference answer and key points generated in one call; nothing checks them against the sources | No validation code exists; context contained the correct facts | `study.py:29-46,256-266,316-353` | Confirmed (no validation); Strong hypothesis (generation error, not retrieval) | Post-generation grounding check (+ one regeneration) |
| B5 | High | Grader scores coverage of stored key points and has no rule for paraphrase or false premises | Prompt text; feedback penalises an equivalent phrase | `study.py:48-86,588-596` | Strong hypothesis | Concept-level rubric, explicit paraphrase acceptance, premise rule; fix B4 first |
| B6 | Medium | One holistic LLM score; no contradiction or numeric check; seats weren't a key point | Prompt text; key points as shown in feedback | `study.py:48-86,106-113,615-619` | Strong hypothesis | Per-key-point verdicts + contradiction list; deterministic score from them |
| B7 | Low | Model emits a heading/label without a blank line after a list; rendered verbatim with `st.markdown` | Rendered output; no post-processing | `rag.py:9-20`; `rag_ui.py:190-201` | Strong hypothesis | Normalise blank lines before headings in `render_ask_result` |
| B8 | Low | Streamlit keeps previous elements until rerun reaches them; Past questions is fetched last, after other API calls | Code order; all tabs run each rerun | `rag_ui.py:332,435-488,624-625,279-305` | Confirmed (display-only) | Clear course-scoped placeholders on switch; lazy-load per tab |

---

## B1 — Duplicate course names and unsafe deletion (High)

**Confidence: Confirmed.** Origin: **database + API + UI combined.**

### Execution path

1. **Create.** `rag_ui.py:394-401` posts the name; `main.py:209-220` → `courses.create_course` (`courses.py:4-21`) strips whitespace and inserts. `schema.sql:15-21` has no unique constraint on `(user_id, name)`. Duplicate created.
2. **Select new course.** The new ID is stored in `session_state.pending_course_id` and applied to `selected_course_id` on the next run (`rag_ui.py:356-358`). This first render correctly shows the new (empty) course — matching the audit.
3. **Picker.** `st.selectbox(options=list(course_names.keys()), format_func=lambda id: course_names[id], key="selected_course_id")` (`rag_ui.py:362-367`). Options are IDs, which looks safe — but:
4. **Streamlit's selectbox serialises by label.** `create_mappings()` builds `formatted_label → index` and, per its own comment, *"If formatted labels are duplicated, the last one wins"* (`options_selector_utils.py:335-339`). `SelectboxSerde.deserialize` maps the browser's label back through that dict (`selectbox.py:157-172`).
5. **Order makes the original course win.** `list_courses` orders `created_at DESC` (`courses.py:33`), so the duplicate (newest) is first and the original is last. Every label lookup returns the original's ID.
6. **Any rerun resolves to the original.** Clicking Cancel (`rag_ui.py:431-433`) reruns; the browser sends the label, which deserialises to the original. Choosing either duplicate in the dropdown does the same. The duplicate is now unreachable — exactly what the audit saw.
7. **Delete targets the resolved ID.** `rag_ui.py:420` sends `DELETE /courses/{selected_course_id}`. After step 6 that is the course **with** materials.
8. **Deletion is soft.** `courses.archive_course` sets `archived_at` only (`courses.py:69-86`). Documents, chunks, questions, attempts and Pinecone vectors are untouched. A wrongly deleted course is recoverable by setting `archived_at = NULL`, but the UI text (`rag_ui.py:415-416`) tells the user it is gone.

### Other checks

| Item | Finding |
|---|---|
| Name normalisation | `strip()` only. `"Biology"` vs `"biology"` vs `"Biology  101"` all allowed |
| Ownership on delete | Correct: `WHERE id = %s AND user_id = %s AND archived_at IS NULL` |
| Same name, different users | No conflict (correct) |
| Archive then recreate same name | Allowed today; must stay allowed after the fix (unique index must be partial on `archived_at IS NULL`) |
| Existing duplicates | At least one in Sarah's account from the audit; production-wide count unknown |

**Evidence still useful (read-only):**
```sql
select user_id, lower(btrim(name)) as key, count(*)
from courses where archived_at is null
group by 1,2 having count(*) > 1;
```
Also: the deployed Streamlit version (Manage app → logs, or `pip freeze` in the build), to confirm the label-based selectbox serialisation is in the running build.

---

## B2 — Repeated submission inflates mastery (High)

**Confidence: Confirmed.**

### Execution path

1. `answer_form` (`rag_ui.py:575-585`) keeps the text and the Submit button enabled after grading; the result renders below (`rag_ui.py:620-622`).
2. Each submit posts `/grade` with the same `question_id` and answer (`rag_ui.py:593-600`). No submission identifier exists.
3. `grade_student_answer` (`study.py:564-659`) always calls the model and always calls `record_attempt`.
4. `record_attempt` (`study.py:434-562`) reads `question_state`, computes new mastery from the **current** value, inserts a new `attempts` row and upserts `question_state`. Second identical submission: 0.68 → 0.788 (audit: 68% → 79%, "answered 2×").

### Concurrency and retries

- **Lost update (confirmed in code, not observed):** the `SELECT` at `study.py:451-458` takes no lock. Two concurrent `/grade` calls for the same question both read the same `old_mastery`/`attempt_count`; both insert attempts; the later upsert wins. Result: two `attempts` rows but `attempt_count` incremented once, and mastery computed from a stale base.
- **Retry amplification (potential):** the UI times out at 120 s (`rag_ui.py:23`), while the server keeps running (OpenAI client default timeout and 2 SDK retries, plus `_call_structured`'s own 2-try loop, `study.py:226-242`). A user who resubmits after a UI timeout can produce two attempts for one intent.
- **Atomicity:** the attempt insert and state upsert share one transaction (`with get_conn()` commits on exit), so a crash between them can't half-apply. A model failure raises before `record_attempt`, so failed grading does not change mastery — correct today; keep it that way.

### Idempotency options

| Option | How | Trade-off |
|---|---|---|
| A. Client submission key (recommended) | UI generates a UUID when a question is shown / answer changes; send `submission_id`; `attempts.submission_id` with `unique(user_id, submission_id)`; on conflict return the stored attempt without regrading | Needs migration and an API field (optional for compatibility) |
| B. Content dedupe | Same user + question + normalised answer hash within N minutes returns the previous attempt | No client change; blocks a legitimate deliberate retry of the same text — product decision |
| C. UI-only | Disable Submit until the answer changes | Trivial; does not protect against double clicks, retries or other clients — use alongside A |

Plus `SELECT … FOR UPDATE` on `question_state` (insert a default row first or use advisory lock on `(user_id, question_id)`) so concurrent different answers serialise.

---

## B3 — Empty answer graded, with fabricated feedback (High)

**Confidence:** *Confirmed* that the grader did not receive an empty answer. *Strong hypothesis* that it received the previous question's answer through stale widget state.

### Why the grader cannot have received an empty answer

- UI guard: `student_answer = (st.session_state.get("quiz_answer") or "").strip()`; empty → notice, no request (`rag_ui.py:588-590`).
- API: `GradeRequest.answer: str = Field(min_length=1)` (`main.py:143-145`).
- Service: `student_answer.strip()`; empty → `StudyValidationError` → HTTP 400 before any model call (`study.py:571-574`).
- Yet an attempt was recorded with score 0 (mastery 0.5 → 0.30, "answered 2×"). `record_attempt` only runs after a successful grade. So a non-empty `answer_text` was stored.

### What was most likely sent

The feedback said the answer "incorrectly referred to a different leader and a different council structure". The previous answer (to the prior question on the same topic) was *"Varn funded shipbuilding to fix the economy, while Ostrel created a parliament with 14 seats"* — a different leader and a different council structure. The grader described real text; it was the wrong text, not fabricated.

### Mechanism (strong hypothesis)

`quiz_answer` is the key of a `st.text_area` **inside `st.form("answer_form")`** (`rag_ui.py:575-583`), and it is also set programmatically to `""` when a new question loads (`rag_ui.py:556`, and `reset_quiz_state`, `rag_ui.py:312`). The box rendered empty, but on the next form submit the browser-side form state still carried the previously submitted value, which Streamlit wrote back into `session_state.quiz_answer`. That value passed every guard. Programmatic assignment to a form widget's key is a known source of browser/server state disagreement in Streamlit.

Ruled out: prompt contamination. `GRADE_PROMPT` (`study.py:48-86`) is built only from the current question row and the current request body; there is no conversation history or cross-request state on the server.

Also possible (unverified): a browser extension (Grammarly was active in that text box in the screenshot) restoring prior text into the textarea's DOM value.

### Evidence that would confirm it (read-only)

```sql
select a.created_at, a.answer_text, a.score
from attempts a
join questions q on q.id = a.question_id
where q.prompt like 'What significant political change occurred in Velmora%'
order by a.created_at;
```
If the 0% attempt's `answer_text` equals the previous answer, the stale-state hypothesis is confirmed. A local reproduction with `streamlit.testing.v1.AppTest` (or the browser with extensions disabled) would pin down the exact mechanism.

### Whitespace

`"   "` passes Pydantic (`min_length=1`) but is rejected by `study.py:571-574` (400). For a 400 with a string detail, `error_message` shows the raw detail (`rag_ui.py:136-145`), so a direct API caller would see "student_answer must not be empty" — safe, but developer wording. The UI guard normally prevents this path.

---

## B4 — Generated question with a false premise (High)

**Confidence:** *Confirmed* that nothing verifies generated questions. *Strong hypothesis* that this was a generation error rather than a retrieval failure.

### Execution path

`study.py:316-353`: `retrieve_chunks(query=topic, k=5)` → `QUIZ_PROMPT` with the student's topic and all chunks → one structured call (`_question_from_prompt`, `study.py:256-266`) → `save_question` stores the draft and links every retrieved chunk as a source (`study.py:199-222`).

### Analysis

| Area | Finding |
|---|---|
| Retrieval | The Kestrel notes are ~900 characters; with 800/100 chunking (`ingest.py:24-32`) that is ~2 chunks. With `k=5` in a 2-document course, both Kestrel chunks were almost certainly in context — the correct facts were available. Confirm via `question_sources` for that question. |
| Prompt | Grounding is instruction-only ("Use ONLY the retrieved context"). The prompt asks for a question about the student's topic ("Kestrel handshake **and** reliability"), which invites merging two sections; nothing warns against combining facts from different sections into one claim. |
| Schema | `QuizDraft` (`study.py:97-103`) has no field for the supporting quote/chunk per claim. Sources are linked wholesale, not per question. |
| Validation | None. No check that the question's premises appear in the sources; no regeneration. `questions.generation` jsonb (`schema.sql:73`) exists but is never written, so the prompt version/model used is not recorded. |
| Reference answer | Generated in the same call. The grader's explanation ("MOLT … requests the retransmission of lost feathers") suggests the **reference answer was correct** while the **question premise was wrong** — a sign of an internally inconsistent draft. |

### Practical validation strategy

1. **Cheap structural checks (no cost):** require `supporting_quotes: list[str]` in `QuizDraft`; verify each quote is a substring (after whitespace normalisation) of a retrieved chunk. Reject otherwise.
2. **One verifier call (moderate cost, ~1 extra small-model call per new question):** given only the chunks and the draft, return `{premises: [{claim, supported: bool, quote}]}`. Any unsupported premise → regenerate once with the failure reason; second failure → return `insufficient_material` with a friendly message.
3. **Store the outcome** in `questions.generation` (model, prompt version, verifier result) so bad questions can be audited and retired (`status = 'retired'`).

Latency: only on the "generate" path; reused questions (`mode == "existing"`) cost nothing.

---

## B5 — Correct answer graded as partial (High)

**Confidence: Strong hypothesis.**

### Execution path

`study.py:576-596`: load stored reference answer, key points, source chunks → `GRADE_PROMPT` → `GradeDraft` → clamp → `verdict_from_score`.

### Analysis

| Area | Finding |
|---|---|
| Ground truth | The prompt tells the grader to use "ONLY the stored reference answer, the required key points, the original source material" (`study.py:50-53`). The question text itself is treated as valid. |
| Premise handling | No rule. When a student correctly rejects the question's premise, the grader still evaluates against "MOLT's purpose **in the handshake process**", and repeated that phrase in its feedback. |
| Paraphrase | No instruction to accept semantically equivalent wording. `missed_points must contain the important key points the student missed` (`study.py:63`) pushes literal matching: "ask for missing feathers to be resent" was not credited as "requests retransmission of lost feathers". |
| Score calculation | Holistic float from the model; verdict thresholds are deterministic (`study.py:115-120`). Thresholds aren't the problem; the score is. |
| Output validation | `GradeDraft` is unconstrained beyond types (`study.py:106-113`); `missed_points` may contain points the student actually covered. Nothing cross-checks them. |

### Shared cause with B4

Yes. Both come from the same design: **one unchecked generation produces the question, the reference and the key points; the grader then treats all three as ground truth.** A wrong premise in generation becomes a wrong rubric in grading. Fix order follows from this: validate generation (B4) before re-tuning the grader (B5), otherwise grader changes are evaluated against corrupted references.

---

## B6 — Misleading partial credit for a wrong answer (Medium)

**Confidence: Strong hypothesis.**

- The grader produces one holistic score (`study.py:57`); there are no explicit concept checks, contradiction detection or numeric verification.
- The swapped answer mentioned a true fact (shipbuilding) attached to the wrong person; the model awarded 25% (≥ 0 but < 0.35 → "incorrect", `study.py:115-120`). The prompt does not say that a misattributed fact earns nothing.
- "14 seats" was not flagged because the question asked how the reforms *differed*; seat count was probably not a key point. The grader only lists *missed* key points, not *wrong* statements.

**Recommendation:** separate concept checks from scoring. Ask the model for `key_point_results: [{point, status: covered|partially|missing|contradicted, evidence}]` and `factual_errors: [{student_claim, source_says}]`; compute the score deterministically in code (e.g. covered = 1, partial = 0.5, contradicted = 0 and caps the verdict at partially_correct). This makes scores explainable and testable with mocked model output.

---

## B7 — Markdown heading merged into list item (Low)

**Confidence: Strong hypothesis** (raw model output wasn't captured).

- `/ask` returns `answer.answer` as a string; `render_ask_result` passes it straight to `st.markdown` (`rag_ui.py:200-201`). No post-processing.
- `GROUNDING_PROMPT_TEMPLATE` (`rag.py:9-20`) gives no formatting guidance.
- Standard Markdown treats a line that directly follows a list item (no blank line) as a continuation of that item ("lazy continuation"), which matches the screenshot: "…(from Lumenite_Chemistry_Notes.pdf) Why storage matters" inside item 3.

**Smallest reliable fix:** in `render_ask_result`, insert a blank line before any line that starts with `#` or is a bold-only line, and after a list block ends. A prompt hint ("separate sections with a blank line") helps but should not be relied on alone.

**Evidence to confirm:** log or inspect the raw `answer.answer` for that question.

---

## B8 — Stale course data after switching (Low)

**Confidence: Confirmed. Display-only; no backend cross-course effect found.**

- Streamlit reruns the whole script on every interaction and keeps old elements on screen until the rerun reaches and replaces them.
- `st.tabs` executes all three tab bodies on every run (`rag_ui.py:435-625`). Before the Past questions list is redrawn (`rag_ui.py:624-625`), the run makes `GET /courses` (`332`), `GET /documents` (`488`) and then `GET /questions` (`281`), each also running `ensure_user` with fresh DB connections. That ordering explains the ~3–5 s of stale content.
- Backend calls in the new run all use the new `selected_course_id`, and `reset_quiz_state` clears the quiz question on course change (`rag_ui.py:369-371`), so a stale `question_id` is not graded against the new course.
- Related leaks observed in the audit, confirmed in code:
  - `reset_quiz_state` resets `quiz_topic` but **not** `quiz_topic_input` (the widget key, `rag_ui.py:533`), so the topic text carries over.
  - The Ask `text_input` has no key and isn't reset (`rag_ui.py:495-498`), so the question text carries over.
- Minor backend gap found while tracing: `/grade` checks question ownership but not that the course is still active (`study.py:367-390`), so questions in an archived course remain gradable if their ID is known.

---

## Additional risks found while tracing (not in the audit)

| Risk | Code path | Status |
|---|---|---|
| Pinecone upsert inside an uncommitted DB transaction; commit failure leaves orphan vectors whose `chunk_id` doesn't exist in Postgres → later `question_sources` FK error on quiz | `ingest.py:153-224` | Potential |
| Vectors deleted before the soft-delete commits; commit failure leaves an "active" document with no vectors | `documents.py:66-90` | Potential |
| Archived courses keep vectors and history while UI says they are removed | `courses.py:69-86`, `rag_ui.py:415` | Confirmed |
| Concurrent first login: two `ensure_user` calls race on `unique(provider, subject)` → one 500 | `users.py:3-42` | Potential (low impact) |
| API impersonation if deployed without `API_SHARED_SECRET` and without `RENDER`/`APP_ENV` flags | `main.py:39-89` | Potential — needs deployed config confirmed |
| Every DB helper opens a new connection; several per request | `db.py:11-15` | Confirmed; latency/pool-limit risk |
| `_call_structured` swallows the first failure silently and retries; no logging of which attempt failed | `study.py:226-242` | Confirmed; observability gap |
| `next_due_at` and `next_due()` are unused; spaced repetition is not active | `adaptive.py:6-15`, `schema.sql:118` | Confirmed |

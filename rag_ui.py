"""Streamlit client for the Adaptive Study Agent FastAPI service.

Run locally:
  streamlit run rag_ui.py

Configuration (environment variable or a top-level key in .streamlit/secrets.toml):
  RAG_API_URL        FastAPI base URL (default http://127.0.0.1:8000)
  API_SHARED_SECRET  must match the API's value (required in production)
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import httpx
import streamlit as st
from pypdf import PdfReader

TIMEOUT_SECONDS = 120.0


def setting(name: str, default: str = "") -> str:
    """Read a setting from the environment, then top-level Streamlit secrets, then the
    [auth] section of secrets (a common place for it to end up by mistake)."""

    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
        auth_section = st.secrets.get("auth", {})
        if name in auth_section:
            return str(auth_section[name]).strip()
    except Exception:  # no secrets file locally
        pass
    return default


API_URL = setting("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_SHARED_SECRET = setting("API_SHARED_SECRET")

APP_CSS = """
<style>
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 3.5rem;
        max-width: 860px;
    }
    h1 {
        letter-spacing: -0.03em;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        font-size: 1.12rem;
        line-height: 1.45;
        opacity: 0.78;
        margin-bottom: 0.35rem;
    }
    .hero-note {
        font-size: 0.95rem;
        line-height: 1.5;
        opacity: 0.68;
        margin-bottom: 1.25rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        margin-bottom: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.55rem 0.9rem;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
    }
</style>
"""


# ── API client ───────────────────────────────────────────────────────

def auth_headers() -> dict[str, str]:
    """Identity headers for the API. Values are URL-encoded so non-ASCII names work."""

    info = st.user.to_dict()
    headers = {"X-Google-Sub": str(info.get("sub", ""))}
    if info.get("email"):
        headers["X-User-Email"] = quote(str(info["email"]))
    if info.get("name"):
        headers["X-User-Name"] = quote(str(info["name"]))
    if API_SHARED_SECRET:
        headers["X-API-Key"] = API_SHARED_SECRET
    return headers


def api_request(
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    params: dict | None = None,
) -> tuple[int, dict | list | str]:
    url = f"{API_URL}{path}"
    try:
        response = httpx.request(
            method,
            url,
            json=payload,
            params=params,
            headers=auth_headers(),
            timeout=TIMEOUT_SECONDS,
        )
        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.HTTPError:
        return 0, {"error": "Can't reach the study service right now. Please try again in a minute."}


def error_message(status: int, data, fallback: str) -> str:
    if isinstance(data, dict):
        if data.get("error"):
            return str(data["error"])
        detail = data.get("detail")
        if isinstance(detail, str) and status != 422:
            return detail
    if status == 401:
        return "Your session could not be verified. Try logging out and back in."
    return fallback


# ── File extraction ──────────────────────────────────────────────────

def extract_txt_text(file_bytes: bytes) -> str:
    """Decode a .txt upload as UTF-8, replacing invalid bytes instead of crashing."""

    return file_bytes.decode("utf-8", errors="replace")


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from every PDF page; skip pages with no extractable text."""

    reader = PdfReader(BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        stripped = extracted.strip()
        if stripped:
            pages.append(stripped)
    return "\n\n".join(pages)


def extract_uploaded_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return extract_txt_text(file_bytes)
    if suffix == ".pdf":
        return extract_pdf_text(file_bytes)
    raise ValueError(f"Unsupported file type: {suffix or '(none)'}")


# ── Rendering helpers ────────────────────────────────────────────────

def render_ingest_file_result(*, filename: str, status: int, data) -> None:
    if status == 200 and isinstance(data, dict):
        if data.get("status") == "unchanged":
            st.info(f"{filename} is already in this course.")
        else:
            st.success(f"Added {filename}.")
    else:
        st.error(f"Could not add {filename}: " + error_message(status, data, "please try again."))


def render_ask_result(data: dict) -> None:
    answer = data.get("answer", {})
    answer_text = answer.get("answer", "")
    sources_needed = answer.get("sources_needed", False)

    if sources_needed:
        st.warning("There isn't enough in your uploaded notes to answer that.")
    else:
        st.success("Answer from your notes")

    with st.container(border=True):
        st.markdown(answer_text)


def render_grade_result(data: dict) -> None:
    verdict = data.get("verdict")
    labels = {
        "correct": ("Correct", st.success),
        "partially_correct": ("Partially correct", st.warning),
        "incorrect": ("Needs review", st.error),
    }
    label, message_fn = labels.get(verdict, (verdict or "Ungraded", st.info))
    message_fn(label)

    score = data.get("score")
    mastery = data.get("mastery")
    if score is not None or mastery is not None:
        cols = st.columns(2)
        cols[0].metric("Score", f"{score:.0%}" if score is not None else "—")
        cols[1].metric("Question mastery", f"{mastery:.0%}" if mastery is not None else "—")

    with st.container(border=True):
        st.markdown("**Feedback**")
        st.write(data.get("feedback") or "—")

        explanation = data.get("explanation")
        if explanation:
            st.markdown("**Explanation**")
            st.write(explanation)

        missed = data.get("missed_points") or []
        if missed:
            st.markdown("**Review these points**")
            for point in missed:
                st.markdown(f"- {point}")

        gap = data.get("identified_gap")
        if gap:
            st.markdown("**Focus for next time**")
            st.write(gap)


def render_materials(course_id: str) -> None:
    st.markdown("#### Your materials")
    status, data = api_request("GET", "/documents", params={"course_id": course_id})

    if status != 200 or not isinstance(data, list):
        st.warning(error_message(status, data, "Couldn't load your materials."))
        return

    if not data:
        st.caption("No materials in this course yet.")
        return

    for document in data:
        doc_id = document["id"]
        cols = st.columns([5, 1])
        added = str(document.get("created_at", ""))[:10]
        cols[0].markdown(f"**{document['filename']}**")
        cols[0].caption(f"Added {added}")

        pending = st.session_state.get("pending_delete") == doc_id
        if not pending:
            if cols[1].button("Delete", key=f"delete_{doc_id}"):
                st.session_state.pending_delete = doc_id
                st.rerun()
        else:
            if cols[1].button("Confirm", key=f"confirm_{doc_id}", type="primary"):
                del_status, del_data = api_request("DELETE", f"/documents/{doc_id}")
                st.session_state.pending_delete = None
                if del_status == 200:
                    st.session_state.materials_notice = f"Deleted {document['filename']}."
                else:
                    st.session_state.materials_notice = error_message(
                        del_status, del_data, "Couldn't delete that file."
                    )
                st.rerun()


def render_past_questions(course_id: str) -> None:
    with st.expander("Past questions"):
        status, data = api_request("GET", "/questions", params={"course_id": course_id})

        if status != 200 or not isinstance(data, list):
            st.warning(error_message(status, data, "Couldn't load your past questions."))
            return

        if not data:
            st.caption("No questions yet. Get a question above to start.")
            return

        for item in data:
            st.markdown(f"**{item['prompt']}**")
            details = [part for part in (item.get("topic"), item.get("concept_label")) if part]
            attempts = item.get("attempt_count") or 0
            if attempts:
                last_score = item.get("last_score")
                mastery = item.get("mastery")
                details.append(f"answered {attempts}×")
                if last_score is not None:
                    details.append(f"last score {last_score:.0%}")
                if mastery is not None:
                    details.append(f"mastery {mastery:.0%}")
            else:
                details.append("not answered yet")
            st.caption(" · ".join(details))


def reset_quiz_state() -> None:
    for key in ("quiz_question", "quiz_question_id", "quiz_grade", "quiz_notice"):
        st.session_state[key] = None
    st.session_state.quiz_topic = ""
    st.session_state.quiz_answer = ""


# ── Page ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Adaptive Study Agent", layout="centered")
st.markdown(APP_CSS, unsafe_allow_html=True)

if not st.user.is_logged_in:
    st.title("Adaptive Study Agent")
    st.write("Sign in to start studying.")
    if st.button("Sign in with Google"):
        st.login()
    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.user.to_dict().get('name') or 'you'}**")
    if st.button("Log out"):
        st.logout()

status, courses = api_request("GET", "/courses")
if status != 200 or not isinstance(courses, list):
    st.title("Adaptive Study Agent")
    st.error(error_message(status, courses, "Couldn't load your courses. Please refresh in a minute."))
    st.stop()

if not courses:
    st.title("Adaptive Study Agent")
    st.subheader("Create your first course")
    course_name = st.text_input("Course name", placeholder="e.g. Biology 101")
    if st.button("Create course", type="primary"):
        if course_name.strip():
            create_status, create_data = api_request("POST", "/courses", payload={"name": course_name})
            if create_status == 200:
                st.rerun()
            st.error(error_message(create_status, create_data, "Couldn't create the course."))
        else:
            st.warning("Enter a course name.")
    st.stop()

course_names = {str(course["id"]): course["name"] for course in courses}

# A course created in the sidebar is selected on the next run (widget state can't
# be changed after the widget is drawn). Drop a stale selection if it vanished.
pending_course_id = st.session_state.pop("pending_course_id", None)
if pending_course_id in course_names:
    st.session_state.selected_course_id = pending_course_id
elif st.session_state.get("selected_course_id") not in course_names:
    st.session_state.pop("selected_course_id", None)

selected_course_id = st.selectbox(
    "Course",
    options=list(course_names.keys()),
    format_func=lambda course_id: course_names[course_id],
    key="selected_course_id",
)

if st.session_state.get("active_course_id") != selected_course_id:
    st.session_state.active_course_id = selected_course_id
    reset_quiz_state()

st.title("Adaptive Study Agent")
st.markdown(
    '<p class="hero-subtitle">Turn your course materials into personalized practice '
    "that adapts to what you need to review.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-note">Upload your notes, ask grounded questions, and practice with '
    "quizzes that remember your weak areas.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**Study workspace**")
    st.caption(f"Current course: {course_names[selected_course_id]}")
    with st.expander("+ New course"):
        new_course_name = st.text_input(
            "New course name",
            placeholder="e.g. Biology 102",
            key="new_course_name",
        )
        if st.button("Create new course", key="create_new_course"):
            if new_course_name.strip():
                create_status, create_data = api_request(
                    "POST", "/courses", payload={"name": new_course_name}
                )
                if create_status == 200:
                    st.session_state.pending_course_id = str(create_data["id"])
                    st.rerun()
                st.error(error_message(create_status, create_data, "Couldn't create the course."))
            else:
                st.warning("Enter a course name.")

    with st.expander("Delete course"):
        current_name = course_names[selected_course_id]
        if st.session_state.get("pending_course_delete") != selected_course_id:
            st.caption(f"Remove **{current_name}** from your course list.")
            if st.button("Delete this course", key="delete_course"):
                st.session_state.pending_course_delete = selected_course_id
                st.rerun()
        else:
            st.error(
                f"Delete **{current_name}**? It disappears from your courses, along with its "
                "materials, questions and progress. This can't be undone in the app."
            )
            confirm_col, cancel_col = st.columns(2)
            if confirm_col.button("Yes, delete", key="confirm_delete_course", type="primary"):
                del_status, del_data = api_request("DELETE", f"/courses/{selected_course_id}")
                st.session_state.pending_course_delete = None
                if del_status == 200:
                    remaining = [cid for cid in course_names if cid != selected_course_id]
                    reset_quiz_state()
                    st.session_state.active_course_id = None
                    st.session_state.pending_delete = None
                    if remaining:
                        st.session_state.pending_course_id = remaining[0]
                    st.rerun()
                st.error(error_message(del_status, del_data, "Couldn't delete the course."))
            if cancel_col.button("Cancel", key="cancel_delete_course"):
                st.session_state.pending_course_delete = None
                st.rerun()

tab_ingest, tab_ask, tab_quiz = st.tabs(["Study Materials", "Ask Your Materials", "Quiz Me"])

with tab_ingest:
    st.subheader("Add study materials")
    st.write("Upload PDF or TXT course materials. Questions and quizzes will be grounded in these sources.")
    st.caption(
        "For class notes only. Don't upload personal, medical or confidential documents. "
        "Your notes are processed by OpenAI and stored in Supabase and Pinecone."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="You can select several .pdf and .txt files at once.",
        label_visibility="collapsed",
    )

    if st.button("Add to study materials", type="primary"):
        if not uploaded_files:
            st.warning("Choose one or more PDF or TXT files first.")
        else:
            for uploaded in uploaded_files:
                filename = uploaded.name or "uploaded-document"
                try:
                    extracted = extract_uploaded_text(filename, uploaded.getvalue()).strip()
                except Exception:  # extraction should never crash the page
                    st.error(f"Could not read {filename}. Is it a valid PDF or TXT file?")
                    continue

                if not extracted:
                    st.error(
                        f"No readable text in {filename}. "
                        "Scanned PDFs without selectable text cannot be used yet."
                    )
                    continue

                with st.spinner(f"Adding {filename}…"):
                    upload_status, upload_data = api_request(
                        "POST",
                        "/ingest",
                        payload={
                            "course_id": selected_course_id,
                            "filename": filename,
                            "text": extracted,
                        },
                    )
                render_ingest_file_result(filename=filename, status=upload_status, data=upload_data)

    st.divider()
    notice = st.session_state.pop("materials_notice", None)
    if notice:
        st.info(notice)
    render_materials(selected_course_id)

with tab_ask:
    st.subheader("Ask your materials")
    st.write("Get answers grounded in the notes you've uploaded.")

    with st.form("ask_form"):
        question = st.text_input(
            "Your question",
            placeholder="What does this material say about the main topic?",
        )
        ask_submitted = st.form_submit_button("Ask", type="primary")

    if ask_submitted:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Looking through your notes…"):
                ask_status, ask_data = api_request(
                    "POST",
                    "/ask",
                    payload={"course_id": selected_course_id, "question": question},
                )

            if ask_status == 200 and isinstance(ask_data, dict):
                render_ask_result(ask_data)
            else:
                st.error(error_message(ask_status, ask_data, "Could not get an answer. Try again."))

with tab_quiz:
    for key in ("quiz_question", "quiz_question_id", "quiz_grade", "quiz_notice"):
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("quiz_topic", "")

    st.subheader("Practice quiz")
    st.write(
        "Choose a topic and get one question at a time. "
        "Your next questions adapt to concepts you've struggled with."
    )

    with st.form("topic_form", border=True):
        st.markdown("**Topic**")
        topic_input = st.text_input(
            "Topic",
            placeholder="e.g. Photosynthesis",
            key="quiz_topic_input",
            label_visibility="collapsed",
        )
        topic_submitted = st.form_submit_button("Get a question", type="primary")

    if topic_submitted:
        topic = topic_input.strip()
        if not topic:
            st.session_state.quiz_notice = "Enter a topic first."
        else:
            with st.spinner("Finding your next question…"):
                quiz_status, quiz_data = api_request(
                    "POST",
                    "/quiz",
                    payload={"course_id": selected_course_id, "topic": topic},
                )

            if quiz_status == 200 and isinstance(quiz_data, dict) and quiz_data.get("status") == "ok":
                st.session_state.quiz_question = quiz_data.get("question")
                st.session_state.quiz_question_id = quiz_data.get("question_id")
                st.session_state.quiz_topic = quiz_data.get("topic") or topic
                st.session_state.quiz_grade = None
                st.session_state.quiz_notice = None
                st.session_state.quiz_answer = ""
            elif quiz_status == 200 and isinstance(quiz_data, dict):
                st.session_state.quiz_question = None
                st.session_state.quiz_grade = None
                st.session_state.quiz_notice = quiz_data.get("detail") or (
                    "Not enough course material was found for that topic."
                )
            else:
                st.session_state.quiz_notice = error_message(
                    quiz_status, quiz_data, "Could not get a question. Try again."
                )

    if st.session_state.quiz_question:
        with st.container(border=True):
            st.markdown("**Question**")
            if st.session_state.quiz_topic:
                st.caption(st.session_state.quiz_topic)
            st.markdown(st.session_state.quiz_question)

        with st.form("answer_form", border=False):
            st.markdown("**Your answer**")
            st.text_area(
                "Your answer",
                key="quiz_answer",
                height=120,
                label_visibility="collapsed",
                placeholder="Write your answer in your own words.",
            )
            st.caption("Press Cmd/Ctrl + Enter or click Submit answer.")
            answer_submitted = st.form_submit_button("Submit answer")

        if answer_submitted:
            student_answer = (st.session_state.get("quiz_answer") or "").strip()
            if not student_answer:
                st.session_state.quiz_notice = "Write an answer before submitting."
            else:
                with st.spinner("Checking your answer…"):
                    grade_status, grade_data = api_request(
                        "POST",
                        "/grade",
                        payload={
                            "question_id": st.session_state.quiz_question_id,
                            "answer": student_answer,
                        },
                    )

                if grade_status == 200 and isinstance(grade_data, dict) and grade_data.get("status") == "ok":
                    st.session_state.quiz_grade = grade_data
                    st.session_state.quiz_notice = None
                elif grade_status == 200 and isinstance(grade_data, dict):
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = (
                        "This answer couldn't be graded against your notes. "
                        "Try submitting again, or get a new question."
                    )
                else:
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = error_message(
                        grade_status, grade_data, "Could not grade that answer. Try again."
                    )

    if st.session_state.quiz_notice:
        st.warning(st.session_state.quiz_notice)

    if st.session_state.quiz_grade:
        st.markdown("**Result**")
        render_grade_result(st.session_state.quiz_grade)

    st.divider()
    render_past_questions(selected_course_id)

"""Streamlit client for the Adaptive Study Agent FastAPI service.

Run:
  export RAG_API_URL=https://your-service.onrender.com
  streamlit run rag_ui.py
"""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import httpx
import streamlit as st
from pypdf import PdfReader

import requests

def auth_headers():
    return {
        "X-Google-Sub": st.user.sub,
        "X-User-Email": st.user.email,
        "X-User-Name": st.user.name,
    }

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 120.0


def get_courses():
    response = requests.get(
        f"{DEFAULT_API_URL.rstrip('/')}/courses",
        headers=auth_headers(),
    )
    response.raise_for_status()
    return response.json()


def create_course(name):
    response = requests.post(
        f"{DEFAULT_API_URL.rstrip('/')}/courses",
        json={"name": name},
        headers=auth_headers(),
    )
    response.raise_for_status()
    return response.json()

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


def api_post(path: str, payload: dict) -> tuple[int, dict | str]:
    url = f"{DEFAULT_API_URL.rstrip('/')}{path}"
    try:
        response = httpx.post(
            url,
            json=payload,
            headers=auth_headers(),
            timeout=TIMEOUT_SECONDS,
        )
        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {url}. Check the API URL and that the service is running."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


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


def render_ingest_file_result(*, filename: str, status: int, data: dict | str) -> None:
    """Show a user-facing result for one uploaded file."""

    if status == 200 and isinstance(data, dict):
        state = data.get("status")
        if state == "unchanged":
            st.info(f"{filename} is already in this course.")
        else:
            chunks = data.get("chunks_indexed")
            suffix = f" ({chunks} chunks indexed)" if chunks is not None else ""
            st.success(f"Added {filename}{suffix}.")
    elif isinstance(data, dict) and data.get("error"):
        st.error(f"Could not add {filename}: {data['error']}")
    elif status >= 400:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        st.error(f"Could not add {filename}: {detail}")
    else:
        st.error(data if isinstance(data, str) else f"Could not add {filename}.")


def render_ask_result(data: dict) -> None:
    answer = data.get("answer", {})
    answer_text = answer.get("answer", "")
    confidence = answer.get("confidence")
    sources_needed = answer.get("sources_needed", False)

    if sources_needed:
        st.warning("There isn't enough in your uploaded notes to answer that.")
    else:
        st.success("Answer from your notes")

    with st.container(border=True):
        st.markdown(answer_text)

    if confidence is not None:
        st.caption(f"Confidence: {confidence:.0%}")


def init_quiz_state() -> None:
    defaults = {
        "quiz_topic": "",
        "quiz_question": None,
        "quiz_question_id": None,
        "quiz_grade": None,
        "quiz_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_grade_result(data: dict) -> None:
    verdict = data.get("verdict")
    labels = {
        "correct": ("Correct", "success"),
        "partially_correct": ("Partially correct", "warning"),
        "incorrect": ("Needs review", "error"),
    }
    label, level = labels.get(verdict, (verdict or "Ungraded", "info"))
    message_fn = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }[level]
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


st.set_page_config(page_title="Adaptive Study Agent", layout="centered")
st.markdown(APP_CSS, unsafe_allow_html=True)

if not st.user.is_logged_in:
    st.title("Adaptive Study Agent")
    st.write("Sign in to start studying.")

    if st.button("Sign in with Google"):
        st.login()

    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.user.name}**")

    if st.button("Log out"):
        st.logout()



courses = get_courses()

if not courses:
    st.title("Adaptive Study Agent")
    st.subheader("Create your first course")

    course_name = st.text_input(
        "Course name",
        placeholder="e.g. Biology 101",
    )

    if st.button("Create course"):
        if course_name.strip():
            create_course(course_name)
            st.rerun()

    st.stop()

course_options = {
    course["name"]: course["id"]
    for course in courses
}

selected_course_name = st.selectbox(
    "Course",
    options=list(course_options.keys()),
)

selected_course_id = course_options[selected_course_name]

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

st.sidebar.markdown("**Study workspace**")
st.sidebar.caption(f"Current course: {selected_course_name}")
with st.sidebar.expander("+ New course"):
    new_course_name = st.text_input(
        "New course name",
        placeholder="e.g. Biology 102",
        key="new_course_name",
    )
    if st.button("Create new course", key="create_new_course"):
        if new_course_name.strip():
            create_course(new_course_name)
            st.rerun()
        else:
            st.warning("Enter a course name.")

tab_ingest, tab_ask, tab_quiz = st.tabs(
    ["Study Materials", "Ask Your Materials", "Quiz Me"]
)

with tab_ingest:
    st.subheader("Add study materials")
    st.write(
        "Upload PDF or TXT course materials. Questions and quizzes will be grounded in these sources."
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
                except Exception as exc:  # extraction should never crash the page
                    st.error(f"Could not read {filename}: {exc}")
                    continue

                if not extracted:
                    st.error(
                        f"No readable text in {filename}. "
                        "Scanned PDFs without selectable text cannot be used yet."
                    )
                    continue

                ingest_payload = {
                    "course_id": selected_course_id,
                    "filename": filename,
                    "text": extracted,
                }
                with st.spinner(f"Adding {filename}…"):
                    status, data = api_post("/ingest", ingest_payload)
                render_ingest_file_result(
                    filename=filename,
                    status=status,
                    data=data,
                )

with tab_ask:
    st.subheader("Ask your materials")
    st.write("Get answers grounded in the notes you've uploaded.")

    question = st.text_input(
        "Your question",
        placeholder="What does this material say about the main topic?",
    )

    if st.button("Ask", type="primary"):
        ask_payload = {
            "course_id": selected_course_id,
            "question": question,
        }
        with st.spinner("Looking through your notes…"):
            status, data = api_post("/ask", ask_payload)

        if status == 200 and isinstance(data, dict):
            render_ask_result(data)
        elif status >= 400:
            st.error(data.get("detail", data) if isinstance(data, dict) else data)
        else:
            st.error("Could not get an answer. Try again.")

with tab_quiz:
    init_quiz_state()
    st.subheader("Practice quiz")
    st.write(
        "Choose a topic and get one question at a time. "
        "Your next questions adapt to concepts you've struggled with."
    )

    with st.container(border=True):
        st.markdown("**Topic**")
        topic_input = st.text_input(
            "Topic",
            placeholder="Japanese greetings",
            key="quiz_topic_input",
            label_visibility="collapsed",
        )
        if st.button("Get a question", type="primary"):
            topic = topic_input.strip()
            if not topic:
                st.session_state.quiz_notice = "Enter a topic first."
            else:
                quiz_payload = {
                    "course_id": selected_course_id,
                    "topic": topic,
                }
                with st.spinner("Writing a question from your notes…"):
                    status, data = api_post("/quiz", quiz_payload)

                if status == 200 and isinstance(data, dict) and data.get("status") == "ok":
                    st.session_state.quiz_question = data.get("question")
                    st.session_state.quiz_question_id = data.get("question_id")
                    st.session_state.quiz_topic = data.get("topic") or topic
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = None
                    st.session_state.quiz_answer = ""
                elif status == 200 and isinstance(data, dict):
                    st.session_state.quiz_question = None
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = data.get("detail") or (
                        "Not enough course material was found for that topic."
                    )
                elif isinstance(data, dict) and data.get("error"):
                    st.session_state.quiz_notice = data["error"]
                elif status >= 400:
                    detail = data.get("detail", data) if isinstance(data, dict) else data
                    st.session_state.quiz_notice = str(detail)
                else:
                    st.session_state.quiz_notice = "Could not generate a question. Try again."

    if st.session_state.quiz_question:
        with st.container(border=True):
            st.markdown("**Question**")
            if st.session_state.quiz_topic:
                st.caption(st.session_state.quiz_topic)
            st.markdown(st.session_state.quiz_question)

        st.markdown("**Your answer**")
        st.text_area(
            "Your answer",
            key="quiz_answer",
            height=120,
            label_visibility="collapsed",
            placeholder="Write your answer in your own words.",
        )

        if st.button("Submit answer"):
            student_answer = (st.session_state.get("quiz_answer") or "").strip()
            if not student_answer:
                st.session_state.quiz_notice = "Write an answer before submitting."
            else:
                grade_payload = {
                    "question_id": st.session_state.quiz_question_id,
                    "answer": student_answer,
                }
                with st.spinner("Checking your answer…"):
                    status, data = api_post("/grade", grade_payload)

                if status == 200 and isinstance(data, dict) and data.get("status") == "ok":
                    st.session_state.quiz_grade = data
                    st.session_state.quiz_notice = None
                elif status == 200 and isinstance(data, dict):
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = data.get("detail") or (
                        "Not enough course material was found to grade this answer."
                    )
                elif isinstance(data, dict) and data.get("error"):
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = data["error"]
                elif status >= 400:
                    st.session_state.quiz_grade = None
                    detail = data.get("detail", data) if isinstance(data, dict) else data
                    st.session_state.quiz_notice = str(detail)
                else:
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = "Could not grade that answer. Try again."

    if st.session_state.quiz_notice:
        st.warning(st.session_state.quiz_notice)

    if st.session_state.quiz_grade:
        st.markdown("**Result**")
        render_grade_result(st.session_state.quiz_grade)



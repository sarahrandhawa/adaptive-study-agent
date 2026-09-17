"""Streamlit client for the Adaptive Study Agent FastAPI service.

Run:
  export RAG_API_URL=https://your-service.onrender.com
  streamlit run rag_ui.py
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path

import httpx
import requests
import streamlit as st
from pypdf import PdfReader

from memory_store import topic_progress

API_BASE = "http://127.0.0.1:8000"
DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 120.0

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


def api_post(base_url: str, path: str, payload: dict) -> tuple[int, dict | str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = httpx.post(url, json=payload, timeout=TIMEOUT_SECONDS)
        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {url}. Check the API URL and that the service is running."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def document_id_from_filename(filename: str) -> str:
    """Derive a Pinecone-safe document_id from an uploaded filename."""

    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return safe or "uploaded-document"


def unique_document_id(base_id: str, used_ids: set[str]) -> str:
    """Keep document_ids unique when several uploads share a filename."""

    candidate = base_id
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base_id}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


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


def render_ingest_file_result(
    *,
    filename: str,
    document_id: str,
    status: int,
    data: dict | str,
) -> None:
    """Show success or failure for one uploaded file after POST /ingest."""

    if status == 200 and isinstance(data, dict):
        st.success(f"Added {filename}.")
    elif isinstance(data, dict) and data.get("error"):
        st.error(f"Could not add {filename}: {data['error']}")
    elif status >= 400:
        detail = data.get("detail", data) if isinstance(data, dict) else data
        st.error(f"Could not add {filename}: {detail}")
    else:
        st.error(data if isinstance(data, str) else f"Could not add {filename}.")

    with st.expander("Details"):
        st.caption(f"Saved as `{document_id}`")
        if isinstance(data, dict):
            st.json(data)
        else:
            st.code(str(data), language="text")


def render_ask_result(data: dict) -> None:
    answer = data.get("answer", {})
    text = answer.get("answer", "")
    confidence = answer.get("confidence")
    sources_needed = answer.get("sources_needed", False)
    chunk_ids = data.get("retrieved_chunk_ids", [])

    if sources_needed:
        st.warning("There isn't enough in your uploaded notes to answer that.")
    else:
        st.success("Answer from your notes")

    with st.container(border=True):
        st.markdown(text)

    with st.expander("View sources & details"):
        if chunk_ids:
            st.markdown("**Sources**")
            st.code("\n".join(chunk_ids), language="text")
        else:
            st.caption("No source IDs returned.")
        cols = st.columns(4)
        cols[0].metric("Confidence", f"{confidence:.2f}" if confidence is not None else "—")
        cols[1].metric("Tokens", data.get("tokens_used", "—"))
        cols[2].metric("Latency (ms)", data.get("latency_ms", "—"))
        cols[3].metric("Cost (USD)", data.get("cost_usd", "—"))
        st.json(data)


def render_agent_result(data: dict) -> None:
    """Show POST /agent answer, model, and steps (Think / Act / Observe from tool events)."""

    st.success("Agent answer")
    st.markdown(data.get("answer") or "(no answer)")
    st.caption(f"Model: `{data.get('model', '—')}`")

    st.markdown("**Think → Act → Observe** (from `/agent` `steps[]`, not hidden model reasoning)")
    steps = data.get("steps") or []
    if not steps:
        st.caption("No tool steps returned.")
    for index, step in enumerate(steps, start=1):
        tool = step.get("tool") or "unknown"
        observation = str(step.get("observation") or "")
        st.markdown(f"**Step {index}** — tool `{tool}`")
        if observation.startswith("called"):
            st.info(f"**Think** — model proposed `{tool}`")
            st.warning(f"**Act** — `{tool}` {observation}")
        else:
            st.success(f"**Observe** — `{tool}` returned a real result")
            st.code(observation, language="text")

    with st.expander("Full JSON response"):
        st.json(data)


def init_quiz_state() -> None:
    defaults = {
        "quiz_topic": "",
        "quiz_question": None,
        "quiz_source_ids": [],
        "quiz_grade": None,
        "quiz_notice": None,
        "quiz_adaptive_focus": None,
        "quiz_progress": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_grade_result(data: dict) -> None:
    verdict = data.get("verdict")
    labels = {
        "correct": ("Correct", "success"),
        "partially_correct": ("Partially correct", "warning"),
        "incorrect": ("Needs review · Incorrect", "error"),
    }
    label, level = labels.get(verdict, (verdict or "Ungraded", "info"))
    message_fn = {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[level]
    message_fn(label)

    with st.container(border=True):
        st.markdown("**Feedback**")
        st.write(data.get("feedback") or "—")
        st.markdown("**Explanation**")
        st.write(data.get("explanation") or "—")
        gap = data.get("identified_gap")
        if gap:
            st.markdown("**Focus for next time**")
            st.write(gap)

    with st.expander("View sources & details"):
        source_ids = data.get("source_document_ids") or []
        if source_ids:
            st.code("\n".join(str(item) for item in source_ids), language="text")
        else:
            st.caption("No source document IDs returned.")
        st.json(data)


def render_learning_progress(data: dict) -> None:
    st.markdown("**Learning progress**")
    topic = data.get("topic") or "This topic"
    st.caption(topic)
    st.markdown(f"Mastery: **{data.get('mastery', 'Not assessed')}**")

    cols = st.columns(4)
    cols[0].metric("Attempts", data.get("attempts", 0))
    cols[1].metric("Correct", data.get("correct", 0))
    cols[2].metric("Partial", data.get("partial", 0))
    cols[3].metric("Incorrect", data.get("incorrect", 0))

    weak = data.get("weak_concepts") or []
    st.markdown("**Focus areas**")
    if weak:
        for concept in weak:
            st.markdown(f"- {concept}")
    else:
        st.caption("None recorded yet.")


st.set_page_config(page_title="Adaptive Study Agent", layout="centered")
st.markdown(APP_CSS, unsafe_allow_html=True)

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
st.sidebar.caption("Upload notes, ask questions, then practice.")

with st.sidebar.expander("Developer settings"):
    base_url = st.text_input(
        "API base URL",
        value=DEFAULT_API_URL,
        help="Set RAG_API_URL in your environment to prefill this (no secrets needed).",
    )
    if st.button("Check connection"):
        try:
            health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=30.0)
            st.write(f"HTTP {health.status_code}: {health.text}")
        except httpx.HTTPError as exc:
            st.error(str(exc))

tab_ingest, tab_ask, tab_quiz, tab_advanced = st.tabs(
    ["Study Materials", "Ask Your Materials", "Quiz Me", "Advanced / Course Demo"]
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
            used_ids: set[str] = set()
            for uploaded in uploaded_files:
                filename = uploaded.name or "uploaded-document"
                document_id = unique_document_id(
                    document_id_from_filename(filename),
                    used_ids,
                )
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

                payload = {
                    "document_id": document_id,
                    "text": extracted,
                    "source": filename,
                }
                with st.spinner(f"Adding {filename}…"):
                    status, data = api_post(base_url, "/ingest", payload)
                render_ingest_file_result(
                    filename=filename,
                    document_id=document_id,
                    status=status,
                    data=data,
                )

    with st.expander("Advanced / Course Demo — paste text"):
        st.caption("Manual ingest for course demos. Same POST /ingest contract as before.")
        document_id = st.text_input("document_id", placeholder="POL-101")
        source = st.text_input("source (optional)", placeholder="doc1_handbook.txt")
        text = st.text_area("text", height=200, placeholder="Paste document text here…")

        if st.button("Ingest pasted text"):
            payload = {"document_id": document_id, "text": text}
            if source.strip():
                payload["source"] = source.strip()

            with st.spinner("Adding pasted text…"):
                status, data = api_post(base_url, "/ingest", payload)

            if status == 200 and isinstance(data, dict):
                st.success("Added pasted text to your study materials.")
            elif status >= 400:
                st.error(data.get("detail", data) if isinstance(data, dict) else data)
            with st.expander("Details"):
                st.json(data)

with tab_ask:
    st.subheader("Ask your materials")
    st.write("Get answers grounded in the notes you've uploaded.")

    question = st.text_input(
        "Your question",
        placeholder="What does this material say about the main topic?",
    )

    with st.expander("Answer options"):
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "o3-mini"], index=0)

    if st.button("Ask", type="primary"):
        payload = {"question": question, "model": model}
        with st.spinner("Looking through your notes…"):
            status, data = api_post(base_url, "/ask", payload)

        if status == 200 and isinstance(data, dict):
            render_ask_result(data)
        elif status >= 400:
            st.error(data.get("detail", data) if isinstance(data, dict) else data)
            with st.expander("View sources & details"):
                st.json(data)
        else:
            with st.expander("View sources & details"):
                st.json(data)

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
                payload = {"topic": topic, "user_id": "demo-user"}
                with st.spinner("Writing a question from your notes…"):
                    status, data = api_post(base_url, "/quiz", payload)

                if status == 200 and isinstance(data, dict) and data.get("status") == "ok":
                    st.session_state.quiz_question = data.get("question")
                    st.session_state.quiz_topic = data.get("topic") or topic
                    st.session_state.quiz_source_ids = data.get("source_document_ids") or []
                    st.session_state.quiz_adaptive_focus = data.get("adaptive_focus")
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_notice = None
                    st.session_state.quiz_answer = ""
                elif status == 200 and isinstance(data, dict):
                    st.session_state.quiz_question = None
                    st.session_state.quiz_grade = None
                    st.session_state.quiz_adaptive_focus = None
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
            if st.session_state.quiz_adaptive_focus:
                st.info(
                    "Reviewing a concept you've struggled with: "
                    f"{st.session_state.quiz_adaptive_focus}"
                )
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
                payload = {
                    "question": st.session_state.quiz_question,
                    "student_answer": student_answer,
                    "topic": st.session_state.quiz_topic,
                    "user_id": "demo-user",
                }
                with st.spinner("Checking your answer…"):
                    status, data = api_post(base_url, "/grade", payload)

                if status == 200 and isinstance(data, dict) and data.get("status") == "ok":
                    st.session_state.quiz_grade = data
                    st.session_state.quiz_progress = data.get("learning_progress")
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

    if st.session_state.quiz_progress:
        with st.container(border=True):
            render_learning_progress(st.session_state.quiz_progress)

with tab_advanced:
    st.caption("Course assignment tools. Not required for the student study flow.")
    tab_agent, tab_trace, tab_memory = st.tabs(["Agent", "TRACE Eval", "Memory"])

    with tab_agent:
        st.subheader("Ask the ADK agent")
        st.caption("Calls POST /agent — Gemini decides whether to run `search_docs` (Pinecone).")
        agent_user_id = st.text_input(
            "User ID",
            value="demo-user",
            key="agent_user_id",
            help="Durable preferences saved for this user are loaded before the agent runs.",
        )
        agent_message = st.text_area(
            "Agent question",
            key="agent_message",
            height=100,
            placeholder="How many annual leave days do Northwind Robotics employees get, and which policy document ID says so?",
        )

        if st.button("Run agent", type="primary"):
            payload = {
                "message": agent_message,
                "user_id": agent_user_id,
            }
            with st.spinner("Calling POST /agent…"):
                status, data = api_post(base_url, "/agent", payload)

            st.markdown(f"**HTTP {status}**")
            if status == 200 and isinstance(data, dict):
                render_agent_result(data)
            elif isinstance(data, dict) and "error" in data:
                st.error(data["error"])
            elif status >= 400:
                st.error(data.get("detail", data) if isinstance(data, dict) else data)
            else:
                st.error(data if isinstance(data, str) else "Unexpected response")
                if isinstance(data, dict):
                    st.json(data)

    with tab_trace:
        st.subheader("TRACE Evaluation")

        st.write(
            "Evaluation of 20 Northwind FAQ agent traces using "
            "deterministic code-based checks."
        )

        st.markdown("### Before / After")

        before_col, after_col = st.columns(2)

        with before_col:
            st.metric(
                label="Before fix",
                value="85%",
                delta="17 / 20 passed",
            )

        with after_col:
            st.metric(
                label="After fix",
                value="100%",
                delta="+15 percentage points",
            )

        st.markdown("### Checks")

        st.write("**1. Non-empty answer** — Agent must return a usable response.")
        st.write("**2. Tool execution** — `search_docs` must be recorded in the agent trace.")

        st.markdown("### Top failure")

        st.warning(
            "Empty agent response / no execution: "
            "3 of the original 20 traces returned no answer and no tool steps."
        )

        st.markdown("### Fix")

        st.success(
            "Added graceful handling so an empty agent result no longer "
            "silently returns a blank response."
        )

        st.markdown("### Result")

        st.write("Baseline: **17/20 passed (85%)**")
        st.write("After fix: **20/20 passed (100%)**")

        if st.button("Show evaluation results"):
            st.success("20 / 20 traces passed")
            st.progress(1.0)
            st.write("Overall pass rate: **100%**")

    with tab_memory:
        st.subheader("Durable Memory Demo")

        st.write(
            "Save a user preference, then retrieve it in a new session. "
            "Memory is stored outside the chat context and survives process restarts."
        )

        memory_user_id = st.text_input(
            "User ID",
            value="demo-user",
            key="memory_user_id",
        )

        st.markdown("### Save preference")

        preference_type = st.selectbox(
            "Preference",
            [
                "preferred_answer_style",
                "preferred_language",
                "preferred_name",
            ],
            key="memory_preference_type",
        )

        preference_value = st.text_input(
            "Value",
            value="Japanese",
            key="memory_preference_value",
        )

        if st.button("Save preference"):
            try:
                response = requests.post(
                    f"{API_BASE}/memory",
                    json={
                        "user_id": memory_user_id,
                        "key": preference_type,
                        "value": preference_value,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                st.success("Preference saved to durable memory.")
                st.json(response.json())
            except requests.RequestException as exc:
                st.error(f"Could not save memory: {exc}")

        st.markdown("### New session recall")

        st.write(
            "This retrieves durable memory from the backend. "
            "It does not use Streamlit chat history."
        )

        if st.button("Recall saved memory"):
            try:
                response = requests.get(
                    f"{API_BASE}/memory/{memory_user_id}",
                    timeout=30,
                )
                response.raise_for_status()

                recalled = response.json()

                st.success("Durable memory retrieved.")

                memory = recalled.get("memory") if isinstance(recalled, dict) else {}
                if isinstance(memory, dict):
                    mastery = memory.get("topic_mastery")
                    if isinstance(mastery, dict) and mastery:
                        st.markdown("### Learning progress")
                        for topic_name, entry in mastery.items():
                            render_learning_progress(topic_progress(str(topic_name), entry))

                st.json(recalled)

            except requests.RequestException as exc:
                st.error(f"Could not retrieve memory: {exc}")

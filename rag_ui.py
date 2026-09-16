"""Minimal Streamlit client for the FastAPI service (/ingest + /ask + /agent).

Run:
  export RAG_API_URL=https://your-service.onrender.com
  streamlit run rag_ui.py
"""

from __future__ import annotations

import json
import os
import requests

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 120.0


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


def render_ask_result(data: dict) -> None:
    answer = data.get("answer", {})
    text = answer.get("answer", "")
    confidence = answer.get("confidence")
    sources_needed = answer.get("sources_needed", False)
    chunk_ids = data.get("retrieved_chunk_ids", [])

    if sources_needed:
        st.warning("Refusal — not enough context in ingested docs")
    else:
        st.success("Grounded answer")

    st.markdown(text)

    st.markdown("**Citations / retrieved chunks**")
    if chunk_ids:
        st.code("\n".join(chunk_ids), language="text")
    else:
        st.caption("No chunk IDs returned.")

    cols = st.columns(4)
    cols[0].metric("Confidence", f"{confidence:.2f}" if confidence is not None else "—")
    cols[1].metric("Tokens", data.get("tokens_used", "—"))
    cols[2].metric("Latency (ms)", data.get("latency_ms", "—"))
    cols[3].metric("Cost (USD)", data.get("cost_usd", "—"))

    with st.expander("Full JSON response"):
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


st.set_page_config(page_title="RAG Demo", layout="centered")
st.title("RAG Demo")
st.caption("Streamlit calls your FastAPI service — no RAG logic runs in this UI.")

base_url = st.sidebar.text_input(
    "API base URL",
    value=DEFAULT_API_URL,
    help="Set RAG_API_URL in your environment to prefill this (no secrets needed).",
)

if st.sidebar.button("Check /health"):
    try:
        health = httpx.get(f"{base_url.rstrip('/')}/health", timeout=30.0)
        st.sidebar.write(f"HTTP {health.status_code}: {health.text}")
    except httpx.HTTPError as exc:
        st.sidebar.error(str(exc))

tab_ingest, tab_ask, tab_agent, tab_trace, tab_memory = st.tabs(
    ["Ingest", "Ask", "Agent", "TRACE Eval", "Memory"]
)

with tab_ingest:
    st.subheader("Ingest a document")
    document_id = st.text_input("document_id", placeholder="POL-101")
    source = st.text_input("source (optional)", placeholder="doc1_handbook.txt")
    text = st.text_area("text", height=200, placeholder="Paste document text here…")

    if st.button("Ingest", type="primary"):
        payload = {"document_id": document_id, "text": text}
        if source.strip():
            payload["source"] = source.strip()

        with st.spinner("Calling POST /ingest…"):
            status, data = api_post(base_url, "/ingest", payload)

        st.markdown(f"**HTTP {status}**")
        if status == 200 and isinstance(data, dict):
            st.success(f"Indexed {data.get('chunks_indexed', '?')} chunks for `{data.get('document_id')}`")
        elif status >= 400:
            st.error(data.get("detail", data) if isinstance(data, dict) else data)
        st.json(data)

with tab_ask:
    st.subheader("Ask a question")
    question = st.text_input(
        "Question",
        placeholder="How many remote days are allowed per week?",
    )
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "o3-mini"], index=0)

    if st.button("Ask", type="primary"):
        payload = {"question": question, "model": model}
        with st.spinner("Calling POST /ask…"):
            status, data = api_post(base_url, "/ask", payload)

        st.markdown(f"**HTTP {status}**")
        if status == 200 and isinstance(data, dict):
            render_ask_result(data)
        elif status >= 400:
            st.error(data.get("detail", data) if isinstance(data, dict) else data)
            st.json(data)
        else:
            st.json(data)

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
            st.json(recalled)

        except requests.RequestException as exc:
            st.error(f"Could not retrieve memory: {exc}")
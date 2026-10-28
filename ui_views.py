"""Streamlit views for the Progress and Question Bank tabs.

Kept separate from rag_ui.py so they can be rendered in tests (streamlit AppTest)
without Google sign-in. `fetch(path, params)` returns (status, data) like
rag_ui.api_request; `error_message(status, data, fallback)` formats failures.
"""

from __future__ import annotations

from collections import OrderedDict

import streamlit as st

BAND_LABELS = {
    "mastered": "Mastered",
    "developing": "Developing",
    "needs_review": "Needs review",
    "not_practiced": "Not practiced yet",
}
UNCATEGORIZED = "Uncategorized"
ALL_TOPICS = "All topics"


def pct(value) -> str:
    return f"{value:.0%}" if value is not None else "—"


def short_date(value) -> str:
    return str(value)[:10] if value else "—"


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


# ── Progress ─────────────────────────────────────────────────────────

def render_progress_overview(fetch, error_message) -> None:
    status, data = fetch("/progress", None)
    if status != 200 or not isinstance(data, list):
        st.warning(error_message(status, data, "Couldn't load your progress."))
        return

    rows = [
        {
            "Course": course["name"],
            "Mastery": pct(course["mastery"]) if course["questions_answered"] else "Not practiced",
            "Questions answered": course["questions_answered"],
            "Average score": pct(course["average_score"]),
            "Last activity": short_date(course["last_activity_at"]),
        }
        for course in data
    ]
    st.dataframe(rows, hide_index=True)


def _topic_caption(topic) -> str:
    if topic["mastery_band"] == "not_practiced":
        return f"Not practiced yet · {plural(topic['questions_in_bank'], 'question')} ready"

    parts = [
        f"{topic['questions_answered']} of {plural(topic['questions_in_bank'], 'question')} answered",
        f"{plural(topic['attempts'], 'answer')} graded",
        f"average score {pct(topic['average_score'])}",
        f"last practiced {short_date(topic['last_attempt_at'])}",
    ]
    if topic["low_data"]:
        parts.append("early estimate")
    return " · ".join(parts)


def render_course_progress(course_id, fetch, error_message) -> None:
    status, data = fetch(f"/courses/{course_id}/progress", None)
    if status != 200 or not isinstance(data, dict):
        st.warning(error_message(status, data, "Couldn't load progress for this course."))
        return

    if not data["attempts"]:
        st.info("No graded answers in this course yet. Practice in **Quiz Me** and your progress will appear here.")
    else:
        cols = st.columns(4)
        cols[0].metric(
            "Course mastery",
            pct(data["mastery"]),
            help="Average mastery of the questions you've answered (the same estimate that picks your next question).",
        )
        cols[1].metric("Questions answered", f"{data['questions_answered']}")
        cols[2].metric("Average score", pct(data["average_score"]))
        cols[3].metric("Answers in last 7 days", f"{data['attempts_last_7_days']}")
        if data["low_data"]:
            st.caption(
                f"Early estimate: based on {plural(data['attempts'], 'graded answer')}. "
                "Mastery becomes more reliable as you practice."
            )

    focus = [t for t in data["topics"] if t["name"] in data["focus_topics"]]
    if focus:
        st.markdown("#### Focus next")
        for topic in focus:
            st.markdown(f"- **{topic['name']}**: {BAND_LABELS[topic['mastery_band']].lower()}, mastery {pct(topic['mastery'])}")

    practiced = [t for t in data["topics"] if t["mastery_band"] != "not_practiced"]
    unpracticed = [t for t in data["topics"] if t["mastery_band"] == "not_practiced"]

    if practiced:
        st.markdown("#### Topic mastery")
        for topic in practiced:
            st.markdown(f"**{topic['name']}** · {BAND_LABELS[topic['mastery_band']]}")
            st.progress(min(max(topic["mastery"], 0.0), 1.0), text=f"Mastery {pct(topic['mastery'])}")
            st.caption(_topic_caption(topic))

    if unpracticed:
        st.markdown("#### Not practiced yet")
        st.caption("These topics have questions but no graded answers, so there's no mastery estimate yet.")
        for topic in unpracticed:
            st.markdown(f"- **{topic['name']}**: {_topic_caption(topic)}")

    if not data["topics"]:
        st.caption("No topics yet. Ask for a question in **Quiz Me** to start a topic.")

    if data["recent_activity"]:
        st.markdown("#### Recent activity")
        verdicts = {"correct": "Correct", "partially_correct": "Partially correct", "incorrect": "Needs review"}
        for item in data["recent_activity"]:
            question = item["question"] or "_Question from a deleted document_"
            label = verdicts.get(item["verdict"], "Ungraded")
            topic = f" · {item['topic']}" if item["topic"] else ""
            st.markdown(f"- {question}")
            st.caption(f"{label} · score {pct(item['score'])}{topic} · {short_date(item['attempted_at'])}")


# ── Question bank ────────────────────────────────────────────────────

def group_by_topic(items) -> "OrderedDict[str, list]":
    groups: OrderedDict[str, list] = OrderedDict()
    for item in items:
        groups.setdefault(item.get("topic") or UNCATEGORIZED, []).append(item)
    return groups


def render_question_bank(course_id, fetch, error_message) -> None:
    status, data = fetch("/question-bank", {"course_id": course_id})
    if status != 200 or not isinstance(data, list):
        st.warning(error_message(status, data, "Couldn't load your question bank."))
        return

    if not data:
        st.info("No saved questions in this course yet. Questions you get in **Quiz Me** are added here.")
        return

    groups = group_by_topic(data)
    choice = st.selectbox("Topic", [ALL_TOPICS, *groups.keys()], key=f"bank_topic_{course_id}")
    st.caption(
        f"{plural(len(data), 'question')} across {plural(len(groups), 'topic')}. "
        "Each question is listed once; model answers are the ones saved when the question was written."
    )

    for topic, items in groups.items():
        if choice not in (ALL_TOPICS, topic):
            continue
        st.markdown(f"#### {topic}")
        for item in items:
            with st.container(border=True):
                st.markdown(item["prompt"])
                meta = [item["concept_label"]] if item.get("concept_label") else []
                attempts = item.get("attempt_count") or 0
                meta.append(f"practiced {attempts}×" if attempts else "not practiced yet")
                meta.append(f"added {short_date(item.get('created_at'))}")
                st.caption(" · ".join(meta))
                with st.expander("Show model answer"):
                    if item.get("reference_answer"):
                        st.markdown(item["reference_answer"])
                    else:
                        st.caption("No model answer was saved for this question.")
                    if item.get("key_points"):
                        st.markdown("**Key points**")
                        for point in item["key_points"]:
                            st.markdown(f"- {point}")

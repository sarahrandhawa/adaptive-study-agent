"""Render the Progress and Question Bank views with Streamlit's AppTest (no sign-in,
no network): `fetch` returns canned API payloads."""

import json

import pytest
from streamlit.testing.v1 import AppTest

COURSE = "44444444-4444-4444-4444-444444444444"


def topic(name, band, mastery, answered=1, attempts=3, in_bank=2):
    return {
        "topic_id": "66666666-6666-6666-6666-666666666666", "name": name, "mastery_band": band,
        "mastery": mastery, "questions_answered": answered, "questions_in_bank": in_bank, "attempts": attempts,
        "average_score": mastery, "correct_rate": None, "low_data": attempts < 3, "last_attempt_at": "2026-10-27T10:00:00Z",
    }


def progress(**overrides):
    data = {
        "course_id": COURSE, "name": "Bio", "questions_in_bank": 4, "questions_answered": 2, "attempts": 5,
        "average_score": 0.6, "correct_rate": 0.4, "mastery": 0.55, "mastery_band": "developing", "low_data": False,
        "last_activity_at": "2026-10-27T10:00:00Z", "attempts_last_7_days": 5,
        "topics": [
            topic("Weak", "needs_review", 0.2),
            topic("Strong", "mastered", 0.9),
            topic("Fresh", "not_practiced", None, answered=0, attempts=0),
        ],
        "focus_topics": ["Weak"], "not_practiced_topics": ["Fresh"],
        "recent_activity": [
            {"attempted_at": "2026-10-27T10:00:00Z", "verdict": "incorrect", "score": 0.1, "question": None, "topic": "Weak", "source_removed": True},
        ],
    }
    data.update(overrides)
    return data


def run_view(call: str, responses: dict) -> AppTest:
    script = f"""
import json
import ui_views

RESPONSES = json.loads({json.dumps(json.dumps(responses))})

def fetch(path, params=None):
    status, data = RESPONSES[path]
    return status, data

def error_message(status, data, fallback):
    return fallback

ui_views.{call}
"""
    return AppTest.from_string(script, default_timeout=30).run()


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [str(m.value) for m in at.metric] + [m.label for m in at.metric]
    return "\n".join(parts)


def test_course_progress_renders_bands_focus_and_unpracticed():
    at = run_view(f'render_course_progress("{COURSE}", fetch, error_message)', {f"/courses/{COURSE}/progress": [200, progress()]})

    assert not at.exception
    out = texts(at)
    assert "Course mastery" in out and "55%" in out
    assert "Focus next" in out and "**Weak**" in out
    assert "Needs review" in out and "Mastered" in out
    assert "Not practiced yet" in out and "**Fresh**" in out
    assert "Question from a deleted document" in out
    assert len(at.get("progress")) == 2  # bars only for practised topics


def test_course_progress_with_no_answers_shows_no_fake_mastery():
    data = progress(attempts=0, questions_answered=0, mastery=None, average_score=None, mastery_band="not_practiced",
                    low_data=True, topics=[topic("Fresh", "not_practiced", None, answered=0, attempts=0)],
                    focus_topics=[], recent_activity=[])
    at = run_view(f'render_course_progress("{COURSE}", fetch, error_message)', {f"/courses/{COURSE}/progress": [200, data]})

    assert not at.exception
    assert at.info and "No graded answers" in at.info[0].value
    assert not at.metric
    assert not at.get("progress")


def test_course_progress_low_data_is_labelled_early_estimate():
    at = run_view(f'render_course_progress("{COURSE}", fetch, error_message)',
                  {f"/courses/{COURSE}/progress": [200, progress(attempts=2, low_data=True)]})

    assert "Early estimate" in texts(at)


def test_progress_error_is_reported():
    at = run_view(f'render_course_progress("{COURSE}", fetch, error_message)', {f"/courses/{COURSE}/progress": [503, {}]})

    assert at.warning[0].value == "Couldn't load progress for this course."


def test_progress_overview_table():
    rows = [progress(), progress(name="Empty", questions_answered=0, mastery=None, average_score=None, last_activity_at=None)]
    at = run_view("render_progress_overview(fetch, error_message)", {"/progress": [200, rows]})

    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["Course"]) == ["Bio", "Empty"]
    assert list(table["Mastery"]) == ["55%", "Not practiced"]


BANK = [
    {"id": "q1", "prompt": "Why does water evaporate?", "reference_answer": "Heat.", "key_points": ["heat", "sun"],
     "concept_label": "evaporation", "created_at": "2026-10-20T00:00:00Z", "topic": "Water", "attempt_count": 3},
    {"id": "q2", "prompt": "Old question", "reference_answer": None, "key_points": [],
     "concept_label": None, "created_at": "2026-10-01T00:00:00Z", "topic": None, "attempt_count": 0},
]


def test_question_bank_groups_by_topic_and_hides_answers_in_expanders():
    at = run_view(f'render_question_bank("{COURSE}", fetch, error_message)', {"/question-bank": [200, BANK]})

    assert not at.exception
    out = texts(at)
    assert "#### Water" in out and "#### Uncategorized" in out
    assert "practiced 3×" in out and "not practiced yet" in out
    assert [e.label for e in at.expander] == ["Show model answer", "Show model answer"]
    assert "Heat." in [m.value for m in at.expander[0].markdown]
    assert "No model answer was saved" in at.expander[1].caption[0].value


def test_question_bank_topic_filter():
    at = run_view(f'render_question_bank("{COURSE}", fetch, error_message)', {"/question-bank": [200, BANK]})
    at.selectbox[0].select("Water").run()

    assert "#### Uncategorized" not in texts(at)
    assert len(at.expander) == 1


@pytest.mark.parametrize("response, expected", [([200, []], "No saved questions"), ([503, {}], "Couldn't load your question bank.")])
def test_question_bank_empty_and_error_states(response, expected):
    at = run_view(f'render_question_bank("{COURSE}", fetch, error_message)', {"/question-bank": response})

    assert expected in texts(at)

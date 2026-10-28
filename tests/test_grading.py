"""Grading logic in study.grade_student_answer, with the LLM and DB mocked."""

import math

import pytest

import study
from fakes import QUESTION_ID, USER_ID, FakeGrader, fake_question


@pytest.fixture
def recorded(monkeypatch):
    """Capture record_attempt calls instead of writing to a database."""

    calls = []

    def fake_record_attempt(**kwargs):
        calls.append(kwargs)
        return {"attempt_id": "a1", "mastery": 0.5, "attempt_count": 1, "correct_streak": 0}

    monkeypatch.setattr(study, "record_attempt", fake_record_attempt)
    monkeypatch.setattr(study, "load_question_for_grading", lambda **_: fake_question())
    return calls


def grade(answer="Evaporation vs precipitation"):
    return study.grade_student_answer(user_id=USER_ID, question_id=QUESTION_ID, student_answer=answer)


# ── Deterministic verdict thresholds ─────────────────────────────────

@pytest.mark.parametrize(
    ("score", "verdict"),
    [
        (1.0, "correct"),
        (0.85, "correct"),
        (0.8499, "partially_correct"),
        (0.35, "partially_correct"),
        (0.3499, "incorrect"),
        (0.0, "incorrect"),
    ],
)
def test_verdict_from_score_boundaries(score, verdict):
    assert study.verdict_from_score(score) == verdict


@pytest.mark.parametrize(
    ("llm_score", "stored_score", "verdict"),
    [
        (0.9, 0.9, "correct"),
        (0.85, 0.85, "correct"),
        (0.5, 0.5, "partially_correct"),
        (0.1, 0.1, "incorrect"),
        (1.7, 1.0, "correct"),
        (-0.3, 0.0, "incorrect"),
    ],
)
def test_grade_uses_code_verdict_and_clamps_score(monkeypatch, recorded, llm_score, stored_score, verdict):
    monkeypatch.setattr(study, "_call_structured", FakeGrader(score=llm_score))

    result = grade()

    assert result["status"] == "ok"
    assert result["score"] == stored_score
    assert result["verdict"] == verdict
    assert len(recorded) == 1
    assert recorded[0]["score"] == stored_score
    assert recorded[0]["verdict"] == verdict


def test_correct_answer_clears_identified_gap(monkeypatch, recorded):
    monkeypatch.setattr(study, "_call_structured", FakeGrader(score=0.95, identified_gap="should be dropped"))

    result = grade()

    assert result["identified_gap"] is None
    assert recorded[0]["identified_gap"] is None


def test_partial_answer_keeps_stripped_gap_and_feedback(monkeypatch, recorded):
    monkeypatch.setattr(study, "_call_structured", FakeGrader(score=0.5, identified_gap="  misses precipitation  "))

    result = grade()

    assert result["identified_gap"] == "misses precipitation"
    assert recorded[0]["feedback"] == "Nice work."
    assert recorded[0]["explanation"] == "It compares evaporation to precipitation."


# ── Malformed / missing model scores ─────────────────────────────────

def test_insufficient_material_records_no_attempt(monkeypatch, recorded):
    monkeypatch.setattr(
        study, "_call_structured", FakeGrader(sufficient_material=False, score=None, insufficient_reason=" off topic ")
    )

    result = grade()

    assert result["status"] == "insufficient_material"
    assert result["detail"] == "off topic"
    assert recorded == []


def test_missing_score_records_no_attempt(monkeypatch, recorded):
    monkeypatch.setattr(study, "_call_structured", FakeGrader(score=None))

    result = grade()

    assert result["status"] == "insufficient_material"
    assert result["verdict"] is None
    assert recorded == []


def test_model_json_with_nan_score_reaches_the_grader():
    """Confirms the NaN path is reachable: pydantic accepts a NaN literal in model JSON."""

    draft = study.GradeDraft.model_validate_json('{"sufficient_material": true, "score": NaN}')

    assert math.isnan(draft.score)


@pytest.mark.regression
@pytest.mark.parametrize("bad_score", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_score_is_not_recorded(monkeypatch, recorded, bad_score):
    """BUG: max(0, min(1, nan)) == 1.0, so a NaN (or +inf) score is graded 'correct'
    and recorded as a perfect attempt, inflating mastery."""

    monkeypatch.setattr(study, "_call_structured", FakeGrader(score=bad_score))

    result = grade()

    assert recorded == [], f"non-finite score was recorded as {recorded[0]['score']!r} / {recorded[0]['verdict']!r}"
    assert result["status"] != "ok"


# ── Blank answers and missing material never reach the LLM ──────────

@pytest.mark.parametrize("answer", ["", "   ", "\n\t  "], ids=["empty", "spaces", "whitespace"])
def test_blank_answer_rejected_before_any_io(monkeypatch, answer):
    grader = FakeGrader()
    monkeypatch.setattr(study, "_call_structured", grader)

    def no_db(**_):
        raise AssertionError("question should not be loaded for a blank answer")

    monkeypatch.setattr(study, "load_question_for_grading", no_db)

    with pytest.raises(study.StudyValidationError):
        grade(answer)

    assert grader.calls == 0


def test_question_without_source_chunks_is_rejected(monkeypatch, recorded):
    grader = FakeGrader()
    monkeypatch.setattr(study, "_call_structured", grader)
    monkeypatch.setattr(study, "load_question_for_grading", lambda **_: fake_question(chunks=False))

    with pytest.raises(study.StudyValidationError):
        grade()

    assert grader.calls == 0
    assert recorded == []


def test_student_answer_is_stripped_before_grading_and_storage(monkeypatch, recorded):
    grader = FakeGrader()
    monkeypatch.setattr(study, "_call_structured", grader)

    grade("   my answer   ")

    assert recorded[0]["answer_text"] == "my answer"
    assert "Student answer:\nmy answer\n" in grader.prompts[0]

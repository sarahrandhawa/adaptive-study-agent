"""generate_quiz_question flow (reuse vs. generate) and select_next_question using the
real priority_score formula. Topic, retrieval, LLM and persistence are faked."""

from datetime import datetime, timedelta, timezone

import pytest

import study
from fakes import COURSE_ID, QUESTION_ID, USER_ID

TOPIC_UUID = "66666666-6666-6666-6666-666666666666"
CHUNKS = [{"chunk_id": "c1", "document_id": "d1", "chunk_index": 0, "source": "notes.txt", "text": "Water evaporates."}]


class QuizLLM:
    def __init__(self, draft):
        self.draft = draft
        self.calls = 0

    def __call__(self, prompt, schema):
        assert schema is study.QuizDraft
        self.calls += 1
        return self.draft


def good_draft(**overrides):
    fields = dict(
        sufficient_material=True,
        question="  Why does water evaporate?  ",
        reference_answer="Heat from the sun.",
        key_points=["heat", "sun"],
        concept_label=" evaporation ",
    )
    fields.update(overrides)
    return study.QuizDraft(**fields)


@pytest.fixture
def flow(monkeypatch):
    state = {"saved": [], "retrieved": 0, "selection": {"mode": "generate", "topic_mastery": 0.5}, "chunks": CHUNKS}

    monkeypatch.setattr(study, "get_or_create_topic", lambda user_id, course_id, name: TOPIC_UUID)
    monkeypatch.setattr(study, "select_next_question", lambda **_: state["selection"])

    def fake_retrieve(**kwargs):
        state["retrieved"] += 1
        return {"chunks": state["chunks"]}

    def fake_save(**kwargs):
        state["saved"].append(kwargs)
        return QUESTION_ID

    monkeypatch.setattr(study, "retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(study, "save_question", fake_save)
    return state


def generate(topic="evaporation"):
    return study.generate_quiz_question(user_id=USER_ID, course_id=COURSE_ID, topic=topic)


def test_existing_question_is_reused_without_retrieval_or_llm(monkeypatch, flow):
    llm = QuizLLM(good_draft())
    monkeypatch.setattr(study, "_call_structured", llm)
    flow["selection"] = {
        "mode": "existing",
        "topic_mastery": 0.4,
        "question": {"question_id": QUESTION_ID, "question": "Old Q", "concept_label": "c"},
    }

    result = generate()

    assert result["mode"] == "existing"
    assert result["question_id"] == QUESTION_ID
    assert result["topic_mastery"] == 0.4
    assert llm.calls == 0
    assert flow["retrieved"] == 0
    assert flow["saved"] == []


def test_generated_question_is_saved_and_never_leaks_answer(monkeypatch, flow):
    monkeypatch.setattr(study, "_call_structured", QuizLLM(good_draft()))

    result = generate()

    assert result["status"] == "ok"
    assert result["mode"] == "generated"
    assert result["question"] == "Why does water evaporate?"
    assert result["concept_label"] == "evaporation"
    assert result["source_document_ids"] == ["d1"]
    assert "reference_answer" not in result
    assert "key_points" not in result
    assert len(flow["saved"]) == 1
    assert flow["saved"][0]["topic_id"] == TOPIC_UUID


def test_no_retrieved_material_skips_llm(monkeypatch, flow):
    llm = QuizLLM(good_draft())
    monkeypatch.setattr(study, "_call_structured", llm)
    flow["chunks"] = []

    result = generate()

    assert result["status"] == "insufficient_material"
    assert llm.calls == 0
    assert flow["saved"] == []


@pytest.mark.parametrize(
    "draft",
    [
        good_draft(sufficient_material=False, insufficient_reason="off topic"),
        good_draft(question="   "),
    ],
    ids=["model_says_insufficient", "blank_question"],
)
def test_unusable_draft_is_not_saved(monkeypatch, flow, draft):
    monkeypatch.setattr(study, "_call_structured", QuizLLM(draft))

    result = generate()

    assert result["status"] == "insufficient_material"
    assert flow["saved"] == []


@pytest.mark.regression
@pytest.mark.parametrize(
    "overrides",
    [{"reference_answer": "  "}, {"key_points": []}],
    ids=["blank_reference_answer", "no_key_points"],
)
def test_draft_without_grading_rubric_is_not_saved(monkeypatch, flow, overrides):
    """BUG: _question_from_prompt only checks sufficient_material and question text.
    A draft with no reference answer or no key points is saved, and every later
    grade of that question is judged against an empty rubric."""

    monkeypatch.setattr(study, "_call_structured", QuizLLM(good_draft(**overrides)))

    result = generate()

    assert flow["saved"] == []
    assert result["status"] == "insufficient_material"


def test_blank_topic_rejected_before_topic_row_is_created(monkeypatch):
    def no_write(*args):
        raise AssertionError("topic row created for a blank topic")

    monkeypatch.setattr(study, "get_or_create_topic", no_write)

    with pytest.raises(study.StudyValidationError):
        generate("   ")


# ── select_next_question with the real priority formula ─────────────

NOW = datetime.now(timezone.utc)


def candidate(qid, mastery, last_attempt_at):
    return {"question_id": qid, "question": f"Q{qid}", "concept_label": "c", "mastery": mastery, "last_attempt_at": last_attempt_at}


def select(monkeypatch, candidates, topic_mastery):
    monkeypatch.setattr(study, "get_question_candidates", lambda **_: candidates)
    monkeypatch.setattr(study, "get_topic_mastery", lambda **_: topic_mastery)
    return study.select_next_question(user_id=USER_ID, topic_id=TOPIC_UUID)


def test_unanswered_question_wins_tie_against_generating(monkeypatch):
    result = select(monkeypatch, [candidate(1, None, None)], 0.6)

    assert result["mode"] == "existing"


def test_recently_answered_question_loses_to_new_question(monkeypatch):
    result = select(monkeypatch, [candidate(1, 0.3, NOW - timedelta(minutes=1))], 0.5)

    assert result["mode"] == "generate"


def test_weak_question_returns_once_recency_penalty_expires(monkeypatch):
    result = select(monkeypatch, [candidate(1, 0.2, NOW - timedelta(hours=25))], 0.6)

    assert result["mode"] == "existing"
    assert result["question"]["question_id"] == 1


def test_weakest_stale_question_is_chosen_among_several(monkeypatch):
    candidates = [
        candidate(1, 0.6, NOW - timedelta(days=3)),
        candidate(2, 0.1, NOW - timedelta(minutes=5)),
        candidate(3, 0.15, NOW - timedelta(days=2)),
    ]

    result = select(monkeypatch, candidates, 0.9)

    assert result["mode"] == "existing"
    assert result["question"]["question_id"] == 3


def test_well_mastered_topic_prefers_reviewing_over_generating(monkeypatch):
    """High topic mastery raises the 'new question' priority (0.5 * mastery), so a
    stale, moderately weak question is reviewed instead of writing another one."""

    result = select(monkeypatch, [candidate(1, 0.4, NOW - timedelta(days=2))], 0.9)

    assert result["mode"] == "existing"

import pytest

import study
from fakes import QUESTION_ID, USER_ID, TOPIC_ID, FakeGrader, fake_question

def test_generate_no_existing_questions(monkeypatch):
    def fake_get_question_candidates(user_id, topic_id):
        return []

    def fake_get_topic_mastery(user_id, topic_id):
        return 0.5

    monkeypatch.setattr(study, "get_question_candidates", fake_get_question_candidates)
    monkeypatch.setattr(study, "get_topic_mastery", fake_get_topic_mastery)
    

    result = study.select_next_question(user_id=USER_ID, topic_id=TOPIC_ID)

    assert result["mode"] == "generate"
    assert result["topic_mastery"] == 0.5

def test_existing_unanswered_question(monkeypatch):
    def fake_get_question_candidates(user_id, topic_id):
        return [
            {
                "question_id": QUESTION_ID,
                "question": "Testing Question",
                "concept_label": "Test Label",
                "mastery": None,
                "last_attempt_at": None,
            }
        ]
    
    def fake_get_topic_mastery(user_id, topic_id):
        return 0.5
    
    def fake_priority_score(question_mastery, topic_mastery, last_attempt_at, now):
        return 0.1
    
    monkeypatch.setattr(study, "get_question_candidates", fake_get_question_candidates)
    monkeypatch.setattr(study, "get_topic_mastery", fake_get_topic_mastery)
    monkeypatch.setattr(study, "priority_score", fake_priority_score)

    result = study.select_next_question(user_id=USER_ID, topic_id=TOPIC_ID)

    assert result["mode"] == "existing"
    assert result["question"] == fake_get_question_candidates(user_id=USER_ID, topic_id=TOPIC_ID)[0]

def test_generate_when_existing_question(monkeypatch):
    def fake_get_question_candidates(user_id, topic_id):
        return [
            {
                "question_id": QUESTION_ID,
                "question": "Testing Question",
                "concept_label": "Test Label",
                "mastery": None,
                "last_attempt_at": None,
            }
        ]
    
    def fake_get_topic_mastery(user_id, topic_id):
        return 0.5
    
    def fake_priority_score(question_mastery, topic_mastery, last_attempt_at, now):
        return 0.3
    
    monkeypatch.setattr(study, "get_question_candidates", fake_get_question_candidates)
    monkeypatch.setattr(study, "get_topic_mastery", fake_get_topic_mastery)
    monkeypatch.setattr(study, "priority_score", fake_priority_score)

    result = study.select_next_question(user_id=USER_ID, topic_id=TOPIC_ID)

    assert result["mode"] == "generate"
    assert result["topic_mastery"] == 0.5

def test_select_lowest_priority(monkeypatch):
    def fake_get_question_candidates(user_id, topic_id):
        return [
            {
                "question_id": 1111,
                "question": "Question 1",
                "concept_label": "Test Label",
                "mastery": 0.5,
                "last_attempt_at": None,
            },
            {
                "question_id": 2222,
                "question": "Question 2",
                "concept_label": "Test Label",
                "mastery": 0.2,
                "last_attempt_at": None,
            },
            {
                "question_id": 3333,
                "question": "Question 3",
                "concept_label": "Test Label",
                "mastery": 0.8,
                "last_attempt_at": None,
            }
        ]
    
    def fake_get_topic_mastery(user_id, topic_id):
        return 0.5
    
    def fake_priority_score(question_mastery, topic_mastery, last_attempt_at, now):
        return question_mastery * topic_mastery
    
    monkeypatch.setattr(study, "get_question_candidates", fake_get_question_candidates)
    monkeypatch.setattr(study, "get_topic_mastery", fake_get_topic_mastery)
    monkeypatch.setattr(study, "priority_score", fake_priority_score)

    result = study.select_next_question(user_id=USER_ID, topic_id=TOPIC_ID)

    assert result["mode"] == "existing"
    assert result["question"] == fake_get_question_candidates(user_id=USER_ID, topic_id=TOPIC_ID)[1]
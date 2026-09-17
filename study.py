"""Grounded quiz generation and grading — reuses retrieve_chunks + OpenAI structured output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import APIError, OpenAI
from pydantic import BaseModel

from memory_store import record_quiz_result, weak_concepts_for_topic
from rag import DEFAULT_RETRIEVAL_K, _format_chunk
from retrieve import retrieve_chunks

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

_client = OpenAI()
DEFAULT_STUDY_MODEL = os.getenv("STUDY_MODEL", "gpt-4o-mini")

QUIZ_PROMPT = """You are a study quiz generator. Create ONE question using ONLY the retrieved course material below.

Rules:
- Use ONLY the retrieved context. Never invent facts, vocabulary, numbers, or examples.
- The question must be answerable from the retrieved context.
- Prefer a question that tests understanding (meaning, when to use something, or choosing the right form) rather than asking the student to copy a sentence.
- Write the question only. Do not include the correct answer, hints, answer choices, or an answer key.
- If the context is missing, off-topic, or too thin to write a fair question about the student's topic, set sufficient_material to false and explain why.

Student topic: {topic}

Retrieved course material:
{context}
"""

ADAPTIVE_QUIZ_PROMPT = """You are a study quiz generator. Create ONE question using ONLY the retrieved course material below.

Rules:
- Use ONLY the retrieved context. Never invent facts, vocabulary, numbers, or examples.
- The question must be answerable from the retrieved context.
- Prefer a question that tests understanding rather than asking the student to copy a sentence.
- Write the question only. Do not include the correct answer, hints, answer choices, or an answer key.
- The student needs review on this concept: {focus}
- If the retrieved material supports that review concept, write a question that tests it.
- If the retrieved material does NOT support that concept, set sufficient_material to false. Do not invent a question from memory of the concept.

Student topic: {topic}
Review focus: {focus}

Retrieved course material:
{context}
"""

GRADE_PROMPT = """You are a grounded study tutor. Grade the student's answer using ONLY the retrieved course material.

Rules:
- Use ONLY the retrieved context. Do not use outside knowledge.
- If the context is too thin or off-topic to grade this question fairly, set sufficient_material to false.
- verdict must be exactly one of: correct, partially_correct, incorrect.
  - correct: the student demonstrates the concept accurately.
  - partially_correct: some understanding, but an important part is missing or wrong.
  - incorrect: the student does not demonstrate the required understanding.
- feedback: short, student-facing comments on how they did.
- explanation: teach from the retrieved material so the student can see what was expected.
- identified_gap: the specific concept they appear weak on (for example "Japanese time-of-day greetings"). Not a generic phrase like "wrong answer". Use null when verdict is correct.

Topic: {topic}
Question: {question}
Student answer: {student_answer}

Retrieved course material:
{context}
"""


class StudyValidationError(ValueError):
    """Raised when quiz/grade input fails validation."""


class StudyModelError(RuntimeError):
    """Raised when the quiz/grade model call fails."""


class QuizDraft(BaseModel):
    sufficient_material: bool
    question: str = ""
    insufficient_reason: str = ""


class GradeDraft(BaseModel):
    sufficient_material: bool
    verdict: Literal["correct", "partially_correct", "incorrect"] | None = None
    feedback: str = ""
    explanation: str = ""
    identified_gap: str | None = None
    insufficient_reason: str = ""


def _document_ids(chunks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        document_id = chunk.get("document_id")
        if not document_id:
            continue
        text_id = str(document_id)
        if text_id not in seen:
            seen.add(text_id)
            ids.append(text_id)
    return ids


def _context_from_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(_format_chunk(chunk) for chunk in chunks)


def _call_structured(prompt: str, schema: type[BaseModel]) -> BaseModel:
    last_error: Exception | None = None
    for _ in range(2):
        try:
            completion = _client.chat.completions.parse(
                model=DEFAULT_STUDY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Model returned no parseable structured output")
            return parsed
        except (ValueError, APIError) as exc:
            last_error = exc
            continue
    raise StudyModelError(str(last_error) if last_error else "Model call failed")


def _insufficient_quiz(topic: str, source_document_ids: list[str], detail: str) -> dict[str, Any]:
    return {
        "status": "insufficient_material",
        "question": None,
        "topic": topic,
        "source_document_ids": source_document_ids,
        "adaptive_focus": None,
        "detail": detail,
    }


def _ok_quiz(
    *,
    topic: str,
    question: str,
    source_document_ids: list[str],
    adaptive_focus: str | None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "question": question,
        "topic": topic,
        "source_document_ids": source_document_ids,
        "adaptive_focus": adaptive_focus,
        "detail": None,
    }


def _question_from_prompt(prompt: str) -> tuple[str | None, str]:
    draft = _call_structured(prompt, QuizDraft)
    assert isinstance(draft, QuizDraft)
    question = draft.question.strip()
    if not draft.sufficient_material or not question:
        return None, (
            draft.insufficient_reason.strip()
            or "The retrieved material is not enough to write a fair question on that topic."
        )
    return question, ""


def generate_quiz_question(*, topic: str, user_id: str = "demo-user") -> dict[str, Any]:
    """Retrieve course chunks for a topic and generate one grounded quiz question."""

    normalized_topic = topic.strip()
    if not normalized_topic:
        raise StudyValidationError("topic must not be empty")

    weak_concepts = weak_concepts_for_topic(user_id, normalized_topic)
    if weak_concepts:
        focus = weak_concepts[-1]
        focused_retrieval = retrieve_chunks(
            query=f"{normalized_topic} {focus}",
            k=DEFAULT_RETRIEVAL_K,
        )
        focused_chunks = focused_retrieval.get("chunks") or []
        if focused_chunks:
            focused_prompt = ADAPTIVE_QUIZ_PROMPT.format(
                topic=normalized_topic,
                focus=focus,
                context=_context_from_chunks(focused_chunks),
            )
            question, _reason = _question_from_prompt(focused_prompt)
            if question:
                return _ok_quiz(
                    topic=normalized_topic,
                    question=question,
                    source_document_ids=_document_ids(focused_chunks),
                    adaptive_focus=focus,
                )

    retrieval = retrieve_chunks(query=normalized_topic, k=DEFAULT_RETRIEVAL_K)
    chunks = retrieval.get("chunks") or []
    source_document_ids = _document_ids(chunks)

    if not chunks:
        return _insufficient_quiz(
            normalized_topic,
            [],
            "No matching course material was found for that topic. "
            "Upload relevant materials and try again.",
        )

    prompt = QUIZ_PROMPT.format(
        topic=normalized_topic,
        context=_context_from_chunks(chunks),
    )
    question, reason = _question_from_prompt(prompt)
    if not question:
        return _insufficient_quiz(normalized_topic, source_document_ids, reason)

    return _ok_quiz(
        topic=normalized_topic,
        question=question,
        source_document_ids=source_document_ids,
        adaptive_focus=None,
    )


def grade_student_answer(
    *,
    question: str,
    student_answer: str,
    topic: str,
    user_id: str = "demo-user",
) -> dict[str, Any]:
    """Retrieve course chunks and grade a student answer against that material only."""

    normalized_question = question.strip()
    normalized_answer = student_answer.strip()
    normalized_topic = topic.strip()

    if not normalized_topic:
        raise StudyValidationError("topic must not be empty")
    if not normalized_question:
        raise StudyValidationError("question must not be empty")
    if not normalized_answer:
        raise StudyValidationError("student_answer must not be empty")

    retrieval_query = f"{normalized_topic}\n{normalized_question}"
    retrieval = retrieve_chunks(query=retrieval_query, k=DEFAULT_RETRIEVAL_K)
    chunks = retrieval.get("chunks") or []
    source_document_ids = _document_ids(chunks)

    insufficient = {
        "status": "insufficient_material",
        "verdict": None,
        "feedback": None,
        "explanation": None,
        "identified_gap": None,
        "source_document_ids": source_document_ids,
        "question": normalized_question,
        "topic": normalized_topic,
        "learning_progress": None,
        "detail": (
            "Not enough matching course material was found to grade this answer fairly."
        ),
    }

    if not chunks:
        return insufficient

    prompt = GRADE_PROMPT.format(
        topic=normalized_topic,
        question=normalized_question,
        student_answer=normalized_answer,
        context=_context_from_chunks(chunks),
    )
    draft = _call_structured(prompt, GradeDraft)
    assert isinstance(draft, GradeDraft)

    if not draft.sufficient_material or draft.verdict is None:
        insufficient["detail"] = (
            draft.insufficient_reason.strip()
            or insufficient["detail"]
        )
        return insufficient

    identified_gap = draft.identified_gap.strip() if draft.identified_gap else None
    if draft.verdict == "correct":
        identified_gap = None

    progress = record_quiz_result(
        user_id=user_id,
        topic=normalized_topic,
        verdict=draft.verdict,
        identified_gap=identified_gap,
    )

    return {
        "status": "ok",
        "verdict": draft.verdict,
        "feedback": draft.feedback.strip(),
        "explanation": draft.explanation.strip(),
        "identified_gap": identified_gap,
        "source_document_ids": source_document_ids,
        "question": normalized_question,
        "topic": normalized_topic,
        "learning_progress": progress,
        "detail": None,
    }

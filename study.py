"""Grounded quiz generation and grading — reuses retrieve_chunks + OpenAI structured output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import APIError, OpenAI
from pydantic import BaseModel

from datetime import datetime, timezone
from psycopg.types.json import Jsonb

from memory_store import get_or_create_topic
from rag import DEFAULT_RETRIEVAL_K, _format_chunk
from retrieve import retrieve_chunks
from db import get_conn
from adaptive import priority_score, update_mastery


_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

_client = OpenAI()
DEFAULT_STUDY_MODEL = os.getenv("STUDY_MODEL", "gpt-4o-mini")

QUIZ_PROMPT = """You are a study quiz generator. Create ONE question using ONLY the retrieved course material below.

Rules:
- Use ONLY the retrieved context. Never invent facts, vocabulary, numbers, or examples.
- The question must be answerable from the retrieved context.
- Prefer a question that tests understanding (meaning, when to use something, or choosing the right form) rather than asking the student to copy a sentence.
- Generate one question for the student.
- Also generate a reference answer using ONLY the retrieved context.
- Generate 2-4 key points that a correct answer should contain.
- Give the specific concept being tested a short concept_label of 6 words or fewer.
- Do not put the reference answer or key points inside the question itself.
- If the context is missing, off-topic, or too thin to write a fair question about the student's topic, set sufficient_material to false and explain why.

Student topic: {topic}

Retrieved course material:
{context}
"""

GRADE_PROMPT = """You are a grounded study tutor.

Grade the student's answer using ONLY:
1. the stored reference answer,
2. the required key points,
3. the original source material.

Rules:
- Do not use outside knowledge.
- Give a score from 0.0 to 1.0 based on how completely and accurately the student answered.
- verdict must be exactly one of: correct, partially_correct, incorrect.
- Use these score ranges:
  - 0.85 to 1.00: correct
  - 0.35 to below 0.85: partially_correct
  - 0.00 to below 0.35: incorrect
- missed_points must contain the important key points the student missed.
- feedback should be short and student-facing.
- explanation should explain what was expected using the source material.
- identified_gap should describe the specific weakness revealed by the answer.
- If the answer is correct, identified_gap should be null.
- sufficient_material is about the SOURCE MATERIAL only, never the student's answer. Set it to true whenever the reference answer, key points and source material are enough to judge the answer — which is almost always.
- An incomplete, vague or wrong student answer is still gradable: keep sufficient_material true and give it a low score.
- Set sufficient_material to false only if the source material itself is missing or unrelated to the question, and explain why in insufficient_reason.

Question:
{question}

Reference answer:
{reference_answer}

Required key points:
{key_points}

Student answer:
{student_answer}

Original source material:
{context}
"""


class StudyValidationError(ValueError):
    """Raised when quiz/grade input fails validation."""


class StudyModelError(RuntimeError):
    """Raised when the quiz/grade model call fails."""


class QuizDraft(BaseModel):
    sufficient_material: bool
    question: str = ""
    reference_answer: str = ""
    key_points: list[str] = []
    concept_label: str = ""
    insufficient_reason: str = ""


class GradeDraft(BaseModel):
    sufficient_material: bool
    score: float | None = None
    missed_points: list[str] = []
    feedback: str = ""
    explanation: str = ""
    identified_gap: str | None = None
    insufficient_reason: str = ""

def verdict_from_score(score: float) -> str:
    if score >= 0.85:
        return "correct"
    if score >= 0.35:
        return "partially_correct"
    return "incorrect"

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

def save_question(
    *,
    user_id,
    course_id,
    topic_id,
    draft: QuizDraft,
    chunks: list[dict[str, Any]],
):
    with get_conn() as conn:
        with conn.cursor() as cur:

            # 1. Save the actual question
            cur.execute(
                """
                INSERT INTO questions (
                    user_id,
                    course_id,
                    question_type,
                    prompt,
                    reference_answer,
                    key_points,
                    concept_label,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'active')
                RETURNING id
                """,
                (
                    user_id,
                    course_id,
                    "short_answer",
                    draft.question.strip(),
                    draft.reference_answer.strip(),
                    Jsonb(draft.key_points),
                    draft.concept_label.strip(),
                ),
            )

            question_id = cur.fetchone()[0]

            # 2. Connect question → topic
            cur.execute(
                """
                INSERT INTO question_topics (
                    user_id,
                    question_id,
                    topic_id,
                    is_primary
                )
                VALUES (%s, %s, %s, true)
                """,
                (
                    user_id,
                    question_id,
                    topic_id,
                ),
            )

            # 3. Connect question → source chunks
            for rank, chunk in enumerate(chunks):
                chunk_id = chunk.get("chunk_id")

                if not chunk_id:
                    continue

                cur.execute(
                    """
                    INSERT INTO question_sources (
                        user_id,
                        question_id,
                        chunk_id,
                        rank
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        question_id,
                        chunk_id,
                        rank,
                    ),
                )

            return question_id

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


def _question_from_prompt(prompt: str) -> tuple[QuizDraft | None, str]:
    draft = _call_structured(prompt, QuizDraft)
    assert isinstance(draft, QuizDraft)

    if not draft.sufficient_material or not draft.question.strip():
        return None, (
            draft.insufficient_reason.strip()
            or "The retrieved material is not enough to write a fair question on that topic."
        )

    return draft, ""


def generate_quiz_question(
    *,
    user_id,
    course_id,
    topic: str,
) -> dict[str, Any]:

    topic = topic.strip()

    if not topic:
        raise StudyValidationError("topic must not be empty")

    # Turn the user's topic string into its permanent topic row.
    topic_id = get_or_create_topic(
        user_id,
        course_id,
        topic,
    )

    # Decide whether to reuse something from the bank
    # or generate a new question.
    selection = select_next_question(
        user_id=user_id,
        topic_id=topic_id,
    )

    # ─────────────────────────────────────────
    # EXISTING QUESTION WINS
    # ─────────────────────────────────────────
    if selection["mode"] == "existing":
        question = selection["question"]

        return {
            "status": "ok",
            "question_id": question["question_id"],
            "question": question["question"],
            "topic": topic,
            "concept_label": question["concept_label"],
            "mode": "existing",
            "topic_mastery": selection["topic_mastery"],
            "detail": None,
        }

    # ─────────────────────────────────────────
    # NEW QUESTION WINS
    # ─────────────────────────────────────────

    retrieval = retrieve_chunks(
        query=topic,
        k=DEFAULT_RETRIEVAL_K,
        user_id=user_id,
        course_id=course_id,
    )

    chunks = retrieval.get("chunks") or []

    if not chunks:
        return _insufficient_quiz(
            topic,
            [],
            "No matching course material was found for that topic. "
            "Upload relevant materials and try again.",
        )

    prompt = QUIZ_PROMPT.format(
        topic=topic,
        context=_context_from_chunks(chunks),
    )

    draft, reason = _question_from_prompt(prompt)

    if draft is None:
        return _insufficient_quiz(
            topic,
            _document_ids(chunks),
            reason,
        )

    question_id = save_question(
        user_id=user_id,
        course_id=course_id,
        topic_id=topic_id,
        draft=draft,
        chunks=chunks,
    )

    return {
        "status": "ok",
        "question_id": question_id,
        "question": draft.question.strip(),
        "topic": topic,
        "concept_label": draft.concept_label.strip(),
        "mode": "generated",
        "topic_mastery": selection["topic_mastery"],
        "source_document_ids": _document_ids(chunks),
        "detail": None,
    }

def load_question_for_grading(*, user_id, question_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    q.id,
                    q.prompt,
                    q.reference_answer,
                    q.key_points,
                    q.concept_label,
                    t.name
                FROM questions q
                JOIN question_topics qt
                    ON qt.question_id = q.id
                   AND qt.is_primary = true
                JOIN topics t
                    ON t.id = qt.topic_id
                WHERE q.id = %s
                  AND q.user_id = %s
                  AND q.status = 'active'
                """,
                (question_id, user_id),
            )

            row = cur.fetchone()

            if row is None:
                raise StudyValidationError("Question not found")

            cur.execute(
                """
                SELECT
                    c.id,
                    c.document_id,
                    c.chunk_index,
                    c.text
                FROM question_sources qs
                JOIN chunks c
                    ON c.id = qs.chunk_id
                WHERE qs.question_id = %s
                  AND qs.user_id = %s
                ORDER BY qs.rank
                """,
                (question_id, user_id),
            )

            source_rows = cur.fetchall()

    return {
        "id": row[0],
        "question": row[1],
        "reference_answer": row[2],
        "key_points": row[3] or [],
        "concept_label": row[4],
        "topic": row[5],
        "chunks": [
            {
                "chunk_id": source[0],
                "document_id": source[1],
                "chunk_index": source[2],
                "text": source[3],
            }
            for source in source_rows
        ],
    }

def record_attempt(
    *,
    user_id,
    question_id,
    answer_text,
    verdict,
    score,
    missed_points,
    feedback,
    explanation,
    identified_gap,
):
    with get_conn() as conn:
        with conn.cursor() as cur:

            # Get the question's current learner state.
            cur.execute(
                """
                SELECT mastery, attempt_count, correct_streak
                FROM question_state
                WHERE user_id = %s
                  AND question_id = %s
                """,
                (user_id, question_id),
            )

            state = cur.fetchone()

            if state is None:
                old_mastery = 0.5
                attempt_count = 0
                correct_streak = 0
            else:
                old_mastery = state[0]
                attempt_count = state[1]
                correct_streak = state[2]

            new_mastery = update_mastery(
                old_mastery,
                score,
            )

            new_attempt_count = attempt_count + 1

            if verdict == "correct":
                new_correct_streak = correct_streak + 1
            else:
                new_correct_streak = 0

            # Preserve the raw historical evidence.
            cur.execute(
                """
                INSERT INTO attempts (
                    user_id,
                    question_id,
                    answer_text,
                    verdict,
                    score,
                    missed_points,
                    feedback,
                    explanation,
                    identified_gap,
                    grader_model,
                    grader_version
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                RETURNING id, created_at
                """,
                (
                    user_id,
                    question_id,
                    answer_text,
                    verdict,
                    score,
                    Jsonb(missed_points),
                    feedback,
                    explanation,
                    identified_gap,
                    DEFAULT_STUDY_MODEL,
                    "v1",
                ),
            )

            attempt_id, attempted_at = cur.fetchone()

            # Update our CURRENT estimate of the student's state.
            cur.execute(
                """
                INSERT INTO question_state (
                    user_id,
                    question_id,
                    mastery,
                    attempt_count,
                    correct_streak,
                    last_score,
                    last_attempt_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())

                ON CONFLICT (user_id, question_id)
                DO UPDATE SET
                    mastery = EXCLUDED.mastery,
                    attempt_count = EXCLUDED.attempt_count,
                    correct_streak = EXCLUDED.correct_streak,
                    last_score = EXCLUDED.last_score,
                    last_attempt_at = EXCLUDED.last_attempt_at,
                    updated_at = now()
                """,
                (
                    user_id,
                    question_id,
                    new_mastery,
                    new_attempt_count,
                    new_correct_streak,
                    score,
                    attempted_at,
                ),
            )

    return {
        "attempt_id": attempt_id,
        "mastery": new_mastery,
        "attempt_count": new_attempt_count,
        "correct_streak": new_correct_streak,
    }

def grade_student_answer(
    *,
    user_id,
    question_id,
    student_answer: str,
) -> dict[str, Any]:

    student_answer = student_answer.strip()

    if not student_answer:
        raise StudyValidationError("student_answer must not be empty")

    question = load_question_for_grading(
        user_id=user_id,
        question_id=question_id,
    )

    chunks = question["chunks"]

    if not chunks:
        raise StudyValidationError(
            "The source material for this question could not be found"
        )

    prompt = GRADE_PROMPT.format(
        question=question["question"],
        reference_answer=question["reference_answer"],
        key_points=question["key_points"],
        student_answer=student_answer,
        context=_context_from_chunks(chunks),
    )

    draft = _call_structured(prompt, GradeDraft)
    assert isinstance(draft, GradeDraft)

    if not draft.sufficient_material or draft.score is None:
        return {
            "status": "insufficient_material",
            "question_id": question_id,
            "verdict": None,
            "score": None,
            "missed_points": [],
            "feedback": None,
            "explanation": None,
            "identified_gap": None,
            "detail": (
                draft.insufficient_reason.strip()
                or "The stored material was not sufficient to grade this answer."
            ),
        }

    # Never allow the model to give us something outside 0–1.
    score = max(0.0, min(1.0, float(draft.score)))

    # OUR deterministic rule, not the LLM's.
    verdict = verdict_from_score(score)

    identified_gap = (
        draft.identified_gap.strip()
        if draft.identified_gap
        else None
    )

    if verdict == "correct":
        identified_gap = None
    
    attempt = record_attempt(
        user_id=user_id,
        question_id=question_id,
        answer_text=student_answer,
        verdict=verdict,
        score=score,
        missed_points=draft.missed_points,
        feedback=draft.feedback.strip(),
        explanation=draft.explanation.strip(),
        identified_gap=identified_gap,
    )

    return {
        "status": "ok",
        "question_id": question_id,
        "attempt_id": attempt["attempt_id"],
        "verdict": verdict,
        "score": score,
        "mastery": attempt["mastery"],
        "attempt_count": attempt["attempt_count"],
        "correct_streak": attempt["correct_streak"],
        "missed_points": draft.missed_points,
        "feedback": draft.feedback.strip(),
        "explanation": draft.explanation.strip(),
        "identified_gap": identified_gap,
        "reference_answer": question["reference_answer"],
        "concept_label": question["concept_label"],
        "topic": question["topic"],
        "detail": None,
    }

def get_question_candidates(*, user_id, topic_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    q.id,
                    q.prompt,
                    q.concept_label,
                    qs.mastery,
                    qs.last_attempt_at
                FROM questions q
                JOIN question_topics qt
                    ON qt.question_id = q.id
                LEFT JOIN question_state qs
                    ON qs.question_id = q.id
                   AND qs.user_id = q.user_id
                WHERE q.user_id = %s
                  AND qt.topic_id = %s
                  AND q.status = 'active'
                ORDER BY q.created_at
                """,
                (user_id, topic_id),
            )

            rows = cur.fetchall()

    return [
        {
            "question_id": row[0],
            "question": row[1],
            "concept_label": row[2],
            "mastery": row[3],
            "last_attempt_at": row[4],
        }
        for row in rows
    ]


def get_topic_mastery(*, user_id, topic_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT AVG(qs.mastery)
                FROM question_state qs
                JOIN question_topics qt
                    ON qt.question_id = qs.question_id
                WHERE qs.user_id = %s
                  AND qt.topic_id = %s
                """,
                (user_id, topic_id),
            )

            average = cur.fetchone()[0]

    if average is None:
        return 0.5

    return float(average)

def select_next_question(*, user_id, topic_id):
    candidates = get_question_candidates(
        user_id=user_id,
        topic_id=topic_id,
    )

    topic_mastery = get_topic_mastery(
        user_id=user_id,
        topic_id=topic_id,
    )

    now = datetime.now(timezone.utc)

    new_question_priority = 0.5 * topic_mastery

    best_question = None
    best_priority = None

    for candidate in candidates:
        priority = priority_score(
            question_mastery=candidate["mastery"],
            topic_mastery=topic_mastery,
            last_attempt_at=candidate["last_attempt_at"],
            now=now,
        )

        if best_priority is None or priority < best_priority:
            best_question = candidate
            best_priority = priority

    # No questions exist yet → obviously generate one.
    if best_question is None:
        return {
            "mode": "generate",
            "topic_mastery": topic_mastery,
        }

    # Existing unanswered question wins ties against generating.
    if (
        best_question["mastery"] is None
        and best_priority <= new_question_priority
    ):
        return {
            "mode": "existing",
            "question": best_question,
            "topic_mastery": topic_mastery,
        }

    # Generate when NEW has strictly better priority.
    if new_question_priority < best_priority:
        return {
            "mode": "generate",
            "topic_mastery": topic_mastery,
        }

    return {
        "mode": "existing",
        "question": best_question,
        "topic_mastery": topic_mastery,
    }
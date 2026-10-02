import os

import time
from pathlib import Path
from typing import NoReturn

from datetime import datetime
from uuid import UUID

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ingest import IngestValidationError, ingest_document
from rag import DEFAULT_RETRIEVAL_K, build_grounding_prompt
from retrieve import RetrieveValidationError, retrieve_chunks
from vectorstore import VectorStoreConfigError, check_pinecone
from users import ensure_user
from courses import create_course, list_courses
from documents import delete_document, list_documents
from study import (
    StudyValidationError,
    generate_quiz_question,
    grade_student_answer,
)

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)
app = FastAPI()
client = OpenAI()  
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-5-mini": (0.00025, 0.002),
}

if CHAT_MODEL not in MODEL_PRICES_PER_1K:
    raise ValueError(f"No pricing configured for CHAT_MODEL={CHAT_MODEL}")

def current_user(
    x_google_sub: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
    x_user_name: str | None = Header(default=None),
):
    if x_google_sub:
        return ensure_user(
            "google",
            x_google_sub,
            email=x_user_email,
            name=x_user_name,
        )

    if os.getenv("ALLOW_DEV_USER", "false").lower() == "true":
        return ensure_user("dev", "dev-local")

    raise HTTPException(status_code=401, detail="Authentication required")

class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    course_id: UUID
    question: str

class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str]


class IngestRequest(BaseModel):
    course_id: UUID
    filename: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    document_id: UUID
    chunks_indexed: int
    status: str


class QuizRequest(BaseModel):
    course_id: UUID
    topic: str = Field(min_length=1)


class QuizResponse(BaseModel):
    status: str
    question: str | None = None
    topic: str
    source_document_ids: list[str]
    adaptive_focus: str | None = None
    detail: str | None = None


class GradeRequest(BaseModel):
    question_id: UUID
    answer: str = Field(min_length=1)


class LearningProgress(BaseModel):
    topic: str
    mastery: str
    attempts: int
    correct: int
    partial: int
    incorrect: int
    weak_concepts: list[str]


class GradeResponse(BaseModel):
    status: str
    verdict: str | None = None
    feedback: str | None = None
    explanation: str | None = None
    identified_gap: str | None = None
    source_document_ids: list[str]
    question: str | None = None
    topic: str | None = None
    learning_progress: LearningProgress | None = None
    detail: str | None = None

class CreateCourseRequest(BaseModel):
    name: str = Field(min_length=1)

class CourseResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime

class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    char_count: int
    chunk_count: int
    created_at: datetime

def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[CHAT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(prompt: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/ingest")
def ingest(
    body: IngestRequest,
    user_id=Depends(current_user),
    ) -> IngestResponse:

    try:
        result = ingest_document(
            user_id=user_id,
            course_id=body.course_id,
            filename=body.filename,
            text=body.text,
        )
    except IngestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestResponse(**result)


def _raise_study_http(exc: Exception) -> NoReturn:
    """Map study/retrieval failures to the same HTTP codes as /ask and /ingest."""

    if isinstance(exc, StudyValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, VectorStoreConfigError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, RetrieveValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, StudyModelError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise exc


@app.post("/quiz")
def quiz(
    body: QuizRequest,
    user_id=Depends(current_user),
):
    try:
        return generate_quiz_question(
            user_id=user_id,
            course_id=body.course_id,
            topic=body.topic,
        )
    except StudyValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post("/grade")
def grade(
    body: GradeRequest,
    user_id=Depends(current_user),
    ):

    try:
        return grade_student_answer(
            user_id=user_id,
            question_id=body.question_id,
            student_answer=body.answer,
        )
    except StudyValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post("/ask")
def ask(
    body: AskRequest,
    user_id=Depends(current_user),
    ) -> AskResponse:
    """Answer one question with retrieval-augmented generation, guardrails, and cost visibility."""

    model = CHAT_MODEL

    try:
        retrieval = retrieve_chunks(
            query=body.question,
            k=DEFAULT_RETRIEVAL_K,
            user_id=user_id,
            course_id=body.course_id,
        )
    except VectorStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    grounding_prompt, retrieved_chunk_ids = build_grounding_prompt(
        question=body.question,
        chunks=retrieval["chunks"],
    )

    try:
        start = time.perf_counter()
            
        answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounding_prompt, model
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

        return AskResponse(
            answer=answer,
            tokens_used=tokens_used,
            model=model,
            latency_ms=latency_ms,
            cost_usd=round(cost_usd, 6),
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
        
@app.post("/courses", response_model=CourseResponse)
def create_course_endpoint(
    body: CreateCourseRequest,
    user_id=Depends(current_user),
    ):
    

    try:
        course_id = create_course(user_id, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    courses = list_courses(user_id)

    for course in courses:
        if course["id"] == course_id:
            return CourseResponse(**course)

    raise HTTPException(status_code=500, detail="Course was created but could not be loaded")


@app.get("/courses", response_model=list[CourseResponse])
def list_courses_endpoint(
    user_id=Depends(current_user)
    ):
    return [CourseResponse(**course) for course in list_courses(user_id)]

@app.get("/documents", response_model=list[DocumentResponse])
def documents(
    course_id: UUID,
    user_id=Depends(current_user),
    ):

    return [
        DocumentResponse(**document)
        for document in list_documents(user_id, course_id)
    ]


@app.delete("/documents/{document_id}")
def remove_document(
    document_id: UUID,
    user_id=Depends(current_user),
    ):

    if not delete_document(user_id, document_id):
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "deleted"}
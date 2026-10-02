import logging
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from uuid import UUID

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from openai import APIError, OpenAI
from pydantic import BaseModel, Field, ValidationError

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

from courses import archive_course, create_course, list_courses, user_owns_course  # noqa: E402
from documents import delete_document, list_documents  # noqa: E402
from ingest import IngestValidationError, ingest_document  # noqa: E402
from questions import list_questions  # noqa: E402
from rag import DEFAULT_RETRIEVAL_K, build_grounding_prompt  # noqa: E402
from retrieve import RetrieveValidationError, retrieve_chunks  # noqa: E402
from study import (  # noqa: E402
    StudyModelError,
    StudyValidationError,
    generate_quiz_question,
    grade_student_answer,
)
from users import ensure_user  # noqa: E402
from vectorstore import VectorStoreConfigError  # noqa: E402

logger = logging.getLogger("adaptive_study_agent")

# ── Deployment safety ────────────────────────────────────────────────
# The API trusts the Google identity headers sent by the Streamlit app.
# In production that is only safe when every request must also carry the
# shared secret that only the Streamlit app knows.
ALLOW_DEV_USER = os.getenv("ALLOW_DEV_USER", "false").strip().lower() == "true"
API_SHARED_SECRET = os.getenv("API_SHARED_SECRET", "").strip()
IS_PRODUCTION = (
    os.getenv("RENDER", "").strip().lower() == "true"
    or os.getenv("APP_ENV", "").strip().lower() == "production"
)

if IS_PRODUCTION and ALLOW_DEV_USER:
    raise RuntimeError("ALLOW_DEV_USER must be false in production.")
if IS_PRODUCTION and not API_SHARED_SECRET:
    raise RuntimeError("API_SHARED_SECRET must be set in production.")
if not API_SHARED_SECRET:
    logger.warning(
        "API_SHARED_SECRET is not set: identity headers are trusted without a key. "
        "Local development only."
    )

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
    x_api_key: str | None = Header(default=None),
    x_google_sub: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
    x_user_name: str | None = Header(default=None),
):
    if API_SHARED_SECRET:
        if not x_api_key or not secrets.compare_digest(x_api_key, API_SHARED_SECRET):
            raise HTTPException(status_code=401, detail="Authentication required")

    if x_google_sub:
        return ensure_user(
            "google",
            x_google_sub,
            email=unquote(x_user_email) if x_user_email else None,
            name=unquote(x_user_name) if x_user_name else None,
        )

    if ALLOW_DEV_USER:
        return ensure_user("dev", "dev-local")

    raise HTTPException(status_code=401, detail="Authentication required")


def require_course(user_id, course_id) -> None:
    if not user_owns_course(user_id, course_id):
        raise HTTPException(status_code=404, detail="Course not found")


def service_error(exc: Exception, message: str = "Something went wrong. Please try again.") -> HTTPException:
    """Log the real error server-side; return a generic message to the client."""

    logger.exception("Request failed: %s", exc)
    return HTTPException(status_code=503, detail=message)


class Answer(BaseModel):
    """Structured model output for /ask."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    course_id: UUID
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
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


class GradeRequest(BaseModel):
    question_id: UUID
    answer: str = Field(min_length=1)


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


class QuestionSummary(BaseModel):
    id: UUID
    prompt: str
    topic: str | None = None
    concept_label: str | None = None
    created_at: datetime
    mastery: float | None = None
    attempt_count: int = 0
    last_score: float | None = None
    last_attempt_at: datetime | None = None


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[CHAT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(prompt: str, model: str) -> tuple[Answer, int, int, int]:
    """OpenAI structured output forces exactly the Answer schema."""

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


@app.post("/courses", response_model=CourseResponse)
def create_course_endpoint(body: CreateCourseRequest, user_id=Depends(current_user)):
    try:
        course_id = create_course(user_id, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for course in list_courses(user_id):
        if course["id"] == course_id:
            return CourseResponse(**course)

    raise HTTPException(status_code=500, detail="Course was created but could not be loaded")


@app.get("/courses", response_model=list[CourseResponse])
def list_courses_endpoint(user_id=Depends(current_user)):
    return [CourseResponse(**course) for course in list_courses(user_id)]


@app.delete("/courses/{course_id}")
def delete_course_endpoint(course_id: UUID, user_id=Depends(current_user)):
    """Archive (soft-delete) a course. Documents, questions and attempts are kept."""

    if not archive_course(user_id, course_id):
        raise HTTPException(status_code=404, detail="Course not found")

    return {"status": "deleted"}


@app.post("/ingest")
def ingest(body: IngestRequest, user_id=Depends(current_user)) -> IngestResponse:
    try:
        result = ingest_document(
            user_id=user_id,
            course_id=body.course_id,
            filename=body.filename,
            text=body.text,
        )
    except IngestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (VectorStoreConfigError, RuntimeError) as exc:
        raise service_error(exc, "Could not add this file right now. Please try again.") from exc

    return IngestResponse(**result)


@app.get("/documents", response_model=list[DocumentResponse])
def documents(course_id: UUID, user_id=Depends(current_user)):
    require_course(user_id, course_id)
    return [DocumentResponse(**document) for document in list_documents(user_id, course_id)]


@app.delete("/documents/{document_id}")
def remove_document(document_id: UUID, user_id=Depends(current_user)):
    try:
        deleted = delete_document(user_id, document_id)
    except (VectorStoreConfigError, RuntimeError) as exc:
        raise service_error(exc, "Could not delete this file right now. Please try again.") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "deleted"}


@app.get("/questions", response_model=list[QuestionSummary])
def questions(course_id: UUID, user_id=Depends(current_user)):
    require_course(user_id, course_id)
    return [QuestionSummary(**question) for question in list_questions(user_id, course_id)]


@app.post("/quiz")
def quiz(body: QuizRequest, user_id=Depends(current_user)):
    require_course(user_id, body.course_id)

    try:
        return generate_quiz_question(
            user_id=user_id,
            course_id=body.course_id,
            topic=body.topic,
        )
    except StudyValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StudyModelError as exc:
        raise service_error(exc, "The AI service had a problem writing a question. Please try again.") from exc
    except (VectorStoreConfigError, RuntimeError) as exc:
        raise service_error(exc) from exc


@app.post("/grade")
def grade(body: GradeRequest, user_id=Depends(current_user)):
    try:
        return grade_student_answer(
            user_id=user_id,
            question_id=body.question_id,
            student_answer=body.answer,
        )
    except StudyValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except StudyModelError as exc:
        raise service_error(exc, "The AI service had a problem grading your answer. Please try again.") from exc
    except RuntimeError as exc:
        raise service_error(exc) from exc


@app.post("/ask")
def ask(body: AskRequest, user_id=Depends(current_user)) -> AskResponse:
    """Answer one question from the user's own notes in the selected course."""

    require_course(user_id, body.course_id)
    model = CHAT_MODEL

    try:
        retrieval = retrieve_chunks(
            query=body.question,
            k=DEFAULT_RETRIEVAL_K,
            user_id=user_id,
            course_id=body.course_id,
        )
    except RetrieveValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (VectorStoreConfigError, RuntimeError) as exc:
        raise service_error(exc) from exc

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
    except (ValidationError, ValueError, APIError) as exc:
        raise service_error(exc, "The AI service had a problem answering. Please try again.") from exc

    return AskResponse(
        answer=answer,
        tokens_used=tokens_used,
        model=model,
        latency_ms=latency_ms,
        cost_usd=round(cost_usd, 6),
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

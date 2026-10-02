import os

import time
from pathlib import Path
from typing import NoReturn

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ingest import IngestValidationError, ingest_text
from rag import DEFAULT_RETRIEVAL_K, build_grounding_prompt
from retrieve import RetrieveValidationError, retrieve_chunks
from vectorstore import VectorStoreConfigError, check_pinecone

from study import (
    StudyModelError,
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


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

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
    document_id: str = Field(min_length=1)
    text: str
    source: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


class QuizRequest(BaseModel):
    topic: str
    user_id: str = "demo-user"


class QuizResponse(BaseModel):
    status: str
    question: str | None = None
    topic: str
    source_document_ids: list[str]
    adaptive_focus: str | None = None
    detail: str | None = None


class GradeRequest(BaseModel):
    question: str
    student_answer: str
    topic: str
    user_id: str = "demo-user"


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


@app.get("/debug/pinecone")
def debug_pinecone() -> dict:
    """Confirm Pinecone credentials, index presence, and basic connectivity."""

    return check_pinecone()


@app.get("/debug/retrieve")
def debug_retrieve(q: str, k: int = 5) -> dict:
    """Embed a question and return top-k chunks with scores — no LLM call.

    Example:
        curl -s "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20RAG%3F"
    """

    try:
        return retrieve_chunks(query=q, k=k)
    except RetrieveValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk, embed, and upsert a document into Pinecone.

    Example:
        curl -s -X POST http://127.0.0.1:8000/ingest \\
          -H "Content-Type: application/json" \\
          -d '{
            "document_id": "rag-intro-001",
            "text": "Retrieval Augmented Generation combines search with generation...",
            "source": "sample_rag_document.txt"
          }'
    """

    try:
        result = ingest_text(
            document_id=body.document_id,
            text=body.text,
            source=body.source,
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
def quiz(body: QuizRequest) -> QuizResponse:
    """Generate one grounded quiz question from ingested course materials."""

    try:
        result = generate_quiz_question(topic=body.topic, user_id=body.user_id)
    except (StudyValidationError, VectorStoreConfigError, RetrieveValidationError, StudyModelError, RuntimeError) as exc:
        _raise_study_http(exc)

    return QuizResponse(**result)


@app.post("/grade")
def grade(body: GradeRequest) -> GradeResponse:
    """Grade a student answer against retrieved course material and record compact mastery."""

    try:
        result = grade_student_answer(
            question=body.question,
            student_answer=body.student_answer,
            topic=body.topic,
            user_id=body.user_id,
        )
    except (StudyValidationError, VectorStoreConfigError, RetrieveValidationError, StudyModelError, RuntimeError) as exc:
        _raise_study_http(exc)

    return GradeResponse(**result)


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question with retrieval-augmented generation, guardrails, and cost visibility."""

    model = CHAT_MODEL

    try:
        retrieval = retrieve_chunks(query=body.question, k=DEFAULT_RETRIEVAL_K)
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
        

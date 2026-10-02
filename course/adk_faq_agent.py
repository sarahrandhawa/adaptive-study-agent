"""
Session 3: Northwind FAQ agent (Google ADK)

One agent, one multi-step job: search company docs, then answer with citations.
The one real tool is search_docs — it calls Session 2 retrieve_chunks (Pinecone).

Patterns copied from:
  ai-engineering-bootcamp/adk-multi-agent-systems/demo1_routing.py
  (Agent + tools=[], Runner, InMemorySessionService, event stream)
  plus RunConfig.max_llm_calls from ADK runtime config.

Run locally (same style as the sample demos):
    cd ai-engineering-bootcamp-v2/week-1
    python adk_faq_agent.py

Needs GOOGLE_API_KEY in this folder's .env (https://aistudio.google.com/apikey).
`adk web` is optional later (ADK package folder); this Python entrypoint is the assignment path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from retrieve import RetrieveValidationError, retrieve_chunks
from vectorstore import VectorStoreConfigError

# ADK logs a full traceback on Gemini 503s; keep the classroom output readable.
logging.getLogger("google.adk").setLevel(logging.CRITICAL)
logging.getLogger("google.genai").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*JSON_SCHEMA_FOR_FUNC_DECL.*")

# Load .env from this folder so keys work regardless of the shell's cwd.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# gemini-2.5-flash is retired for new keys. gemini-3.6-flash is often 503 (high demand).
# Prefer a lighter Flash model; override with GEMINI_MODEL= in .env.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = [
    MODEL,
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.6-flash",
]
APP_NAME = "northwind_faq"
MAX_LLM_CALLS = 6  # hard stop: search + answer should take 2–3 model calls
AGENT_INSTRUCTION = (
    "You are northwind_faq_agent, a grounded FAQ bot for this company's docs.\n\n"
    "Goal: Answer the user's question using only retrieved document chunks, "
    "and cite those sources.\n\n"
    "Constraints:\n"
    "- Always call search_docs before you answer.\n"
    "- Never invent policies, numbers, or citations.\n"
    "- If search_docs returns status=ok, answer from the chunk text and cite "
    "document_id (for example [POL-101]).\n"
    "- If search_docs returns status=empty, say the answer is not in the docs.\n"
    "- If search_docs returns status=error, tell the user the tool error. Do not guess.\n"
    "- Call search_docs at most twice per question.\n\n"
    "Done looks like: a short answer plus citations (document_id or source), "
    "OR a clear 'not in the docs' / tool-error message after searching."
)

# --- Real tool: Session 2 retrieval ---


def search_docs(query: str) -> dict:
    """Search ingested Northwind company docs (Pinecone) and return the top matching chunks.

    Use this whenever the user asks about company policy, handbook rules, expenses,
    security, product, IT, or facilities. Returns real retrieved text plus
    document_id / source so you can cite. On failure, returns status=error
    instead of raising.
    """

    try:
        result = retrieve_chunks(query=query, k=5)
    except RetrieveValidationError as exc:
        return {"status": "error", "query": query, "error": str(exc)}
    except VectorStoreConfigError as exc:
        return {"status": "error", "query": query, "error": str(exc)}
    except Exception as exc:  # surface Pinecone/embed failures as observations
        return {
            "status": "error",
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
        }

    chunks = []
    for chunk in result.get("chunks") or []:
        chunks.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "source": chunk.get("source"),
                "score": chunk.get("score"),
                "text": chunk.get("text"),
            }
        )

    return {
        "status": "ok" if chunks else "empty",
        "query": result.get("query", query),
        "k": result.get("k", 5),
        "chunks": chunks,
    }


search_docs_tool = FunctionTool(search_docs)


def build_agent(model: str) -> Agent:
    return Agent(
        name="northwind_faq_agent",
        model=model,
        description="Answers internal FAQ questions from company docs, with citations.",
        instruction=AGENT_INSTRUCTION,
        tools=[search_docs_tool],
    )


# Exported for ADK tooling (`adk web` later). The CLI may swap models on 503.
root_agent = build_agent(MODEL)


# --- Think / Act / Observe logging ---


def _format_observe(result) -> str:
    """Compact, real observation for the terminal (model still gets the full tool payload)."""

    if not isinstance(result, dict):
        return str(result)[:800]

    if result.get("status") == "error":
        return f"status=error error={result.get('error')!r}"

    chunks = result.get("chunks") or []
    lines = [
        f"status={result.get('status')} query={result.get('query')!r} "
        f"chunk_count={len(chunks)}"
    ]
    for chunk in chunks[:3]:
        text = " ".join((chunk.get("text") or "").split())[:180]
        lines.append(
            f"  - {chunk.get('chunk_id')} source={chunk.get('source')} "
            f"score={chunk.get('score')} text={text!r}"
        )
    if len(chunks) > 3:
        lines.append(f"  - … {len(chunks) - 3} more chunks")
    return "\n" + "\n".join(lines)


def _print_event(event) -> str | None:
    """Map ADK's event stream to Think / Act / Observe (or final Answer)."""

    final_text = None
    author = getattr(event, "author", "northwind_faq_agent")
    is_final = bool(event.is_final_response())
    parts = event.content.parts if event.content and event.content.parts else []

    for part in parts:
        function_call = getattr(part, "function_call", None)
        function_response = getattr(part, "function_response", None)
        text = getattr(part, "text", None)

        if function_call:
            args = dict(function_call.args) if function_call.args else {}
            # Gemini often skips a text "think" turn; the function_call *is* the plan.
            print(f"  THINK    [{author}] propose tool call {function_call.name}")
            print(f"  ACT      [{author}] {function_call.name}({args})")
        elif function_response:
            result = function_response.response
            print(f"  OBSERVE  [{author}] {function_response.name} -> {_format_observe(result)}")
        elif text and text.strip():
            label = "ANSWER" if is_final else "THINK"
            print(f"  {label:<8} [{author}] {text.strip()}")
            if is_final:
                final_text = text

    return final_text


def _unique_models() -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    for name in FALLBACK_MODELS:
        if name and name not in seen:
            seen.add(name)
            models.append(name)
    return models


def _is_overload(exc: BaseException) -> bool:
    text = str(exc)
    return "503" in text or "high demand" in text.lower() or "UNAVAILABLE" in text


def _is_missing_model(exc: BaseException) -> bool:
    text = str(exc)
    return "no longer available" in text or "NOT_FOUND" in text or "404" in text


# --- Runner ---


class AgentConfigError(RuntimeError):
    """Raised when the agent cannot start (missing key, empty message)."""


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*\S+"
)
_KEY_PREFIX_RE = re.compile(r"\b(?:sk-|AIza)[A-Za-z0-9_\-]{8,}")
_OBSERVE_LIMIT = 240


def _redact(text: str) -> str:
    """Strip key-shaped values so HTTP logs never leak secrets."""

    cleaned = _SECRET_ASSIGNMENT_RE.sub(r"\1=***", text)
    return _KEY_PREFIX_RE.sub("***", cleaned)


def _truncate(text: str, limit: int = _OBSERVE_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _safe_observation(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("status") == "error":
            return _truncate(f"status=error error={_redact(str(result.get('error', '')))}")
        chunks = result.get("chunks") or []
        ids = [str(chunk.get("chunk_id") or "") for chunk in chunks if chunk.get("chunk_id")]
        snippet = ""
        if chunks:
            snippet = " ".join(str(chunks[0].get("text") or "").split())
        summary = (
            f"status={result.get('status')} chunk_count={len(chunks)} "
            f"ids={','.join(ids[:5])}"
        )
        if snippet:
            summary = f"{summary} text={snippet}"
        return _truncate(_redact(summary))
    return _truncate(_redact(str(result)))


def _record_event_steps(event) -> tuple[str | None, list[dict[str, str]]]:
    """Collect secret-free tool steps from one ADK event. Also prints the CLI loop."""

    final_text = _print_event(event)
    steps: list[dict[str, str]] = []
    parts = event.content.parts if event.content and event.content.parts else []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        function_response = getattr(part, "function_response", None)
        if function_call:
            args = dict(function_call.args) if function_call.args else {}
            query = args.get("query", args)
            steps.append(
                {
                    "tool": function_call.name,
                    "observation": _truncate(f"called query={_redact(str(query))}"),
                }
            )
        elif function_response:
            steps.append(
                {
                    "tool": function_response.name,
                    "observation": _safe_observation(function_response.response),
                }
            )
    return final_text, steps


async def _run_once(agent: Agent, message: str) -> tuple[str, list[dict[str, str]]]:
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])
    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
    final = "(no response)"
    steps: list[dict[str, str]] = []
    step = 0

    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=content,
        run_config=run_config,
    ):
        step += 1
        print(f"  --- step {step} ---")
        maybe_final, event_steps = _record_event_steps(event)
        steps.extend(event_steps)
        if maybe_final:
            final = maybe_final
    return final, steps


async def run_agent(message: str) -> dict[str, Any]:
    """Run the FAQ agent and return answer + truncated tool steps (no secrets)."""

    goal = message.strip()
    if not goal:
        raise AgentConfigError("message must not be empty")
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        raise AgentConfigError("GOOGLE_API_KEY is not set")

    models = _unique_models()
    last_error: BaseException | None = None

    for index, model in enumerate(models):
        print(f"  (model={model}, max_llm_calls={MAX_LLM_CALLS})")
        try:
            answer, steps = await _run_once(build_agent(model), goal)
            return {"answer": answer, "steps": steps, "model": model}
        except (ClientError, ServerError, APIError) as exc:
            last_error = exc
            print(f"  ERROR    Gemini {model}: {exc}")
            has_next = index < len(models) - 1
            if (_is_overload(exc) or _is_missing_model(exc)) and has_next:
                next_model = models[index + 1]
                print(f"  hint: that model is busy or retired. Trying {next_model}...")
                continue
            break

    raise RuntimeError(_redact(str(last_error) if last_error else "Gemini API error"))


async def ask(message: str) -> str:
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        sys.exit(
            "Set GOOGLE_API_KEY in .env "
            "(https://aistudio.google.com/apikey)"
        )

    try:
        result = await run_agent(message)
    except RuntimeError as exc:
        print(
            "  hint: Google's free Gemini quota is busy. Wait a few minutes, or set "
            "GEMINI_MODEL=gemini-3.5-flash-lite in .env, then rerun."
        )
        return f"(Gemini API error: {exc})"
    return str(result["answer"])


PROOF_HEADER = """
Session 3 — prove the loop (no API keys)
Stack: ADK
This is an agent because the model chooses to call search_docs, observes live Pinecone chunks, then writes an answer from that result — the next step is not a fixed script.
""".strip()

# Company-specific fact in ingested docs; not answerable from Gemini's weights alone.
PROOF_QUERY = (
    "How many annual leave days do Northwind Robotics employees get, "
    "and which policy document ID says so?"
)


async def main():
    print(PROOF_HEADER)
    print(f"\n=== USER ===\n{PROOF_QUERY}\n")
    print("=== AGENT LOOP (Think / Act / Observe) ===")
    answer = await ask(PROOF_QUERY)
    print(f"\n=== DONE ===\n{answer}\n")


if __name__ == "__main__":
    asyncio.run(main())

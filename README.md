# Week 1 — `/ask` Demo (5 stages)

Build a typed LLM endpoint step by step. Each stage is a standalone FastAPI app you can run and compare.

## Setup

```bash
cp .env.example .env          # OPENAI_API_KEY=sk-...
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo stages

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `completions.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Streamlit demo runner

Interactive UI for all five stages:

```bash
streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal.

## Test with curl

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG in one sentence?"}'
```

Stage 5 example (model + cost):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "gpt-4o-mini"}'
```

## Smoke-test all stages

Requires `.venv` and a valid `OPENAI_API_KEY`:

```bash
python test_all_stages.py
```

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined)
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── adk_faq_agent.py     # Session 3 — Google ADK FAQ agent + POST /agent helper
├── requirements.txt
├── .env.example
└── .gitignore
```

## Session 3 — ADK FAQ agent

One Gemini agent that searches your docs with Session 2 `retrieve_chunks`, then answers with citations.

Needs `GOOGLE_API_KEY` (Gemini) plus the Session 2 vars (`OPENAI_API_KEY`, `PINECONE_*`) and ingested Northwind docs.

```bash
# from this folder, with .venv active
python adk_faq_agent.py
```

That question (`How many PTO days do employees get?`) forces a `search_docs` tool call. You should see `THINK` / `ACT` / `OBSERVE` / `ANSWER`, with real Pinecone chunks in OBSERVE. The run is capped at `max_llm_calls=6`.

### POST /agent (same FastAPI app as `/ask`)

`POST /ask` is unchanged. `POST /agent` runs the ADK loop and returns `{ answer, steps, model }`. `steps` are tool names plus truncated observations — no API keys.

Local:

```bash
curl -s -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "How many annual leave days do Northwind employees get?"}'
```

Render (replace with your service URL):

```bash
curl -s -X POST https://YOUR-SERVICE.onrender.com/agent \
  -H "Content-Type: application/json" \
  -d '{"goal": "How many annual leave days do Northwind employees get?"}'
```

`message` and `goal` are interchangeable. Add `GOOGLE_API_KEY` (and the existing OpenAI / Pinecone vars) on Render so `/agent` can run.

## Deploy to Render

Create a Render Web Service connected to the GitHub repository and use:

- Root Directory: `ai-engineering-bootcamp-v2/week-1`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

Add `OPENAI_API_KEY`, `GOOGLE_API_KEY`, and the Pinecone vars as environment variables in Render rather than committing them to the repository.

After deployment, test the endpoint with:

```bash
curl -s -X POST https://YOUR-SERVICE.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG in one sentence?"}'

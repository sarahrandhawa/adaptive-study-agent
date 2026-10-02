# Adaptive Study Agent

Adaptive Study Agent is an AI-powered study application that turns course materials into personalized study support. Students can upload their own materials, ask grounded questions about their notes, generate quizzes, and receive practice that adapts to their learning progress and weak areas.

## Features

- Upload PDF and TXT study materials
- Ask questions grounded in uploaded materials using RAG
- Generate quizzes from course content
- Grade quiz responses and track performance
- Track topic mastery and weak areas
- Adapt future practice based on previous performance

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **LLM:** OpenAI API
- **Vector database:** Pinecone
- **RAG:** LangChain
- **Database:** PostgreSQL / Supabase
- **Authentication:** Google OAuth

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and configure the required environment variables.

## Run Locally

Start the FastAPI backend:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal, activate the virtual environment and start the Streamlit frontend:

```bash
source .venv/bin/activate
streamlit run rag_ui.py
```

The Streamlit application will be available locally at `http://localhost:8501`.

## Project Structure

```text
main.py              FastAPI application and API endpoints
rag_ui.py            Streamlit frontend
ingest.py            Document ingestion
retrieve.py          Document retrieval
rag.py               RAG generation
vectorstore.py       Pinecone vector storage
memory_store.py      Student learning memory
study.py             Quiz and study logic
course/              Archived course exercises and demos
```

## Development Status

Adaptive Study Agent is currently under active development. The application is being extended with persistent per-user learning memory, authentication, user-isolated study materials, and adaptive quiz generation.

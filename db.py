import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Load .env next to this file, so it works no matter which folder the server starts in.
load_dotenv(Path(__file__).resolve().parent / ".env")


def get_conn():
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(database_url)

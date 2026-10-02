#!/usr/bin/env python3
"""Ingest Northwind sample docs via POST /ingest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

DOCS_DIR = Path(__file__).resolve().parent / "northwind-sample-docs"
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

# Stable IDs from northwind-sample-docs/README.md
DOCUMENTS: list[tuple[str, str, Path]] = [
    ("POL-101", "doc1_handbook.txt", DOCS_DIR / "doc1_handbook.txt"),
    ("POL-114", "doc2_expenses.txt", DOCS_DIR / "doc2_expenses.txt"),
    ("POL-207", "doc3_security.txt", DOCS_DIR / "doc3_security.txt"),
    ("SPEC-WB9", "doc4_product.txt", DOCS_DIR / "doc4_product.txt"),
    ("POL-220", "doc5_it_acceptable_use.txt", DOCS_DIR / "doc5_it_acceptable_use.txt"),
    ("POL-118", "doc6_facilities.txt", DOCS_DIR / "doc6_facilities.txt"),
]


def main() -> int:
    total_indexed = 0

    with httpx.Client(base_url=BASE_URL.rstrip("/"), timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()

        for document_id, filename, path in DOCUMENTS:
            text = path.read_text(encoding="utf-8")
            response = client.post(
                "/ingest",
                json={
                    "document_id": document_id,
                    "text": text,
                    "source": filename,
                },
            )
            response.raise_for_status()
            data = response.json()
            chunks = data["chunks_indexed"]
            total_indexed += chunks
            print(f"{document_id} ({filename}): {chunks} chunks")

        debug = client.get("/debug/pinecone")
        debug.raise_for_status()
        stats = debug.json().get("stats", {})
        store_total = stats.get("total_vector_count")

    print(f"\nIndexed this run: {total_indexed} chunks")
    print(f"Total vectors in store: {store_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

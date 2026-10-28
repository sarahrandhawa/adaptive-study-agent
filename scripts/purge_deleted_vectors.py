"""Retry Pinecone vector removal for deleted/replaced documents (ops script).

Deleting a document commits the Postgres change first, then removes its vectors.
If Pinecone was unavailable, the document is left with vectors_deleted_at = NULL.
Those stale vectors are never served (retrieval re-checks every match against
Postgres), but this script removes them to keep the index clean.

Dry run by default; pass --apply to delete. Uses DATABASE_URL and PINECONE_* from
the environment / .env. Only ever deletes chunk ids of non-active documents, in
the owning user's namespace.

    python scripts/purge_deleted_vectors.py            # list pending purges
    python scripts/purge_deleted_vectors.py --apply    # purge them
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import documents  # noqa: E402


def run(*, apply: bool, limit: int, out=print) -> dict[str, int]:
    pending = documents.pending_vector_purges(limit=limit)
    summary = {"pending": len(pending), "purged": 0, "failed": 0}

    for user_id, document_id in pending:
        if not apply:
            out(f"pending  user={user_id} document={document_id}")
            continue
        if documents.try_purge_document_vectors(user_id, document_id):
            summary["purged"] += 1
            out(f"purged   user={user_id} document={document_id}")
        else:
            summary["failed"] += 1
            out(f"FAILED   user={user_id} document={document_id}")

    mode = "applied" if apply else "dry run (pass --apply to purge)"
    out(f"{summary['pending']} pending, {summary['purged']} purged, {summary['failed']} failed — {mode}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="actually delete vectors (default: dry run)")
    parser.add_argument("--limit", type=int, default=500, help="maximum documents to process")
    args = parser.parse_args(argv)
    summary = run(apply=args.apply, limit=args.limit)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

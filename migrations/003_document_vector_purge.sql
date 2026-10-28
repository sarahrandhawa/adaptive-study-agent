-- 003: track Pinecone vector removal for deleted/replaced documents.
--
-- Document deletion now commits the Postgres change first (so content stops being
-- retrievable, gradable and practicable immediately) and removes vectors afterwards.
-- vectors_deleted_at records that the second step finished; NULL on a deleted or
-- replaced document means the purge is still pending and will be retried.
--
-- Backfill: before this change, vectors were deleted *before* the document was marked
-- deleted/replaced, inside the same transaction, so every existing non-active document
-- already had its vectors removed. Mark those as purged.
--
-- Apply BEFORE deploying the code that reads/writes documents.vectors_deleted_at.
-- Additive and safe to re-run; the previous code ignores the new column.

BEGIN;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS vectors_deleted_at timestamptz;

UPDATE documents
SET vectors_deleted_at = COALESCE(deleted_at, now())
WHERE status IN ('deleted', 'replaced')
  AND vectors_deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS documents_pending_vector_purge
    ON documents (deleted_at)
    WHERE status IN ('deleted', 'replaced')
      AND vectors_deleted_at IS NULL;

COMMIT;

-- Rollback for 003. Run only after rolling the API back to a version that does not
-- use documents.vectors_deleted_at. Pending purges are forgotten; run
-- scripts/purge_deleted_vectors.py first if any are listed.

BEGIN;

DROP INDEX IF EXISTS documents_pending_vector_purge;

ALTER TABLE documents
    DROP COLUMN IF EXISTS vectors_deleted_at;

COMMIT;

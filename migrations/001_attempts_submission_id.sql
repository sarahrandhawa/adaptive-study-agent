-- 001: idempotent grade submissions.
--
-- Adds a client-generated submission_id to attempts so a retried /grade
-- request is recorded once. Existing rows keep NULL (NULLs never conflict in
-- a Postgres unique index), so this is backward compatible with the old code.
--
-- Apply BEFORE deploying the code that writes attempts.submission_id.
-- Safe to re-run.

BEGIN;

ALTER TABLE attempts
    ADD COLUMN IF NOT EXISTS submission_id uuid;

CREATE UNIQUE INDEX IF NOT EXISTS attempts_user_submission_key
    ON attempts (user_id, submission_id);

COMMIT;

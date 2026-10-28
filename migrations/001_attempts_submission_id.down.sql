-- Rollback for 001. Only run after rolling the API back to a version that does
-- not write attempts.submission_id. Drops stored submission ids (retry history
-- only; attempts and mastery are untouched).

BEGIN;

DROP INDEX IF EXISTS attempts_user_submission_key;

ALTER TABLE attempts
    DROP COLUMN IF EXISTS submission_id;

COMMIT;

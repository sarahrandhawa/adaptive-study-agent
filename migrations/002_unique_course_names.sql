CREATE UNIQUE INDEX IF NOT EXISTS courses_user_normalized_name_unique
ON courses (
    user_id,
    LOWER(
        REGEXP_REPLACE(
            BTRIM(name),
            '[[:space:]]+',
            ' ',
            'g'
        )
    )
)
WHERE archived_at IS NULL;
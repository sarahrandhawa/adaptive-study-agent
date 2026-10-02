"""Topic helpers backed by Postgres."""

from db import get_conn


def normalize_topic(topic: str) -> str:
    return " ".join(topic.split()).casefold()

def get_or_create_topic(user_id, course_id, name):
    name = name.strip()
    normalized_key = normalize_topic(name)

    if not normalized_key:
        raise ValueError("Topic must not be empty")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO topics (
                    user_id,
                    course_id,
                    name,
                    normalized_key,
                    created_by
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (course_id, normalized_key)
                DO UPDATE SET name = topics.name
                RETURNING id
                """,
                (
                    user_id,
                    course_id,
                    name,
                    normalized_key,
                    "user",
                ),
            )

            return cur.fetchone()[0]

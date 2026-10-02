from db import get_conn


def create_course(user_id, name):
    name = name.strip()

    if not name:
        raise ValueError("Course name must not be empty")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO courses (user_id, name)
                VALUES (%s, %s)
                RETURNING id
                """,
                (user_id, name),
            )

            return cur.fetchone()[0]


def list_courses(user_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, created_at
                FROM courses
                WHERE user_id = %s
                  AND archived_at IS NULL
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "name": row[1],
            "created_at": row[2],
        }
        for row in rows
    ]


def user_owns_course(user_id, course_id) -> bool:
    """True only when the course exists, is not archived, and belongs to this user."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM courses
                WHERE id = %s
                  AND user_id = %s
                  AND archived_at IS NULL
                """,
                (course_id, user_id),
            )

            return cur.fetchone() is not None


def archive_course(user_id, course_id) -> bool:
    """Soft-delete: hide the course from the user's list. Study history is kept."""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE courses
                SET archived_at = now()
                WHERE id = %s
                  AND user_id = %s
                  AND archived_at IS NULL
                RETURNING id
                """,
                (course_id, user_id),
            )

            return cur.fetchone() is not None

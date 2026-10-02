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
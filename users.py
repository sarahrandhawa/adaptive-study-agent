from db import get_conn


def ensure_user(provider, subject, email=None, name=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Check whether this login identity already belongs to a user
            cur.execute(
                """
                SELECT user_id
                FROM user_identities
                WHERE provider = %s AND subject = %s
                """,
                (provider, subject),
            )

            result = cur.fetchone()

            # Existing user: just return their UUID
            if result is not None:
                return result[0]

            # New user: create the user and get their generated UUID
            cur.execute(
                """
                INSERT INTO users (email, display_name)
                VALUES (%s, %s)
                RETURNING id
                """,
                (email, name),
            )

            user_id = cur.fetchone()[0]

            # Connect their login identity to their new user UUID
            cur.execute(
                """
                INSERT INTO user_identities (user_id, provider, subject)
                VALUES (%s, %s, %s)
                """,
                (user_id, provider, subject),
            )

            return user_id
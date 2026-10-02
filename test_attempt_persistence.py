from db import get_conn


USER_PROVIDER = "dev"
USER_SUBJECT = "dev-local"
QUESTION_ID = "7bb0dc06-40f6-4605-bccd-da9fbd91f790"


def test_attempts_are_history_and_state_is_summary():
    with get_conn() as conn:
        with conn.cursor() as cur:

            # Find our dev user.
            cur.execute(
                """
                SELECT user_id
                FROM user_identities
                WHERE provider = %s
                  AND subject = %s
                """,
                (USER_PROVIDER, USER_SUBJECT),
            )
            user_id = cur.fetchone()[0]

            # Our manual test already created one attempt.
            cur.execute(
                """
                SELECT COUNT(*)
                FROM attempts
                WHERE user_id = %s
                  AND question_id = %s
                """,
                (user_id, QUESTION_ID),
            )
            attempt_count = cur.fetchone()[0]

            # question_state should summarize those attempts.
            cur.execute(
                """
                SELECT attempt_count, last_score, mastery
                FROM question_state
                WHERE user_id = %s
                  AND question_id = %s
                """,
                (user_id, QUESTION_ID),
            )
            state = cur.fetchone()

    assert state is not None

    state_attempt_count, last_score, mastery = state

    assert attempt_count == 1
    assert state_attempt_count == attempt_count
    assert last_score == 0.65
    assert round(mastery, 2) == 0.56
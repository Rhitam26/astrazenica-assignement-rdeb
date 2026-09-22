"""Read-only discovery of persisted top-level LangGraph threads."""

from psycopg.rows import dict_row


def checkpoint_candidates(pool, offset: int, limit: int) -> list[dict]:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(
            """
            SELECT thread_id, checkpoint_id, checkpoint->>'ts' AS updated_at
            FROM (
                SELECT DISTINCT ON (thread_id) thread_id, checkpoint_id, checkpoint
                FROM conversation.checkpoints
                WHERE checkpoint_ns = ''
                ORDER BY thread_id, checkpoint_id DESC
            ) AS latest
            ORDER BY checkpoint_id DESC, thread_id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        ).fetchall()

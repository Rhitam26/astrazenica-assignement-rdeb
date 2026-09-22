"""Durable job and parsed-chunk persistence for asynchronous ingestion."""

from datetime import timedelta
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.ingestion.chunker import ChunkRecord


def _one(conn, sql, values=()):
    with conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(sql, values).fetchone()


def _all(conn, sql, values=()):
    with conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(sql, values).fetchall()


def create_job(conn, *, job_id: UUID, original_filename: str, source_path: str, file_hash: str) -> dict:
    row = _one(
        conn,
        """
        INSERT INTO ingestion.jobs (job_id, doc_id, original_filename, source_path, file_hash, status)
        VALUES (%s, %s, %s, %s, %s, 'queued')
        RETURNING *
        """,
        (job_id, f"upload-{job_id}", original_filename, source_path, file_hash),
    )
    return row


def get_job_by_hash(conn, file_hash: str) -> dict | None:
    return _one(conn, "SELECT * FROM ingestion.jobs WHERE file_hash = %s", (file_hash,))


def get_job(conn, job_id: UUID) -> dict | None:
    return _one(conn, "SELECT * FROM ingestion.jobs WHERE job_id = %s", (job_id,))


def list_jobs(conn, offset: int, limit: int) -> list[dict]:
    return _all(
        conn,
        "SELECT * FROM ingestion.jobs ORDER BY created_at DESC OFFSET %s LIMIT %s",
        (offset, limit + 1),
    )


def list_job_chunks(conn, job_id: UUID, offset: int, limit: int) -> list[dict]:
    return _all(
        conn,
        "SELECT * FROM ingestion.job_chunks WHERE job_id = %s ORDER BY chunk_index OFFSET %s LIMIT %s",
        (job_id, offset, limit + 1),
    )


def retry_job(conn, job_id: UUID) -> dict | None:
    return _one(
        conn,
        """
        UPDATE ingestion.jobs
        SET status = 'queued', lease_expires_at = NULL, error_type = NULL, updated_at = now()
        WHERE job_id = %s AND status = 'failed'
        RETURNING *
        """,
        (job_id,),
    )


def cancel_job(conn, job_id: UUID) -> dict | None:
    return _one(
        conn,
        """
        UPDATE ingestion.jobs SET status = 'cancelled', updated_at = now()
        WHERE job_id = %s AND status = 'queued'
        RETURNING *
        """,
        (job_id,),
    )


def claim_next_job(conn, *, lease_seconds: int, max_attempts: int) -> dict | None:
    conn.execute(
        """
        UPDATE ingestion.jobs
        SET status = 'queued', lease_expires_at = NULL, updated_at = now()
        WHERE status IN ('parsing', 'chunked', 'embedding')
          AND lease_expires_at < now()
          AND attempts < %s
        """,
        (max_attempts,),
    )
    row = _one(
        conn,
        """
        SELECT * FROM ingestion.jobs
        WHERE status = 'queued' AND attempts < %s
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """,
        (max_attempts,),
    )
    if row is None:
        return None
    return _one(
        conn,
        """
        UPDATE ingestion.jobs
        SET status = 'parsing', attempts = attempts + 1,
            lease_expires_at = now() + %s::interval, updated_at = now()
        WHERE job_id = %s
        RETURNING *
        """,
        (timedelta(seconds=lease_seconds), row["job_id"]),
    )


def set_stage(conn, job_id: UUID, status: str, *, lease_seconds: int) -> None:
    conn.execute(
        """
        UPDATE ingestion.jobs
        SET status = %s, lease_expires_at = now() + %s::interval, updated_at = now()
        WHERE job_id = %s AND status <> 'cancelled'
        """,
        (status, timedelta(seconds=lease_seconds), job_id),
    )


def is_cancelled(conn, job_id: UUID) -> bool:
    row = conn.execute(
        "SELECT status = 'cancelled' FROM ingestion.jobs WHERE job_id = %s", (job_id,)
    ).fetchone()
    return bool(row and row[0])


def replace_job_chunks(conn, job_id: UUID, records: list[ChunkRecord]) -> None:
    conn.execute("DELETE FROM ingestion.job_chunks WHERE job_id = %s", (job_id,))
    for record in records:
        conn.execute(
            """
            INSERT INTO ingestion.job_chunks (
                job_id, chunk_index, content_hash, heading_path, chapter_title, section_title,
                subsection_title, page_numbers, page_start, page_end, content_type, table_id, text, token_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                job_id,
                record.chunk_index,
                record.content_hash,
                Jsonb(record.heading_path),
                record.chapter_title,
                record.section_title,
                record.subsection_title,
                record.page_numbers,
                record.page_start,
                record.page_end,
                record.content_type,
                record.table_id,
                record.text,
                record.token_count,
            ),
        )
    conn.execute(
        "UPDATE ingestion.jobs SET total_chunks = %s, updated_at = now() WHERE job_id = %s",
        (len(records), job_id),
    )


def set_embedding_results(
    conn,
    job_id: UUID,
    *,
    embedded_hashes: set[str],
    skipped_hashes: set[str],
    embedded_count: int,
    skipped_count: int,
) -> None:
    conn.execute(
        "UPDATE ingestion.job_chunks SET embedding_status = 'embedded' WHERE job_id = %s AND content_hash = ANY(%s)",
        (job_id, list(embedded_hashes)),
    )
    conn.execute(
        "UPDATE ingestion.job_chunks SET embedding_status = 'skipped' WHERE job_id = %s AND content_hash = ANY(%s)",
        (job_id, list(skipped_hashes)),
    )
    conn.execute(
        """
        UPDATE ingestion.jobs
        SET status = 'completed', lease_expires_at = NULL, embedded_chunks = %s, skipped_chunks = %s,
            completed_at = now(), updated_at = now()
        WHERE job_id = %s
        """,
        (embedded_count, skipped_count, job_id),
    )


def fail_job(conn, job_id: UUID, error_type: str) -> None:
    conn.execute(
        """
        UPDATE ingestion.jobs
        SET status = 'failed', lease_expires_at = NULL, error_type = %s, updated_at = now()
        WHERE job_id = %s AND status <> 'cancelled'
        """,
        (error_type[:100], job_id),
    )

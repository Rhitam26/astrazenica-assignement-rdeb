"""Postgres/pgvector access layer.

Kept deliberately thin (raw SQL via psycopg, no ORM) since the schema is
small and stable; this also makes the HNSW/vector-specific SQL easy to
reason about during the interview.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: list[object]
    page_numbers: list[int]
    content_type: str
    chunk_index: int
    text: str
    similarity: float

    @property
    def citation(self) -> str:
        pages = f" p. {min(self.page_numbers)}" if self.page_numbers else ""
        heading = " > ".join(str(part) for part in self.heading_path) if self.heading_path else None
        location = f", {heading}" if heading else ""
        return f"{self.doc_title}{pages}{location}"


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    settings = get_settings()
    conn = psycopg.connect(settings.database_url, autocommit=False)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_document(conn: psycopg.Connection, doc_id: str, title: str, source_path: str) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, title, source_path)
        VALUES (%s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE
            SET title = EXCLUDED.title,
                source_path = EXCLUDED.source_path
        """,
        (doc_id, title, source_path),
    )


def get_existing_hashes(conn: psycopg.Connection, content_hashes: list[str]) -> set[str]:
    """Return which of the given content hashes are already indexed.

    Used to skip re-embedding unchanged chunks on re-ingestion, which saves
    both OpenAI API cost and DB writes.
    """
    if not content_hashes:
        return set()
    rows = conn.execute(
        "SELECT content_hash FROM chunks WHERE content_hash = ANY(%s)",
        (content_hashes,),
    ).fetchall()
    return {row[0] for row in rows}


def insert_chunk(conn, chunk: dict) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            content_hash,
            doc_id,
            doc_title,
            heading_path,
            chapter_title,
            section_title,
            subsection_title,
            page_numbers,
            page_start,
            page_end,
            content_type,
            table_id,
            chunk_index,
            text,
            token_count,
            embedding_model,
            embedding_dims,
            embedding
        )
        VALUES (
            %(content_hash)s,
            %(doc_id)s,
            %(doc_title)s,
            %(heading_path)s,
            %(chapter_title)s,
            %(section_title)s,
            %(subsection_title)s,
            %(page_numbers)s,
            %(page_start)s,
            %(page_end)s,
            %(content_type)s,
            %(table_id)s,
            %(chunk_index)s,
            %(text)s,
            %(token_count)s,
            %(embedding_model)s,
            %(embedding_dims)s,
            %(embedding)s
        )
        ON CONFLICT (content_hash) DO NOTHING
        """,
        chunk,
    )


def retrieve_relevant_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    limit: int = 5,
    doc_id: str | None = None,
    content_type: str | None = None,
) -> list[RetrievedChunk]:
    """Return nearest chunks for a query vector using cosine similarity.

    pgvector's `<=>` operator returns cosine distance for a cosine index, so
    similarity is reported as `1 - distance` for easier reading.
    """
    filters = ["embedding_model = %s", "embedding_dims = %s"]
    settings = get_settings()
    filter_params: list[object] = [settings.embedding_model, settings.embedding_dims]

    if doc_id:
        filters.append("doc_id = %s")
        filter_params.append(doc_id)

    if content_type:
        filters.append("content_type = %s")
        filter_params.append(content_type)

    query_vector = Vector(query_embedding)
    params: list[object] = [query_vector, *filter_params, query_vector, limit]
    rows = conn.execute(
        f"""
        SELECT
            chunk_id::text,
            doc_id,
            doc_title,
            heading_path,
            page_numbers,
            content_type,
            chunk_index,
            text,
            1 - (embedding <=> %s) AS similarity
        FROM chunks
        WHERE {" AND ".join(filters)}
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        params,
    ).fetchall()

    return [
        RetrievedChunk(
            chunk_id=row[0],
            doc_id=row[1],
            doc_title=row[2],
            heading_path=row[3] or [],
            page_numbers=row[4] or [],
            content_type=row[5],
            chunk_index=row[6],
            text=row[7],
            similarity=float(row[8]),
        )
        for row in rows
    ]

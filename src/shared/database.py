"""Database connectivity and common readiness checks."""

from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from src.shared.config import Settings, get_settings


@contextmanager
def get_connection(settings: Settings | None = None) -> Iterator[psycopg.Connection]:
    settings = settings or get_settings()
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        yield conn


def validate_embeddings(conn: psycopg.Connection, settings: Settings) -> None:
    row = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid='public.chunks'::regclass AND attname='embedding'"
    ).fetchone()
    if not row or row[0] != f"vector({settings.embedding_dims})":
        raise ValueError(
            "Schema embedding dimensions do not match configuration; migrate and re-index explicitly"
        )
    incompatible = conn.execute(
        "SELECT 1 FROM chunks WHERE embedding_model <> %s OR embedding_dims <> %s LIMIT 1",
        (settings.embedding_model, settings.embedding_dims),
    ).fetchone()
    if incompatible:
        raise ValueError("Corpus embedding model/dimensions mismatch; re-index explicitly")


def check_ready(settings: Settings) -> None:
    with get_connection(settings) as conn:
        validate_embeddings(conn, settings)
        index = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='public' AND indexname='idx_chunks_embedding_hnsw'"
        ).fetchone()
        if not index or "halfvec" not in index[0]:
            raise ValueError("Missing compatible HNSW index; run python -m src.shared.migrate")
        conn.execute("SELECT 1 FROM conversation.checkpoints LIMIT 1")

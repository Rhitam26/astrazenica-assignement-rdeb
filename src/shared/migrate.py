"""Repeatable, data-preserving schema/index/checkpointer initialization."""

import logging
from pathlib import Path

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from src.shared.config import Settings, get_settings


def migrate(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    schema = Path(__file__).resolve().parents[2] / "sql/schema.sql"
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7193471)")
        index = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE schemaname='public' AND indexname='idx_chunks_embedding_hnsw'"
        ).fetchone()
        if index and "halfvec" not in index[0]:
            conn.execute("DROP INDEX public.idx_chunks_embedding_hnsw")
        conn.execute(schema.read_text())
        conn.execute("CREATE SCHEMA IF NOT EXISTS conversation")
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        conn.execute("SET search_path TO conversation, public")
        PostgresSaver(conn).setup()
    logging.getLogger(__name__).info("Schema and checkpoint migrations complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()

"""Small synchronous retrieval boundary; no duplicate SQL implementation."""

from contextlib import AbstractContextManager
from typing import Callable, Protocol

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from src.assistant.models import MetadataFilters, RetrievedChunk
from src.assistant.repository import retrieve_relevant_chunks
from src.shared.config import Settings
from src.shared.database import get_connection, validate_embeddings
from src.shared.embedder import Embedder
from src.shared.observability import Telemetry, preview, value_hash


class Retriever(Protocol):
    def search(
        self, query: str, top_k: int = 5, filters: MetadataFilters | None = None
    ) -> list[RetrievedChunk]: ...


def retrieval_pool(settings: Settings) -> ConnectionPool:
    def configure(conn):
        register_vector(conn)
        conn.commit()

    # Transactional borrowing preserves retrieval's SET LOCAL HNSW setting.
    return ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=8,
        timeout=5,
        configure=configure,
        open=False,
    )


class PgRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        telemetry: Telemetry | None = None,
        connection_factory: Callable[[], AbstractContextManager[psycopg.Connection]] | None = None,
    ):
        self.settings, self.embedder = settings, embedder
        self.telemetry = telemetry or Telemetry(settings)
        self.connection_factory = connection_factory or (lambda: get_connection(settings))

    def search(
        self, query: str, top_k: int = 5, filters: MetadataFilters | None = None
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Empty retrieval query")
        with self.connection_factory() as conn:
            validate_embeddings(conn, self.settings)
        vector = self.embedder.embed_texts([query])[0]
        with self.telemetry.observation(
            "retrieval.database",
            input={
                "query_hash": value_hash(query),
                "query": preview(query, self.settings.langfuse_preview_chars),
            },
            metadata={"top_k": top_k, "filters": filters.model_dump(exclude_none=True) if filters else {}},
        ) as observation:
            with self.connection_factory() as conn:
                chunks = retrieve_relevant_chunks(
                    conn, vector, limit=top_k, filters=filters, settings=self.settings
                )
            scores = [chunk.score for chunk in chunks]
            observation.update(
                output={"result_count": len(chunks), "chunk_ids": [str(c.chunk_id) for c in chunks]},
                metadata={
                    "top_score": max(scores, default=None),
                    "mean_score": sum(scores) / len(scores) if scores else None,
                    "document_ids": list(dict.fromkeys(c.doc_id for c in chunks)),
                },
            )
            return chunks

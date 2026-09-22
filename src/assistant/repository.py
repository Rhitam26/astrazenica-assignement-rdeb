"""Read-side vector retrieval SQL used only by the assistant runtime."""

import psycopg
from pgvector import Vector
from psycopg.rows import dict_row

from src.assistant.models import MetadataFilters, RetrievedChunk
from src.shared.config import Settings, get_settings


def filter_sql(filters: MetadataFilters | None) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if filters is None:
        return clauses, params
    for field, value in filters.model_dump(exclude_none=True).items():
        if field == "doc_ids":
            clauses.append("c.doc_id = ANY(%s)")
        elif field == "doc_titles":
            clauses.append("d.title = ANY(%s)")
        elif field == "page_numbers":
            clauses.append("c.page_numbers && %s::integer[]")
        else:
            # Field names originate exclusively in the validated allow-list model.
            clauses.append(f"c.{field} = %s")
        params.append(value)
    return clauses, params


def retrieve_relevant_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    limit: int = 5,
    filters: MetadataFilters | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    if not 1 <= limit <= 20 or len(query_embedding) != settings.embedding_dims:
        raise ValueError("Invalid retrieval limit or embedding dimension")
    clauses, params = filter_sql(filters)
    clauses = ["c.embedding_model = %s", "c.embedding_dims = %s", *clauses]
    vector = Vector(query_embedding)
    # Iterative scanning improves recall when metadata filtering removes ANN candidates.
    conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
    sql = f"""
        WITH candidates AS MATERIALIZED (
            SELECT c.*, d.source_path, d.title AS canonical_title
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE {" AND ".join(clauses)}
            ORDER BY c.embedding::halfvec(3072) <=> %s::halfvec(3072)
            LIMIT %s
        )
        SELECT chunk_id, content_hash, doc_id, canonical_title AS doc_title, source_path,
               heading_path, chapter_title, section_title, subsection_title,
               page_numbers, page_start, page_end, content_type, table_id, chunk_index, text,
               1 - (embedding <=> %s) AS score
        FROM candidates
        WHERE 1 - (embedding <=> %s) >= %s
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    values = [
        settings.embedding_model,
        settings.embedding_dims,
        *params,
        vector,
        max(limit, settings.retrieval_candidate_pool),
        vector,
        vector,
        settings.retrieval_score_threshold,
        vector,
        limit,
    ]
    with conn.cursor(row_factory=dict_row) as cursor:
        return [RetrievedChunk.model_validate(row) for row in cursor.execute(sql, values).fetchall()]

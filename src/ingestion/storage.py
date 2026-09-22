"""Write-side knowledge-base persistence used only by offline ingestion."""

import psycopg


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
    if not content_hashes:
        return set()
    rows = conn.execute(
        "SELECT content_hash FROM chunks WHERE content_hash = ANY(%s)", (content_hashes,)
    ).fetchall()
    return {row[0] for row in rows}


def insert_chunk(conn, chunk: dict) -> None:
    conn.execute(
        """
        INSERT INTO chunks (
            content_hash, doc_id, doc_title, heading_path, chapter_title, section_title, subsection_title,
            page_numbers, page_start, page_end, content_type, table_id, chunk_index, text, token_count,
            embedding_model, embedding_dims, embedding
        ) VALUES (
            %(content_hash)s, %(doc_id)s, %(doc_title)s, %(heading_path)s, %(chapter_title)s,
            %(section_title)s, %(subsection_title)s, %(page_numbers)s, %(page_start)s, %(page_end)s,
            %(content_type)s, %(table_id)s, %(chunk_index)s, %(text)s, %(token_count)s,
            %(embedding_model)s, %(embedding_dims)s, %(embedding)s
        ) ON CONFLICT (content_hash) DO NOTHING
        """,
        chunk,
    )

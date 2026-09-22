"""Shared PDF-to-pgvector processing used by the worker and batch CLI."""

from pathlib import Path
from uuid import UUID

from psycopg.types.json import Jsonb

from src.ingestion.chunker import ChunkRecord, chunk_pdf
from src.ingestion.jobs import is_cancelled, replace_job_chunks, set_embedding_results, set_stage
from src.ingestion.storage import get_existing_hashes, insert_chunk, upsert_document
from src.shared.config import Settings
from src.shared.database import get_connection, validate_embeddings
from src.shared.embedder import Embedder


def persist_records(
    conn, records: list[ChunkRecord], settings: Settings, embedder: Embedder, source_path: str
) -> tuple[set[str], set[str], int]:
    """Write records and return new hashes, existing hashes, and skipped-record count."""
    if not records:
        return set(), set(), 0
    upsert_document(conn, records[0].doc_id, records[0].doc_title, source_path)
    existing = get_existing_hashes(conn, [record.content_hash for record in records])
    unique = {record.content_hash: record for record in records}
    new_records = [record for content_hash, record in unique.items() if content_hash not in existing]
    if new_records:
        vectors = embedder.embed_texts([record.text for record in new_records])
        if len(vectors) != len(new_records):
            raise ValueError("Embedding count mismatch")
        for record, vector in zip(new_records, vectors, strict=True):
            insert_chunk(
                conn,
                {
                    "content_hash": record.content_hash,
                    "doc_id": record.doc_id,
                    "doc_title": record.doc_title,
                    "heading_path": Jsonb(record.heading_path),
                    "chapter_title": record.chapter_title,
                    "section_title": record.section_title,
                    "subsection_title": record.subsection_title,
                    "page_numbers": record.page_numbers,
                    "page_start": record.page_start,
                    "page_end": record.page_end,
                    "content_type": record.content_type,
                    "table_id": record.table_id,
                    "chunk_index": record.chunk_index,
                    "text": record.text,
                    "token_count": record.token_count,
                    "embedding_model": settings.embedding_model,
                    "embedding_dims": settings.embedding_dims,
                    "embedding": vector,
                },
            )
    return {record.content_hash for record in new_records}, set(existing), len(records) - len(new_records)


def process_uploaded_job(settings: Settings, job: dict, embedder: Embedder) -> None:
    """Process one claimed job; each durable stage is committed independently."""
    job_id: UUID = job["job_id"]
    source_path = Path(job["source_path"])
    records = chunk_pdf(source_path, doc_id=job["doc_id"], doc_title=Path(job["original_filename"]).stem)
    if not records:
        raise ValueError("NoChunksProduced")
    with get_connection(settings) as conn:
        if is_cancelled(conn, job_id):
            return
        replace_job_chunks(conn, job_id, records)
        set_stage(conn, job_id, "chunked", lease_seconds=settings.ingestion_job_lease_seconds)
        set_stage(conn, job_id, "embedding", lease_seconds=settings.ingestion_job_lease_seconds)
    with get_connection(settings) as conn:
        if is_cancelled(conn, job_id):
            return
        validate_embeddings(conn, settings)
        embedded_hashes, skipped_hashes, skipped_count = persist_records(
            conn, records, settings, embedder, str(source_path)
        )
        set_embedding_results(
            conn,
            job_id,
            embedded_hashes=embedded_hashes,
            skipped_hashes=skipped_hashes,
            embedded_count=len(embedded_hashes),
            skipped_count=skipped_count,
        )

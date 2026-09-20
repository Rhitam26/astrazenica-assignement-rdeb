"""Standalone ingestion entrypoint.

Run separately from the query-time API, e.g.:

    python -m src.ingestion.ingest --source-dir ./corpus

Idempotent by design: each chunk's content_hash is checked against the DB
*before* it is sent to the embeddings API, so re-running on an unchanged
corpus costs nothing, and adding one new/changed document does not
re-embed or re-process the rest of the corpus.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from psycopg.types.json import Jsonb

from config import get_settings
from db import get_connection, get_existing_hashes, insert_chunk, upsert_document
from chunker import chunk_pdf
from embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_directory(source_dir: Path) -> None:
    settings = get_settings()
    embedder = Embedder()

    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s", source_dir)
        return

    with get_connection() as conn:
        for pdf_path in pdf_paths:
            doc_id = pdf_path.stem
            doc_title = pdf_path.stem.replace("_", " ")
            logger.info("Processing %s", pdf_path.name)

            records = chunk_pdf(pdf_path, doc_id=doc_id, doc_title=doc_title)
            if not records:
                logger.warning("No chunks produced for %s, skipping", pdf_path.name)
                continue

            upsert_document(conn, doc_id=doc_id, title=doc_title, source_path=str(pdf_path))

            # Skip embedding work entirely for chunks already indexed.
            existing = get_existing_hashes(conn, [r.content_hash for r in records])
            new_records = [r for r in records if r.content_hash not in existing]
            skipped = len(records) - len(new_records)

            if new_records:
                vectors = embedder.embed_texts([r.text for r in new_records])
                for record, vector in zip(new_records, vectors):
                    insert_chunk(
                        conn,
                        {
                        "content_hash": record.content_hash,
                        "doc_id": record.doc_id,
                        "doc_title": record.doc_title,

                        "heading_path":Jsonb(record.heading_path),

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
                conn.commit()

            logger.info(
                "%s: %d chunks total, %d newly embedded, %d already indexed (skipped)",
                pdf_path.name, len(records), len(new_records), skipped,
            )


def main() -> None:
    source_dir = Path("pdfs")
    ingest_directory(source_dir)


if __name__ == "__main__":
    main()

"""Standalone ingestion entrypoint.

Run separately from the query-time API, e.g.:

    python -m src.ingestion.ingest --source-dir ./pdfs

Idempotent by design: each chunk's content_hash is checked against the DB
*before* it is sent to the embeddings API, so re-running on an unchanged
corpus costs nothing, and adding one new/changed document does not
re-embed or re-process the rest of the corpus.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ingestion.chunker import chunk_pdf
from src.ingestion.processor import persist_records
from src.shared.config import get_settings
from src.shared.database import get_connection, validate_embeddings
from src.shared.embedder import Embedder
from src.shared.observability import Telemetry, preview, value_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_directory(source_dir: Path) -> None:
    settings = get_settings()
    telemetry = Telemetry(settings)
    try:
        embedder = Embedder(settings, telemetry)
    except TypeError:
        # Preserve compatibility with simple zero-argument test doubles.
        embedder = Embedder()

    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s", source_dir)
        telemetry.flush()
        telemetry.shutdown()
        return

    try:
        with telemetry.trace(
            "ingestion-run",
            session_id=f"ingestion:{value_hash(source_dir)}",
            metadata={"source_dir": preview(source_dir, 200), "document_count": len(pdf_paths)},
        ) as run:
            with get_connection() as conn:
                validate_embeddings(conn, settings)
                for pdf_path in pdf_paths:
                    doc_id = pdf_path.stem
                    doc_title = pdf_path.stem.replace("_", " ")
                    logger.info("Processing %s", pdf_path.name)

                    with telemetry.observation(
                        "ingestion.document",
                        metadata={"document_id": doc_id, "filename": preview(pdf_path.name, 200)},
                    ) as document:
                        records = chunk_pdf(pdf_path, doc_id=doc_id, doc_title=doc_title)
                        if not records:
                            logger.warning("No chunks produced for %s, skipping", pdf_path.name)
                            document.update(metadata={"total_chunks": 0, "skipped": True})
                            continue

                        embedded_hashes, _, skipped = persist_records(
                            conn, records, settings, embedder, str(pdf_path)
                        )
                        conn.commit()

                        document.update(
                            metadata={
                                "total_chunks": len(records),
                                "new_chunks": len(embedded_hashes),
                                "skipped_chunks": skipped,
                                "token_count": sum(getattr(r, "token_count", 0) for r in records),
                            }
                        )
                        logger.info(
                            "%s: %d chunks total, %d newly embedded, %d already indexed (skipped)",
                            pdf_path.name,
                            len(records),
                            len(embedded_hashes),
                            skipped,
                        )
            run.update(metadata={"completed_documents": len(pdf_paths)})
    finally:
        telemetry.flush()
        telemetry.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs with content-hash deduplication")
    parser.add_argument("--source-dir", type=Path, default=Path(get_settings().pdf_dir))
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        parser.error("source directory does not exist")
    ingest_directory(args.source_dir)


if __name__ == "__main__":
    main()

"""Admin-only HTTP API for durable asynchronous PDF ingestion."""

import hashlib
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.ingestion.jobs import (
    cancel_job,
    create_job,
    get_job,
    get_job_by_hash,
    list_job_chunks,
    list_jobs,
    retry_job,
)
from src.ingestion.models import IngestionChunkPage, IngestionJob, IngestionPage
from src.shared.config import Settings, get_settings
from src.shared.database import get_connection, validate_embeddings

logger = logging.getLogger(__name__)
UPLOAD_BLOCK_SIZE = 1024 * 1024


def job_page(rows: list[dict], offset: int, limit: int) -> IngestionPage:
    return IngestionPage(
        items=[IngestionJob.model_validate(row) for row in rows[:limit]],
        next_offset=offset + limit if len(rows) > limit else None,
    )


def chunk_page(rows: list[dict], offset: int, limit: int) -> IngestionChunkPage:
    return IngestionChunkPage(
        items=[row for row in rows[:limit]], next_offset=offset + limit if len(rows) > limit else None
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.ingestion_api_key.get_secret_value():
            raise ValueError("INGESTION_API_KEY must be configured")
        upload_dir = Path(settings.ingestion_upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        if not upload_dir.is_dir():
            raise ValueError("INGESTION_UPLOAD_DIR is not a directory")
        app.state.upload_dir = upload_dir
        yield

    app = FastAPI(
        title="Document Ingestion API",
        version="1.0.0",
        description="Admin-only asynchronous PDF upload, chunking, embedding, and pgvector ingestion.",
        lifespan=lifespan,
    )

    def require_admin(x_ingestion_key: str = Header(default="")) -> None:
        expected = settings.ingestion_api_key.get_secret_value()
        if not expected or not secrets.compare_digest(x_ingestion_key, expected):
            raise HTTPException(401, "Invalid ingestion API key.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        try:
            with get_connection(settings) as conn:
                validate_embeddings(conn, settings)
                conn.execute("SELECT 1 FROM ingestion.jobs LIMIT 1")
        except Exception as exc:
            logger.warning("ingestion_readiness_failed", extra={"error_type": type(exc).__name__})
            raise HTTPException(503, "Ingestion service is not ready.") from None
        return {"status": "ready"}

    @app.post(
        "/v1/ingestions", response_model=IngestionJob, status_code=202, dependencies=[Depends(require_admin)]
    )
    async def upload_pdf(file: UploadFile = File(...)):
        filename = Path(file.filename or "upload.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(422, "Only PDF files are accepted.")
        upload_dir: Path = app.state.upload_dir
        temporary_path = upload_dir / f".{uuid4()}.uploading"
        digest = hashlib.sha256()
        size = 0
        starts_with = b""
        try:
            with temporary_path.open("xb") as destination:
                while block := await file.read(UPLOAD_BLOCK_SIZE):
                    if len(starts_with) < 5:
                        starts_with += block[: 5 - len(starts_with)]
                    size += len(block)
                    if size > settings.ingestion_max_upload_bytes:
                        raise HTTPException(413, "PDF exceeds the configured upload limit.")
                    digest.update(block)
                    destination.write(block)
            if starts_with != b"%PDF-":
                raise HTTPException(422, "Uploaded file is not a valid PDF.")
            file_hash = digest.hexdigest()
            with get_connection(settings) as conn:
                existing = get_job_by_hash(conn, file_hash)
                if existing is not None:
                    temporary_path.unlink(missing_ok=True)
                    return JSONResponse(
                        status_code=200, content=IngestionJob.model_validate(existing).model_dump(mode="json")
                    )
                job_id = uuid4()
                # create_job owns the document ID; use a preallocated safe path and create the DB row atomically.
                final_path = upload_dir / f"{job_id}.pdf"
                temporary_path.replace(final_path)
                try:
                    row = create_job(
                        conn,
                        job_id=job_id,
                        original_filename=filename,
                        source_path=str(final_path),
                        file_hash=file_hash,
                    )
                except Exception:
                    final_path.unlink(missing_ok=True)
                    raise
            return IngestionJob.model_validate(row)
        finally:
            await file.close()
            temporary_path.unlink(missing_ok=True)

    @app.get("/v1/ingestions", response_model=IngestionPage, dependencies=[Depends(require_admin)])
    def jobs(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        with get_connection(settings) as conn:
            return job_page(list_jobs(conn, offset, limit), offset, limit)

    @app.get("/v1/ingestions/{job_id}", response_model=IngestionJob, dependencies=[Depends(require_admin)])
    def job(job_id: UUID):
        with get_connection(settings) as conn:
            row = get_job(conn, job_id)
        if row is None:
            raise HTTPException(404, "Ingestion job not found.")
        return row

    @app.get(
        "/v1/ingestions/{job_id}/chunks",
        response_model=IngestionChunkPage,
        dependencies=[Depends(require_admin)],
    )
    def chunks(job_id: UUID, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        with get_connection(settings) as conn:
            if get_job(conn, job_id) is None:
                raise HTTPException(404, "Ingestion job not found.")
            return chunk_page(list_job_chunks(conn, job_id, offset, limit), offset, limit)

    @app.post(
        "/v1/ingestions/{job_id}/retry", response_model=IngestionJob, dependencies=[Depends(require_admin)]
    )
    def retry(job_id: UUID):
        with get_connection(settings) as conn:
            row = retry_job(conn, job_id)
        if row is None:
            raise HTTPException(409, "Only failed ingestion jobs can be retried.")
        return row

    @app.post(
        "/v1/ingestions/{job_id}/cancel", response_model=IngestionJob, dependencies=[Depends(require_admin)]
    )
    def cancel(job_id: UUID):
        with get_connection(settings) as conn:
            row = cancel_job(conn, job_id)
        if row is None:
            raise HTTPException(409, "Only queued ingestion jobs can be cancelled.")
        return row

    return app


app = create_app()

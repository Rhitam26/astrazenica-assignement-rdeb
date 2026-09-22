"""Public contracts for the protected PDF-ingestion API."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

IngestionStatus = Literal["queued", "parsing", "chunked", "embedding", "completed", "failed", "cancelled"]


class IngestionJob(BaseModel):
    job_id: UUID
    doc_id: str
    original_filename: str
    status: IngestionStatus
    attempts: int
    total_chunks: int
    embedded_chunks: int
    skipped_chunks: int
    error_type: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class IngestionPage(BaseModel):
    items: list[IngestionJob]
    next_offset: int | None = None


class IngestionChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    content_hash: str
    heading_path: list[str]
    chapter_title: str | None
    section_title: str | None
    subsection_title: str | None
    page_numbers: list[int]
    page_start: int | None
    page_end: int | None
    content_type: Literal["text", "table"]
    table_id: str | None
    text: str
    token_count: int | None
    embedding_status: Literal["pending", "embedded", "skipped"]


class IngestionChunkPage(BaseModel):
    items: list[IngestionChunk]
    next_offset: int | None = None

"""Validated database, graph, and HTTP contracts."""

import re
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

Route = Literal["direct", "clarify", "simple_rag", "agentic_rag"]
CHUNK_PREVIEW_CHARS = 400


class MetadataFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_ids: list[str] | None = None
    doc_titles: list[str] | None = None
    chapter_title: str | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    content_type: Literal["text", "table"] | None = None
    table_id: str | None = None
    page_numbers: list[int] | None = None


class Citation(BaseModel):
    marker: int = Field(ge=1)
    chunk_id: UUID
    doc_id: str
    doc_title: str
    source_path: str | None
    heading_path: list[str]
    page_numbers: list[int]
    page_start: int | None
    page_end: int | None
    chapter_title: str | None
    section_title: str | None
    subsection_title: str | None
    content_type: Literal["text", "table"]
    table_id: str | None
    score: float
    text_preview: str


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    content_hash: str
    doc_id: str
    doc_title: str
    source_path: str | None
    heading_path: list[str]
    chapter_title: str | None
    section_title: str | None
    subsection_title: str | None
    page_numbers: list[int]
    page_start: int | None
    page_end: int | None
    content_type: Literal["text", "table"]
    table_id: str | None
    chunk_index: int
    text: str
    score: float

    def citation(self, marker: int) -> Citation:
        text = re.sub(r"\s+", " ", self.text).strip()
        if len(text) > CHUNK_PREVIEW_CHARS:
            text = text[:CHUNK_PREVIEW_CHARS].rstrip() + "…"
        return Citation.model_validate({**self.model_dump(), "marker": marker, "text_preview": text})


class ContextDecision(BaseModel):
    standalone_question: str
    unresolved: bool
    clarification: str


class RouteDecision(BaseModel):
    route: Route
    reason: str
    confidence: float = Field(ge=0, le=1)


class ContextRouteDecision(ContextDecision, RouteDecision):
    """Resolve the latest message and classify the resolved question in one call."""


class EvidenceDecision(BaseModel):
    sufficient: bool
    relevant_markers: list[int]
    missing_information: str


class GeneratedAnswer(BaseModel):
    answer: str
    abstained: bool


class GroundedBlock(BaseModel):
    text: str
    evidence_markers: list[int]


class GroundedAnswer(BaseModel):
    blocks: list[GroundedBlock]
    abstained: bool


class SupportDecision(BaseModel):
    supported: bool
    reason: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID | None = None
    message: str = Field(min_length=1, max_length=100000)

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    workflow: Route
    route_reason: str
    citations: list[Citation]
    abstained: bool
    latency_ms: int


class SavedAnswer(BaseModel):
    version: int = 1
    workflow: Route
    route_reason: str
    citations: list[Citation]
    abstained: bool


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: SavedAnswer | None = None


class ConversationHistory(BaseModel):
    conversation_id: UUID
    messages: list[ConversationMessage]


class ConversationSummary(BaseModel):
    conversation_id: UUID
    title: str
    updated_at: datetime
    message_count: int


class ConversationPage(BaseModel):
    items: list[ConversationSummary]
    next_offset: int | None = None

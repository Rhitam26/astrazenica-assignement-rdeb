"""Application-owned graph lifecycle and cross-worker conversation serialization."""

from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable
from uuid import UUID, uuid4

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.assistant.conversations import checkpoint_candidates
from src.assistant.models import (
    ChatResponse,
    ConversationHistory,
    ConversationMessage,
    ConversationPage,
    ConversationSummary,
    SavedAnswer,
)
from src.assistant.provider import OpenAIModel
from src.assistant.retrieval import PgRetriever, retrieval_pool
from src.assistant.workflow import RunContext, create_graph
from src.shared.config import Settings
from src.shared.database import check_ready
from src.shared.embedder import Embedder
from src.shared.observability import Telemetry, preview


class ConversationBusy(Exception):
    pass


class ConversationNotFound(Exception):
    pass


@contextmanager
def conversation_lock(settings: Settings, conversation_id: str):
    # A transaction lock is released automatically on failure/disconnect.
    with psycopg.connect(settings.database_url) as conn:
        lock_row = conn.execute(
            "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))", (conversation_id,)
        ).fetchone()
        if not lock_row or not lock_row[0]:
            raise ConversationBusy("Conversation is already processing a request")
        yield


@dataclass
class ChatService:
    settings: Settings
    graph: Any
    lock: Callable[[str], AbstractContextManager]
    ready_check: Callable[[], None]
    telemetry: Telemetry | None = None
    history_candidates: Callable[[int, int], list[dict]] | None = None

    def history(self, cid: UUID, checkpoint_id: str | None = None) -> ConversationHistory:
        config = {"configurable": {"thread_id": str(cid)}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        state = self.graph.get_state(config)
        messages = []
        pending = None
        for message in state.values.get("messages", []):
            if not isinstance(message.content, str):
                continue
            if message.type == "human":
                pending = ConversationMessage(role="user", content=message.content)
            elif message.type == "ai" and pending is not None and not message.tool_calls:
                raw = message.additional_kwargs.get("saved_answer")
                metadata = SavedAnswer.model_validate(raw) if raw and raw.get("version") == 1 else None
                messages.extend(
                    [
                        pending,
                        ConversationMessage(role="assistant", content=message.content, metadata=metadata),
                    ]
                )
                pending = None
        if not messages:
            raise ConversationNotFound()
        return ConversationHistory(conversation_id=cid, messages=messages)

    def conversations(self, offset: int = 0, limit: int = 20) -> ConversationPage:
        items: list[ConversationSummary] = []
        if self.history_candidates is None:
            return ConversationPage(items=[])
        while len(items) < limit:
            rows = self.history_candidates(offset, limit)
            if not rows:
                return ConversationPage(items=items)
            for row in rows:
                offset += 1
                try:
                    cid = UUID(row["thread_id"])
                except ValueError:
                    continue
                try:
                    history = self.history(cid, row["checkpoint_id"])
                except ConversationNotFound:
                    continue
                title = " ".join(history.messages[0].content.split())
                items.append(
                    ConversationSummary(
                        conversation_id=cid,
                        title=title[:60],
                        updated_at=row["updated_at"],
                        message_count=len(history.messages),
                    )
                )
                if len(items) == limit:
                    return ConversationPage(items=items, next_offset=offset)
        return ConversationPage(items=items)

    def ready(self) -> None:
        self.ready_check()

    def chat(self, message: str, conversation_id: UUID | None = None) -> ChatResponse:
        conversation_id = conversation_id or uuid4()
        start = perf_counter()
        telemetry = self.telemetry or Telemetry(self.settings)
        with telemetry.trace(
            "chat-request",
            session_id=str(conversation_id),
            metadata={"message": preview(message, self.settings.langfuse_preview_chars)},
        ) as trace:
            with self.lock(str(conversation_id)):
                state = self.graph.invoke(
                    {"user_message": message},
                    {
                        "configurable": {"thread_id": str(conversation_id)},
                        "recursion_limit": self.settings.langgraph_recursion_limit,
                    },
                    context=RunContext(),
                )
            latency_ms = round((perf_counter() - start) * 1000)
            trace.update(
                output={"answer": preview(state["answer"], self.settings.langfuse_preview_chars)},
                metadata={
                    "workflow": state["route"],
                    "route_reason": preview(state.get("route_reason", ""), 200),
                    "search_count": state.get("tool_call_count", 0),
                    "citation_count": len(state.get("citations", [])),
                    "abstained": state["abstained"],
                    "latency_ms": latency_ms,
                },
            )
        return ChatResponse(
            conversation_id=conversation_id,
            answer=state["answer"],
            workflow=state["route"],
            route_reason=state["route_reason"],
            citations=state["citations"],
            abstained=state["abstained"],
            latency_ms=latency_ms,
        )


@contextmanager
def production_service(settings: Settings):
    def configure(conn):
        conn.execute("SET search_path TO conversation, public")

    with (
        ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=8,
            timeout=5,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=configure,
            open=True,
        ) as pool,
        retrieval_pool(settings) as search_pool,
    ):
        pool.wait(timeout=10)
        search_pool.wait(timeout=10)
        telemetry = Telemetry(settings)
        embedder = Embedder(settings, telemetry)
        try:
            graph = create_graph(
                settings,
                OpenAIModel(settings, telemetry),
                PgRetriever(settings, embedder, telemetry, search_pool.connection),
                PostgresSaver(pool),
                telemetry,
            )
            yield ChatService(
                settings,
                graph,
                lambda cid: conversation_lock(settings, cid),
                lambda: check_ready(settings),
                telemetry,
                history_candidates=lambda offset, limit: checkpoint_candidates(pool, offset, limit),
            )
        finally:
            embedder.close()
            telemetry.shutdown()

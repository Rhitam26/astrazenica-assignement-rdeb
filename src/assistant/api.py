"""Non-streaming HTTP boundary. Blocking graph/psycopg work runs in FastAPI's thread pool."""

import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.assistant.logging_config import configure_logging, conversation_id, request_id
from src.assistant.models import ChatRequest, ChatResponse, ConversationHistory, ConversationPage
from src.assistant.service import ChatService, ConversationBusy, ConversationNotFound, production_service
from src.shared.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: ChatService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        if service is not None:
            app.state.service = service
            yield
        else:
            with production_service(settings) as live_service:
                app.state.service = live_service
                yield

    app = FastAPI(
        title="DocuSense",
        version="1.0.0",
        description="Document-grounded conversational RAG with persistent threads and database citations.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def identify_request(request: Request, call_next):
        rid = str(uuid4())
        token = request_id.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id.reset(token)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default response echoes rejected input; do not leak user content.
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request. Check message, conversation_id, and pagination."},
        )

    @app.get("/health", summary="Process liveness")
    def health():
        return {"status": "ok"}

    @app.get("/ready", summary="Database, schema, index, and checkpoint readiness")
    def ready(request: Request):
        try:
            request.app.state.service.ready()
        except Exception as exc:
            logger.warning("readiness_failed", extra={"error_type": type(exc).__name__})
            raise HTTPException(503, "Knowledge service is not ready.") from None
        return {"status": "ready"}

    @app.get("/v1/conversations", response_model=ConversationPage)
    def conversations(request: Request, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
        try:
            return request.app.state.service.conversations(offset, limit)
        except Exception as exc:
            logger.error("history_list_failed", extra={"error_type": type(exc).__name__})
            raise HTTPException(503, "Conversation history is temporarily unavailable.") from None

    @app.get("/v1/conversations/{cid}", response_model=ConversationHistory)
    def history(cid: UUID, request: Request):
        try:
            return request.app.state.service.history(cid)
        except ConversationNotFound:
            raise HTTPException(404, "Conversation not found.") from None
        except Exception as exc:
            logger.error("history_load_failed", extra={"error_type": type(exc).__name__})
            raise HTTPException(503, "Conversation history is temporarily unavailable.") from None

    @app.post(
        "/v1/chat",
        response_model=ChatResponse,
        summary="Ask a grounded question or continue a conversation",
        responses={
            409: {"description": "Conversation busy"},
            422: {"description": "Invalid input"},
            503: {"description": "Dependency unavailable"},
        },
    )
    def chat(body: ChatRequest, request: Request):
        if len(body.message) > settings.max_message_chars:
            raise HTTPException(422, f"Message exceeds {settings.max_message_chars} characters.")
        cid = body.conversation_id or uuid4()
        token = conversation_id.set(str(cid))
        try:
            return request.app.state.service.chat(body.message, cid)
        except ConversationBusy:
            raise HTTPException(
                409, "This conversation is processing another request. Try again shortly."
            ) from None
        except Exception as exc:
            logger.error("chat_failed", extra={"error_type": type(exc).__name__})
            raise HTTPException(
                503, "The knowledge service is temporarily unavailable. Please try again later."
            ) from None
        finally:
            conversation_id.reset(token)

    return app


app = create_app()

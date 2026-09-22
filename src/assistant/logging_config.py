"""Small structured-log adapter, ready for a future telemetry sink."""

import json
import logging
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="")
conversation_id: ContextVar[str] = ContextVar("conversation_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": request_id.get(),
            "conversation_id": conversation_id.get(),
        }
        for key in ("workflow", "search_count", "abstained", "error_type"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        return json.dumps(data)


def configure_logging(level: str):
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    # Provider HTTP logging can contain request URLs; keep it quiet by default.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

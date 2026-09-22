"""Optional, metadata-first Langfuse instrumentation.

The application must remain useful when Langfuse is not configured or when the
telemetry service is unavailable.  This module therefore exposes small no-op
compatible context managers and keeps the SDK import lazy.
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def preview(value: Any, limit: int = 1000) -> str:
    """Return a bounded, lightly redacted representation for telemetry."""
    text = str(value)
    text = re.sub(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    return text[:limit] + ("…" if len(text) > limit else "")


def value_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


class _NullObservation:
    trace_id = ""

    def update(self, **kwargs: Any) -> None:
        return None

    def score(self, **kwargs: Any) -> None:
        return None


@dataclass
class Telemetry:
    """Thin adapter around the Langfuse v4 SDK with a safe disabled mode."""

    settings: Any

    def __post_init__(self) -> None:
        self.client = None
        if not getattr(self.settings, "langfuse_enabled", False):
            return
        try:
            from langfuse import get_client

            self.client = get_client()
        except Exception as exc:  # telemetry must never prevent application startup
            logger.warning("langfuse_unavailable", extra={"error_type": type(exc).__name__})

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _metadata(self, metadata: dict[str, Any] | None) -> dict[str, str]:
        result = {}
        for key, value in (metadata or {}).items():
            result[str(key)] = preview(value, 200)
        return result

    @staticmethod
    def _update(observation: Any, **kwargs: Any) -> None:
        try:
            observation.update(**kwargs)
        except Exception:
            logger.debug("langfuse_observation_update_failed")

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        kind: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Iterator[Any]:
        if not self.client:
            yield _NullObservation()
            return
        started = perf_counter()
        application_error = False
        try:
            kwargs: dict[str, Any] = {
                "as_type": kind,
                "name": name,
                "metadata": self._metadata(metadata),
            }
            if input is not None:
                kwargs["input"] = input
            if model:
                kwargs["model"] = model
            with self.client.start_as_current_observation(**kwargs) as observation:
                try:
                    yield observation
                except Exception as exc:
                    application_error = True
                    self._update(
                        observation,
                        level="ERROR",
                        status_message=type(exc).__name__,
                        metadata={"error_type": type(exc).__name__},
                    )
                    raise
                finally:
                    self._update(
                        observation, metadata={"duration_ms": round((perf_counter() - started) * 1000, 2)}
                    )
        except Exception as exc:
            if application_error:
                raise
            # SDK/network errors are deliberately swallowed; application work has priority.
            if not isinstance(exc, (ValueError, TypeError)):
                logger.debug("langfuse_observation_failed", extra={"error_type": type(exc).__name__})
            yield _NullObservation()

    @contextmanager
    def trace(self, name: str, *, session_id: str, metadata: dict[str, Any] | None = None) -> Iterator[Any]:
        if not self.client:
            yield _NullObservation()
            return
        application_error = False
        try:
            from langfuse import propagate_attributes

            with ExitStack() as stack:
                stack.enter_context(
                    propagate_attributes(
                        session_id=session_id,
                        environment=getattr(self.settings, "app_env", None),
                        metadata=self._metadata(metadata),
                    )
                )
                with self.observation(name, metadata=metadata) as root:
                    try:
                        yield root
                    except Exception:
                        application_error = True
                        raise
        except Exception as exc:
            if application_error:
                raise
            logger.debug("langfuse_trace_failed", extra={"error_type": type(exc).__name__})
            yield _NullObservation()

    def score(self, observation: Any, *, name: str, value: float, comment: str = "") -> None:
        try:
            observation.score(
                name=name, value=float(value), data_type="NUMERIC", comment=preview(comment, 200)
            )
        except Exception:
            logger.debug("langfuse_score_failed", extra={"score": name})

    def flush(self) -> None:
        if self.client:
            try:
                self.client.flush()
            except Exception:
                logger.debug("langfuse_flush_failed")

    def shutdown(self) -> None:
        if self.client:
            try:
                self.client.shutdown()
            except Exception:
                logger.debug("langfuse_shutdown_failed")

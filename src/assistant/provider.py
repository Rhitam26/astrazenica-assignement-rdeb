"""Small provider boundary replaceable by deterministic test doubles."""

import json
from typing import Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.assistant.prompts import PROMPTS
from src.shared.config import Settings
from src.shared.observability import Telemetry, preview
from src.shared.reliability import retry_policy

T = TypeVar("T", bound=BaseModel)


class Model(Protocol):
    def structured(self, task: str, payload: dict, schema: type[T]) -> T: ...
    def agent(self, messages: list[BaseMessage], tools: list[BaseTool]) -> AIMessage: ...


class OpenAIModel:
    def __init__(self, settings: Settings, telemetry: Telemetry | None = None):
        self.settings = settings
        self.telemetry = telemetry or Telemetry(settings)
        self.client = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_retries=0,
            timeout=settings.provider_timeout,
            max_completion_tokens=settings.max_output_tokens,
        )

    def structured(self, task: str, payload: dict[str, Any], schema: type[T]) -> T:
        with self.telemetry.observation(
            f"llm.{task}",
            kind="generation",
            model=self.settings.llm_model,
            input={"task": task, "payload": preview(payload, self.settings.langfuse_preview_chars)},
            metadata={"schema": schema.__name__, "temperature": self.settings.llm_temperature},
        ) as generation:
            runnable = self.client.with_structured_output(schema, method="json_schema", strict=True)
            result = retry_policy(self.settings)(
                runnable.invoke,
                [
                    SystemMessage(PROMPTS[task]),
                    HumanMessage(json.dumps(payload, ensure_ascii=False, default=str)),
                ],
            )
            parsed = schema.model_validate(result)
            usage = getattr(result, "usage_metadata", None) or getattr(result, "response_metadata", {}).get(
                "token_usage"
            )
            generation.update(
                output=preview(parsed.model_dump(), self.settings.langfuse_preview_chars),
                metadata={"usage": usage or {}},
            )
            return parsed

    def agent(self, messages: list[BaseMessage], tools: list[BaseTool]) -> AIMessage:
        with self.telemetry.observation(
            "llm.agent",
            kind="generation",
            model=self.settings.llm_model,
            input={"message_count": len(messages), "tool_count": len(tools)},
        ) as generation:
            result = retry_policy(self.settings)(
                self.client.bind_tools(tools, parallel_tool_calls=False).invoke,
                [SystemMessage(PROMPTS["agent"]), *messages],
            )
            if not isinstance(result, AIMessage):
                raise ValueError("Invalid agent response")
            generation.update(
                output=preview(result.content, self.settings.langfuse_preview_chars),
                metadata={"tool_calls": len(result.tool_calls), "usage": result.usage_metadata or {}},
            )
            return result

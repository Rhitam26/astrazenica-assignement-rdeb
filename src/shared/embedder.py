"""Existing OpenAI embedding adapter, with strict dimensions and bounded retries."""

import math

from openai import OpenAI

from src.shared.config import Settings, get_settings
from src.shared.observability import Telemetry
from src.shared.reliability import retry_policy


class Embedder:
    def __init__(self, settings: Settings | None = None, telemetry: Telemetry | None = None) -> None:
        self.settings = settings or get_settings()
        self.telemetry = telemetry or Telemetry(self.settings)
        self._client = OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            max_retries=0,
            timeout=self.settings.provider_timeout,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        telemetry = getattr(self, "telemetry", Telemetry(self.settings))
        for start in range(0, len(texts), self.settings.embed_batch_size):
            batch = texts[start : start + self.settings.embed_batch_size]
            with telemetry.observation(
                "embedding.create",
                kind="generation",
                model=self.settings.embedding_model,
                input={"batch_size": len(batch), "text_lengths": [len(text) for text in batch]},
                metadata={"dimensions": self.settings.embedding_dims},
            ) as generation:
                response = retry_policy(self.settings)(
                    self._client.embeddings.create,
                    model=self.settings.embedding_model,
                    input=batch,
                    dimensions=self.settings.embedding_dims,
                )
                generation.update(
                    metadata={
                        "batch_size": len(batch),
                        "usage": getattr(response, "usage", None) or {},
                    }
                )
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(batch))):
                raise ValueError("Embedding response does not match input batch")
            for item in items:
                if len(item.embedding) != self.settings.embedding_dims or not all(
                    map(math.isfinite, item.embedding)
                ):
                    raise ValueError("Invalid embedding dimensions or values")
                vectors.append(item.embedding)
        return vectors

    def close(self) -> None:
        self._client.close()

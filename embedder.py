"""OpenAI embedding client with batching and retry handling."""
from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._batch_size = settings.embed_batch_size

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        # OpenAI preserves input order in the response payload.
        return [item.embedding for item in response.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
            logger.info("Embedded %d/%d chunks", min(start + self._batch_size, len(texts)), len(texts))
        return vectors

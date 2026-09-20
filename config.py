"""Centralized configuration for the RAG pipeline.

Values are loaded from environment variables / a local .env file so nothing
is hard-coded and no secrets are committed to source control.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI ---
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072
    chat_model: str = "gpt-4o-mini"  # used later by the query-time app

    # --- Postgres ---
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_db: str = "rag_knowledge_base"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # --- Chunking / embedding ---
    max_chunk_tokens: int = 512
    embed_batch_size: int = 128

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

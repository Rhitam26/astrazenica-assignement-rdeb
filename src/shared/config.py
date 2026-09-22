"""Environment configuration shared by ingestion and query-time services."""

from functools import lru_cache

from psycopg.conninfo import make_conninfo
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)
    openai_api_key: SecretStr = SecretStr("")
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = Field(3072, ge=3072, le=3072)
    llm_model: str = Field("gpt-4o-mini", validation_alias=AliasChoices("LLM_MODEL", "CHAT_MODEL"))
    llm_temperature: float = Field(0, ge=0, le=2)
    postgres_user: str = "rag_user"
    postgres_password: SecretStr = SecretStr("rag_password")
    postgres_db: str = "rag_knowledge_base"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    db_statement_timeout_ms: int = Field(10000, gt=0)
    db_connect_timeout: int = Field(5, gt=0)
    provider_timeout: float = Field(30, gt=0)
    provider_max_attempts: int = Field(3, ge=1, le=5)
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"
    cors_origins: list[str] = []
    pdf_dir: str = "./pdfs"
    ingestion_api_key: SecretStr = SecretStr("")
    ingestion_upload_dir: str = "./uploads"
    ingestion_max_upload_bytes: int = Field(50 * 1024 * 1024, ge=1)
    ingestion_worker_poll_seconds: float = Field(2, gt=0)
    ingestion_job_lease_seconds: int = Field(900, ge=60)
    ingestion_job_max_attempts: int = Field(3, ge=1, le=10)
    max_chunk_tokens: int = Field(512, gt=0)
    embed_batch_size: int = Field(128, ge=1, le=2048)
    retrieval_top_k: int = Field(5, ge=1, le=20)
    retrieval_candidate_pool: int = Field(25, ge=20, le=200)
    retrieval_score_threshold: float = Field(0.0, ge=-1, le=1)
    router_confidence_threshold: float = Field(0.6, ge=0, le=1)
    agent_max_tool_calls: int = Field(4, ge=1, le=8)
    langgraph_recursion_limit: int = Field(32, ge=16)
    history_turns: int = Field(6, ge=1, le=20)
    history_token_budget: int = Field(6000, ge=100)
    max_message_chars: int = Field(8000, ge=1)
    max_output_tokens: int = Field(2000, ge=100)
    langfuse_enabled: bool = False
    langfuse_preview_chars: int = Field(1000, ge=100, le=10000)

    @property
    def database_url(self) -> str:
        return make_conninfo(
            host=self.postgres_host,
            port=self.postgres_port,
            user=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            dbname=self.postgres_db,
            connect_timeout=self.db_connect_timeout,
            options=f"-c statement_timeout={self.db_statement_timeout_ms}",
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Repeatable schema setup required by the standalone ingestion service."""

from pathlib import Path

from src.shared.config import Settings, get_settings
from src.shared.database import get_connection


def migrate(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    schema = Path(__file__).resolve().parents[2] / "sql/schema.sql"
    with get_connection(settings) as conn:
        conn.execute(schema.read_text())


if __name__ == "__main__":
    migrate()

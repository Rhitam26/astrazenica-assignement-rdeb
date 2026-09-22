FROM python:3.13-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements/base.txt ./requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt
RUN useradd --create-home app

FROM base AS assistant
COPY requirements/assistant.txt ./requirements/assistant.txt
RUN pip install --no-cache-dir -r requirements/assistant.txt
COPY --chown=app:app src/shared ./src/shared
COPY --chown=app:app src/assistant ./src/assistant
COPY --chown=app:app sql ./sql
USER app
CMD ["sh", "-c", "python -m src.shared.migrate && exec uvicorn src.assistant.api:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]

FROM base AS ingestion
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements/ingestion.txt ./requirements/ingestion.txt
RUN pip install --no-cache-dir -r requirements/ingestion.txt
COPY --chown=app:app src/shared ./src/shared
COPY --chown=app:app src/ingestion ./src/ingestion
COPY --chown=app:app sql ./sql
USER app
CMD ["python", "-m", "src.ingestion.ingest"]

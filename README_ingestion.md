# Ingestion quick reference

The Docling + HybridChunker pipeline lives in `src/ingestion/`; shared infrastructure is in `src/shared/` and the chat/agent runtime is isolated in `src/assistant/`.
From the project root, with the virtual environment active and `.env` configured:

```bash
docker compose up -d postgres
python -m src.shared.migrate
python -m src.ingestion.ingest --source-dir ./pdfs
```

`PDF_DIR` defaults to `./pdfs`; the CLI option overrides it. The batch command is
still available, and a separate protected ingestion API is available for uploaded PDFs.

## Asynchronous ingestion API

Set a long random `INGESTION_API_KEY` in `.env`, then start the dedicated API and worker:

```bash
docker compose --profile ingestion up -d ingestion-api ingestion-worker
```

The API listens on `http://localhost:8001`. It accepts only PDFs, stores them in the
managed `ingestion_uploads` volume, and returns a job immediately. The worker then runs
Docling parsing, structural chunking, deduplicated embeddings, and pgvector insertion.

```bash
curl -X POST http://localhost:8001/v1/ingestions \
  -H "X-Ingestion-Key: $INGESTION_API_KEY" \
  -F "file=@./pdfs/example.pdf"

curl http://localhost:8001/v1/ingestions/<job-id> \
  -H "X-Ingestion-Key: $INGESTION_API_KEY"

curl "http://localhost:8001/v1/ingestions/<job-id>/chunks?limit=20" \
  -H "X-Ingestion-Key: $INGESTION_API_KEY"
```

`POST /v1/ingestions/{job_id}/retry` requeues failed jobs. `POST /v1/ingestions/{job_id}/cancel`
cancels queued jobs. Uploads are idempotent by whole-file SHA-256: resubmitting identical
PDF bytes returns the existing job. The API never accepts caller-provided chunks or vectors,
and it never returns embedding values.

The pipeline extracts headings, all contributing pages, content type and table
identity before checking hashes in PostgreSQL. Only new hashes reach the embedding
API. Re-running the unchanged supplied corpus preserves all 146 existing chunks
and creates zero new embeddings. PDFs are still parsed on each run.

Metadata follows `sql/schema.sql`, including `heading_path`, `page_numbers`,
`page_start`, `page_end`, `table_id`, and all embedding provenance fields.
`text-embedding-3-large` and 3072 dimensions remain unchanged. The HNSW index uses
a half-vector expression while the stored embeddings remain full precision.

Model changes require explicit re-indexing; dimension changes also require schema
and code changes. The existing text-only global hash can collapse identical text
across locations, and deleted/changed documents do not remove old chunks. Run one
ingestion process at a time.

See [README.md](README.md) for architecture, setup, Docker ingestion, tests,
security considerations, and the full known limitations.

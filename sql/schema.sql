-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document.
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per chunk.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash      TEXT NOT NULL UNIQUE,

    doc_id            TEXT NOT NULL
                      REFERENCES documents(doc_id)
                      ON DELETE CASCADE,

    doc_title         TEXT NOT NULL,

    heading_path      JSONB NOT NULL DEFAULT '[]'::jsonb,

    chapter_title     TEXT,
    section_title     TEXT,
    subsection_title  TEXT,

    page_numbers      INTEGER[] NOT NULL DEFAULT '{}',
    page_start        INTEGER,
    page_end          INTEGER,

    content_type      TEXT NOT NULL DEFAULT 'text'
                      CHECK (content_type IN ('text', 'table')),

    table_id          TEXT,

    chunk_index       INTEGER NOT NULL,
    text              TEXT NOT NULL,

    token_count       INTEGER,

    embedding_model   TEXT NOT NULL,
    embedding_dims    INTEGER NOT NULL,

    embedding         vector(3072) NOT NULL,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Vector similarity index
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Metadata indexes
CREATE INDEX IF NOT EXISTS idx_chunks_section
    ON chunks (section_title);

CREATE INDEX IF NOT EXISTS idx_chunks_subsection
    ON chunks (subsection_title);

CREATE INDEX IF NOT EXISTS idx_chunks_page_numbers
    ON chunks USING GIN (page_numbers);

CREATE INDEX IF NOT EXISTS idx_chunks_heading_path
    ON chunks USING GIN (heading_path);

CREATE INDEX IF NOT EXISTS idx_chunks_table_id
    ON chunks (table_id);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON chunks (doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_content_type
    ON chunks (content_type);

CREATE INDEX IF NOT EXISTS idx_chunks_chapter
    ON chunks (chapter_title);

-- Durable asynchronous PDF ingestion jobs and source-reviewed chunk snapshots.
CREATE SCHEMA IF NOT EXISTS ingestion;

CREATE TABLE IF NOT EXISTS ingestion.jobs (
    job_id              UUID PRIMARY KEY,
    doc_id              TEXT NOT NULL UNIQUE,
    original_filename   TEXT NOT NULL,
    source_path         TEXT NOT NULL,
    file_hash           TEXT NOT NULL UNIQUE,
    status              TEXT NOT NULL CHECK (status IN ('queued', 'parsing', 'chunked', 'embedding', 'completed', 'failed', 'cancelled')),
    attempts            INTEGER NOT NULL DEFAULT 0,
    lease_expires_at    TIMESTAMPTZ,
    total_chunks        INTEGER NOT NULL DEFAULT 0,
    embedded_chunks     INTEGER NOT NULL DEFAULT 0,
    skipped_chunks      INTEGER NOT NULL DEFAULT 0,
    error_type          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_queue
    ON ingestion.jobs (status, created_at);

CREATE TABLE IF NOT EXISTS ingestion.job_chunks (
    job_id              UUID NOT NULL REFERENCES ingestion.jobs(job_id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    content_hash        TEXT NOT NULL,
    heading_path        JSONB NOT NULL DEFAULT '[]'::jsonb,
    chapter_title       TEXT,
    section_title       TEXT,
    subsection_title    TEXT,
    page_numbers        INTEGER[] NOT NULL DEFAULT '{}',
    page_start          INTEGER,
    page_end            INTEGER,
    content_type        TEXT NOT NULL CHECK (content_type IN ('text', 'table')),
    table_id            TEXT,
    text                TEXT NOT NULL,
    token_count         INTEGER,
    embedding_status    TEXT NOT NULL DEFAULT 'pending' CHECK (embedding_status IN ('pending', 'embedded', 'skipped')),
    PRIMARY KEY (job_id, chunk_index)
);

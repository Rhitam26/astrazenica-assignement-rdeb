from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from src.ingestion import ingest
from src.shared.config import Settings
from src.shared.embedder import Embedder
from tests.fakes import chunk


def test_unchanged_chunks_never_embedded(monkeypatch, tmp_path):
    (tmp_path / "test.pdf").touch()
    record = chunk()
    processed = []
    monkeypatch.setattr(ingest, "chunk_pdf", lambda *a, **k: [record])
    monkeypatch.setattr(ingest, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(ingest, "get_connection", lambda: nullcontext(SimpleNamespace(commit=lambda: None)))
    monkeypatch.setattr(ingest, "validate_embeddings", lambda *a: None)
    monkeypatch.setattr(
        ingest,
        "persist_records",
        lambda conn, records, settings, embedder, source_path: (
            processed.append(records) or set(),
            {record.content_hash},
            1,
        ),
    )
    ingest.ingest_directory(tmp_path)
    assert processed == [[record]]


def test_new_duplicate_hash_embedded_once_with_metadata(monkeypatch, tmp_path):
    (tmp_path / "test.pdf").touch()
    record = SimpleNamespace(**chunk().model_dump(), token_count=12)
    calls = []
    monkeypatch.setattr(ingest, "chunk_pdf", lambda *a, **k: [record, record])
    monkeypatch.setattr(ingest, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(ingest, "get_connection", lambda: nullcontext(SimpleNamespace(commit=lambda: None)))
    monkeypatch.setattr(ingest, "validate_embeddings", lambda *a: None)
    monkeypatch.setattr(
        ingest,
        "persist_records",
        lambda conn, records, settings, embedder, source_path: (
            calls.extend(records) or {record.content_hash},
            set(),
            1,
        ),
    )
    ingest.ingest_directory(tmp_path)
    assert calls == [record, record]


def test_persist_records_embeds_new_unique_chunks_once(monkeypatch):
    from src.ingestion import processor

    record = SimpleNamespace(**chunk().model_dump(), token_count=12)
    writes, calls = [], []
    monkeypatch.setattr(processor, "upsert_document", lambda *a, **k: None)
    monkeypatch.setattr(processor, "get_existing_hashes", lambda *a: set())
    monkeypatch.setattr(processor, "insert_chunk", lambda conn, values: writes.append(values))
    embedder = SimpleNamespace(embed_texts=lambda texts: calls.append(texts) or [[1.0] * 3072 for _ in texts])

    embedded, skipped, skipped_count = processor.persist_records(
        object(), [record, record], Settings(_env_file=None), embedder, "upload.pdf"
    )
    assert embedded == {record.content_hash}
    assert skipped == set() and skipped_count == 1
    assert calls == [[record.text]]
    assert writes[0]["page_numbers"] == [4, 5]
    assert writes[0]["table_id"] == "#/tables/1"
    assert writes[0]["heading_path"].obj == ["Indexes", "ANN"]


def test_wrong_embedding_dimensions_fail():
    embedder = Embedder.__new__(Embedder)
    embedder.settings = Settings(_env_file=None)
    embedder._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **k: SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])
        )
    )
    with pytest.raises(ValueError, match="dimensions"):
        embedder.embed_texts(["hello"])


def test_chunk_metadata_helpers():
    from src.ingestion.chunker import _get_heading_path, _get_page_numbers, _get_page_range, _get_table_id

    item = SimpleNamespace(
        prov=[SimpleNamespace(page_no=3), SimpleNamespace(page_no=5)], self_ref="#/tables/2"
    )
    assert _get_page_numbers([item, item]) == [3, 5]
    assert _get_page_range([3, 5]) == (3, 5)
    assert _get_page_range([]) == (None, None)
    assert _get_table_id(item) == "#/tables/2"
    assert _get_heading_path(SimpleNamespace(meta=SimpleNamespace(headings=[" Chapter ", "Section"]))) == [
        "Chapter",
        "Section",
    ]

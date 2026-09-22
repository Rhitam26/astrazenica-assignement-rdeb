"""Real PostgreSQL + graph + HTTP tests. Never uses paid providers or the production KB."""

import os
from contextlib import contextmanager
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import PoolTimeout

from src.assistant.api import create_app
from src.assistant.conversations import checkpoint_candidates
from src.assistant.models import MetadataFilters
from src.assistant.repository import retrieve_relevant_chunks
from src.assistant.retrieval import PgRetriever, retrieval_pool
from src.assistant.service import ChatService, ConversationBusy, conversation_lock
from src.assistant.workflow import create_graph
from src.ingestion.storage import insert_chunk, upsert_document
from src.shared.config import get_settings
from src.shared.database import check_ready, get_connection
from src.shared.migrate import migrate
from tests.fakes import DeterministicEmbedder, FakeModel, chunk

pytestmark = pytest.mark.integration


def test_retrieval_pool_reuse_timeout_and_close(database):
    settings, _ = database
    pool = retrieval_pool(settings)
    with pool:
        pool.wait(timeout=10)
        pool.resize(min_size=1, max_size=1)
        with pool.connection() as first:
            pid = first.info.backend_pid
            with pytest.raises(PoolTimeout):
                with pool.connection(timeout=0.05):
                    pass
        with pytest.raises(RuntimeError):
            with pool.connection() as conn:
                conn.execute("SELECT 1")
                raise RuntimeError("application failure")
        with pool.connection() as second:
            assert second.info.backend_pid == pid
            assert second.execute("SELECT 1").fetchone()[0] == 1
    assert pool.closed


@pytest.fixture(scope="module")
def database():
    name = os.getenv("TEST_DATABASE_NAME")
    if not name:
        pytest.skip("Set TEST_DATABASE_NAME to a dedicated disposable database")
    base = get_settings()
    if name == base.postgres_db or not name.endswith("_test"):
        pytest.fail("Test database must differ from application DB and end in _test")
    settings = base.model_copy(update={"postgres_db": name})
    migrate(settings)
    migrate(settings)
    record = chunk(content_hash=str(uuid4()), doc_id="integration-" + str(uuid4()))
    with get_connection(settings) as conn:
        upsert_document(conn, record.doc_id, record.doc_title, record.source_path)
        values = record.model_dump(exclude={"chunk_id", "score", "source_path"})
        values.update(
            heading_path=Jsonb(record.heading_path),
            token_count=10,
            embedding_model=settings.embedding_model,
            embedding_dims=3072,
            embedding=DeterministicEmbedder().embed_texts(["x"])[0],
        )
        insert_chunk(conn, values)
    yield settings, record
    with get_connection(settings) as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = %s", (record.doc_id,))


@contextmanager
def live_test_service(settings, model=None):
    with (
        psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn,
        retrieval_pool(settings) as pool,
    ):
        pool.wait(timeout=10)
        conn.execute("SET search_path TO conversation, public")
        graph = create_graph(
            settings,
            model or FakeModel(),
            PgRetriever(settings, DeterministicEmbedder(), connection_factory=pool.connection),
            PostgresSaver(conn),
        )
        yield ChatService(
            settings,
            graph,
            lambda cid: conversation_lock(settings, cid),
            lambda: check_ready(settings),
            history_candidates=lambda offset, limit: checkpoint_candidates(pool, offset, limit),
        )


def test_real_retrieval_metadata_and_filters(database):
    settings, record = database
    with get_connection(settings) as conn:
        rows = retrieve_relevant_chunks(
            conn,
            DeterministicEmbedder().embed_texts(["x"])[0],
            settings=settings,
            filters=MetadataFilters(doc_ids=[record.doc_id], page_numbers=[5], content_type="table"),
        )
        assert len(rows) == 1
        assert rows[0].page_numbers == [4, 5]
        assert rows[0].heading_path == record.heading_path
        assert rows[0].table_id == record.table_id
        assert rows[0].score == pytest.approx(1)
        assert not retrieve_relevant_chunks(
            conn,
            DeterministicEmbedder().embed_texts(["x"])[0],
            settings=settings,
            filters=MetadataFilters(section_title="'; DROP TABLE chunks; --"),
        )
        definitions = conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename='chunks'").fetchall()
        assert any("halfvec" in row[0] for row in definitions)


def test_end_to_end_api_restart_and_isolation(database):
    settings, _ = database
    cid = str(uuid4())
    with live_test_service(settings) as service, TestClient(create_app(settings, service)) as client:
        assert client.get("/ready").status_code == 200
        first = client.post("/v1/chat", json={"message": "Explain HNSW and IVF", "conversation_id": cid})
        assert first.status_code == 200
        assert first.json()["citations"][0]["table_id"] == "#/tables/1"
    model = FakeModel()
    with live_test_service(settings, model) as service, TestClient(create_app(settings, service)) as client:
        restored = client.get(f"/v1/conversations/{cid}")
        assert restored.status_code == 200
        messages = restored.json()["messages"]
        assert len(messages) == 2
        assert messages[1]["metadata"]["citations"][0]["text_preview"]
        offset = 0
        listed = []
        while True:
            page = client.get(f"/v1/conversations?offset={offset}").json()
            listed.extend(item["conversation_id"] for item in page["items"])
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert cid in listed
        follow = client.post(
            "/v1/chat", json={"message": "I did not understand, explain with example", "conversation_id": cid}
        )
        assert follow.status_code == 200
        assert len(client.get(f"/v1/conversations/{cid}").json()["messages"]) == 4
        assert any("HNSW" in item["content"] for item in model.payloads[0][1]["history"])
        assert follow.json()["workflow"] == "simple_rag"
        isolated = client.post("/v1/chat", json={"message": "Which one?"})
        assert isolated.json()["workflow"] == "clarify"


def test_conversation_lock(database):
    settings, _ = database
    cid = str(uuid4())
    with conversation_lock(settings, cid):
        with pytest.raises(ConversationBusy), conversation_lock(settings, cid):
            pass
        with conversation_lock(settings, str(uuid4())):
            pass

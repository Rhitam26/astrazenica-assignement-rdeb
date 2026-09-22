from uuid import uuid4

import httpx
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def empty_history_api(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("GET", url), json={"items": [], "next_offset": None}
        ),
    )


def test_resume_saved_conversation_and_preserve_it_on_load_error(monkeypatch):
    cid = str(uuid4())
    other = str(uuid4())
    requests = []

    def get(url, **kwargs):
        if url.endswith("/v1/conversations"):
            data = {
                "items": [
                    {"conversation_id": cid, "title": "Saved HNSW chat", "updated_at": "2026-01-01"},
                    {"conversation_id": other, "title": "Unavailable chat", "updated_at": "2026-01-01"},
                ],
                "next_offset": None,
            }
        elif url.endswith(cid):
            data = {
                "conversation_id": cid,
                "messages": [
                    {"role": "user", "content": "Explain HNSW"},
                    {"role": "assistant", "content": "HNSW uses a graph.", "metadata": None},
                ],
            }
        else:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, request=httpx.Request("GET", url), json=data)

    def post(url, **kwargs):
        requests.append(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "conversation_id": cid,
                "answer": "An example.",
                "workflow": "simple_rag",
                "latency_ms": 10,
                "citations": [],
                "abstained": False,
            },
        )

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    app = AppTest.from_file("../../src/assistant/ui.py").run()
    app.sidebar.button(key=f"conversation_{cid}").click().run()
    assert app.session_state.conversation_id == cid
    assert app.query_params["conversation_id"] == [cid]
    assert any("older response" in item.value for item in app.caption)
    app.sidebar.button(key=f"conversation_{other}").click().run()
    assert app.session_state.conversation_id == cid
    assert len(app.session_state.history) == 2
    app.chat_input[0].set_value("Give an example").run()
    assert requests[-1]["conversation_id"] == cid
    assert len(app.session_state.history) == 4
    assert not app.exception
    refreshed = AppTest.from_file("../../src/assistant/ui.py")
    refreshed.query_params["conversation_id"] = cid
    refreshed.run()
    assert refreshed.session_state.conversation_id == cid
    assert len(refreshed.session_state.history) == 2


def test_ui_calls_http_and_retains_thread(monkeypatch):
    cid = str(uuid4())
    requests = []

    def post(url, **kwargs):
        requests.append((url, kwargs["json"]))
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "conversation_id": cid,
                "answer": "Hello!",
                "workflow": "direct",
                "route_reason": "Greeting",
                "citations": [],
                "abstained": False,
                "latency_ms": 10,
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    app = AppTest.from_file("../../src/assistant/ui.py").run()
    app.chat_input[0].set_value("hi").run()
    assert not app.exception
    assert app.session_state.conversation_id == cid
    app.chat_input[0].set_value("thanks").run()
    assert requests[1][1]["conversation_id"] == cid
    assert requests[0][0].endswith("/v1/chat")
    app.sidebar.button[0].click().run()
    assert app.session_state.conversation_id is None
    assert not app.session_state.history


def test_ui_safe_error(monkeypatch):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("internal secret")

    monkeypatch.setattr(httpx, "post", fail)
    app = AppTest.from_file("../../src/assistant/ui.py").run()
    app.chat_input[0].set_value("hi").run()
    assert not app.exception
    assert "internal secret" not in app.error[0].value


def test_ui_displays_retrieved_text_preview(monkeypatch):
    cid = str(uuid4())

    def post(url, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "conversation_id": cid,
                "answer": "HNSW uses a graph. [1]",
                "workflow": "simple_rag",
                "route_reason": "Test",
                "citations": [
                    {
                        "marker": 1,
                        "chunk_id": str(uuid4()),
                        "doc_id": "vectors",
                        "doc_title": "Vector databases",
                        "source_path": "pdfs/vectors.pdf",
                        "heading_path": ["Indexes", "ANN"],
                        "page_numbers": [4],
                        "page_start": 4,
                        "page_end": 4,
                        "chapter_title": "Indexes",
                        "section_title": "ANN",
                        "subsection_title": None,
                        "content_type": "text",
                        "table_id": None,
                        "score": 0.9,
                        "text_preview": "HNSW uses a graph for approximate nearest-neighbor search.",
                    }
                ],
                "abstained": False,
                "latency_ms": 10,
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    app = AppTest.from_file("../../src/assistant/ui.py").run()
    app.chat_input[0].set_value("Explain HNSW").run()
    assert not app.exception
    assert any(
        "HNSW uses a graph for approximate nearest-neighbor search." in item.value for item in app.text
    )

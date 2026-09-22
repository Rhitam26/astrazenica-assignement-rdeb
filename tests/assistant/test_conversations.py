from contextlib import nullcontext
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.assistant.api import create_app
from src.assistant.service import ChatService
from src.assistant.workflow import create_graph
from src.shared.config import Settings
from tests.fakes import FakeModel, FakeRetriever


def test_list_restore_legacy_and_continue_after_service_recreation():
    settings = Settings(_env_file=None)
    saver = InMemorySaver()
    graph = create_graph(settings, FakeModel(), FakeRetriever(), saver)
    legacy, fresh, partial = uuid4(), uuid4(), uuid4()
    graph.update_state(
        {"configurable": {"thread_id": str(legacy)}},
        {"messages": [HumanMessage("Explain HNSW and IVF"), AIMessage("Both are ANN methods.")]},
        as_node="finish",
    )
    graph.update_state(
        {"configurable": {"thread_id": str(partial)}},
        {"user_message": "unfinished", "answer": "not completed"},
    )

    def candidates(offset, limit):
        rows = []
        for cid in (partial, fresh, legacy):
            snapshot = graph.get_state({"configurable": {"thread_id": str(cid)}})
            if snapshot.values:
                rows.append(
                    {
                        "thread_id": str(cid),
                        "checkpoint_id": snapshot.config["configurable"]["checkpoint_id"],
                        "updated_at": snapshot.created_at,
                    }
                )
        return rows[offset : offset + limit]

    service = ChatService(
        settings, graph, lambda _: nullcontext(), lambda: None, history_candidates=candidates
    )
    service.chat("Explain HNSW", fresh)
    # Rebuild the graph/service over the same persistent checkpoint adapter.
    service = ChatService(
        settings,
        create_graph(settings, FakeModel(), FakeRetriever(), saver),
        lambda _: nullcontext(),
        lambda: None,
        history_candidates=candidates,
    )
    with TestClient(create_app(settings, service)) as client:
        page = client.get("/v1/conversations?limit=1").json()
        assert page["items"][0]["conversation_id"] == str(fresh)
        assert page["items"][0]["message_count"] == 2
        page2 = client.get(f"/v1/conversations?limit=1&offset={page['next_offset']}").json()
        assert page2["items"][0]["conversation_id"] == str(legacy)
        history = client.get(f"/v1/conversations/{fresh}").json()
        assert history["messages"][1]["metadata"]["citations"][0]["text_preview"]
        old = client.get(f"/v1/conversations/{legacy}").json()
        assert old["messages"][1]["metadata"] is None
        reply = client.post(
            "/v1/chat", json={"conversation_id": str(legacy), "message": "explain with example"}
        )
        assert reply.status_code == 200
        assert len(client.get(f"/v1/conversations/{legacy}").json()["messages"]) == 4
        assert client.get(f"/v1/conversations/{uuid4()}").status_code == 404
        assert client.get(f"/v1/conversations/{partial}").status_code == 404
        assert client.get("/v1/conversations/not-a-uuid").status_code == 422
        assert client.get("/v1/conversations?limit=0").status_code == 422

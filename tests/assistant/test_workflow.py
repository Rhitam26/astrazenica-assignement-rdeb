from contextlib import nullcontext
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from src.assistant.api import create_app
from src.assistant.service import ChatService
from src.assistant.workflow import RunContext, assess_evidence, create_graph
from src.shared.config import Settings
from tests.fakes import FakeModel, FakeRetriever, chunk


def setup_graph(model=None, retriever=None, settings=None):
    settings = settings or Settings(_env_file=None)
    model, retriever = model or FakeModel(), retriever or FakeRetriever()
    graph = create_graph(settings, model, retriever, InMemorySaver())
    return graph, model, retriever


def invoke(graph, question, cid="test"):
    return graph.invoke(
        {"user_message": question}, {"configurable": {"thread_id": cid}}, context=RunContext()
    )


@pytest.mark.parametrize(
    "question,route",
    [
        ("Explain HNSW", "simple_rag"),
        ("Compare pgvector and Pinecone", "agentic_rag"),
        ("hi", "direct"),
        ("Which one?", "clarify"),
    ],
)
def test_routes(question, route):
    graph, _, retrieval = setup_graph()
    result = invoke(graph, question)
    assert result["route"] == route
    if route in ("clarify", "direct"):
        assert not retrieval.queries


@pytest.mark.parametrize(
    "first,follow,subjects",
    [
        ("Explain HNSW and IVF", "I did not understand, explain with example", ["HNSW", "IVF"]),
        ("Compare pgvector and Pinecone", "Which one supports ACID?", ["pgvector", "Pinecone"]),
    ],
)
def test_context_is_resolved_before_routing_and_retrieval(first, follow, subjects):
    graph, model, retrieval = setup_graph()
    invoke(graph, first)
    result = invoke(graph, follow)
    assert all(subject in result["standalone_question"] for subject in subjects)
    intent_payloads = [payload for task, payload in model.payloads if task == "context_route"]
    assert len(intent_payloads) == 2
    assert intent_payloads[-1]["latest_message"] == follow
    assert not any(task in ("context", "route") for task, _ in model.payloads)
    assert all(query != follow for query, _ in retrieval.queries)
    assert len(result["messages"]) == 4
    assert "retrieved_chunks" not in result


def test_isolation_and_no_evidence_in_checkpoints():
    graph, _, retrieval = setup_graph()
    invoke(graph, "Explain HNSW and IVF", "one")
    before = len(retrieval.queries)
    result = invoke(graph, "Which one?", "two")
    assert result["route"] == "clarify"
    assert len(retrieval.queries) == before
    assert len(result["messages"]) == 2
    for checkpoint in graph.get_state_history({"configurable": {"thread_id": "one"}}):
        assert "evidence" not in checkpoint.values
        assert "agent_messages" not in checkpoint.values


@pytest.mark.parametrize("searches,expected", [(["a", "b", "c", "d", "e"], 4), (["a", "a"], 1)])
def test_agent_bound_and_duplicate_prevention(searches, expected):
    graph, _, retrieval = setup_graph(FakeModel(searches=searches))
    result = invoke(graph, "Compare pgvector and Pinecone")
    assert result["tool_call_count"] == expected == len(retrieval.queries)


@pytest.mark.parametrize(
    "model,retriever",
    [
        (FakeModel(), FakeRetriever([])),
        (FakeModel(unsupported=True), FakeRetriever()),
        (FakeModel(fake_citations=True), FakeRetriever()),
    ],
)
def test_abstention(model, retriever):
    graph, _, _ = setup_graph(model, retriever)
    result = invoke(graph, "Explain HNSW")
    assert result["abstained"]
    assert result["citations"] == []


def test_api_flow_and_validation():
    settings = Settings(_env_file=None, max_message_chars=100)
    graph, _, _ = setup_graph(settings=settings)
    service = ChatService(settings, graph, lambda _: nullcontext(), lambda: None)
    with TestClient(create_app(settings, service)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        response = client.post("/v1/chat", json={"message": "Explain HNSW and IVF"})
        assert response.status_code == 200
        body = response.json()
        assert body["citations"][0]["page_numbers"] == [4, 5]
        assert body["citations"][0]["text_preview"].startswith("HNSW uses a graph")
        assert response.headers["X-Request-ID"]
        follow = client.post(
            "/v1/chat", json={"message": "explain with example", "conversation_id": body["conversation_id"]}
        )
        assert follow.status_code == 200
        for invalid in (
            {"message": " "},
            {"message": "x" * 101},
            {"message": "hi", "conversation_id": "bad"},
        ):
            assert client.post("/v1/chat", json=invalid).status_code == 422
        other = client.post("/v1/chat", json={"message": "Which one?", "conversation_id": str(uuid4())})
        assert other.json()["workflow"] == "clarify"


def test_context_failure_does_not_search():
    class Broken(FakeModel):
        def structured(self, task, payload, schema):
            raise RuntimeError("secret password")

    graph, _, retrieval = setup_graph(Broken())
    settings = Settings(_env_file=None)
    with TestClient(
        create_app(settings, ChatService(settings, graph, lambda _: nullcontext(), lambda: None))
    ) as client:
        response = client.post("/v1/chat", json={"message": "Explain HNSW"})
        assert response.status_code == 503
        assert "secret" not in response.text
        assert not retrieval.queries


def test_model_cannot_invent_empty_thread_referent():
    class InventingModel(FakeModel):
        def structured(self, task, payload, schema):
            assert task != "context_route", "Empty-thread guard must precede the model"
            return super().structured(task, payload, schema)

    graph, _, retrieval = setup_graph(InventingModel())
    result = invoke(graph, "Which one supports ACID?")
    assert result["route"] == "clarify"
    assert not retrieval.queries


def test_support_check_rejects_unsupported_answer_with_valid_marker():
    class UnsupportedAnswer(FakeModel):
        def structured(self, task, payload, schema):
            if task == "support":
                return schema(supported=False, reason="Claim not supported by cited passage")
            return super().structured(task, payload, schema)

    graph, _, _ = setup_graph(UnsupportedAnswer())
    result = invoke(graph, "Explain HNSW")
    assert result["abstained"] and not result["citations"]


def test_bounded_history():
    from langchain_core.messages import HumanMessage

    from src.assistant.workflow import bounded_history

    settings = Settings(_env_file=None, history_turns=2, history_token_budget=100)
    history = [HumanMessage("x " * 1000)] + [HumanMessage(str(i)) for i in range(6)]
    result = bounded_history(history, settings)
    assert len(result) == 2
    assert result[0]["content"] == "4"


def test_simple_rag_uses_four_model_calls():
    graph, model, _ = setup_graph()
    assert not invoke(graph, "Explain HNSW")["abstained"]
    assert [task for task, _ in model.payloads] == ["context_route", "evidence", "answer", "support"]


@pytest.mark.parametrize(
    "unresolved,confidence,expected", [(False, 0.1, "simple_rag"), (True, 0.9, "clarify")]
)
def test_combined_intent_policy_overrides(unresolved, confidence, expected):
    class IntentModel(FakeModel):
        def structured(self, task, payload, schema):
            result = super().structured(task, payload, schema)
            if task == "context_route":
                result.unresolved = unresolved
                result.confidence = confidence
                result.route = "direct"
            return result

    graph, _, _ = setup_graph(IntentModel())
    assert invoke(graph, "Explain HNSW")["route"] == expected


def test_agent_final_evidence_reuses_last_assessment():
    class SufficientModel(FakeModel):
        def structured(self, task, payload, schema):
            result = super().structured(task, payload, schema)
            if task == "evidence":
                result.sufficient = True
            return result

    graph, model, _ = setup_graph(SufficientModel())
    result = invoke(graph, "Compare pgvector and Pinecone")
    assert not result["abstained"]
    assert result["tool_call_count"] == 1
    assert sum(task == "evidence" for task, _ in model.payloads) == 1


@pytest.mark.parametrize("unsupported", [False, True])
def test_assessment_reuse_and_invalidation(unsupported):
    settings = Settings(_env_file=None)
    model = FakeModel(unsupported=unsupported)
    run = RunContext(evidence=[chunk(), chunk(text="Other passage")])
    assess_evidence("question", run, settings, model)
    assess_evidence("question", run, settings, model)
    assert len(model.payloads) == 1
    run.evidence.reverse()
    assess_evidence("question", run, settings, model)
    run.evidence[0] = run.evidence[0].model_copy(update={"text": "changed"})
    assess_evidence("question", run, settings, model)
    run.evidence[0] = run.evidence[0].model_copy(update={"score": 0.8})
    assess_evidence("question", run, settings, model)
    assess_evidence("new question", run, settings, model)
    run.evidence.pop()
    assess_evidence("new question", run, settings, model)
    assert len(model.payloads) == 6
    run.evidence.clear()
    assess_evidence("new question", run, settings, model)
    assert not run.sufficient
    assert len(model.payloads) == 6


def test_invalid_markers_cannot_mark_agent_evidence_sufficient():
    class InvalidModel(FakeModel):
        def structured(self, task, payload, schema):
            result = super().structured(task, payload, schema)
            if task == "evidence":
                result.sufficient = True
                result.relevant_markers = [99]
            return result

    graph, _, _ = setup_graph(InvalidModel())
    result = invoke(graph, "Compare pgvector and Pinecone")
    assert result["abstained"]
    assert not result["citations"]

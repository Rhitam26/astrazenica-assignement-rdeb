from langchain_core.messages import AIMessage, HumanMessage

from src.assistant.workflow import bounded_history
from src.shared.config import Settings
from tests.assistant.test_workflow import invoke, setup_graph
from tests.fakes import FakeModel


def test_oversized_answer_keeps_question_and_budget():
    import tiktoken

    settings = Settings(_env_file=None, history_token_budget=100)
    history = bounded_history(
        [HumanMessage("Compare pgvector and Pinecone"), AIMessage("details " * 2000)], settings
    )
    assert len(history) == 2
    assert "pgvector and Pinecone" in history[0]["content"]
    assert sum(len(tiktoken.get_encoding("cl100k_base").encode(m["content"])) for m in history) <= 100


class RecoveryModel(FakeModel):
    def __init__(self, question, quotes):
        super().__init__()
        self.question, self.quotes = question, quotes

    def structured(self, task, payload, schema):
        if task == "context_repair":
            self.payloads.append((task, payload))
            return schema(
                standalone_question=self.question,
                unresolved=False,
                clarification="",
                route="simple_rag",
                reason="Resolved from conversation",
                confidence=0.9,
                supporting_quotes=self.quotes,
            )
        result = super().structured(task, payload, schema)
        if task == "context_route" and payload["history"]:
            result.unresolved, result.route = True, "clarify"
        return result


def test_architecture_followup_recovers_before_retrieval():
    first = "What retrieval architecture would you recommend for a production RAG application?"
    resolved = "Which agentic frameworks suit complex questions in a production RAG application?"
    graph, model, retriever = setup_graph(RecoveryModel(resolved, ["production RAG application"]))
    invoke(graph, first)
    result = invoke(graph, "Now add an agentic layer for complex questions. Which frameworks?")
    assert result["route"] == "simple_rag"
    assert retriever.queries[-1][0] == resolved
    assert sum(task == "context_repair" for task, _ in model.payloads) == 1


def test_comparison_then_selected_option_followup():
    class SelectionModel(RecoveryModel):
        def structured(self, task, payload, schema):
            if task == "answer":
                return schema(
                    blocks=[{"text": "pgvector provides ACID guarantees.", "evidence_markers": [1]}],
                    abstained=False,
                )
            return super().structured(task, payload, schema)

    model = SelectionModel("Between pgvector and Pinecone, which provides ACID?", ["pgvector", "Pinecone"])
    graph, _, retriever = setup_graph(model)
    invoke(graph, "Compare pgvector and Pinecone")
    invoke(graph, "Which one gives me ACID guarantees?")
    assert "Pinecone" in retriever.queries[-1][0]
    model.question = "What are the main limitations of pgvector?"
    model.quotes = ["pgvector provides ACID guarantees."]
    result = invoke(graph, "What are the main limitations of that option?")
    assert retriever.queries[-1][0] == model.question
    assert result["messages"][-2].additional_kwargs["resolved_question"] == model.question


def test_repair_cannot_invent_supporting_context():
    graph, _, retriever = setup_graph(RecoveryModel("What are Milvus limitations?", ["We selected Milvus"]))
    invoke(graph, "Explain HNSW")
    count = len(retriever.queries)
    result = invoke(graph, "What about that option?")
    assert result["route"] == "clarify"
    assert len(retriever.queries) == count


def test_recovery_failure_keeps_clarification():
    class FailingRecovery(RecoveryModel):
        def structured(self, task, payload, schema):
            if task == "context_repair":
                raise TimeoutError("provider unavailable")
            return super().structured(task, payload, schema)

    graph, _, _ = setup_graph(FailingRecovery("unused", []))
    invoke(graph, "Explain HNSW")
    assert invoke(graph, "What about that option?")["route"] == "clarify"


def test_genuine_ambiguity_stays_unresolved():
    class AmbiguousRecovery(RecoveryModel):
        def structured(self, task, payload, schema):
            result = super().structured(task, payload, schema)
            if task == "context_repair":
                result.unresolved, result.route = True, "clarify"
            return result

    graph, _, retriever = setup_graph(AmbiguousRecovery("Which option?", ["pgvector", "Pinecone"]))
    invoke(graph, "Compare pgvector and Pinecone")
    count = len(retriever.queries)
    assert invoke(graph, "What are the limitations of that option?")["route"] == "clarify"
    assert len(retriever.queries) == count


def test_explicit_new_topic_uses_normal_path():
    graph, model, retriever = setup_graph()
    invoke(graph, "Explain HNSW")
    invoke(graph, "Explain CrewAI")
    assert retriever.queries[-1][0] == "Explain CrewAI"
    assert not any(task == "context_repair" for task, _ in model.payloads)

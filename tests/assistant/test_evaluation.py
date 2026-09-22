import pytest
from langgraph.checkpoint.memory import InMemorySaver

from scripts.evaluate_rag import (
    RecordingRetriever,
    Turn,
    collect_turn,
    load_cases,
    metric_specs,
    retrieval_metrics,
    summarize,
)
from src.assistant.workflow import create_graph
from src.shared.config import Settings
from tests.fakes import FakeModel, FakeRetriever, chunk


def test_eval_captures_full_evidence_and_isolates_conversations():
    settings = Settings(_env_file=None)
    full_text = "HNSW uses a graph. " * 100
    recorder = RecordingRetriever(FakeRetriever([chunk(text=full_text)]))
    graph = create_graph(settings, FakeModel(), recorder, InMemorySaver())
    turn = Turn(
        message="Explain HNSW", evaluation_question="Explain HNSW", routes=["simple_rag"], abstained=False
    )
    record = collect_turn(graph, recorder, turn, "one", settings)
    assert all(record["checks"].values())
    assert record["generation_context"] == [full_text]
    assert record["searches"][0]["chunks"][0]["text"] == full_text
    assert len(record["citations"][0]["text_preview"]) < len(full_text)
    assert [m[0] for m in metric_specs(record)] == [
        "answer_relevancy",
        "faithfulness",
        "contextual_relevancy",
    ]
    isolated = Turn(
        message="Which one?", evaluation_question="Which one?", routes=["clarify"], abstained=False
    )
    record = collect_turn(graph, recorder, isolated, "two", settings)
    assert all(record["checks"].values())
    assert record["searches"] == []
    assert metric_specs(record) == []


def test_abstention_does_not_receive_vacuous_faithfulness_score():
    settings = Settings(_env_file=None)
    recorder = RecordingRetriever(FakeRetriever([]))
    graph = create_graph(settings, FakeModel(), recorder, InMemorySaver())
    turn = Turn(
        message="Explain HNSW", evaluation_question="Explain HNSW", routes=["simple_rag"], abstained=True
    )
    record = collect_turn(graph, recorder, turn, "test", settings)
    assert metric_specs(record) == []
    assert all(record["checks"].values())
    record.update(passed=True, metrics=[])
    summary = summarize([record, {"passed": False, "error_type": "TimeoutError"}])
    assert summary["turns"] == 2 and summary["passed"] == 1


def test_seed_dataset_validates():
    from pathlib import Path

    assert len(load_cases(Path("evals/cases.json"))) == 6


def test_gold_dataset_has_thirty_source_reviewed_turns():
    from pathlib import Path

    cases = load_cases(Path("evals/gold_cases.json"))
    turns = [turn for case in cases for turn in case.turns]
    assert len(turns) == 30
    answerable = [
        turn
        for turn in turns
        if not turn.abstained and turn.routes[0] != "direct" and turn.routes[0] != "clarify"
    ]
    assert all(turn.expected_output and turn.gold_evidence for turn in answerable)
    assert all(len(evidence.content_hash) == 64 for turn in answerable for evidence in turn.gold_evidence)


def test_retrieval_precision_and_recall_use_exact_gold_hashes():
    record = {
        "expected": {
            "retrieval_evaluation": True,
            "gold_evidence": [{"content_hash": "a" * 64}, {"content_hash": "b" * 64}],
        },
        "searches": [
            {"chunks": [{"content_hash": "a" * 64}, {"content_hash": "x" * 64}, {"content_hash": "b" * 64}]}
        ],
    }
    precision, recall = retrieval_metrics(record)
    assert precision["score"] == 0.4 and precision["passed"]
    assert recall["score"] == 1.0 and recall["passed"]

    record["searches"] = []
    precision, recall = retrieval_metrics(record)
    assert precision["score"] == 0.0 and not precision["passed"]
    assert recall["score"] == 0.0 and not recall["passed"]

    record["searches"] = [{"chunks": [{"content_hash": "a" * 64}] * 5}]
    precision, recall = retrieval_metrics(record)
    assert precision["score"] == 0.2
    assert recall["score"] == 0.5


def test_real_deepeval_metrics_with_local_judge(monkeypatch):
    """Exercise the installed SDK without OpenAI or network requests."""
    monkeypatch.setenv("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    pytest.importorskip("deepeval")
    from deepeval.models import DeepEvalBaseLLM

    from scripts.evaluate_rag import judge_record

    class LocalJudge(DeepEvalBaseLLM):
        def load_model(self):
            return self

        def get_model_name(self):
            return "deterministic-test-judge"

        def generate(self, prompt, schema=None, **kwargs):
            values = {}
            for field in schema.model_fields:
                if field in ("claims", "truths", "statements"):
                    values[field] = ["HNSW uses a graph."]
                elif field == "verdicts":
                    values[field] = [{"verdict": "yes", "statement": "HNSW uses a graph."}]
                elif field == "reason":
                    values[field] = "Supported and relevant."
            return schema.model_validate(values)

        async def a_generate(self, prompt, schema=None, **kwargs):
            return self.generate(prompt, schema, **kwargs)

    record = {
        "expected": {"abstained": False},
        "workflow": "simple_rag",
        "abstained": False,
        "evaluation_question": "Explain HNSW",
        "answer": "HNSW uses a graph.",
        "generation_context": ["HNSW uses a graph."],
        "searches": [{"chunks": [{"text": "HNSW uses a graph."}]}],
    }
    metrics = judge_record(record, LocalJudge())
    assert len(metrics) == 3
    assert all(m.get("score") == 1 and m["passed"] for m in metrics), metrics

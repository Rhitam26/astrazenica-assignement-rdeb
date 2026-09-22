"""Opt-in DeepEval evaluation over the real graph, isolated from saved user chats.

Run from the repository root: python -m scripts.evaluate_rag --help
"""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field

from src.assistant.grounding import build_citations
from src.assistant.models import Route
from src.assistant.prompts import PROMPTS
from src.assistant.provider import OpenAIModel
from src.assistant.retrieval import PgRetriever, retrieval_pool
from src.assistant.workflow import RunContext, create_graph
from src.shared.config import Settings
from src.shared.embedder import Embedder


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1)
    evaluation_question: str = Field(min_length=1)
    routes: list[Route] = Field(min_length=1)
    abstained: bool
    subjects: list[str] = []
    expected_output: str | None = None
    gold_evidence: list["GoldEvidence"] = []
    retrieval_evaluation: bool = False


class GoldEvidence(BaseModel):
    """A source-reviewed relevant indexed chunk for one evaluation turn."""

    model_config = ConfigDict(extra="forbid")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    doc_id: str = Field(min_length=1)
    page_numbers: list[int] = Field(min_length=1)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    turns: list[Turn] = Field(min_length=1)


class Dataset(BaseModel):
    """Versioned evaluation data; list input remains supported for smoke cases."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    corpus_chunks: int | None = Field(default=None, ge=1)
    cases: list[Scenario] = Field(min_length=1)


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(path.read_text())
    dataset = (
        Dataset.model_validate(raw)
        if isinstance(raw, dict)
        else Dataset(name="smoke", version=1, cases=[Scenario.model_validate(item) for item in raw])
    )
    cases = dataset.cases
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Dataset must be nonempty and have unique scenario IDs")
    return dataset


def load_cases(path: Path) -> list[Scenario]:
    """Backward-compatible convenience used by lightweight dataset validation."""
    return load_dataset(path).cases


class RecordingRetriever:
    def __init__(self, delegate):
        self.delegate = delegate
        self.searches = []

    def search(self, query, top_k=5, filters=None):
        chunks = self.delegate.search(query, top_k, filters)
        self.searches.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters.model_dump(exclude_none=True) if filters else {},
                "chunks": [c.model_dump(mode="json") for c in chunks],
            }
        )
        return chunks


def collect_turn(graph, recorder, turn, cid, settings):
    recorder.searches = []
    run = RunContext()
    start = perf_counter()
    state = graph.invoke(
        {"user_message": turn.message},
        {
            "configurable": {"thread_id": cid},
            "recursion_limit": settings.langgraph_recursion_limit,
        },
        context=run,
    )
    latency_ms = round((perf_counter() - start) * 1000)
    checks = {
        "route": state["route"] in turn.routes,
        "abstention": state["abstained"] == turn.abstained,
        "resolved_subjects": all(s.lower() in state["standalone_question"].lower() for s in turn.subjects),
        "search_budget": state["tool_call_count"] <= settings.agent_max_tool_calls,
        "no_search_for_social_or_clarification": bool(recorder.searches) is False
        if state["route"] in ("direct", "clarify")
        else True,
    }
    if state["route"] in ("simple_rag", "agentic_rag") and not state["abstained"]:
        try:
            expected = [c.model_dump(mode="json") for c in build_citations(state["answer"], run.evidence)]
            checks["citations"] = expected == state["citations"] and bool(expected)
        except ValueError:
            checks["citations"] = False
    else:
        checks["citations"] = not state["citations"]
    return {
        "input": turn.message,
        "evaluation_question": turn.evaluation_question,
        "standalone_question": state["standalone_question"],
        "answer": state["answer"],
        "workflow": state["route"],
        "abstained": state["abstained"],
        "expected": turn.model_dump(),
        "citations": state["citations"],
        "checks": checks,
        "latency_ms": latency_ms,
        "searches": recorder.searches,
        "generation_context": [c.text for c in run.evidence],
    }


def retrieval_metrics(record, k=5):
    """Exact ranked scores for source-labelled, single-search turns only."""
    expected = record["expected"]
    if not expected.get("retrieval_evaluation"):
        return []
    relevant = {item["content_hash"] for item in expected["gold_evidence"]}
    searches = record["searches"]
    retrieved = searches[0]["chunks"][:k] if len(searches) == 1 else []
    hits = len({chunk["content_hash"] for chunk in retrieved} & relevant)
    recall = hits / len(relevant) if relevant else 0.0
    return [
        {
            "name": f"retrieval_precision_at_{k}",
            "score": hits / k,
            "threshold": 1 / k,
            "passed": len(searches) == 1 and hits >= 1,
            "relevant_retrieved": hits,
            "relevant_total": len(relevant),
        },
        {
            "name": f"retrieval_recall_at_{k}",
            "score": recall,
            "threshold": 1.0,
            "passed": len(searches) == 1 and recall == 1.0,
            "relevant_retrieved": hits,
            "relevant_total": len(relevant),
        },
    ]


def metric_specs(record):
    """Applicability is explicit; abstention is not rewarded as a faithful answer."""
    specs = []
    if record["expected"].get("expected_output") and record["workflow"] in ("simple_rag", "agentic_rag"):
        specs.append(("factual_correctness", 0.8, None))
    if not record["expected"]["abstained"] and record["workflow"] in ("simple_rag", "agentic_rag"):
        specs.append(("answer_relevancy", 0.7, None))
        if not record["abstained"] and record["generation_context"]:
            specs.append(("faithfulness", 0.8, record["generation_context"]))
        raw_context = list(dict.fromkeys(c["text"] for s in record["searches"] for c in s["chunks"]))
        if raw_context:
            specs.append(("contextual_relevancy", 0.5, raw_context))
    return specs


def judge_record(record, model):
    # Lazy import keeps normal tests and dataset validation free of DeepEval side effects.
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    classes = {
        "answer_relevancy": AnswerRelevancyMetric,
        "faithfulness": FaithfulnessMetric,
        "contextual_relevancy": ContextualRelevancyMetric,
    }
    results = []
    for name, threshold, context in metric_specs(record):
        try:
            if name == "factual_correctness":
                metric = GEval(
                    name="factual_correctness",
                    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
                    evaluation_steps=[
                        "Compare the actual output with the reference answer.",
                        "Penalize factual contradictions, missing required facts, unsupported specificity, and an answer "
                        "when the reference requires abstention.",
                        "Accept concise paraphrases and additional statements only when they do not contradict the reference.",
                    ],
                    threshold=threshold,
                    model=model,
                    async_mode=False,
                    include_reason=True,
                )
            else:
                metric = classes[name](
                    threshold=threshold, model=model, async_mode=False, include_reason=True
                )
            case = LLMTestCase(
                input=record["evaluation_question"],
                actual_output=record["answer"],
                expected_output=record["expected"].get("expected_output"),
                retrieval_context=context,
            )
            metric.measure(case)
            score = metric.score
            if score is None or not math.isfinite(score):
                raise ValueError("Missing metric score")
            results.append(
                {
                    "name": name,
                    "score": score,
                    "threshold": threshold,
                    "passed": score >= threshold,
                    "reason": metric.reason,
                }
            )
        except Exception as exc:
            results.append({"name": name, "passed": False, "error_type": type(exc).__name__})
    return results


def summarize(records):
    values = sorted(r["latency_ms"] for r in records if "latency_ms" in r)
    metrics = {}
    for record in records:
        for metric in record.get("metrics", []):
            group = metrics.setdefault(metric["name"], {"scores": [], "errors": 0})
            if "score" in metric:
                group["scores"].append(metric["score"])
            else:
                group["errors"] += 1
    return {
        "turns": len(records),
        "passed": sum(r["passed"] for r in records),
        "latency_p50_ms": values[math.ceil(len(values) * 0.5) - 1] if values else None,
        "latency_p95_ms": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        "metrics": {
            name: {
                "count": len(g["scores"]),
                "errors": g["errors"],
                "mean": sum(g["scores"]) / len(g["scores"]) if g["scores"] else None,
            }
            for name, g in metrics.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/gold_cases.json"))
    parser.add_argument("--live", action="store_true", help="Run paid application and judge calls")
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--scenario", help="Run one scenario ID")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evals"))
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    cases = dataset.cases
    if args.scenario:
        cases = [c for c in cases if c.id == args.scenario]
        if not cases:
            parser.error("Unknown scenario ID")
    if not args.live:
        print(
            f"Validated {len(cases)} scenarios / {sum(len(c.turns) for c in cases)} turns. Use --live to evaluate."
        )
        return 0
    load_dotenv()
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    settings = Settings(langfuse_enabled=False)
    if not settings.openai_api_key.get_secret_value():
        parser.error("OPENAI_API_KEY is required")
    # Ensure the SDK judge receives the same key loaded by Settings.
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
    try:
        import deepeval  # noqa: F401
    except ImportError:
        parser.error("Install requirements/evaluation.txt first")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = args.output_dir / run_id
    output.mkdir(parents=True)
    records = []
    report = {
        "run_id": run_id,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "judge_model": args.judge_model,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "prompt_sha256": hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
        "config": settings.model_dump(
            include={
                "retrieval_top_k",
                "retrieval_score_threshold",
                "agent_max_tool_calls",
                "history_turns",
                "history_token_budget",
                "llm_temperature",
            }
        ),
        "records": records,
    }

    def save():
        report["summary"] = summarize(records)
        (output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    with retrieval_pool(settings) as pool:
        pool.wait(timeout=10)
        with pool.connection() as conn:
            corpus = conn.execute(
                "SELECT doc_id, content_hash, embedding_model, embedding_dims FROM public.chunks "
                "ORDER BY doc_id, content_hash"
            ).fetchall()
        if not corpus:
            parser.error("The knowledge base is empty; ingest the evaluation corpus first")
        report["corpus_sha256"] = hashlib.sha256(json.dumps(corpus).encode()).hexdigest()
        report["corpus_chunks"] = len(corpus)
        if dataset.corpus_sha256 and report["corpus_sha256"] != dataset.corpus_sha256:
            parser.error("Indexed corpus fingerprint differs from this source-reviewed benchmark")
        if dataset.corpus_chunks and report["corpus_chunks"] != dataset.corpus_chunks:
            parser.error("Indexed corpus chunk count differs from this source-reviewed benchmark")
        indexed = {row[1] for row in corpus}
        gold_hashes = {
            item.content_hash for scenario in cases for turn in scenario.turns for item in turn.gold_evidence
        }
        missing = sorted(gold_hashes - indexed)
        if missing:
            parser.error(f"Indexed corpus is missing {len(missing)} gold evidence chunk(s)")
        embedder = Embedder(settings)
        model = OpenAIModel(settings)
        try:
            recorder = RecordingRetriever(PgRetriever(settings, embedder, connection_factory=pool.connection))
            graph = create_graph(settings, model, recorder, InMemorySaver())
            for scenario in cases:
                cid = str(uuid4())
                for index, turn in enumerate(scenario.turns):
                    try:
                        record = collect_turn(graph, recorder, turn, cid, settings)
                        record["metrics"] = retrieval_metrics(record) + judge_record(record, args.judge_model)
                        record["passed"] = all(record["checks"].values()) and all(
                            m["passed"] for m in record["metrics"]
                        )
                    except Exception as exc:
                        record = {"passed": False, "error_type": type(exc).__name__}
                    record.update(scenario=scenario.id, turn=index + 1)
                    records.append(record)
                    save()
                    print(f"{scenario.id}/{index + 1}: {'PASS' if record['passed'] else 'FAIL'}")
                    if "error_type" in record:
                        # A failed setup turn invalidates the rest of this conversation.
                        for skipped in range(index + 1, len(scenario.turns)):
                            records.append(
                                {
                                    "scenario": scenario.id,
                                    "turn": skipped + 1,
                                    "passed": False,
                                    "error_type": "PriorTurnFailed",
                                }
                            )
                        save()
                        break
        finally:
            embedder.close()
    print(f"Report: {output / 'results.json'}")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

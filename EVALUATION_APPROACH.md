# DeepEval evaluation approach

## Purpose

This project evaluates the real LangGraph RAG workflow on a source-reviewed, 30-turn gold benchmark. It separates three questions:

1. Did the system retrieve the labelled supporting chunks?
2. Is the final answer factually correct against a reference answer?
3. Is the answer relevant, grounded in selected evidence, routed correctly, cited correctly, and produced within an acceptable latency envelope?

The runner is `scripts/evaluate_rag.py`. DeepEval is optional (`requirements/evaluation.txt`) and never runs in the normal API/UI request path.

## Datasets

`evals/gold_cases.json` is the default benchmark. It has exactly 30 source-reviewed turns grounded in the four supplied PDFs. Each answerable turn contains a reference answer plus exact relevant `content_hash`, document ID, and page labels. It covers RAG design, retrieval, vector-store selection, agentic frameworks, security, latency, cross-document decisions, ambiguity, greetings, and unsupported questions.

The benchmark declares the reviewed 146-chunk corpus fingerprint. A live run stops before model calls when its indexed corpus differs or a labelled chunk is missing. Re-ingesting changed source text, changing the embedding corpus, or adding documents requires review and a new benchmark version; this prevents scores from being compared across different knowledge bases.

`evals/cases.json` remains the six-scenario smoke suite for quick schema and regression validation. It is deliberately not a factual release gate.

“Source-reviewed” means the reference answer and labels were checked against the supplied corpus. It is not a claim of independent external subject-matter review.

## Run

Use a separate environment because DeepEval 4.2.3 has a `click` constraint that conflicts with the ingestion stack:

```bash
python3.13 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r requirements/evaluation.txt

# No database, model, or provider requests.
.venv-eval/bin/python -m scripts.evaluate_rag
.venv-eval/bin/python -m scripts.evaluate_rag --dataset evals/cases.json

# Paid application, embedding, and judge calls; PostgreSQL must be migrated and ingested.
.venv-eval/bin/python -m scripts.evaluate_rag --live --scenario hnsw-ivf
.venv-eval/bin/python -m scripts.evaluate_rag --live --judge-model gpt-4.1
```

Live runs load `.env`, use the normal application graph and PostgreSQL retriever, create in-memory LangGraph checkpoints, and never save evaluation messages to user conversation history. Langfuse is disabled and DeepEval telemetry is opted out for these runs. The API and Streamlit do not need to be running.

## Measurements

| Measurement | Method | Passing rule | Applicability |
| --- | --- | --- | --- |
| Factual correctness | DeepEval G-Eval compares generated answer with `expected_output` | >= 0.80 | RAG turns with a reference answer, including expected abstentions |
| Retrieval precision@5 | Exact relevant hashes in the one returned ranked top-5 list divided by 5 | >= 0.20 and at least one hit | Labelled single-search turns |
| Retrieval recall@5 | Exact relevant hashes found in the ranked top-5 list divided by all labels | 1.00 | Labelled single-search turns |
| Answer relevancy | DeepEval judge against the standalone evaluation question | >= 0.70 | Expected answerable RAG turns |
| Faithfulness | DeepEval judge against complete selected generation evidence | >= 0.80 | Non-abstaining RAG turns with evidence |
| Contextual relevancy | DeepEval judge against complete raw retrieved text | >= 0.50 | Expected answerable RAG turns with results |

Precision and recall are deterministic and use content hashes, never citation previews or an LLM judge. They intentionally apply only where the graph makes one retrieval. Agentic turns can issue several searches with different intents; merging them into a single ranked list would produce misleading precision/recall. Those turns still receive factual correctness, grounding, citations, route, abstention, and tool-budget checks.

Every turn additionally checks allowed route, expected abstention, required subjects preserved by question rewriting, tool-call budget, no retrieval on direct/clarify paths, and deterministic citation integrity. Unsupported questions receive a factual-correctness reference requiring abstention; no retrieval metric is applied.

## Reports and interpretation

Each live run writes `artifacts/evals/<UTC timestamp>-<suffix>/results.json`, including dataset/prompt/corpus fingerprints, configuration, reference answer and gold evidence labels, ordered searches and chunks, answer, citations, deterministic checks, metric scores/reasons, and p50/p95 graph latency. Results are saved after each turn.

Treat a failing deterministic check or retrieval metric as a regression. For judge metrics, inspect the answer, reference, retrieved chunks, and judge reason before changing prompts or thresholds. Compare only runs that use the same dataset version, corpus fingerprint, application model, judge model, and retrieval configuration. Run paid evaluations outside ordinary CI; unit tests validate the benchmark schema and scoring logic without provider calls.

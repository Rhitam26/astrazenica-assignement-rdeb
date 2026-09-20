"""Interactive RAG chat over chunks stored in Postgres/pgvector.

Examples:
    python chat.py
    python chat.py "What are the main RAG architecture patterns?"
    python chat.py --top-k 8 --show-context "Compare the vector databases"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import get_settings
from db import RetrievedChunk, get_connection, retrieve_relevant_chunks
from embedder import Embedder


SYSTEM_PROMPT = """You are a careful RAG assistant.
Answer using the provided retrieved document context and the conversation history.
Use numbered citations like [1] or [2] for claims from retrieved documents.
You may also use explicit facts the user gave earlier in the conversation; those do not need citations.
If neither the retrieved context nor the conversation history contains the answer, say so clearly.
Keep answers concise.
"""

QUESTION_REWRITE_PROMPT = """Rewrite the latest user message for RAG retrieval.
Use recent conversation to resolve pronouns, implied topics, and follow-up requests.

Return only valid JSON with this exact shape:
{
  "standalone_question": "fully contextualized question for vector search",
  "needs_retrieval": true,
  "reason": "short reason"
}

Set needs_retrieval to false only when the answer can be handled entirely from explicit
conversation facts, such as the user's own earlier statement.
"""

MAX_HISTORY_TURNS = 6


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]
    standalone_question: str
    needs_retrieval: bool
    rewrite_reason: str


@dataclass(frozen=True)
class ConversationTurn:
    question: str
    answer: str
    standalone_question: str
    sources: list[RetrievedChunk]


@dataclass(frozen=True)
class RetrievalPlan:
    standalone_question: str
    needs_retrieval: bool
    reason: str


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            "\n".join(
                [
                    f"[{idx}] {chunk.citation}",
                    f"chunk_id: {chunk.chunk_id}",
                    f"similarity: {chunk.similarity:.3f}",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def format_history(history: list[ConversationTurn]) -> str:
    if not history:
        return "No previous conversation."

    parts = []
    recent_history = history[-MAX_HISTORY_TURNS:]
    for idx, turn in enumerate(recent_history, start=1):
        source_summary = ", ".join(chunk.citation for chunk in turn.sources[:3])
        if not source_summary:
            source_summary = "no retrieved document sources"
        parts.append(
            "\n".join(
                [
                    f"Turn {idx}",
                    f"User: {turn.question}",
                    f"Retrieval query: {turn.standalone_question}",
                    f"Assistant: {turn.answer}",
                    f"Sources used: {source_summary}",
                ]
            )
        )
    return "\n\n".join(parts)


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return json.loads(cleaned)


def _coerce_bool(value: object, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def prepare_retrieval_plan(
    client: OpenAI,
    question: str,
    history: list[ConversationTurn],
) -> RetrievalPlan:
    if not history:
        return RetrievalPlan(
            standalone_question=question,
            needs_retrieval=True,
            reason="No prior conversation.",
        )

    response = client.chat.completions.create(
        model=get_settings().chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": QUESTION_REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{format_history(history)}\n\n"
                    f"Latest user message: {question}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    payload = _parse_json_object(content)

    standalone_question = str(payload.get("standalone_question") or question).strip()
    needs_retrieval = _coerce_bool(payload.get("needs_retrieval"), default=True)
    reason = str(payload.get("reason") or "").strip()
    return RetrievalPlan(
        standalone_question=standalone_question or question,
        needs_retrieval=needs_retrieval,
        reason=reason,
    )


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(4))
def generate_answer(
    client: OpenAI,
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ConversationTurn],
) -> str:
    settings = get_settings()
    context = format_context(chunks) if chunks else "No document context retrieved for this turn."
    response = client.chat.completions.create(
        model=settings.chat_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{format_history(history)}\n\n"
                    f"Retrieved document context:\n{context}\n\n"
                    f"Latest question: {question}\n\n"
                    "Answer with citations:"
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def answer_question(
    question: str,
    *,
    history: list[ConversationTurn] | None = None,
    top_k: int = 5,
    doc_id: str | None = None,
    content_type: str | None = None,
) -> RagAnswer:
    history = history or []
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        retrieval_plan = prepare_retrieval_plan(client, question, history)
    except Exception:
        retrieval_plan = RetrievalPlan(
            standalone_question=question,
            needs_retrieval=True,
            reason="Question rewrite failed; used the latest question as-is.",
        )

    chunks: list[RetrievedChunk] = []
    if retrieval_plan.needs_retrieval:
        query_embedding = Embedder().embed_texts([retrieval_plan.standalone_question])[0]

        with get_connection() as conn:
            chunks = retrieve_relevant_chunks(
                conn,
                query_embedding,
                limit=top_k,
                doc_id=doc_id,
                content_type=content_type,
            )

    if retrieval_plan.needs_retrieval and not chunks:
        return RagAnswer(
            answer="I could not find any matching chunks in the indexed documents.",
            sources=[],
            standalone_question=retrieval_plan.standalone_question,
            needs_retrieval=retrieval_plan.needs_retrieval,
            rewrite_reason=retrieval_plan.reason,
        )

    answer = generate_answer(client, question, chunks, history)
    return RagAnswer(
        answer=answer,
        sources=chunks,
        standalone_question=retrieval_plan.standalone_question,
        needs_retrieval=retrieval_plan.needs_retrieval,
        rewrite_reason=retrieval_plan.reason,
    )


def print_answer(result: RagAnswer, *, show_context: bool = False) -> None:
    print("\nAnswer\n------")
    print(result.answer)

    if show_context:
        print("\nRetrieval Plan\n--------------")
        print(f"needs_retrieval={result.needs_retrieval}")
        print(f"standalone_question={result.standalone_question}")
        if result.rewrite_reason:
            print(f"reason={result.rewrite_reason}")

    if not result.sources:
        return

    print("\nSources\n-------")
    for idx, chunk in enumerate(result.sources, start=1):
        print(f"[{idx}] {chunk.citation} | similarity={chunk.similarity:.3f}")

    if show_context:
        print("\nRetrieved Context\n-----------------")
        print(format_context(result.sources))


def run_interactive(args: argparse.Namespace) -> None:
    print("RAG chat ready. Type 'exit' or 'quit' to stop.\n")
    history: list[ConversationTurn] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return

        result = answer_question(
            question,
            history=history,
            top_k=args.top_k,
            doc_id=args.doc_id,
            content_type=args.content_type,
        )
        print_answer(result, show_context=args.show_context)
        history.append(
            ConversationTurn(
                question=question,
                answer=result.answer,
                standalone_question=result.standalone_question,
                sources=result.sources,
            )
        )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with your pgvector-backed RAG index.")
    parser.add_argument("question", nargs="*", help="Optional one-off question to ask.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--doc-id", help="Restrict retrieval to a document id.")
    parser.add_argument(
        "--content-type",
        choices=["text", "table"],
        help="Restrict retrieval to text or table chunks.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the full retrieved chunks after the answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")

    question = " ".join(args.question).strip()
    if question:
        result = answer_question(
            question,
            top_k=args.top_k,
            doc_id=args.doc_id,
            content_type=args.content_type,
        )
        print_answer(result, show_context=args.show_context)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()

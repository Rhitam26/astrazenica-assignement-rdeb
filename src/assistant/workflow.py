"""Explicit LangGraph workflow; evidence and tool transcripts live only for one request."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from src.assistant.grounding import ABSTENTION, build_citations, evidence_payload, render_grounded_answer
from src.assistant.models import (
    Citation,
    ContextRouteDecision,
    EvidenceDecision,
    GeneratedAnswer,
    GroundedAnswer,
    MetadataFilters,
    RetrievedChunk,
    Route,
    RouteDecision,
    SupportDecision,
)
from src.assistant.provider import Model
from src.assistant.retrieval import Retriever
from src.shared.config import Settings
from src.shared.observability import Telemetry, preview, value_hash

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    standalone_question: str
    unresolved: bool
    clarification: str
    route: Route
    route_reason: str
    route_confidence: float
    tool_call_count: int
    chunk_ids: list[str]
    answer: str
    citations: list[dict]
    abstained: bool
    error: str | None
    continue_agent: bool


@dataclass
class RunContext:
    evidence: list[RetrievedChunk] = field(default_factory=list)
    agent_messages: list[BaseMessage] = field(default_factory=list)
    seen_searches: set[str] = field(default_factory=set)
    search_queries: list[str] = field(default_factory=list)
    pending: AIMessage | None = None
    sufficient: bool = False
    assessment_signature: str | None = None
    assessment: EvidenceDecision | None = None


def assess_evidence(question: str, run: RunContext, settings: Settings, model: Model) -> EvidenceDecision:
    """Reuse only an assessment of the exact same ordered, filtered input."""
    run.evidence = [c for c in run.evidence if c.score >= settings.retrieval_score_threshold]
    payload = {"question": question, "evidence": evidence_payload(run.evidence)}
    signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if not run.evidence:
        decision = EvidenceDecision(sufficient=False, relevant_markers=[], missing_information="No evidence")
    elif run.assessment is not None and run.assessment_signature == signature:
        decision = run.assessment
    else:
        decision = model.structured("evidence", payload, EvidenceDecision)
    run.assessment, run.assessment_signature = decision, signature
    run.sufficient = (
        decision.sufficient
        and bool(decision.relevant_markers)
        and all(1 <= marker <= len(run.evidence) for marker in decision.relevant_markers)
    )
    return decision


def bounded_history(messages: list[BaseMessage], settings: Settings) -> list[dict[str, str]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    remaining = settings.history_token_budget
    result = []
    for msg in reversed(messages[-settings.history_turns * 2 :]):
        if msg.type not in ("human", "ai") or not isinstance(msg.content, str):
            continue
        tokens = encoding.encode(msg.content)
        if len(tokens) > remaining:
            break
        result.append({"role": msg.type, "content": msg.content})
        remaining -= len(tokens)
    return list(reversed(result))


def create_graph(
    settings: Settings,
    model: Model,
    retriever: Retriever,
    checkpointer: Any,
    telemetry: Telemetry | None = None,
):
    telemetry = telemetry or Telemetry(settings)

    def contextualize(state: ChatState):
        history = bounded_history(state.get("messages", []), settings)
        latest = state["user_message"]
        # No model may invent a comparison set for an empty thread.
        referential = re.match(
            r"(?i)^(which one|what about that|why[?.!]*$|explain (that|it)|give me an example|i did not understand)",
            latest.strip(),
        )
        if not history and referential and " or " not in latest.lower():
            result = ContextRouteDecision(
                standalone_question=latest,
                unresolved=True,
                clarification="Which subject or technologies are you referring to?",
                route="clarify",
                reason="The reference needs clarification.",
                confidence=1,
            )
        else:
            result = model.structured(
                "context_route",
                {
                    "history": history,
                    "latest_message": state["user_message"],
                },
                ContextRouteDecision,
            )
        if not result.standalone_question.strip():
            raise ValueError("Empty contextualized question")
        return {
            "standalone_question": result.standalone_question,
            "unresolved": result.unresolved,
            "clarification": result.clarification,
            "route": result.route,
            "route_reason": result.reason,
            "route_confidence": result.confidence,
            "answer": "",
            "citations": [],
            "chunk_ids": [],
            "tool_call_count": 0,
            "abstained": False,
            "error": None,
            "continue_agent": False,
        }

    def route(state: ChatState):
        if state["unresolved"]:
            decision = RouteDecision(
                route="clarify", reason="The reference needs clarification.", confidence=1
            )
        else:
            decision = RouteDecision(
                route=state["route"], reason=state["route_reason"], confidence=state["route_confidence"]
            )
            if decision.confidence < settings.router_confidence_threshold and decision.route != "clarify":
                decision = RouteDecision(
                    route="simple_rag",
                    reason="Use grounded retrieval for uncertain intent.",
                    confidence=decision.confidence,
                )
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "route_confidence": decision.confidence,
        }

    def direct(state: ChatState):
        answer = model.structured("direct", {"question": state["standalone_question"]}, GeneratedAnswer)
        return {"answer": answer.answer, "abstained": answer.abstained}

    def clarify(state: ChatState):
        return {
            "answer": state.get("clarification")
            or "Which subject or comparison would you like me to explain?"
        }

    def simple(state: ChatState, runtime: Runtime[RunContext]):
        with telemetry.observation(
            "workflow.simple-retrieval",
            metadata={"query_hash": value_hash(state["standalone_question"])},
        ):
            runtime.context.search_queries.append(state["standalone_question"])
            runtime.context.evidence = retriever.search(
                state["standalone_question"], settings.retrieval_top_k
            )
        return {"tool_call_count": 1}

    def make_tool(run: RunContext):
        @tool
        def search_knowledge_base(query: str, top_k: int = 5, filters: MetadataFilters | None = None) -> str:
            """Search supplied documents with a standalone focused question and optional metadata filters."""
            if not query.strip() or len(query) > settings.max_message_chars or not 1 <= top_k <= 20:
                raise ValueError("Invalid search arguments")
            with telemetry.observation(
                "workflow.search-tool",
                input={"query": preview(query, settings.langfuse_preview_chars)},
                metadata={"query_hash": value_hash(query), "top_k": top_k},
            ) as observation:
                chunks = retriever.search(query, top_k, filters)
                observation.update(
                    output={"result_count": len(chunks), "chunk_ids": [str(c.chunk_id) for c in chunks]},
                    metadata={"filters": filters.model_dump(exclude_none=True) if filters else {}},
                )
            known = {chunk.chunk_id for chunk in run.evidence}
            for chunk in chunks:
                if chunk.chunk_id not in known:
                    run.evidence.append(chunk)
                    known.add(chunk.chunk_id)
            run.search_queries.append(query)
            return json.dumps(evidence_payload(chunks))

        return search_knowledge_base

    def agent(state: ChatState, runtime: Runtime[RunContext]):
        run = runtime.context
        count = state["tool_call_count"]
        if count >= settings.agent_max_tool_calls or run.sufficient:
            return {"continue_agent": False}
        if not run.agent_messages:
            run.agent_messages.append(
                HumanMessage(
                    json.dumps(
                        {
                            "question": state["standalone_question"],
                            "search_budget": settings.agent_max_tool_calls,
                        }
                    )
                )
            )
        response = model.agent(run.agent_messages, [make_tool(run)])
        if not response.tool_calls:
            return {"continue_agent": False}
        # Execute one call only; never allow an oversized parallel batch to bypass the budget.
        call = response.tool_calls[0]
        if call["name"] != "search_knowledge_base":
            return {"continue_agent": False}
        args = call["args"]
        filters = MetadataFilters.model_validate(args.get("filters") or {})
        signature = json.dumps(
            {
                "query": " ".join(str(args.get("query", "")).lower().split()),
                "filters": filters.model_dump(exclude_none=True),
            },
            sort_keys=True,
        )
        if signature in run.seen_searches:
            return {"continue_agent": False}
        run.seen_searches.add(signature)
        run.pending = AIMessage(content="", tool_calls=[call])
        return {"continue_agent": True}

    def search(state: ChatState, runtime: Runtime[RunContext]):
        run = runtime.context
        with telemetry.observation("workflow.agent-search") as observation:
            assert run.pending is not None
            node = ToolNode([make_tool(run)], handle_tool_errors=False)
            output = node.invoke({"messages": [run.pending]})
            run.agent_messages.extend([run.pending, *output["messages"]])
            assessment = assess_evidence(state["standalone_question"], run, settings, model)
            if run.evidence:
                observation.update(metadata={"sufficient": run.sufficient})
                if not run.sufficient:
                    run.agent_messages.append(
                        HumanMessage(
                            "Evidence is still insufficient. Search a focused query for: "
                            + assessment.missing_information
                        )
                    )
            return {"tool_call_count": state["tool_call_count"] + 1}

    def evidence(state: ChatState, runtime: Runtime[RunContext]):
        run = runtime.context
        with telemetry.observation("workflow.evidence") as observation:
            decision = assess_evidence(state["standalone_question"], run, settings, model)
            if run.evidence:
                valid = set(decision.relevant_markers)
                observation.update(metadata={"sufficient": run.sufficient, "relevant_markers": len(valid)})
                if run.sufficient:
                    run.evidence = [chunk for i, chunk in enumerate(run.evidence, 1) if i in valid]
            return {"chunk_ids": [str(c.chunk_id) for c in run.evidence]}

    def generate(state: ChatState, runtime: Runtime[RunContext]):
        run = runtime.context
        if not run.sufficient:
            return {"answer": ABSTENTION, "abstained": True, "citations": []}
        payload = {
            "question": state["standalone_question"],
            "latest_message": state["user_message"],
            "evidence": evidence_payload(run.evidence),
        }
        for _ in range(2):
            result = model.structured("answer", payload, GroundedAnswer)
            if result.abstained:
                break
            try:
                answer = render_grounded_answer(result, run.evidence)
                citations: list[Citation] = build_citations(answer, run.evidence)
            except ValueError as exc:
                payload["repair_feedback"] = str(exc)
                continue
            support = model.structured("support", {**payload, "answer": answer}, SupportDecision)
            if support.supported:
                return {
                    "answer": answer,
                    "abstained": False,
                    "citations": [c.model_dump(mode="json") for c in citations],
                }
            payload["repair_feedback"] = support.reason
        return {"answer": ABSTENTION, "abstained": True, "citations": []}

    def finish(state: ChatState, runtime: Runtime[RunContext]):
        logger.info(
            "chat_completed",
            extra={
                "workflow": state["route"],
                "search_count": state["tool_call_count"],
                "abstained": state["abstained"],
            },
        )
        # Only visible messages persist; no tool transcripts or large evidence blobs.
        return {
            "messages": [
                HumanMessage(state["user_message"]),
                AIMessage(
                    state["answer"],
                    additional_kwargs={
                        "saved_answer": {
                            "version": 1,
                            "workflow": state["route"],
                            "route_reason": state["route_reason"],
                            "citations": state["citations"],
                            "abstained": state["abstained"],
                        }
                    },
                ),
            ]
        }

    graph = StateGraph(ChatState, context_schema=RunContext)
    for name, node in [
        ("contextualize", contextualize),
        ("router", route),
        ("direct", direct),
        ("clarify", clarify),
        ("simple_rag", simple),
        ("agentic_rag", agent),
        ("search", search),
        ("evidence", evidence),
        ("generate", generate),
        ("finish", finish),
    ]:
        graph.add_node(name, node)  # type: ignore[call-overload]
    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {name: name for name in ("direct", "clarify", "simple_rag", "agentic_rag")},
    )
    graph.add_edge("simple_rag", "evidence")
    graph.add_conditional_edges(
        "agentic_rag", lambda state: "search" if state["continue_agent"] else "evidence"
    )
    graph.add_edge("search", "agentic_rag")
    graph.add_edge("evidence", "generate")
    for name in ("generate", "direct", "clarify"):
        graph.add_edge(name, "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)

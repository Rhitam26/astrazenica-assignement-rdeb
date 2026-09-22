"""Deterministic provider boundary; all production graph nodes still execute."""

from uuid import uuid4

from langchain_core.messages import AIMessage

from src.assistant.models import RetrievedChunk


def chunk(**updates):
    fields = dict(
        chunk_id=uuid4(),
        content_hash="abc",
        doc_id="vectors",
        doc_title="Vector databases",
        source_path="pdfs/vectors.pdf",
        heading_path=["Indexes", "ANN"],
        chapter_title="Indexes",
        section_title="ANN",
        subsection_title=None,
        page_numbers=[4, 5],
        page_start=4,
        page_end=5,
        content_type="table",
        table_id="#/tables/1",
        chunk_index=0,
        text="HNSW uses a graph. IVF partitions vectors. PostgreSQL provides ACID transactions.",
        score=0.7,
    )
    return RetrievedChunk(**{**fields, **updates})


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = [chunk()] if chunks is None else chunks
        self.queries = []

    def search(self, query, top_k=5, filters=None):
        self.queries.append((query, filters))
        return self.chunks


class FakeModel:
    def __init__(self, searches=None, unsupported=False, fake_citations=False):
        self.payloads = []
        self.agent_calls = 0
        self.searches = searches or ["pgvector ACID", "Pinecone operations"]
        self.unsupported, self.fake_citations = unsupported, fake_citations

    def structured(self, task, payload, schema):
        self.payloads.append((task, payload))
        if task == "context_route":
            latest = payload["latest_message"]
            history = " ".join(m["content"] for m in payload["history"])
            vague = any(x in latest.lower() for x in ("which one", "example", "why?"))
            unresolved = vague and not history
            question = latest
            if vague and "HNSW" in history:
                question = "Explain HNSW and IVF with examples."
            if vague and "Pinecone" in history:
                question = "Between pgvector and Pinecone, which supports ACID?"
            result = dict(
                standalone_question=question,
                unresolved=unresolved,
                clarification="Which technologies do you mean?" if unresolved else "",
            )
            q = question.lower()
            route = (
                "direct"
                if q in ("hi", "thanks")
                else "agentic_rag"
                if any(x in q for x in ("compare", "between"))
                else "simple_rag"
            )
            result.update(
                route="clarify" if unresolved else route, reason="Test classification", confidence=0.95
            )
        elif task == "evidence":
            result = dict(
                sufficient=not self.unsupported
                and ("compare" not in payload["question"].lower() or self.agent_calls >= len(self.searches)),
                relevant_markers=[1],
                missing_information="",
            )
        elif task == "direct":
            result = dict(answer="Hello!", abstained=False)
        elif task == "answer":
            result = dict(
                blocks=[dict(text="HNSW uses a graph.", evidence_markers=[99 if self.fake_citations else 1])],
                abstained=False,
            )
        elif task == "support":
            result = dict(supported=not self.unsupported, reason="Supported by passage")
        else:
            raise AssertionError(task)
        return schema.model_validate(result)

    def agent(self, messages, tools):
        self.agent_calls += 1
        n = sum(message.type == "tool" for message in messages)
        if n >= len(self.searches):
            return AIMessage(content="Evidence collected")
        return AIMessage(
            content="",
            tool_calls=[
                dict(
                    name="search_knowledge_base",
                    args={"query": self.searches[n], "top_k": 5},
                    id=str(n),
                    type="tool_call",
                )
            ],
        )


class DeterministicEmbedder:
    def embed_texts(self, texts):
        return [[1.0, *([0.0] * 3071)] for _ in texts]

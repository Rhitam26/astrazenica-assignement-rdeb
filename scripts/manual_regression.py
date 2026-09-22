"""Opt-in live API regression runner; performs paid provider calls through FastAPI."""

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx


def main():
    base = os.getenv("API_BASE_URL", "http://localhost:8000")
    a, b, isolated = str(uuid4()), str(uuid4()), str(uuid4())
    cases = [
        ("A", a, "Explain HNSW and IVF."),
        ("B", a, "I did not understand, explain with example."),
        ("C1", b, "Compare pgvector and Pinecone."),
        ("C2", b, "Which one gives me ACID guarantees?"),
        (
            "D",
            str(uuid4()),
            "Compare pgvector, Pinecone and Weaviate for a production RAG application considering cost, hybrid search, operations and scalability.",
        ),
        (
            "E",
            str(uuid4()),
            "Which NVIDIA GPU should I buy? Recommend an exact model and price.",
        ),
        ("F", isolated, "Which one gives me ACID guarantees?"),
    ]
    results = []
    with httpx.Client(base_url=base, timeout=900) as client:
        for name, cid, message in cases:
            response = client.post("/v1/chat", json={"message": message, "conversation_id": cid})
            result = {"scenario": name, "status": response.status_code, "response": response.json()}
            results.append(result)
            print(json.dumps(result), flush=True)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/manual_regression.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

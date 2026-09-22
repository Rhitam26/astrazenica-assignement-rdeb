from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from src.assistant.grounding import build_citations, page_label
from src.assistant.models import MetadataFilters
from src.assistant.repository import filter_sql, retrieve_relevant_chunks
from src.shared.config import Settings
from tests.fakes import chunk


@pytest.mark.parametrize("failure", [None, "validate", "embed", "search"])
def test_retriever_releases_connections_around_provider_and_errors(monkeypatch, failure):
    from src.assistant import retrieval

    active = 0
    borrowed = 0
    connection = object()

    @contextmanager
    def factory():
        nonlocal active, borrowed
        active += 1
        borrowed += 1
        try:
            yield connection
        finally:
            active -= 1

    def validate(conn, settings):
        assert conn is connection and active == 1
        if failure == "validate":
            raise RuntimeError("validation")

    class Embedder:
        def embed_texts(self, texts):
            assert active == 0
            if failure == "embed":
                raise RuntimeError("embedding")
            return [[1.0] * 3072]

    def search(conn, vector, **kwargs):
        assert conn is connection and active == 1
        if failure == "search":
            raise RuntimeError("search")
        return [chunk()]

    monkeypatch.setattr(retrieval, "validate_embeddings", validate)
    monkeypatch.setattr(retrieval, "retrieve_relevant_chunks", search)
    retriever = retrieval.PgRetriever(Settings(_env_file=None), Embedder(), connection_factory=factory)
    if failure:
        with pytest.raises(RuntimeError):
            retriever.search("question")
    else:
        assert len(retriever.search("question")) == 1
    assert active == 0
    assert borrowed == (1 if failure in ("validate", "embed") else 2)


def test_metadata_and_citations():
    record = chunk()
    citation = build_citations("A graph [1].", [record])[0]
    assert citation.chunk_id == record.chunk_id
    assert citation.heading_path == ["Indexes", "ANN"]
    assert citation.page_numbers == [4, 5]
    assert citation.table_id == "#/tables/1"
    assert citation.text_preview == record.text
    assert page_label(citation.model_dump()) == "pp. 4–5"
    assert page_label({"page_numbers": [1, 3]}) == "pp. 1, 3"
    with pytest.raises(ValueError):
        build_citations("Invented [99]", [record])
    with pytest.raises(ValueError):
        build_citations("Invented [fake-id]", [record])


def test_citation_text_preview_is_normalized_and_bounded():
    record = chunk(text="  First line.\n\nSecond   line. " + ("x" * 500))
    citation = build_citations("A graph [1].", [record])[0]
    assert citation.text_preview.startswith("First line. Second line.")
    assert len(citation.text_preview) == 401
    assert citation.text_preview.endswith("…")


def test_parameterized_filters():
    attack = "'; DROP TABLE chunks; --"
    clauses, values = filter_sql(MetadataFilters(section_title=attack, page_numbers=[4], doc_ids=["x"]))
    assert attack not in " ".join(clauses)
    assert attack in values
    assert "c.page_numbers && %s::integer[]" in clauses
    with pytest.raises(ValidationError):
        MetadataFilters.model_validate({"raw_sql": "true"})


def test_row_mapping_and_sql():
    record = chunk()

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, values):
            assert "halfvec(3072)" in sql
            assert "JOIN documents" in sql
            assert "private-title" not in sql
            assert ["private-title"] in values
            return self

        def fetchall(self):
            return [record.model_dump()]

    class Conn:
        def execute(self, *args):
            pass

        def cursor(self, **kwargs):
            return Cursor()

    result = retrieve_relevant_chunks(
        Conn(),
        [1.0] * 3072,
        filters=MetadataFilters(doc_titles=["private-title"]),
        settings=Settings(_env_file=None),
    )
    assert result == [record]


def test_block_citations_strip_only_valid_duplicates():
    from src.assistant.grounding import render_grounded_answer
    from src.assistant.models import GroundedAnswer, GroundedBlock

    evidence = [chunk()]
    result = GroundedAnswer(
        blocks=[GroundedBlock(text="A graph. Evidence markers: [1]", evidence_markers=[1])], abstained=False
    )
    assert render_grounded_answer(result, evidence) == "A graph. [1]"
    result.blocks[0].text = "A graph. Evidence markers: 1."
    assert render_grounded_answer(result, evidence) == "A graph. [1]"
    result.blocks[0].text = "A graph. Evidence markers: 99."
    with pytest.raises(ValueError):
        render_grounded_answer(result, evidence)
    result.blocks[0].text = "A graph [99]."
    with pytest.raises(ValueError):
        render_grounded_answer(result, evidence)
    result.blocks[0].text = "Uncited"
    result.blocks[0].evidence_markers = []
    with pytest.raises(ValueError):
        render_grounded_answer(result, evidence)

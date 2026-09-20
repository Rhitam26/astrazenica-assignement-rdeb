# """Docling-based structural chunking.

# Uses docling's HybridChunker so chunk boundaries respect the document's
# actual heading hierarchy (chapter / section / subsection) instead of a
# fixed character count, and tables are detected and tagged as their own
# chunk type rather than being split mid-row by a naive text splitter.
# """
# from __future__ import annotations

# import hashlib
# import logging
# from dataclasses import dataclass, field
# from pathlib import Path

# import tiktoken
# from docling.chunking import HybridChunker
# from docling.document_converter import DocumentConverter
# from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
# from docling_core.types.doc.labels import DocItemLabel

# from src.config import get_settings

# logger = logging.getLogger(__name__)


# @dataclass
# class ChunkRecord:
#     doc_id: str
#     doc_title: str
#     chapter_title: str | None
#     section_title: str | None
#     subsection_title: str | None
#     page_number: int | None
#     content_type: str  # "text" | "table"
#     chunk_index: int
#     text: str
#     token_count: int
#     content_hash: str = field(init=False)

#     def __post_init__(self) -> None:
#         # Idempotency key: unchanged text -> unchanged hash -> ingestion skips it.
#         self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# def _get_tiktoken_encoding(model_name: str) -> tiktoken.Encoding:
#     try:
#         return tiktoken.encoding_for_model(model_name)
#     except KeyError:
#         # text-embedding-3-* models use cl100k_base; fall back to it safely.
#         return tiktoken.get_encoding("cl100k_base")


# def _classify_content_type(doc_items) -> str:
#     labels = {item.label for item in doc_items}
#     if DocItemLabel.TABLE in labels:
#         return "table"
#     return "text"


# def _heading_levels(headings: list[str]) -> tuple[str | None, str | None, str | None]:
#     """Docling returns the heading trail outermost-first, e.g.
#     ["3 Pinecone", "Metadata Filtering"]. Map onto chapter / section /
#     subsection columns so they're independently filterable in SQL.
#     """
#     chapter = headings[0] if len(headings) > 0 else None
#     section = headings[1] if len(headings) > 1 else None
#     subsection = headings[2] if len(headings) > 2 else None
#     return chapter, section, subsection


# def chunk_pdf(pdf_path: Path, doc_id: str, doc_title: str) -> list[ChunkRecord]:
#     """Convert a single PDF and return structurally-aware chunks with metadata."""
#     settings = get_settings()
#     encoding = _get_tiktoken_encoding(settings.embedding_model)

#     converter = DocumentConverter()
#     doc = converter.convert(str(pdf_path)).document

#     tokenizer = OpenAITokenizer(tokenizer=encoding, max_tokens=settings.max_chunk_tokens)
#     chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
#     raw_chunks = list(chunker.chunk(dl_doc=doc))

#     records: list[ChunkRecord] = []
#     for idx, chunk in enumerate(raw_chunks):
#         # contextualize() prefixes the chunk text with its heading trail,
#         # which measurably improves embedding quality for section-scoped
#         # content (e.g. a table of numbers means little without its heading).
#         enriched_text = chunker.contextualize(chunk=chunk)

#         doc_items = chunk.meta.doc_items
#         content_type = _classify_content_type(doc_items)

#         page_number = None
#         for item in doc_items:
#             if item.prov:
#                 page_number = item.prov[0].page_no
#                 break

#         chapter, section, subsection = _heading_levels(chunk.meta.headings or [])

#         records.append(
#             ChunkRecord(
#                 doc_id=doc_id,
#                 doc_title=doc_title,
#                 chapter_title=chapter,
#                 section_title=section,
#                 subsection_title=subsection,
#                 page_number=page_number,
#                 content_type=content_type,
#                 chunk_index=idx,
#                 text=enriched_text,
#                 token_count=len(encoding.encode(enriched_text)),
#             )
#         )

#     table_count = sum(1 for r in records if r.content_type == "table")
#     logger.info("Chunked %s -> %d chunks (%d table chunks)", pdf_path.name, len(records), table_count)
#     return records



from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import tiktoken

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc.labels import DocItemLabel

from config import get_settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    doc_id: str
    doc_title: str

    # Complete hierarchy rather than only chapter/section/subsection.
    heading_path: list[str]

    # Convenience fields for SQL filtering.
    chapter_title: str | None
    section_title: str | None
    subsection_title: str | None

    # Provenance.
    page_numbers: list[int]
    page_start: int | None
    page_end: int | None

    # Content metadata.
    content_type: str  # "text" | "table"
    chunk_index: int
    text: str
    token_count: int

    # Optional logical table identifier.
    table_id: str | None = None

    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """
        Stable content hash used for ingestion idempotency.
        """
        self.content_hash = hashlib.sha256(
            self.text.encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _get_tiktoken_encoding(model_name: str) -> tiktoken.Encoding:
    """
    Get the tokenizer associated with the embedding model.

    This is local tokenization. It does NOT call an API.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Safe fallback for embedding models that aren't registered
        # directly in tiktoken.
        return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Heading handling
# ---------------------------------------------------------------------------

def _get_heading_path(chunk) -> list[str]:
    """
    Return the complete heading hierarchy for a chunk.

    Example:

        [
            "1 Introduction",
            "1.1 Background",
            "1.1.1 Disease mechanism",
            "1.1.1.1 Molecular pathway",
        ]

    We deliberately do NOT truncate this to 3 levels.
    """
    headings = chunk.meta.headings or []

    return [
        heading.strip()
        for heading in headings
        if heading and heading.strip()
    ]


def _heading_fields(
    heading_path: list[str],
) -> tuple[str | None, str | None, str | None]:
    """
    Populate convenience columns.

    The complete hierarchy remains in `heading_path`.

    These fields are primarily useful for SQL filtering.
    """

    chapter = heading_path[0] if len(heading_path) > 0 else None
    section = heading_path[1] if len(heading_path) > 1 else None
    subsection = heading_path[2] if len(heading_path) > 2 else None

    return chapter, section, subsection


# ---------------------------------------------------------------------------
# Provenance / page handling
# ---------------------------------------------------------------------------

def _get_page_numbers(doc_items) -> list[int]:
    """
    Return ALL unique PDF pages represented by the document items
    belonging to this chunk.

    Example:

        [42]

    or:

        [42, 43, 44]
    """

    pages: set[int] = set()

    for item in doc_items:
        for prov in (item.prov or []):
            if prov.page_no is not None:
                pages.add(prov.page_no)

    return sorted(pages)


def _get_page_range(
    page_numbers: list[int],
) -> tuple[int | None, int | None]:
    """
    Return first and last page represented by the chunk.
    """

    if not page_numbers:
        return None, None

    return min(page_numbers), max(page_numbers)


# ---------------------------------------------------------------------------
# Content classification
# ---------------------------------------------------------------------------

def _classify_content_type(doc_items) -> str:
    """
    Identify whether the chunk contains a table.

    If any document item is a TABLE, classify the chunk as a table.
    """

    labels = {
        item.label
        for item in doc_items
        if item.label is not None
    }

    if DocItemLabel.TABLE in labels:
        return "table"

    return "text"


# ---------------------------------------------------------------------------
# Table identification
# ---------------------------------------------------------------------------

def _get_table_items(doc_items):
    """
    Return all table items contained in this chunk.
    """

    return [
        item
        for item in doc_items
        if item.label == DocItemLabel.TABLE
    ]


def _get_table_id(table_item) -> str | None:
    """
    Return a stable identifier for the Docling table.

    Docling TableItems expose `self_ref`, for example:

        #/tables/3

    We use that as the logical table identifier.
    """

    return getattr(table_item, "self_ref", None)


# ---------------------------------------------------------------------------
# Table serialization
# ---------------------------------------------------------------------------

def _serialize_table(table_item, doc) -> str:
    """
    Serialize a Docling table to Markdown.

    This preserves the logical table structure sufficiently well
    for normal RAG / embedding use.

    For downstream workflows requiring exact rowspan/colspan semantics,
    HTML or Docling's structured representation can be preferable.
    """

    return table_item.export_to_markdown(doc=doc)


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_pdf(
    pdf_path: Path,
    doc_id: str,
    doc_title: str,
) -> list[ChunkRecord]:
    """
    Convert a PDF and produce structurally-aware RAG chunks.

    Features:

    1. Heading-aware chunking.
    2. Complete heading trail.
    3. All pages represented in each chunk.
    4. Page start/end range.
    5. Table-aware chunking.
    6. Repeated table headers when tables span chunks.
    7. Stable content hash.
    """

    settings = get_settings()

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------

    encoding = _get_tiktoken_encoding(
        settings.embedding_model
    )

    tokenizer = OpenAITokenizer(
        tokenizer=encoding,
        max_tokens=settings.max_chunk_tokens,
    )

    # ------------------------------------------------------------
    # PDF conversion
    # ------------------------------------------------------------

    converter = DocumentConverter()

    conversion_result = converter.convert(
        str(pdf_path)
    )

    doc = conversion_result.document

    # ------------------------------------------------------------
    # Hybrid chunker
    # ------------------------------------------------------------

    chunker = HybridChunker(
        tokenizer=tokenizer,

        # Allows compatible neighbouring chunks to be merged.
        merge_peers=True,

        # IMPORTANT:
        # If a table is too large for one chunk, repeat its
        # header in subsequent chunks.
        repeat_table_header=True,
    )

    raw_chunks = list(
        chunker.chunk(dl_doc=doc)
    )

    # ------------------------------------------------------------
    # Build ChunkRecords
    # ------------------------------------------------------------

    records: list[ChunkRecord] = []

    for idx, chunk in enumerate(raw_chunks):

        doc_items = chunk.meta.doc_items

        # --------------------------------------------------------
        # Content type
        # --------------------------------------------------------

        content_type = _classify_content_type(
            doc_items
        )

        # --------------------------------------------------------
        # Complete heading hierarchy
        # --------------------------------------------------------

        heading_path = _get_heading_path(chunk)
        (chapter,section,subsection,) = _heading_fields(heading_path)

        # --------------------------------------------------------
        # Page provenance
        # --------------------------------------------------------

        page_numbers = _get_page_numbers(doc_items)

        (page_start,page_end,) = _get_page_range(page_numbers)

        # --------------------------------------------------------
        # Table ID
        # --------------------------------------------------------

        table_items = _get_table_items(
            doc_items
        )

        table_id = None

        if table_items:
            table_id = _get_table_id(
                table_items[0]
            )

        # --------------------------------------------------------
        # Contextualized text
        # --------------------------------------------------------

        enriched_text = chunker.contextualize(
            chunk=chunk
        )

        # --------------------------------------------------------
        # Token count
        # --------------------------------------------------------

        token_count = len(
            encoding.encode(enriched_text)
        )

        # --------------------------------------------------------
        # Record
        # --------------------------------------------------------

        records.append(
            ChunkRecord(
                doc_id=doc_id,
                doc_title=doc_title,

                heading_path=heading_path,

                chapter_title=chapter,
                section_title=section,
                subsection_title=subsection,

                page_numbers=page_numbers,
                page_start=page_start,
                page_end=page_end,

                content_type=content_type,

                chunk_index=idx,

                text=enriched_text,

                token_count=token_count,

                table_id=table_id,
            )
        )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    table_count = sum(
        1
        for record in records
        if record.content_type == "table"
    )

    multi_page_count = sum(
        1
        for record in records
        if len(record.page_numbers) > 1
    )

    logger.info(
        "Chunked %s -> %d chunks "
        "(%d table chunks, %d multi-page chunks)",
        pdf_path.name,
        len(records),
        table_count,
        multi_page_count,
    )

    return records
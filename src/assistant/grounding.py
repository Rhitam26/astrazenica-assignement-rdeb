"""Deterministic evidence identifiers and citation validation."""

import re

from src.assistant.models import Citation, GroundedAnswer, RetrievedChunk

ABSTENTION = "I could not find sufficient information in the supplied knowledge base to answer that question."


def build_citations(answer: str, evidence: list[RetrievedChunk]) -> list[Citation]:
    groups = re.findall(r"\[([^\]]+)\]", answer)
    if not groups or any(not group.isdigit() for group in groups):
        raise ValueError("Missing or malformed citation markers")
    markers = sorted({int(group) for group in groups})
    if any(marker < 1 or marker > len(evidence) for marker in markers):
        raise ValueError("Unknown citation marker")
    return [evidence[marker - 1].citation(marker) for marker in markers]


def evidence_payload(chunks: list[RetrievedChunk]) -> list[dict]:
    # No paths/pages/IDs to copy or fabricate. The model only sees stable ordinal labels.
    return [
        {"marker": i, "text": chunk.text, "score": chunk.score, "content_type": chunk.content_type}
        for i, chunk in enumerate(chunks, 1)
    ]


def page_label(citation: dict) -> str:
    pages = sorted(set(citation.get("page_numbers", [])))
    if not pages:
        return "page unavailable"
    if len(pages) == 1:
        return f"p. {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pp. {pages[0]}–{pages[-1]}"
    return "pp. " + ", ".join(map(str, pages))


def render_grounded_answer(result: GroundedAnswer, evidence: list[RetrievedChunk]) -> str:
    if not result.blocks:
        raise ValueError("No grounded answer blocks")
    rendered = []
    for block in result.blocks:
        markers = sorted(set(block.evidence_markers))
        if not block.text.strip() or not markers:
            raise ValueError("Every answer block must have evidence markers and no inline citations")
        if any(marker < 1 or marker > len(evidence) for marker in markers):
            raise ValueError("Unknown citation marker")

        # Some providers redundantly include inline citations despite the block schema.
        # Remove only validated duplicates; never turn a fabricated reference into a valid one.
        def remove_duplicate(match: re.Match) -> str:
            inline = {int(value.strip()) for value in match.group(1).split(",")}
            if not inline.issubset(set(markers)):
                raise ValueError("Inline citation is not supported by the block markers")
            return ""

        text = re.sub(r"(?i)(?:evidence markers?:\s*)?\[(\d+(?:\s*,\s*\d+)*)\]", remove_duplicate, block.text)
        if "[" in text or "]" in text:
            raise ValueError("Malformed inline citation")
        rendered.append(text.strip() + " " + "".join(f"[{marker}]" for marker in markers))
    return "\n\n".join(rendered)

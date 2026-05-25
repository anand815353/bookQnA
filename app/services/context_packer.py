# app/services/context_packer.py
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.services.observability import (
    maybe_traceable,
    trace_process_inputs_pack_context,
    trace_process_outputs_pack_context,
)
from app.settings import CONTEXT_MAX_CHARS, INCLUDE_PARENT_CONTEXT

CHILD_SNIPPET_CHARS = 850
PARENT_CONTEXT_CHARS = 2300


def _compact_text(text: str, max_chars: int) -> str:
    compacted = re.sub(r"\s+", " ", (text or "").strip())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max(0, max_chars - 3)].rstrip() + "..."


def _citation_trace_id(citation: dict) -> str:
    return (
        f"{citation.get('book_id', '')}:"
        f"p{citation.get('page_start', 0)}-{citation.get('page_end', 0)}:"
        f"{citation.get('section_title') or citation.get('chapter_title') or 'unknown'}"
    )


def _snippet_fingerprint(book_id: str, page_start: int, page_end: int, snippet: str) -> str:
    raw = f"{book_id}|{page_start}|{page_end}|{snippet[:200]}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


@maybe_traceable(
    run_type="chain",
    name="pack_context_from_parent_evidence",
    process_inputs=trace_process_inputs_pack_context,
    process_outputs=trace_process_outputs_pack_context,
)
def pack_context_from_parent_evidence(
    parent_evidence: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
    include_parent_context: bool | None = None,
) -> dict[str, Any]:
    """
    Build LLM context string and parallel citations from parent_evidence.
    Enforces max_chars; dedupes overlapping child snippets; preserves citation metadata.
    """
    budget = int(max_chars if max_chars is not None else CONTEXT_MAX_CHARS)
    include_parent = INCLUDE_PARENT_CONTEXT if include_parent_context is None else bool(include_parent_context)

    sorted_parents = sorted(
        parent_evidence,
        key=lambda p: float(p.get("parent_score", 0.0)),
        reverse=True,
    )

    blocks_data: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()

    for parent in sorted_parents:
        children = parent.get("children") or []
        if not children:
            continue

        best_child = children[0]["doc"]
        best_child_score = float(children[0].get("score", 0.0))
        book_id = str(parent.get("book_id", best_child.metadata.get("book_id", "")))
        book_title = str(parent.get("book_title", best_child.metadata.get("book_title", "Unknown")))
        page_start = int(parent.get("page_start", best_child.metadata.get("page_start", 0)) or 0)
        page_end = int(parent.get("page_end", best_child.metadata.get("page_end", page_start)) or page_start)
        snippet = _compact_text(best_child.page_content, CHILD_SNIPPET_CHARS)
        fp = _snippet_fingerprint(book_id, page_start, page_end, snippet)
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        citation = {
            "book_id": book_id,
            "book_title": book_title,
            "page_start": page_start,
            "page_end": page_end,
            "snippet": snippet,
            "chapter_title": parent.get("chapter_title", best_child.metadata.get("chapter_title")),
            "subchapter_title": parent.get("subchapter_title", best_child.metadata.get("subchapter_title")),
            "section_title": parent.get("section_title", best_child.metadata.get("section_title")),
            "page_category": best_child.metadata.get("page_category"),
            "structure_source": parent.get("structure_source", best_child.metadata.get("structure_source")),
            "confidence": parent.get("confidence", best_child_score),
        }
        cid = _citation_trace_id(
            {
                "book_id": book_id,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": parent.get("section_title"),
                "chapter_title": parent.get("chapter_title"),
            }
        )

        child_lines = []
        for child in children:
            child_doc = child["doc"]
            child_page_start = int(child_doc.metadata.get("page_start", page_start) or page_start)
            child_page_end = int(child_doc.metadata.get("page_end", child_page_start) or child_page_start)
            child_snippet = _compact_text(child_doc.page_content, CHILD_SNIPPET_CHARS)
            child_lines.append(f"[p{child_page_start}-{child_page_end}] {child_snippet}")

        parent_excerpt = _compact_text(parent.get("parent_excerpt", ""), PARENT_CONTEXT_CHARS)
        child_block = "\n".join(child_lines)
        block = (
            f"Book: {book_title}\n"
            f"Chapter: {parent.get('chapter_title') or 'Unknown'}\n"
            f"Subchapter: {parent.get('subchapter_title') or 'Unknown'}\n"
            f"Section: {parent.get('section_title') or 'Unknown'}\n"
            f"Section Pages: {page_start}-{page_end}\n"
            f"Best Child Snippets:\n{child_block}\n"
        )
        if include_parent:
            block += f"Parent Excerpt: {parent_excerpt}"

        part = block.rstrip()
        blocks_data.append({"part": part, "citation": citation, "citation_id": cid})

    chars_before = len("\n\n".join(b["part"] for b in blocks_data)) if blocks_data else 0

    citations: list[dict] = []
    citation_ids: list[str] = []
    context_parts: list[str] = []
    for item in blocks_data:
        part = item["part"]
        projected = len("\n\n".join(context_parts)) + len(part) + (2 if context_parts else 0)
        if projected > budget:
            break
        context_parts.append(part)
        citations.append(item["citation"])
        citation_ids.append(item["citation_id"])

    context = "\n\n".join(context_parts)
    chars_after = len(context)
    return {
        "context": context,
        "citations": citations,
        "citation_ids": citation_ids,
        "context_chars_before": chars_before,
        "context_chars_after": chars_after,
    }

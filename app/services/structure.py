import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

FALLBACK_SECTION_PAGE_SPAN = 8
PARENT_EXCERPT_CHARS = 900
LOW_VALUE_PAGE_CATEGORIES = {
    "front_matter",
    "copyright",
    "acknowledgments",
    "toc",
    "index",
    "back_matter",
}


@dataclass
class SectionMetadata:
    section_id: str
    book_id: str
    level: int
    title: str
    chapter_title: str | None
    subchapter_title: str | None
    section_title: str
    page_start: int
    page_end: int
    text: str
    parent_section_id: str | None
    confidence: float | None
    source_of_structure: str

    def to_record(self) -> dict:
        return asdict(self)


def _normalize_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", (title or "").strip())
    return cleaned[:180]


def infer_page_category(*, page_text: str, page_no: int, total_pages: int) -> str:
    text = (page_text or "").strip()
    lower = text.lower()
    first_lines = [line.strip().lower() for line in text.splitlines()[:20] if line.strip()]

    def _has_any(terms: tuple[str, ...]) -> bool:
        return any(term in lower for term in terms)

    dot_leader_hits = len(re.findall(r"\.{4,}", text))
    toc_hits = len(re.findall(r"\bchapter\b|\bpart\b|\bsection\b", lower))
    if any(line in {"contents", "table of contents"} for line in first_lines) or (
        dot_leader_hits >= 3 and toc_hits >= 2
    ):
        return "toc"

    if _has_any(("copyright", "all rights reserved", "isbn", "published by")):
        return "copyright"

    if _has_any(("acknowledgments", "acknowledgements")):
        return "acknowledgments"

    if (first_lines and first_lines[0] == "index") or (
        page_no >= max(1, total_pages - 0.2 * total_pages)
        and _has_any(("index", "see also", "entries"))
    ):
        return "index"

    if page_no <= max(6, int(total_pages * 0.08)) and _has_any(
        (
            "preface",
            "foreword",
            "dedication",
            "contents",
            "introduction",
            "title page",
        )
    ):
        return "front_matter"

    if page_no >= max(1, int(total_pages * 0.85)) and _has_any(
        (
            "appendix",
            "bibliography",
            "references",
            "glossary",
            "about the author",
            "afterword",
            "notes",
        )
    ):
        return "back_matter"

    return "content"


def _page_number(doc: Document, fallback: int) -> int:
    if "page_start" in doc.metadata:
        return int(doc.metadata["page_start"])
    return int(doc.metadata.get("page", fallback - 1)) + 1


def _compute_page_end(entries: list[dict], index: int, total_pages: int) -> int:
    current = entries[index]
    page_start = int(current["page_start"])
    page_end = total_pages
    for next_entry in entries[index + 1 :]:
        if int(next_entry["level"]) <= int(current["level"]):
            page_end = int(next_entry["page_start"]) - 1
            break
    if page_end < page_start:
        page_end = page_start
    return max(page_start, min(page_end, total_pages))


def extract_toc_structure(file_path: Path, total_pages: int) -> list[dict]:
    entries: list[dict] = []
    try:
        with fitz.open(file_path) as pdf:
            toc = pdf.get_toc(simple=True)
    except Exception:
        logger.exception("structure_toc_read_failed file=%s", file_path)
        return entries

    seen = set()
    for item in toc:
        if len(item) < 3:
            continue
        level, title, page_start = item[0], item[1], int(item[2])
        normalized = _normalize_title(title)
        if not normalized:
            continue
        if page_start < 1 or page_start > total_pages:
            continue
        dedupe_key = (int(level), normalized, page_start)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        entries.append(
            {
                "level": max(1, int(level)),
                "title": normalized,
                "page_start": page_start,
            }
        )
    return entries


def _extract_heading_line(page_text: str) -> tuple[str, int] | None:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines[:14]:
        if len(line) < 5 or len(line) > 120:
            continue
        if re.fullmatch(r"[\d\W_]+", line):
            continue
        lower = line.lower()
        if lower.startswith(("chapter ", "part ", "book ")):
            return _normalize_title(line.title()), 1
        if lower.startswith(("section ", "subsection ")):
            return _normalize_title(line), 2
        if re.match(r"^(chapter|part)\s+[ivxlcdm\d]+(\b|:)", lower):
            return _normalize_title(line.title()), 1
        if re.match(r"^\d+(\.\d+){0,2}\s+\S+", line):
            return _normalize_title(line), 2
        words = line.split()
        if line.isupper() and 1 <= len(words) <= 10:
            return _normalize_title(line.title()), 1
    return None


def extract_heading_candidates(pages: list[Document]) -> list[dict]:
    entries: list[dict] = []
    last_page = 0
    last_title = None
    for idx, page in enumerate(pages, start=1):
        candidate = _extract_heading_line(page.page_content or "")
        if not candidate:
            continue
        title, level = candidate
        if title == last_title:
            continue
        if idx - last_page < 2 and level > 1:
            continue
        entries.append({"level": level, "title": title, "page_start": idx})
        last_page = idx
        last_title = title
    return entries


def _fallback_entries(total_pages: int) -> list[dict]:
    entries: list[dict] = []
    for page_start in range(1, total_pages + 1, FALLBACK_SECTION_PAGE_SPAN):
        page_end = min(total_pages, page_start + FALLBACK_SECTION_PAGE_SPAN - 1)
        entries.append(
            {
                "level": 1,
                "title": f"Pages {page_start}-{page_end}",
                "page_start": page_start,
            }
        )
    return entries


def _entries_to_sections(
    *,
    book_id: str,
    pages: list[Document],
    entries: list[dict],
    source_of_structure: str,
    confidence: float,
) -> list[SectionMetadata]:
    if not entries:
        return []

    total_pages = len(pages)
    sections: list[SectionMetadata] = []
    parent_stack: list[dict] = []

    for i, entry in enumerate(entries):
        level = max(1, int(entry["level"]))
        title = _normalize_title(str(entry["title"]))
        page_start = max(1, min(int(entry["page_start"]), total_pages))
        page_end = _compute_page_end(entries, i, total_pages)

        while parent_stack and int(parent_stack[-1]["level"]) >= level:
            parent_stack.pop()
        parent_section_id = parent_stack[-1]["section_id"] if parent_stack else None

        chapter_title = None
        if level == 1:
            chapter_title = title
        else:
            for parent in reversed(parent_stack):
                if int(parent["level"]) == 1:
                    chapter_title = parent["title"]
                    break

        subchapter_title = None
        if level == 2:
            subchapter_title = title
        elif level > 2:
            for parent in reversed(parent_stack):
                if int(parent["level"]) == 2:
                    subchapter_title = parent["title"]
                    break

        section_id = f"{book_id}::section::{source_of_structure}::{i + 1}"
        section_text = "\n\n".join(
            pages[page_no - 1].page_content
            for page_no in range(page_start, page_end + 1)
            if 1 <= page_no <= total_pages
        )

        section = SectionMetadata(
            section_id=section_id,
            book_id=book_id,
            level=level,
            title=title,
            chapter_title=chapter_title,
            subchapter_title=subchapter_title,
            section_title=title,
            page_start=page_start,
            page_end=page_end,
            text=section_text,
            parent_section_id=parent_section_id,
            confidence=confidence,
            source_of_structure=source_of_structure,
        )
        sections.append(section)
        parent_stack.append(
            {
                "section_id": section_id,
                "title": title,
                "level": level,
            }
        )

    return sections


def build_parent_sections(
    *,
    book_id: str,
    file_path: Path,
    pages: list[Document],
) -> list[SectionMetadata]:
    total_pages = len(pages)
    toc_entries = extract_toc_structure(file_path, total_pages)
    if toc_entries:
        logger.info("structure_source_selected source=toc entries=%s", len(toc_entries))
        return _entries_to_sections(
            book_id=book_id,
            pages=pages,
            entries=toc_entries,
            source_of_structure="toc",
            confidence=0.95,
        )

    heading_entries = extract_heading_candidates(pages)
    if heading_entries:
        logger.info("structure_source_selected source=heuristic entries=%s", len(heading_entries))
        return _entries_to_sections(
            book_id=book_id,
            pages=pages,
            entries=heading_entries,
            source_of_structure="heuristic",
            confidence=0.65,
        )

    fallback_entries = _fallback_entries(total_pages)
    logger.info("structure_source_selected source=fallback entries=%s", len(fallback_entries))
    return _entries_to_sections(
        book_id=book_id,
        pages=pages,
        entries=fallback_entries,
        source_of_structure="fallback",
        confidence=0.35,
    )


def _section_for_page(page_no: int, sections: list[SectionMetadata]) -> SectionMetadata | None:
    candidates = [
        section
        for section in sections
        if section.page_start <= page_no <= section.page_end
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda section: (
            section.level,
            -(section.page_end - section.page_start),
        ),
    )


def apply_section_metadata_to_pages(pages: list[Document], sections: list[SectionMetadata]) -> None:
    if not sections:
        return
    default_section = sections[0]
    total_pages = len(pages)
    for idx, page in enumerate(pages, start=1):
        page_no = _page_number(page, idx)
        section = _section_for_page(page_no, sections) or default_section
        page_category = infer_page_category(
            page_text=page.page_content,
            page_no=page_no,
            total_pages=total_pages,
        )
        page.metadata["page_start"] = page_no
        page.metadata["page_end"] = page_no
        page.metadata["section_id"] = section.section_id
        page.metadata["parent_section_id"] = section.section_id
        page.metadata["chapter_title"] = section.chapter_title or section.title
        page.metadata["subchapter_title"] = section.subchapter_title
        page.metadata["section_title"] = section.section_title
        page.metadata["structure_source"] = section.source_of_structure
        page.metadata["structure_confidence"] = section.confidence
        page.metadata["page_category"] = page_category
        page.metadata["is_low_value_page"] = page_category in LOW_VALUE_PAGE_CATEGORIES


def split_parent_sections_into_child_chunks(*, pages, parent_sections, splitter):
    if not pages:
        return []
    apply_section_metadata_to_pages(pages, parent_sections)

    chunks = []
    pages_by_section: dict[str, list[Document]] = {}
    for page in pages:
        section_id = page.metadata.get("section_id")
        if not section_id:
            continue
        pages_by_section.setdefault(section_id, []).append(page)

    for section in parent_sections:
        section_pages = pages_by_section.get(section.section_id)
        if not section_pages:
            continue
        parent_excerpt = re.sub(r"\s+", " ", (section.text or "").strip())[:PARENT_EXCERPT_CHARS]
        section_chunks = splitter.split_documents(section_pages)
        for chunk in section_chunks:
            chunk.metadata["section_id"] = section.section_id
            chunk.metadata["parent_section_id"] = section.section_id
            chunk.metadata["chapter_title"] = section.chapter_title or section.title
            chunk.metadata["subchapter_title"] = section.subchapter_title
            chunk.metadata["section_title"] = section.section_title
            chunk.metadata["structure_source"] = section.source_of_structure
            chunk.metadata["structure_confidence"] = section.confidence
            chunk.metadata["page_category"] = chunk.metadata.get("page_category", "content")
            chunk.metadata["is_low_value_page"] = bool(chunk.metadata.get("is_low_value_page", False))
            chunk.metadata["page_start"] = int(chunk.metadata.get("page_start", section.page_start))
            chunk.metadata["page_end"] = int(chunk.metadata.get("page_end", chunk.metadata["page_start"]))
            chunk.metadata["parent_page_start"] = section.page_start
            chunk.metadata["parent_page_end"] = section.page_end
            chunk.metadata["parent_excerpt"] = parent_excerpt
            chunk.metadata["contextual_text"] = (
                f"Book: {chunk.metadata.get('book_title', '')}. "
                f"Chapter: {chunk.metadata.get('chapter_title', '')}. "
                f"Section: {chunk.metadata.get('section_title', '')}."
            ).strip()
            chunks.append(chunk)
    return chunks

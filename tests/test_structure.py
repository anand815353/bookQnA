from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.structure import (
    build_parent_sections,
    extract_heading_candidates,
    split_parent_sections_into_child_chunks,
)


def _make_pages(texts: list[str]) -> list[Document]:
    pages = []
    for i, text in enumerate(texts):
        pages.append(Document(page_content=text, metadata={"page": i}))
    return pages


def test_extract_heading_candidates_detects_chapter_lines():
    pages = _make_pages(
        [
            "CHAPTER 1 INTRODUCTION\nThis is page one.",
            "Body only text.",
            "CHAPTER 2 METHODS\nMore content.",
        ]
    )

    headings = extract_heading_candidates(pages)

    assert len(headings) >= 2
    assert headings[0]["title"].lower().startswith("chapter 1")
    assert headings[0]["level"] == 1
    assert headings[0]["page_start"] == 1


def test_build_parent_sections_falls_back_when_no_structure():
    pages = _make_pages(
        [
            "plain text only",
            "another plain page",
            "more content",
            "still no heading",
        ]
    )

    sections = build_parent_sections(
        book_id="b1",
        file_path=Path("missing-file.pdf"),
        pages=pages,
    )

    assert sections
    assert sections[0].source_of_structure == "fallback"
    assert sections[0].page_start == 1
    assert sections[0].page_end >= sections[0].page_start


def test_split_parent_sections_into_child_chunks_carries_section_metadata():
    pages = _make_pages(
        [
            "CHAPTER 1 START\n" + ("A " * 700),
            "Continuation on page 2.\n" + ("B " * 700),
        ]
    )
    for page in pages:
        page.metadata["book_id"] = "book-1"
        page.metadata["book_title"] = "Book One"
        page.metadata["source_file"] = "original.pdf"

    sections = build_parent_sections(
        book_id="book-1",
        file_path=Path("missing-file.pdf"),
        pages=pages,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = split_parent_sections_into_child_chunks(
        pages=pages,
        parent_sections=sections,
        splitter=splitter,
    )

    assert chunks
    first = chunks[0].metadata
    assert first["section_id"].startswith("book-1::section::")
    assert first["parent_section_id"] == first["section_id"]
    assert "chapter_title" in first
    assert "section_title" in first
    assert "structure_source" in first
    assert "structure_confidence" in first
    assert "page_category" in first
    assert "is_low_value_page" in first
    assert "contextual_text" in first
    assert "parent_excerpt" in first
    assert "parent_page_start" in first
    assert "parent_page_end" in first
    assert first["page_start"] >= 1
    assert first["page_end"] >= first["page_start"]

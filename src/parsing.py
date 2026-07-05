"""
PDF text extraction with multiple parser backends.

Each parser function has the same signature:
    parse_<lib>(pdf_path: str) -> list[tuple[int, str]]

Returns a list of (page_number, text) tuples, where page_number is 0-indexed.
This common interface lets the Phase 1 pre-grid iterate over parsers generically.

Citations:
  - _instructions.md L26 (parse PDFs using multiple extraction libraries)
  - _instructions.md L82 (pdfplumber, PyPDF2, PyMuPDF)
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable


_CHAPTER_RE = re.compile(
    r"^(?:chapter|chap\.?|chapitre|cap[ií]tulo|part|title|t[ií]tulo)\s+[\w.\-]+|^第\s*[一二三四五六七八九十百千万\d]+\s*章",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(
    r"^(?:article|art[ií]culo|artigo|rule|regulation)\s+[\w.\-]+|^第\s*[一二三四五六七八九十百千万\d]+\s*条",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(
    r"^(?:section|sec\.|subsection|clause|paragraph|secci[oó]n|se[cç][aã]o)\s*[\w.\-]+|^第\s*[一二三四五六七八九十百千万\d]+\s*款",
    re.IGNORECASE,
)


def _legal_heading_level(text: str) -> int | None:
    """Infer Markdown heading depth from common legal heading labels."""
    stripped = text.strip(" #\t:")
    if _CHAPTER_RE.match(stripped):
        return 1
    if _ARTICLE_RE.match(stripped):
        return 2
    if _SECTION_RE.match(stripped):
        return 3
    return None


def _to_markdown_block(text: str, category: str | None = None) -> str:
    """Convert one parser element/block into Markdown while preserving headings."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return ""

    level = _legal_heading_level(text)
    if level is None and category:
        normalized_category = category.lower()
        if "title" in normalized_category or "heading" in normalized_category:
            level = 2

    if level is not None:
        return f"{'#' * level} {text.lstrip('#').strip()}"
    return text


def _group_markdown_by_page(elements: list[tuple[int, str, str | None]]) -> list[tuple[int, str]]:
    """Group (page, text, category) elements into page-level Markdown strings."""
    pages: dict[int, list[str]] = defaultdict(list)
    for page_number, text, category in elements:
        block = _to_markdown_block(text, category=category)
        if block:
            pages[page_number].append(block)

    if not pages:
        return []

    return [
        (page_number, "\n\n".join(pages.get(page_number, [])))
        for page_number in range(max(pages) + 1)
    ]


def parse_pdfplumber(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text using pdfplumber — good at tables and structured layouts."""
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def parse_pypdf2(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text using PyPDF2 (via pypdf) — lightweight, fast."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def parse_pymupdf(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text using PyMuPDF (fitz) — fast, good general-purpose extraction."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        pages.append((i, text))
    doc.close()
    return pages


def parse_pymupdf_layout_markdown(pdf_path: str) -> list[tuple[int, str]]:
    """Extract page text using PyMuPDF layout blocks and emit Markdown.

    This is a dependency-free fallback for policy PDFs. It preserves block
    order better than plain text extraction and promotes legal headings such
    as Chapter, Article, Section, Capítulo, Artículo, and 第X章 to Markdown.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(doc):
        blocks = page.get_text("blocks") or []
        elements: list[tuple[int, str, str | None]] = []
        for block in sorted(blocks, key=lambda b: (round(float(b[1]) / 5), float(b[0]))):
            text = str(block[4] or "").strip()
            elements.append((0, text, None))
        markdown = _group_markdown_by_page(elements)
        pages.append((page_number, markdown[0][1] if markdown else ""))
    doc.close()
    return pages


def parse_unstructured_markdown(pdf_path: str) -> list[tuple[int, str]]:
    """Extract layout-aware Markdown with Unstructured.

    Requires ``unstructured[pdf]``. Use ``strategy='hi_res'`` in production
    when legal PDFs contain complex tables or multi-column layouts.
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise ImportError(
            "parse_unstructured_markdown requires the optional dependency "
            "'unstructured[pdf]'."
        ) from exc

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
    )
    grouped_elements: list[tuple[int, str, str | None]] = []
    for element in elements:
        text = str(element).strip()
        metadata = getattr(element, "metadata", None)
        page_number = getattr(metadata, "page_number", 1) or 1
        category = getattr(element, "category", None)
        grouped_elements.append((int(page_number) - 1, text, category))
    return _group_markdown_by_page(grouped_elements)


def parse_marker_markdown(pdf_path: str) -> list[tuple[int, str]]:
    """Extract layout-aware Markdown with Marker when installed.

    Marker's public API has changed across releases, so this adapter supports
    the current Python API and raises a clear ImportError if unavailable.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise ImportError(
            "parse_marker_markdown requires the optional dependency 'marker-pdf'."
        ) from exc

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    markdown_text, _, _ = text_from_rendered(rendered)
    pages = markdown_text.split("\n\n---\n\n")
    return [(i, page.strip()) for i, page in enumerate(pages)]


def _page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


def parse_layout_markdown(pdf_path: str) -> list[tuple[int, str]]:
    """Preferred parser for compliance PDFs: layout-aware Markdown output.

    Resolution order:
    1. Marker, which is strongest at reconstructing Markdown document layout.
    2. Unstructured hi-res partitioning.
    3. PyMuPDF block-layout Markdown fallback for local tests/dev machines.
    """
    expected_pages = _page_count(pdf_path)
    for parser in (parse_marker_markdown, parse_unstructured_markdown):
        try:
            pages = parser(pdf_path)
            if len(pages) == expected_pages and any(text.strip() for _, text in pages):
                return pages
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
            continue
    return parse_pymupdf_layout_markdown(pdf_path)


# Registry for programmatic iteration in Phase 1 pre-grid
PARSERS: dict[str, Callable[[str], list[tuple[int, str]]]] = {
    "layout_markdown": parse_layout_markdown,
    "pdfplumber": parse_pdfplumber,
    "pypdf2": parse_pypdf2,
    "pymupdf": parse_pymupdf,
}

"""Contract tests for PDF parsers.

Verifies all parsers return the same shape — not testing library internals,
just that our interface contract holds across all three backends.
"""

import pytest

from src.parsing import PARSERS

PDF_PATH = "data/2022-annual-report.pdf"


@pytest.fixture(scope="module")
def parsed_results() -> dict[str, list[tuple[int, str]]]:
    """Parse the PDF once per module with all parsers."""
    return {name: fn(PDF_PATH) for name, fn in PARSERS.items()}


class TestParserContract:
    """All parsers must satisfy the same interface contract."""

    @pytest.mark.parametrize("parser_name", list(PARSERS.keys()))
    def test_returns_list_of_tuples(self, parsed_results, parser_name):
        """Each parser returns a list of (int, str) tuples."""
        pages = parsed_results[parser_name]
        assert isinstance(pages, list)
        for page_num, text in pages:
            assert isinstance(page_num, int)
            assert isinstance(text, str)

    @pytest.mark.parametrize("parser_name", list(PARSERS.keys()))
    def test_page_numbers_are_sequential(self, parsed_results, parser_name):
        """Page numbers should be 0-indexed and sequential."""
        pages = parsed_results[parser_name]
        page_nums = [p[0] for p in pages]
        assert page_nums == list(range(len(pages)))

    @pytest.mark.parametrize("parser_name", list(PARSERS.keys()))
    def test_no_none_text(self, parsed_results, parser_name):
        """Text should never be None (empty string is OK for blank pages)."""
        pages = parsed_results[parser_name]
        for _, text in pages:
            assert text is not None

    def test_all_parsers_same_page_count(self, parsed_results):
        """All parsers should find the same number of pages."""
        counts = {name: len(pages) for name, pages in parsed_results.items()}
        values = list(counts.values())
        assert all(v == values[0] for v in values), f"Page counts differ: {counts}"

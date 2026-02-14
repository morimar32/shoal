"""Tests for _parsers.py."""

from shoal._models import DocFormat
from shoal._parsers import parse_document


class TestMarkdownParser:
    def test_basic_sections(self):
        md = "# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
        sections = parse_document(md, DocFormat.markdown)
        assert len(sections) == 3
        assert sections[0].header_hierarchy == ["# Title"]
        assert "Intro text" in sections[0].text
        assert sections[1].header_hierarchy == ["# Title", "## Section A"]
        assert "Body A" in sections[1].text
        assert sections[2].header_hierarchy == ["# Title", "## Section B"]
        assert "Body B" in sections[2].text

    def test_nested_headers(self):
        md = "# H1\n\nA\n\n## H2\n\nB\n\n### H3\n\nC\n\n## H2b\n\nD\n"
        sections = parse_document(md, DocFormat.markdown)
        assert len(sections) == 4
        # H3 should have full hierarchy
        assert sections[2].header_hierarchy == ["# H1", "## H2", "### H3"]
        # H2b should pop H3 and H2 from stack
        assert sections[3].header_hierarchy == ["# H1", "## H2b"]

    def test_strips_edit_lines(self):
        md = "# Section\n\n[edit]\n\nSome text here.\n"
        sections = parse_document(md, DocFormat.markdown)
        assert len(sections) == 1
        assert "[edit]" not in sections[0].text

    def test_strips_main_article(self):
        md = "# Section\n\nMain article: Something\n\nActual text.\n"
        sections = parse_document(md, DocFormat.markdown)
        assert len(sections) == 1
        assert "Main article" not in sections[0].text

    def test_preamble_before_first_header(self):
        md = "Preamble text.\n\n# First Header\n\nBody.\n"
        sections = parse_document(md, DocFormat.markdown)
        assert len(sections) == 2
        assert sections[0].header_hierarchy == []
        assert "Preamble" in sections[0].text

    def test_empty_sections_skipped(self):
        md = "# Empty\n\n# Has Content\n\nSome text.\n"
        sections = parse_document(md, DocFormat.markdown)
        # The empty section between headers should be skipped
        assert all(s.text.strip() for s in sections)

    def test_character_offsets(self):
        md = "# Title\n\nBody text.\n"
        sections = parse_document(md, DocFormat.markdown)
        assert sections[0].start_char == 0
        assert sections[0].end_char == len(md)

    def test_no_headers(self):
        md = "Just some text without any headers.\n"
        sections = parse_document(md, DocFormat.markdown)
        assert len(sections) == 1
        assert sections[0].header_hierarchy == []

    def test_strips_boilerplate_sections(self):
        md = (
            "# Article\n\nContent here.\n\n"
            "## Details\n\nMore content.\n\n"
            "## References\n\n1. Some reference.\n2. Another ref.\n\n"
            "## External links\n\n* Link one\n* Link two\n"
        )
        sections = parse_document(md, DocFormat.markdown)
        # Only "Article" and "Details" should survive
        assert len(sections) == 2
        headers = [s.header_hierarchy[-1] for s in sections]
        assert "## References" not in headers
        assert "## External links" not in headers

    def test_strips_boilerplate_with_subsections(self):
        md = (
            "# Article\n\nContent.\n\n"
            "## See also\n\nStuff.\n\n"
            "### Related topics\n\nMore stuff.\n\n"
            "## Real section after\n\nGood content.\n"
        )
        sections = parse_document(md, DocFormat.markdown)
        texts = [s.text for s in sections]
        assert any("Content" in t for t in texts)
        assert any("Good content" in t for t in texts)
        assert not any("Stuff" in t for t in texts)
        assert not any("More stuff" in t for t in texts)

    def test_boilerplate_case_insensitive(self):
        md = "# Title\n\nBody.\n\n## REFERENCES\n\nJunk.\n"
        sections = parse_document(md, DocFormat.markdown)
        # "REFERENCES" lowercased matches "references"
        assert len(sections) == 1


class TestPlaintextParser:
    def test_single_section(self):
        text = "This is plain text.\nAnother line.\n"
        sections = parse_document(text, DocFormat.plaintext)
        assert len(sections) == 1
        assert sections[0].text == text
        assert sections[0].header_hierarchy == []
        assert sections[0].start_char == 0
        assert sections[0].end_char == len(text)

    def test_empty_text(self):
        sections = parse_document("", DocFormat.plaintext)
        assert len(sections) == 0

    def test_whitespace_only(self):
        sections = parse_document("   \n  \n  ", DocFormat.plaintext)
        assert len(sections) == 0

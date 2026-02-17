"""Tests for _parsers.py."""

from shoal._models import DocFormat
from shoal._parsers import parse_document, parse_sections


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


class TestParseSectionsMarkdown:
    def test_h2_sections(self):
        md = (
            "# Title\n\n"
            "## Section A\n\nBody A is here with some text.\n\n"
            "## Section B\n\nBody B is here with some text.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        assert len(sections) >= 2
        titles = [s.title for s in sections]
        assert "Section A" in titles
        assert "Section B" in titles
        assert all(s.depth == 0 for s in sections)

    def test_h3_subsections_long_enough(self):
        # H3 with enough text should become a depth-1 section
        long_text = "A " * 150  # 300 chars, above default min_section_length=200
        md = (
            "# Title\n\n"
            "## Parent\n\nParent body.\n\n"
            f"### Child\n\n{long_text}\n\n"
            "## Next\n\nNext body.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        child_sections = [s for s in sections if s.depth == 1]
        assert len(child_sections) == 1
        assert child_sections[0].title == "Child"
        assert child_sections[0].parent_index is not None

    def test_h3_folded_when_short(self):
        # Short H3 should be folded into parent H2
        md = (
            "# Title\n\n"
            "## Parent\n\nParent body text.\n\n"
            "### Short Child\n\nBrief.\n\n"
            "## Next\n\nNext body.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        # "Short Child" should not appear as a section
        titles = [s.title for s in sections]
        assert "Short Child" not in titles
        # But its text should be folded into Parent
        parent = next(s for s in sections if s.title == "Parent")
        assert "Short Child" in parent.analysis_text
        assert "Brief" in parent.analysis_text

    def test_h4_always_folded(self):
        long_text = "A " * 150  # Long enough normally
        md = (
            "# Title\n\n"
            "## Parent\n\nParent body.\n\n"
            f"#### Deep\n\n{long_text}\n\n"
            "## Next\n\nNext body.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        titles = [s.title for s in sections]
        assert "Deep" not in titles
        parent = next(s for s in sections if s.title == "Parent")
        assert "Deep" in parent.analysis_text

    def test_preamble_as_introduction(self):
        # Long preamble under H1 should become "Introduction" section
        long_preamble = "Intro text. " * 30  # Well above 200 chars
        md = (
            f"# Title\n\n{long_preamble}\n\n"
            "## Section A\n\nBody A.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        intro = [s for s in sections if s.title == "Introduction"]
        assert len(intro) == 1
        assert intro[0].depth == 0

    def test_short_preamble_prepended_to_first_h2(self):
        md = (
            "# Title\n\nBrief intro.\n\n"
            "## Section A\n\nBody A with some text.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        # "Introduction" should not be a separate section
        titles = [s.title for s in sections]
        assert "Introduction" not in titles
        # But preamble text should be in the first H2's analysis_text
        first = sections[0]
        assert "Brief intro" in first.analysis_text

    def test_no_headers_single_section(self):
        md = "Just some text without any headers.\n"
        sections = parse_sections(md, DocFormat.markdown)
        assert len(sections) == 1
        assert sections[0].title == "Document"

    def test_boilerplate_skipped(self):
        md = (
            "# Title\n\n"
            "## Content\n\nGood stuff.\n\n"
            "## References\n\nJunk.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        titles = [s.title for s in sections]
        assert "References" not in titles

    def test_section_char_offsets(self):
        md = (
            "# Title\n\n"
            "## Section A\n\nBody A.\n\n"
            "## Section B\n\nBody B.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        for s in sections:
            assert s.start_char >= 0
            assert s.end_char > s.start_char

    def test_parent_index_resolution(self):
        long_text = "A " * 150
        md = (
            "# Title\n\n"
            "## Parent\n\nParent body.\n\n"
            f"### Child\n\n{long_text}\n\n"
            "## Other\n\nOther body.\n"
        )
        sections = parse_sections(md, DocFormat.markdown)
        child = next((s for s in sections if s.title == "Child"), None)
        if child is not None:
            parent_idx = child.parent_index
            assert parent_idx is not None
            assert sections[parent_idx].title == "Parent"

    def test_empty_content(self):
        sections = parse_sections("", DocFormat.markdown)
        assert len(sections) == 0


class TestParseSectionsPlaintext:
    def test_chapter_detection(self):
        content = (
            "CHAPTER I. The Beginning\n\n"
            + "A " * 200 + "\n\n"
            + "CHAPTER II. The Middle\n\n"
            + "B " * 200 + "\n\n"
            + "CHAPTER III. The End\n\n"
            + "C " * 200
        )
        sections = parse_sections(content, DocFormat.plaintext)
        assert len(sections) >= 3
        titles = [s.title for s in sections]
        assert any("CHAPTER I" in t for t in titles)
        assert any("CHAPTER II" in t for t in titles)

    def test_chapter_with_front_matter(self):
        front = "Title Page\n\n" + "X " * 200 + "\n\n"
        content = (
            front
            + "CHAPTER I. First\n\n"
            + "A " * 200 + "\n\n"
            + "CHAPTER II. Second\n\n"
            + "B " * 200
        )
        sections = parse_sections(content, DocFormat.plaintext)
        if sections[0].title == "Front Matter":
            assert sections[0].depth == 0
            assert sections[0].position == 0

    def test_toc_filtering(self):
        # Dense chapter listings (TOC) should be filtered out
        toc = "\n".join(f"CHAPTER {num}. Title" for num in ["I", "II", "III", "IV", "V"])
        body1 = "A " * 200
        body2 = "B " * 200
        content = (
            f"{toc}\n\n"
            f"CHAPTER I. First Real\n\n{body1}\n\n"
            f"CHAPTER II. Second Real\n\n{body2}\n"
        )
        sections = parse_sections(content, DocFormat.plaintext)
        # TOC entries should be filtered, real chapters should survive
        assert len(sections) >= 2

    def test_no_chapters_single_section(self):
        content = "Just plain text without any chapter markers. Nothing special here."
        sections = parse_sections(content, DocFormat.plaintext)
        assert len(sections) == 1
        assert sections[0].title == "Document"
        assert sections[0].depth == 0

    def test_empty_text(self):
        sections = parse_sections("", DocFormat.plaintext)
        assert len(sections) == 0

    def test_whitespace_only(self):
        sections = parse_sections("   \n  \n  ", DocFormat.plaintext)
        assert len(sections) == 0

    def test_numeric_chapter_pattern(self):
        content = (
            "Chapter 1: Introduction\n\n"
            + "A " * 200 + "\n\n"
            + "Chapter 2: Methods\n\n"
            + "B " * 200 + "\n\n"
            + "Chapter 3: Results\n\n"
            + "C " * 200
        )
        sections = parse_sections(content, DocFormat.plaintext)
        assert len(sections) >= 3

"""Document format parsers for shoal."""

from __future__ import annotations

import re

from ._models import DocFormat, ParsedSection

# Patterns for Wikipedia markdown artifacts to strip
_EDIT_LINE = re.compile(r"^\[edit\]\s*$", re.MULTILINE)
_MAIN_ARTICLE = re.compile(r"^Main articles?:.*$", re.MULTILINE)

# Wikipedia boilerplate sections to skip entirely.
# These contain references, external links, navigation templates, etc.
# that produce massive junk chunks with inflated scores.
_BOILERPLATE_HEADERS = frozenset({
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "notes and references",
    "sources",
    "citations",
})


def parse_document(content: str, fmt: DocFormat) -> list[ParsedSection]:
    """Parse a document into sections based on format."""
    if fmt == DocFormat.markdown:
        return _parse_markdown(content)
    return _parse_plaintext(content)


def _parse_markdown(content: str) -> list[ParsedSection]:
    """Split markdown at #-level headers, tracking header hierarchy."""
    sections: list[ParsedSection] = []
    # Split into (header_line, body_text) pairs
    # Match lines starting with one or more # characters
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    matches = list(header_pattern.finditer(content))

    if not matches:
        # No headers — treat entire content as a single section
        text = _clean_markdown(content)
        if text.strip():
            sections.append(ParsedSection(
                text=text,
                header_hierarchy=[],
                start_char=0,
                end_char=len(content),
            ))
        return sections

    # If there's content before the first header, emit it as a section
    first_match = matches[0]
    if first_match.start() > 0:
        preamble = content[:first_match.start()]
        cleaned = _clean_markdown(preamble)
        if cleaned.strip():
            sections.append(ParsedSection(
                text=cleaned,
                header_hierarchy=[],
                start_char=0,
                end_char=first_match.start(),
            ))

    # Track header hierarchy as a stack
    hierarchy: list[tuple[int, str]] = []  # (level, full_header_line)

    # Track boilerplate skipping: when we hit a boilerplate header at level N,
    # skip everything until we see a header at level < N (i.e. a parent).
    skip_until_level: int | None = None

    for i, match in enumerate(matches):
        level = len(match.group(1))
        header_text = match.group(0).strip()
        header_name = match.group(2).strip().lower()

        # Check if we should stop skipping (hit a header shallower than skip level)
        if skip_until_level is not None:
            if level <= skip_until_level:
                skip_until_level = None
            else:
                continue

        # Check if this header starts a boilerplate section
        if header_name in _BOILERPLATE_HEADERS:
            skip_until_level = level
            continue

        # Update hierarchy: pop headers at same or deeper level
        while hierarchy and hierarchy[-1][0] >= level:
            hierarchy.pop()
        hierarchy.append((level, header_text))

        # Body text: from end of header line to start of next header (or EOF)
        body_start = match.end()
        if i + 1 < len(matches):
            body_end = matches[i + 1].start()
        else:
            body_end = len(content)

        body = content[body_start:body_end]
        cleaned = _clean_markdown(body)

        if cleaned.strip():
            sections.append(ParsedSection(
                text=cleaned,
                header_hierarchy=[h for _, h in hierarchy],
                start_char=match.start(),
                end_char=body_end,
            ))

    return sections


def _clean_markdown(text: str) -> str:
    """Clean Wikipedia markdown artifacts from text."""
    text = _EDIT_LINE.sub("", text)
    text = _MAIN_ARTICLE.sub("", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_plaintext(content: str) -> list[ParsedSection]:
    """Return the full text as a single section."""
    if not content.strip():
        return []
    return [ParsedSection(
        text=content,
        header_hierarchy=[],
        start_char=0,
        end_char=len(content),
    )]

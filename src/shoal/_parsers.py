"""Document format parsers for shoal."""

from __future__ import annotations

import re

from ._models import DocFormat, ParsedSection, SectionData

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

# Chapter detection patterns for plaintext, tried in priority order.
_CHAPTER_PATTERNS = [
    re.compile(r"^(CHAPTER\s+[IVXLC]+\.?)(\s+.*)?$", re.MULTILINE),     # CHAPTER XIV. TITLE
    re.compile(r"^(Chapter\s+\d+)(\s*[.:]\s*.*)?$", re.MULTILINE),       # Chapter 14: Title
    re.compile(r"^(PART\s+[IVXLC]+\.?)(\s+.*)?$", re.MULTILINE),         # PART III. TITLE
    re.compile(r"^(BOOK\s+[IVXLC]+\.?)(\s+.*)?$", re.MULTILINE),         # BOOK II
]


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


# ── Section tree building ──


def parse_sections(
    content: str,
    fmt: DocFormat,
    *,
    max_section_depth: int = 2,
    min_section_length: int = 200,
) -> list[SectionData]:
    """Parse a document into a hierarchical section tree.

    Returns a flat list of SectionData with tree structure encoded via parent_index.
    """
    if fmt == DocFormat.markdown:
        return _build_section_tree_markdown(
            content,
            max_section_depth=max_section_depth,
            min_section_length=min_section_length,
        )
    return _build_section_tree_plaintext(content, min_section_length=min_section_length)


def _build_section_tree_markdown(
    content: str,
    *,
    max_section_depth: int = 2,
    min_section_length: int = 200,
) -> list[SectionData]:
    """Build a section tree from markdown content.

    Heuristics:
    - H1 -> document title, not a section. Content between H1 and first H2
      becomes "Introduction" if substantial.
    - H2 -> always a top-level section (depth 0).
    - H3 -> section (depth 1) if text >= min_section_length, else folded into parent H2.
    - H4+ -> always folded into nearest ancestor section.
    """
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(header_pattern.finditer(content))

    if not matches:
        # No headers — entire document is a single section
        cleaned = _clean_markdown(content)
        if not cleaned.strip():
            return []
        return [SectionData(
            title="Document",
            depth=0,
            position=0,
            start_char=0,
            end_char=len(content),
            parent_index=None,
            analysis_text=cleaned,
        )]

    sections: list[SectionData] = []
    position_at_depth: dict[int, int] = {}  # depth -> next position counter

    # Collect all header entries with their raw body text and char ranges
    headers: list[dict] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        name_lower = title.lower()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body_text = _clean_markdown(content[body_start:body_end])

        headers.append({
            "level": level,
            "title": title,
            "name_lower": name_lower,
            "header_start": match.start(),
            "body_start": body_start,
            "body_end": body_end,
            "body_text": body_text,
            "boilerplate": name_lower in _BOILERPLATE_HEADERS,
        })

    # Handle preamble (content before first header)
    preamble_text = ""
    if matches[0].start() > 0:
        preamble_text = _clean_markdown(content[:matches[0].start()])

    # Handle H1 preamble (content between H1 and first H2)
    h1_preamble_text = ""
    first_h2_idx = None

    # Find the first H1 and first H2
    first_h1_idx = None
    for i, h in enumerate(headers):
        if h["level"] == 1 and first_h1_idx is None:
            first_h1_idx = i
        if h["level"] == 2 and first_h2_idx is None:
            first_h2_idx = i
            break

    # Collect H1 body text (and any sub-headers before first H2) as preamble
    if first_h1_idx is not None:
        # Gather text from H1 body and any headers before first H2 that are deeper than H1
        h1_parts = []
        if headers[first_h1_idx]["body_text"].strip():
            h1_parts.append(headers[first_h1_idx]["body_text"])
        # Also include text from headers between H1 and first H2 that aren't H2-level
        for j in range(first_h1_idx + 1, len(headers)):
            if headers[j]["level"] <= 2:
                break
            if not headers[j]["boilerplate"] and headers[j]["body_text"].strip():
                h1_parts.append(f"{headers[j]['title']}\n\n{headers[j]['body_text']}")
        h1_preamble_text = "\n\n".join(h1_parts)

    # Combine pre-header preamble and H1 preamble
    combined_preamble = "\n\n".join(
        p for p in [preamble_text, h1_preamble_text] if p.strip()
    )

    # Track which header indices have been "consumed" (folded into a parent)
    consumed: set[int] = set()

    # Mark boilerplate headers and their children as consumed
    skip_until_level: int | None = None
    for i, h in enumerate(headers):
        if skip_until_level is not None:
            if h["level"] <= skip_until_level:
                skip_until_level = None
            else:
                consumed.add(i)
                continue
        if h["boilerplate"]:
            consumed.add(i)
            skip_until_level = h["level"]
            continue

    # Mark H1 headers as consumed (they're document title, not sections)
    for i, h in enumerate(headers):
        if h["level"] == 1:
            consumed.add(i)

    # Build section tree from H2+ headers
    # First pass: identify which H3+ headers get folded
    # An H3 is folded if its body_text < min_section_length or depth would exceed max_section_depth
    # H4+ are always folded

    # Map from header index to section index (for parent resolution)
    header_to_section: dict[int, int] = {}

    # Track the current parent at each depth level (header index)
    parent_stack: list[tuple[int, int]] = []  # (header_level, header_index)

    def _find_parent_header_idx(level: int) -> int | None:
        """Find the header index of the nearest ancestor at a shallower level."""
        for plevel, pidx in reversed(parent_stack):
            if plevel < level:
                return pidx
        return None

    # Emit preamble as "Introduction" section if substantial
    preamble_emitted = False
    if combined_preamble.strip() and len(combined_preamble) >= min_section_length:
        pos = position_at_depth.get(0, 0)
        sections.append(SectionData(
            title="Introduction",
            depth=0,
            position=pos,
            start_char=0,
            end_char=matches[first_h2_idx].start() if first_h2_idx is not None else headers[0]["body_end"],
            parent_index=None,
            analysis_text=combined_preamble,
        ))
        position_at_depth[0] = pos + 1
        preamble_emitted = True

    # Process each header
    for i, h in enumerate(headers):
        if i in consumed:
            continue

        level = h["level"]

        # Update parent stack
        while parent_stack and parent_stack[-1][0] >= level:
            parent_stack.pop()

        if level == 2:
            # H2 -> always a top-level section (depth 0)
            section_depth = 0
            parent_section_idx = None

            # Collect body text, including folded children
            analysis_parts = [h["body_text"]] if h["body_text"].strip() else []

            # Look ahead for children that will be folded
            folded_end = h["body_end"]
            for j in range(i + 1, len(headers)):
                if j in consumed:
                    continue
                child = headers[j]
                if child["level"] <= 2:
                    break  # Hit next H2 or shallower
                if child["level"] == 3:
                    # H3: fold if too short or would exceed depth limit
                    child_text_len = len(child["body_text"])
                    if child_text_len < min_section_length or max_section_depth < 2:
                        # Fold this H3 into the H2
                        if child["body_text"].strip():
                            analysis_parts.append(f"{child['title']}\n\n{child['body_text']}")
                        consumed.add(j)
                        folded_end = max(folded_end, child["body_end"])
                        # Also fold any H4+ children of this H3
                        for k in range(j + 1, len(headers)):
                            if k in consumed:
                                continue
                            if headers[k]["level"] <= 3:
                                break
                            if headers[k]["body_text"].strip():
                                analysis_parts.append(f"{headers[k]['title']}\n\n{headers[k]['body_text']}")
                            consumed.add(k)
                            folded_end = max(folded_end, headers[k]["body_end"])
                    # else: H3 is long enough, will become its own section
                elif child["level"] >= 4:
                    # H4+ directly under H2: fold
                    if child["body_text"].strip():
                        analysis_parts.append(f"{child['title']}\n\n{child['body_text']}")
                    consumed.add(j)
                    folded_end = max(folded_end, child["body_end"])

            # If preamble was not emitted as its own section, prepend to first H2
            if not preamble_emitted and combined_preamble.strip() and position_at_depth.get(0, 0) == 0:
                analysis_parts.insert(0, combined_preamble)
                preamble_emitted = True

            analysis_text = "\n\n".join(p for p in analysis_parts if p.strip())

            pos = position_at_depth.get(section_depth, 0)
            section_idx = len(sections)
            sections.append(SectionData(
                title=h["title"],
                depth=section_depth,
                position=pos,
                start_char=h["header_start"],
                end_char=max(h["body_end"], folded_end),
                parent_index=parent_section_idx,
                analysis_text=analysis_text,
            ))
            position_at_depth[section_depth] = pos + 1
            # Reset child depth counters
            for d in list(position_at_depth.keys()):
                if d > section_depth:
                    position_at_depth[d] = 0
            header_to_section[i] = section_idx
            parent_stack.append((level, i))

        elif level == 3 and i not in consumed:
            # H3 -> section (depth 1) if long enough
            section_depth = 1
            parent_header_idx = _find_parent_header_idx(level)
            parent_section_idx = header_to_section.get(parent_header_idx) if parent_header_idx is not None else None

            # Collect body text, folding H4+ children
            analysis_parts = [h["body_text"]] if h["body_text"].strip() else []
            folded_end = h["body_end"]

            for j in range(i + 1, len(headers)):
                if j in consumed:
                    continue
                child = headers[j]
                if child["level"] <= 3:
                    break
                # H4+ under this H3: always fold
                if child["body_text"].strip():
                    analysis_parts.append(f"{child['title']}\n\n{child['body_text']}")
                consumed.add(j)
                folded_end = max(folded_end, child["body_end"])

            analysis_text = "\n\n".join(p for p in analysis_parts if p.strip())

            pos = position_at_depth.get(section_depth, 0)
            section_idx = len(sections)
            sections.append(SectionData(
                title=h["title"],
                depth=section_depth,
                position=pos,
                start_char=h["header_start"],
                end_char=max(h["body_end"], folded_end),
                parent_index=parent_section_idx,
                analysis_text=analysis_text,
            ))
            position_at_depth[section_depth] = pos + 1
            header_to_section[i] = section_idx
            parent_stack.append((level, i))

    # If no sections were created (e.g., only H1 with no H2s), create a single section
    if not sections:
        all_text = _clean_markdown(content)
        if all_text.strip():
            sections.append(SectionData(
                title="Document",
                depth=0,
                position=0,
                start_char=0,
                end_char=len(content),
                parent_index=None,
                analysis_text=all_text,
            ))

    return sections


def _build_section_tree_plaintext(
    content: str,
    *,
    min_section_length: int = 200,
) -> list[SectionData]:
    """Build a section tree from plaintext content using chapter detection."""
    if not content.strip():
        return []

    chapters = _detect_chapters(content, min_body_length=min_section_length)
    if chapters:
        return chapters

    # Final fallback: single section with the whole document
    return [SectionData(
        title="Document",
        depth=0,
        position=0,
        start_char=0,
        end_char=len(content),
        parent_index=None,
        analysis_text=content,
    )]


def _detect_chapters(
    content: str,
    *,
    min_body_length: int = 200,
) -> list[SectionData] | None:
    """Detect chapters in plaintext using regex patterns.

    Returns list of SectionData if chapters found, None otherwise.
    """
    for pattern in _CHAPTER_PATTERNS:
        matches = list(pattern.finditer(content))
        if len(matches) < 2:
            continue

        # Filter out TOC entries: skip matches in dense clusters at start of document.
        # A TOC has many chapter headings in few lines with little text between them.
        filtered = _filter_toc_matches(matches, content, min_body_length=min_body_length)
        if len(filtered) < 2:
            continue

        sections: list[SectionData] = []
        position = 0

        # Front matter: content before first chapter
        first_start = filtered[0].start()
        if first_start > 0:
            front_text = content[:first_start].strip()
            if len(front_text) >= min_body_length:
                sections.append(SectionData(
                    title="Front Matter",
                    depth=0,
                    position=position,
                    start_char=0,
                    end_char=first_start,
                    parent_index=None,
                    analysis_text=front_text,
                ))
                position += 1

        # Each chapter match becomes a section
        for idx, match in enumerate(filtered):
            title = match.group(0).strip()
            chapter_start = match.start()
            chapter_end = filtered[idx + 1].start() if idx + 1 < len(filtered) else len(content)
            body_text = content[match.end():chapter_end].strip()

            sections.append(SectionData(
                title=title,
                depth=0,
                position=position,
                start_char=chapter_start,
                end_char=chapter_end,
                parent_index=None,
                analysis_text=body_text,
            ))
            position += 1

        return sections

    return None


def _filter_toc_matches(
    matches: list[re.Match],
    content: str,
    *,
    min_body_length: int = 200,
) -> list[re.Match]:
    """Filter out TOC entries from chapter pattern matches.

    Chapter patterns in dense clusters (many matches in few lines) at the start
    of the document indicate a table of contents. Only keep matches followed by
    substantial body text.
    """
    filtered = []
    for i, match in enumerate(matches):
        # Check how much text follows before the next match
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body_between = content[match.end():next_start].strip()
        if len(body_between) >= min_body_length:
            filtered.append(match)

    return filtered

"""Ingestion pipeline for shoal (Phase 1 — single pass, no vocabulary extension)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._models import ChunkData, DocFormat, IngestResult, ParsedSection
from ._parsers import parse_document
from ._storage import Storage

if TYPE_CHECKING:
    from lagoon import ReefScorer


def ingest_document(
    scorer: ReefScorer,
    storage: Storage,
    *,
    title: str,
    content: str,
    fmt: DocFormat,
    tags: list[str] | None = None,
    source_path: str | None = None,
    metadata: dict | None = None,
    sensitivity: float = 1.0,
    smooth_window: int = 2,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 30,
) -> IngestResult:
    """Ingest a document: parse, analyze with lagoon, store in SQLite.

    Phase 1 implementation: single pass (no vocabulary extension).
    """
    # 1. Insert document record
    doc_id = storage.insert_document(
        title=title,
        content=content,
        fmt=fmt.value,
        source_path=source_path,
        metadata=metadata,
    )

    # 2. Set tags
    tag_list = tags or []
    if tag_list:
        storage.set_tags(doc_id, tag_list)

    # 3. Parse into sections
    sections = parse_document(content, fmt)

    # 4. Analyze each section and build chunks
    all_chunks: list[ChunkData] = []
    chunk_index = 0

    for section in sections:
        section_chunks = _analyze_section(
            scorer=scorer,
            section=section,
            chunk_index_start=chunk_index,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences,
        )
        all_chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    # 5. Store chunks
    if all_chunks:
        storage.insert_chunks(doc_id, all_chunks)

    return IngestResult(
        id=doc_id,
        title=title,
        n_chunks=len(all_chunks),
        tags=tag_list,
    )


def _analyze_section(
    scorer: ReefScorer,
    section: ParsedSection,
    chunk_index_start: int,
    sensitivity: float,
    smooth_window: int,
    min_chunk_sentences: int,
    max_chunk_sentences: int,
) -> list[ChunkData]:
    """Analyze a section with lagoon and return ChunkData objects."""
    text = section.text
    if not text.strip():
        return []

    analysis = scorer.analyze(
        text,
        sensitivity=sensitivity,
        smooth_window=smooth_window,
        min_chunk_sentences=min_chunk_sentences,
        max_chunk_sentences=max_chunk_sentences,
    )

    chunks: list[ChunkData] = []
    cursor = 0  # Track position in section text for offset computation

    for seg_idx, segment in enumerate(analysis.segments):
        chunk_text = " ".join(segment.sentences)
        sentence_count = len(segment.sentences)
        topic = segment.topic

        # Compute character offsets by finding sentences in section text
        start_char, end_char = _compute_offsets(
            section_text=text,
            sentences=segment.sentences,
            cursor=cursor,
            is_last=(seg_idx == len(analysis.segments) - 1),
            section_end=len(text),
        )

        # Advance cursor past this segment
        if end_char > cursor:
            cursor = end_char

        # Absolute offsets relative to original document
        abs_start = section.start_char + start_char
        abs_end = section.start_char + end_char

        # Extract reef data from TopicResult
        top_reefs = [
            (r.reef_id, r.name, r.z_score, r.raw_bm25, r.n_contributing_words)
            for r in topic.top_reefs
        ]
        top_islands = [
            (isl.island_id, isl.name, isl.aggregate_z, isl.n_contributing_reefs)
            for isl in topic.top_islands
        ]

        chunks.append(ChunkData(
            text=chunk_text,
            chunk_index=chunk_index_start + seg_idx,
            start_char=abs_start,
            end_char=abs_end,
            sentence_count=sentence_count,
            confidence=topic.confidence,
            coverage=topic.coverage,
            matched_words=topic.matched_words,
            top_reef_id=topic.top_reefs[0].reef_id if topic.top_reefs else -1,
            top_reef_name=topic.top_reefs[0].name if topic.top_reefs else "",
            arch_scores=list(topic.arch_scores),
            top_reefs=top_reefs,
            top_islands=top_islands,
            metadata={"headers": section.header_hierarchy},
        ))

    return chunks


def _compute_offsets(
    section_text: str,
    sentences: list[str],
    cursor: int,
    is_last: bool,
    section_end: int,
) -> tuple[int, int]:
    """Compute start_char and end_char for a segment within section text.

    Walks through the section text with a cursor to find each sentence position,
    advancing past previous sentences to handle duplicates correctly.
    """
    if not sentences:
        return cursor, cursor

    # Find start: position of first sentence
    first_pos = section_text.find(sentences[0], cursor)
    if first_pos == -1:
        # Fallback: use cursor position
        first_pos = cursor

    start_char = first_pos

    # Find end: position after last sentence
    pos = first_pos
    for sentence in sentences:
        found = section_text.find(sentence, pos)
        if found != -1:
            pos = found + len(sentence)
        else:
            # Sentence not found exactly — advance by sentence length
            pos += len(sentence)

    if is_last:
        end_char = section_end
    else:
        end_char = pos

    return start_char, end_char

"""Ingestion pipeline for shoal (Phase 1 — single pass, no vocabulary extension)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ._models import ChunkData, DocFormat, IngestResult, SectionData
from ._parsers import parse_sections
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
    max_chunk_sentences: int = 12,
    min_chunk_chars: int = 175,
    max_section_depth: int = 2,
    min_section_length: int = 200,
    min_reef_z: float = 2.0,
    reef_top_k: int = 38,
) -> IngestResult:
    """Ingest a document: parse sections, analyze with lagoon, store in SQLite.

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
    sections = parse_sections(
        content, fmt,
        max_section_depth=max_section_depth,
        min_section_length=min_section_length,
    )

    if not sections:
        return IngestResult(
            id=doc_id,
            title=title,
            n_chunks=0,
            n_sections=0,
            tags=tag_list,
        )

    # 4. Insert section tree
    section_ids = storage.insert_sections(doc_id, sections)

    # 5. Analyze each section and build chunks
    all_chunks: list[ChunkData] = []
    chunk_index = 0

    for sec_idx, section in enumerate(sections):
        section_chunks = _analyze_section(
            scorer=scorer,
            section=section,
            section_index=sec_idx,
            chunk_index_start=chunk_index,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences,
            min_reef_z=min_reef_z,
        )
        all_chunks.extend(section_chunks)
        chunk_index += len(section_chunks)

    # 6. Filter out tiny chunks that lack meaningful reef signal
    if min_chunk_chars > 0:
        all_chunks = [c for c in all_chunks if len(c.text) >= min_chunk_chars]
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i

    # 7. Store chunks
    if all_chunks:
        storage.insert_chunks(doc_id, all_chunks, section_ids=section_ids, reef_top_k=reef_top_k)

    # 8. Update section chunk counts
    chunk_counts: Counter[int] = Counter()
    for chunk in all_chunks:
        chunk_counts[chunk.section_index] += 1
    for sec_idx, count in chunk_counts.items():
        storage.update_section_chunk_count(section_ids[sec_idx], count)

    return IngestResult(
        id=doc_id,
        title=title,
        n_chunks=len(all_chunks),
        n_sections=len(sections),
        tags=tag_list,
    )


def _analyze_section(
    scorer: ReefScorer,
    section: SectionData,
    section_index: int,
    chunk_index_start: int,
    sensitivity: float,
    smooth_window: int,
    min_chunk_sentences: int,
    max_chunk_sentences: int,
    min_reef_z: float = 2.0,
) -> list[ChunkData]:
    """Analyze a section with lagoon and return ChunkData objects."""
    text = section.analysis_text
    if not text.strip():
        return []

    analysis = scorer.analyze(
        text,
        sensitivity=sensitivity,
        smooth_window=smooth_window,
        min_chunk_sentences=min_chunk_sentences,
        max_chunk_sentences=max_chunk_sentences,
        min_reef_z=min_reef_z,
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
            section_index=section_index,
            metadata={"section_title": section.title},
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

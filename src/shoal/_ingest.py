"""Two-pass ingestion with vocabulary extension."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from ._models import ChunkData, DocFormat, IngestResult, SectionData
from ._parsers import parse_sections
from ._storage import Storage
from ._vocab import build_vocabulary, collect_word_observations, get_custom_word_ids_for_chunk

if TYPE_CHECKING:
    from lagoon import DocumentAnalysis, ReefScorer


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
    enable_vocab_extension: bool = True,
    confidence_threshold: float = 1.5,
    association_z_threshold: float = 0.5,
) -> IngestResult:
    """Ingest a document: parse sections, analyze with lagoon, store in SQLite.

    When enable_vocab_extension is True, uses a two-pass pipeline:
    - Pass 1: Analyze to discover unknown words and collect reef context
    - Vocabulary build: Create custom words from observations
    - Pass 2: Re-analyze with extended vocabulary (only if new words found)
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

    # 5. Compute total word count for vocabulary threshold
    total_word_count = sum(
        len(s.analysis_text.split()) for s in sections
    )

    # 6. Analyze sections and build chunks
    all_chunks: list[ChunkData] = []
    chunk_custom_word_map: dict[int, list[int]] = {}  # chunk_index -> [custom_word_ids]
    n_new_words = 0
    two_pass = False

    analyze_kwargs = dict(
        sensitivity=sensitivity,
        smooth_window=smooth_window,
        min_chunk_sentences=min_chunk_sentences,
        max_chunk_sentences=max_chunk_sentences,
        min_reef_z=min_reef_z,
    )

    if enable_vocab_extension:
        # PASS 1: Analyze and collect word observations
        pass1_analyses: list[tuple[int, DocumentAnalysis]] = []
        all_observations = []

        for sec_idx, section in enumerate(sections):
            text = section.analysis_text
            if not text.strip():
                continue
            analysis = scorer.analyze(text, **analyze_kwargs)
            pass1_analyses.append((sec_idx, analysis))
            observations = collect_word_observations(
                scorer, analysis, doc_id,
                confidence_threshold=confidence_threshold,
            )
            all_observations.extend(observations)

        # Store observations
        if all_observations:
            storage.insert_word_observations(all_observations)

        # VOCABULARY BUILD
        n_new_words = build_vocabulary(
            scorer, storage, doc_id, total_word_count,
            association_z_threshold=association_z_threshold,
        )

        if n_new_words > 0:
            # PASS 2: Re-analyze with extended vocabulary
            two_pass = True
            chunk_index = 0
            for sec_idx, section in enumerate(sections):
                section_chunks, cw_map = _analyze_section_pass2(
                    scorer=scorer,
                    section=section,
                    section_index=sec_idx,
                    chunk_index_start=chunk_index,
                    **analyze_kwargs,
                )
                all_chunks.extend(section_chunks)
                # Offset cw_map keys by chunk_index_start
                for ci, cw_ids in cw_map.items():
                    chunk_custom_word_map[chunk_index + ci] = cw_ids
                chunk_index += len(section_chunks)
        else:
            # No new words — reuse Pass 1 analyses
            chunk_index = 0
            for sec_idx, analysis in pass1_analyses:
                section_chunks = _chunks_from_analysis(
                    analysis, sections[sec_idx], sec_idx, chunk_index,
                )
                all_chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
    else:
        # Single-pass: unchanged Phase 1 path
        chunk_index = 0
        for sec_idx, section in enumerate(sections):
            section_chunks = _analyze_section(
                scorer=scorer,
                section=section,
                section_index=sec_idx,
                chunk_index_start=chunk_index,
                **analyze_kwargs,
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

    # 7. Filter out tiny chunks
    if min_chunk_chars > 0:
        filtered: list[ChunkData] = []
        old_to_new: dict[int, int] = {}
        for chunk in all_chunks:
            if len(chunk.text) >= min_chunk_chars:
                old_idx = chunk.chunk_index
                chunk.chunk_index = len(filtered)
                old_to_new[old_idx] = chunk.chunk_index
                filtered.append(chunk)
        all_chunks = filtered

        # Remap chunk_custom_word_map
        remapped: dict[int, list[int]] = {}
        for old_idx, cw_ids in chunk_custom_word_map.items():
            if old_idx in old_to_new:
                remapped[old_to_new[old_idx]] = cw_ids
        chunk_custom_word_map = remapped

    # 8. Store chunks
    chunk_ids: list[int] = []
    if all_chunks:
        chunk_ids = storage.insert_chunks(
            doc_id, all_chunks, section_ids=section_ids, reef_top_k=reef_top_k,
        )

    # 9. Record chunk custom words (if Pass 2 was run)
    if two_pass and chunk_ids:
        for i, db_chunk_id in enumerate(chunk_ids):
            chunk_idx = all_chunks[i].chunk_index
            cw_ids = chunk_custom_word_map.get(chunk_idx, [])
            if cw_ids:
                storage.insert_chunk_custom_words(db_chunk_id, cw_ids)

    # 10. Update section chunk counts
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
        n_new_words=n_new_words,
        two_pass=two_pass,
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

    return _build_chunks_from_segments(analysis, section, section_index, chunk_index_start)


def _analyze_section_pass2(
    scorer: ReefScorer,
    section: SectionData,
    section_index: int,
    chunk_index_start: int,
    sensitivity: float,
    smooth_window: int,
    min_chunk_sentences: int,
    max_chunk_sentences: int,
    min_reef_z: float = 2.0,
) -> tuple[list[ChunkData], dict[int, list[int]]]:
    """Analyze a section (Pass 2) and track custom word IDs per chunk.

    Returns (chunks, {chunk_local_index: [custom_word_ids]}).
    The custom_word_ids are custom_words.id values (tags), not runtime word_ids.
    """
    text = section.analysis_text
    if not text.strip():
        return [], {}

    analysis = scorer.analyze(
        text,
        sensitivity=sensitivity,
        smooth_window=smooth_window,
        min_chunk_sentences=min_chunk_sentences,
        max_chunk_sentences=max_chunk_sentences,
        min_reef_z=min_reef_z,
    )

    chunks = _build_chunks_from_segments(analysis, section, section_index, chunk_index_start)

    # Track custom words per chunk
    cw_map: dict[int, list[int]] = {}
    for seg_idx, segment in enumerate(analysis.segments):
        # Collect all matched_word_ids from segment topic + sentence results
        all_word_ids: set[int] = set(segment.topic.matched_word_ids)
        for sent_result in segment.sentence_results:
            all_word_ids.update(sent_result.matched_word_ids)

        tagged = get_custom_word_ids_for_chunk(scorer, frozenset(all_word_ids))
        if tagged:
            # Values are custom_words.id (the tags)
            cw_map[seg_idx] = list(set(tagged.values()))

    return chunks, cw_map


def _chunks_from_analysis(
    analysis: DocumentAnalysis,
    section: SectionData,
    section_index: int,
    chunk_index_start: int,
) -> list[ChunkData]:
    """Convert a saved DocumentAnalysis to ChunkData list without re-analyzing.

    Used when Pass 1 analysis can be reused (no new words discovered).
    """
    return _build_chunks_from_segments(analysis, section, section_index, chunk_index_start)


def _build_chunks_from_segments(
    analysis: DocumentAnalysis,
    section: SectionData,
    section_index: int,
    chunk_index_start: int,
) -> list[ChunkData]:
    """Build ChunkData objects from analysis segments."""
    text = section.analysis_text
    chunks: list[ChunkData] = []
    cursor = 0

    for seg_idx, segment in enumerate(analysis.segments):
        chunk_text = " ".join(segment.sentences)
        sentence_count = len(segment.sentences)
        topic = segment.topic

        start_char, end_char = _compute_offsets(
            section_text=text,
            sentences=segment.sentences,
            cursor=cursor,
            is_last=(seg_idx == len(analysis.segments) - 1),
            section_end=len(text),
        )

        if end_char > cursor:
            cursor = end_char

        abs_start = section.start_char + start_char
        abs_end = section.start_char + end_char

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

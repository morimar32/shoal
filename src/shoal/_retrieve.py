"""Retrieval module for shoal with lightning rod pre-filtering."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from lagoon import STOP_WORDS

from ._models import QueryInfo, SearchResponse, SearchResult
from ._storage import Storage

if TYPE_CHECKING:
    from lagoon import ReefScorer, TopicResult


_QUERY_REEF_COUNT = 10  # number of reefs to use from scorer (independent of result count)
_MIN_REEF_CONFIDENCE = 0.5  # reef-only results require at least this query confidence (top raw BM25)
_LIGHTNING_ROD_BASE_BOOST = 1.15  # base multiplier for lightning rod candidates
_LIGHTNING_ROD_TERM_BOOST = 0.20  # additional multiplier scaled by query-term frequency score
_LIGHTNING_ROD_DOC_BOOST = 0.25   # additional multiplier for document-level concentration
_LIGHTNING_ROD_LIMIT = 50   # max candidate chunks from word lookup


def _is_stopword_query(query: str) -> bool:
    """Check if query consists entirely of stop words (or has no real words)."""
    words = [w.lower() for w in query.split() if any(c.isalnum() for c in w)]
    if not words:
        return True
    return all(w in STOP_WORDS for w in words)


def _apply_doc_diversity(
    results: list[SearchResult],
    top_k: int,
    max_per_doc: int | None,
) -> list[SearchResult]:
    """Filter results so no single document exceeds max_per_doc entries."""
    if max_per_doc is None or max_per_doc <= 0:
        return results[:top_k]
    doc_counts: dict[int, int] = {}
    filtered: list[SearchResult] = []
    for r in results:
        count = doc_counts.get(r.document_id, 0)
        if count < max_per_doc:
            filtered.append(r)
            doc_counts[r.document_id] = count + 1
            if len(filtered) >= top_k:
                break
    return filtered


def search(
    scorer: ReefScorer,
    storage: Storage,
    query: str,
    *,
    top_k: int = 10,
    tags: list[str] | None = None,
    min_confidence: float | None = None,
    include_scores: bool = False,
    enable_lightning_rod: bool = True,
    max_per_doc: int | None = 3,
) -> SearchResponse:
    """Score a query with lagoon and retrieve matching chunks via reef overlap.

    When enable_lightning_rod is True and the query contains custom vocabulary
    words, pre-filters to chunks containing those words for better precision.

    Uses reef overlap scoring: the query is scored to produce a reef profile,
    then chunks sharing those reefs are ranked by weighted overlap
    (SUM of chunk_z * query_z for each shared reef).

    Quality gating: when query confidence is below _MIN_REEF_CONFIDENCE,
    results are suppressed — low-confidence reef overlap is noise.
    """
    # 0. Stop word check — return empty response for stop-word-only queries
    total_words = len([w for w in query.split() if any(c.isalnum() for c in w)])
    if _is_stopword_query(query):
        return SearchResponse(
            results=[],
            query_info=QueryInfo(
                top_reef="",
                confidence=0.0,
                coverage=0.0,
                matched_words=0,
                total_words=total_words,
                top_reefs=[],
                top_islands=[],
            ),
        )

    # 1. Score the query — use fixed reef count, independent of result count
    result = scorer.score(query, top_k=_QUERY_REEF_COUNT)

    # 2. Detect custom words early — needed before quality gate decision
    tagged_words: dict[int, int] | None = None
    has_custom_words = False
    if enable_lightning_rod and result.matched_word_ids:
        tagged_words = scorer.get_word_tags(result.matched_word_ids)
        has_custom_words = bool(tagged_words)

    # 3. Quality gate — skip when custom words are present (lightning rod
    #    provides word-level precision that doesn't depend on reef confidence)
    if result.confidence < _MIN_REEF_CONFIDENCE and not has_custom_words:
        query_info = _build_query_info(result, total_words, tagged_words=tagged_words)
        return SearchResponse(results=[], query_info=query_info)

    # 4. Extract query reef pairs, filtering out negative z-scores
    query_reefs = [
        (r.reef_id, r.z_score)
        for r in result.top_reefs
        if r.z_score > 0
    ]

    # 5. Lightning rod: pre-filter by custom words
    lightning_results: list[SearchResult] = []
    lightning_chunk_ids: set[int] = set()

    if has_custom_words and tagged_words:
        custom_word_ids = list(tagged_words.values())
        candidate_chunk_ids = storage.get_chunks_by_custom_words(
            custom_word_ids, limit=_LIGHTNING_ROD_LIMIT,
        )
        if candidate_chunk_ids and query_reefs:
            lightning_results = storage.score_chunks_by_reef_overlap(
                candidate_chunk_ids, query_reefs,
            )
            # Apply boost scaled by query-term overlap:
            # chunks containing more query words get a stronger boost
            query_tokens = {
                w.lower() for w in query.split()
                if any(c.isalnum() for c in w)
            } - STOP_WORDS
            n_query_tokens = max(len(query_tokens), 1)

            # Document concentration: count how many candidate chunks
            # come from each document. More chunks → stronger signal
            # that the document is about the query's custom words.
            doc_chunk_counts: dict[int, int] = {}
            for lr in lightning_results:
                doc_chunk_counts[lr.document_id] = (
                    doc_chunk_counts.get(lr.document_id, 0) + 1
                )
            max_doc_chunks = max(doc_chunk_counts.values()) if doc_chunk_counts else 1

            for lr in lightning_results:
                chunk_lower = lr.text.lower()
                # Frequency-weighted: log(1+count) rewards repeated
                # occurrences of query terms, not just binary presence
                freq_score = sum(
                    math.log1p(chunk_lower.count(t))
                    for t in query_tokens
                ) / n_query_tokens
                # Document concentration: fraction of candidate chunks
                # from this document vs the most represented document
                doc_concentration = (
                    doc_chunk_counts[lr.document_id] / max_doc_chunks
                )
                lr.match_score *= (
                    _LIGHTNING_ROD_BASE_BOOST
                    + _LIGHTNING_ROD_TERM_BOOST * freq_score
                    + _LIGHTNING_ROD_DOC_BOOST * doc_concentration
                )
            lightning_chunk_ids = {r.chunk_id for r in lightning_results}

    # 6. Standard reef overlap retrieval (request extra to cover dedup + diversity filtering)
    if max_per_doc is not None and max_per_doc > 0:
        standard_top_k = max(top_k * 3, top_k + len(lightning_chunk_ids))
    else:
        standard_top_k = top_k + len(lightning_chunk_ids)
    standard_results = storage.search_by_reef_overlap(
        query_reefs,
        top_k=standard_top_k,
        tags=tags,
        min_confidence=min_confidence,
        include_scores=include_scores,
    )

    # 7. Merge: lightning results + standard results (dedup by chunk_id)
    if lightning_results:
        merged_map: dict[int, SearchResult] = {}
        # Lightning results take priority (boosted scores)
        for lr in lightning_results:
            merged_map[lr.chunk_id] = lr
        # Add standard results that aren't already in lightning set
        for sr in standard_results:
            if sr.chunk_id not in merged_map:
                merged_map[sr.chunk_id] = sr

        # Re-sort by match_score, apply diversity filter, truncate to top_k
        merged = sorted(merged_map.values(), key=lambda r: r.match_score, reverse=True)
        results = _apply_doc_diversity(merged, top_k, max_per_doc)

        # 8. If include_scores: fetch shared_reefs for lightning results that lack them
        if include_scores:
            for r in results:
                if r.chunk_id in lightning_chunk_ids and not r.shared_reefs:
                    r.shared_reefs = storage._get_shared_reefs(r.chunk_id, query_reefs)
    else:
        results = _apply_doc_diversity(standard_results, top_k, max_per_doc)

    # 9. Build query info with tagged_words
    query_info = _build_query_info(result, total_words, tagged_words=tagged_words)

    return SearchResponse(results=results, query_info=query_info)


def _build_query_info(
    result: TopicResult,
    total_words: int,
    *,
    tagged_words: dict[int, int] | None = None,
) -> QueryInfo:
    """Build QueryInfo from a lagoon TopicResult."""
    tw_list = []
    if tagged_words:
        tw_list = [{"word_id": wid, "tag": tag} for wid, tag in tagged_words.items()]

    return QueryInfo(
        top_reef=result.top_reefs[0].name if result.top_reefs else "",
        confidence=result.confidence,
        coverage=result.coverage,
        matched_words=result.matched_words,
        total_words=total_words,
        top_reefs=[
            {"reef_name": r.name, "z_score": r.z_score}
            for r in result.top_reefs
        ],
        top_islands=[
            {"island_name": isl.name, "aggregate_z": isl.aggregate_z}
            for isl in result.top_islands
        ],
        tagged_words=tw_list,
    )

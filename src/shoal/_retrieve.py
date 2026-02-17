"""Retrieval module for shoal with lightning rod pre-filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lagoon import STOP_WORDS

from ._models import QueryInfo, SearchResponse, SearchResult
from ._storage import Storage

if TYPE_CHECKING:
    from lagoon import ReefScorer, TopicResult


_QUERY_REEF_COUNT = 10  # number of reefs to use from scorer (independent of result count)
_MIN_REEF_CONFIDENCE = 0.1  # reef-only results require at least this query confidence
_LIGHTNING_ROD_BOOST = 1.3  # score multiplier for chunks containing custom query words
_LIGHTNING_ROD_LIMIT = 50   # max candidate chunks from word lookup


def _is_stopword_query(query: str) -> bool:
    """Check if query consists entirely of stop words (or has no real words)."""
    words = [w.lower() for w in query.split() if any(c.isalnum() for c in w)]
    if not words:
        return True
    return all(w in STOP_WORDS for w in words)


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

    # 2. Quality gate — if confidence is too low, reef overlap is noise
    if result.confidence < _MIN_REEF_CONFIDENCE:
        query_info = _build_query_info(result, total_words)
        return SearchResponse(results=[], query_info=query_info)

    # 3. Extract query reef pairs, filtering out negative z-scores
    query_reefs = [
        (r.reef_id, r.z_score)
        for r in result.top_reefs
        if r.z_score > 0
    ]

    # 4. Lightning rod: pre-filter by custom words
    tagged_words: dict[int, int] | None = None
    lightning_results: list[SearchResult] = []
    lightning_chunk_ids: set[int] = set()

    if enable_lightning_rod and result.matched_word_ids:
        tagged_words = scorer.get_word_tags(result.matched_word_ids)
        if tagged_words:
            custom_word_ids = list(tagged_words.values())
            candidate_chunk_ids = storage.get_chunks_by_custom_words(
                custom_word_ids, limit=_LIGHTNING_ROD_LIMIT,
            )
            if candidate_chunk_ids and query_reefs:
                lightning_results = storage.score_chunks_by_reef_overlap(
                    candidate_chunk_ids, query_reefs,
                )
                # Apply boost
                for lr in lightning_results:
                    lr.match_score *= _LIGHTNING_ROD_BOOST
                lightning_chunk_ids = {r.chunk_id for r in lightning_results}

    # 5. Standard reef overlap retrieval (request extra to cover dedup)
    standard_top_k = top_k + len(lightning_chunk_ids)
    standard_results = storage.search_by_reef_overlap(
        query_reefs,
        top_k=standard_top_k,
        tags=tags,
        min_confidence=min_confidence,
        include_scores=include_scores,
    )

    # 6. Merge: lightning results + standard results (dedup by chunk_id)
    if lightning_results:
        merged_map: dict[int, SearchResult] = {}
        # Lightning results take priority (boosted scores)
        for lr in lightning_results:
            merged_map[lr.chunk_id] = lr
        # Add standard results that aren't already in lightning set
        for sr in standard_results:
            if sr.chunk_id not in merged_map:
                merged_map[sr.chunk_id] = sr

        # Re-sort by match_score, truncate to top_k
        merged = sorted(merged_map.values(), key=lambda r: r.match_score, reverse=True)
        results = merged[:top_k]

        # 7. If include_scores: fetch shared_reefs for lightning results that lack them
        if include_scores:
            for r in results:
                if r.chunk_id in lightning_chunk_ids and not r.shared_reefs:
                    r.shared_reefs = storage._get_shared_reefs(r.chunk_id, query_reefs)
    else:
        results = standard_results[:top_k]

    # 8. Build query info with tagged_words
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

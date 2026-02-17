"""Retrieval module for shoal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lagoon import STOP_WORDS

from ._models import QueryInfo, SearchResponse, SearchResult
from ._storage import Storage

if TYPE_CHECKING:
    from lagoon import ReefScorer, TopicResult


_QUERY_REEF_COUNT = 10  # number of reefs to use from scorer (independent of result count)
_MIN_REEF_CONFIDENCE = 0.1  # reef-only results require at least this query confidence


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
) -> SearchResponse:
    """Score a query with lagoon and retrieve matching chunks via reef overlap.

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

    # 4. Reef overlap retrieval
    results = storage.search_by_reef_overlap(
        query_reefs,
        top_k=top_k,
        tags=tags,
        min_confidence=min_confidence,
        include_scores=include_scores,
    )

    # 5. Build query info
    query_info = _build_query_info(result, total_words)

    return SearchResponse(results=results, query_info=query_info)


def _build_query_info(result: TopicResult, total_words: int) -> QueryInfo:
    """Build QueryInfo from a lagoon TopicResult."""
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
    )

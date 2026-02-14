"""Retrieval module for shoal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._models import QueryInfo, SearchResponse
from ._storage import Storage

if TYPE_CHECKING:
    from lagoon import ReefScorer


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
    """Score a query with lagoon and retrieve matching chunks via reef overlap."""
    # 1. Score the query
    result = scorer.score(query, top_k=top_k)

    # 2. Extract query reef pairs
    query_reefs = [(r.reef_id, r.z_score) for r in result.top_reefs]

    # 3. SQL reef overlap retrieval
    results = storage.search_by_reef_overlap(
        query_reefs,
        top_k=top_k,
        tags=tags,
        min_confidence=min_confidence,
        include_scores=include_scores,
    )

    # 4. Build query info
    query_info = QueryInfo(
        top_reef=result.top_reefs[0].name if result.top_reefs else "",
        confidence=result.confidence,
        coverage=result.coverage,
        top_reefs=[
            {"reef_name": r.name, "z_score": r.z_score}
            for r in result.top_reefs
        ],
        top_islands=[
            {"island_name": isl.name, "aggregate_z": isl.aggregate_z}
            for isl in result.top_islands
        ],
    )

    return SearchResponse(results=results, query_info=query_info)

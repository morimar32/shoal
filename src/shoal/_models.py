"""Data structures for shoal."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DocFormat(str, Enum):
    """Supported document formats."""
    markdown = "markdown"
    plaintext = "plaintext"


@dataclass(slots=True)
class ParsedSection:
    """A section of a parsed document."""
    text: str
    header_hierarchy: list[str]
    start_char: int
    end_char: int


@dataclass(slots=True)
class ChunkData:
    """Intermediate chunk before storage."""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    sentence_count: int
    confidence: float
    coverage: float
    matched_words: int
    top_reef_id: int
    top_reef_name: str
    arch_scores: list[float]
    top_reefs: list[tuple[int, str, float, float, int]]  # (reef_id, name, z_score, raw_bm25, n_contributing_words)
    top_islands: list[tuple[int, str, float, int]]  # (island_id, name, aggregate_z, n_contributing_reefs)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class SharedReef:
    """A reef shared between a query and a chunk result."""
    reef_name: str
    chunk_z: float
    query_z: float


@dataclass(slots=True)
class SearchResult:
    """A single search result."""
    text: str
    match_score: float
    n_shared_reefs: int
    document_id: int
    document_title: str
    chunk_index: int
    start_char: int
    end_char: int
    confidence: float
    coverage: float
    top_reef_name: str
    arch_scores: list[float] = field(default_factory=list)
    shared_reefs: list[SharedReef] = field(default_factory=list)


@dataclass(slots=True)
class QueryInfo:
    """Information about the scored query."""
    top_reef: str
    confidence: float
    coverage: float
    top_reefs: list[dict]  # [{"reef_name": str, "z_score": float}, ...]
    top_islands: list[dict] = field(default_factory=list)  # [{"island_name": str, "aggregate_z": float}, ...]


@dataclass(slots=True)
class SearchResponse:
    """Full search response with results and query info."""
    results: list[SearchResult]
    query_info: QueryInfo


@dataclass(slots=True)
class IngestResult:
    """Result of ingesting a document."""
    id: int
    title: str
    n_chunks: int
    tags: list[str]

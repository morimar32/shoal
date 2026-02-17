"""Shoal: retrieval engine built on lagoon."""

from ._engine import Engine
from ._models import (
    DocFormat,
    IngestResult,
    QueryInfo,
    SearchResponse,
    SearchResult,
    SectionData,
    SharedReef,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DocFormat",
    "Engine",
    "IngestResult",
    "QueryInfo",
    "SearchResponse",
    "SearchResult",
    "SectionData",
    "SharedReef",
]

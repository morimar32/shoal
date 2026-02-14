"""Engine: top-level orchestrator for shoal."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lagoon

from ._ingest import ingest_document
from ._models import DocFormat, IngestResult, SearchResponse
from ._retrieve import search
from ._storage import Storage
from ._vocab import inject_custom_vocabulary_at_startup

if TYPE_CHECKING:
    from lagoon import ReefScorer


class Engine:
    """Top-level shoal orchestrator.

    Usage::

        engine = Engine(db_path="shoal.db")
        engine.start()
        result = engine.ingest(title="My Doc", content="...", format=DocFormat.plaintext)
        response = engine.search("my query")
        engine.close()

    Or as a context manager::

        with Engine(db_path="shoal.db") as engine:
            engine.ingest(...)
            engine.search(...)
    """

    def __init__(
        self,
        db_path: str | Path = "shoal.db",
        lagoon_data_dir: str | Path | None = None,
    ):
        self._db_path = str(db_path)
        self._lagoon_data_dir = lagoon_data_dir
        self._scorer: ReefScorer | None = None
        self._storage: Storage | None = None

    def start(self) -> None:
        """Load lagoon scorer, connect to storage, inject custom vocabulary."""
        self._scorer = lagoon.load(self._lagoon_data_dir)
        self._storage = Storage(self._db_path)
        self._storage.connect()
        inject_custom_vocabulary_at_startup(self._scorer, self._storage)

    def close(self) -> None:
        """Close storage connection."""
        if self._storage:
            self._storage.close()
            self._storage = None
        self._scorer = None

    def __enter__(self) -> Engine:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def scorer(self) -> ReefScorer:
        if self._scorer is None:
            raise RuntimeError("Engine not started. Call start() first.")
        return self._scorer

    @property
    def storage(self) -> Storage:
        if self._storage is None:
            raise RuntimeError("Engine not started. Call start() first.")
        return self._storage

    def ingest(
        self,
        *,
        title: str,
        content: str,
        format: DocFormat = DocFormat.plaintext,
        tags: list[str] | None = None,
        source_path: str | None = None,
        metadata: dict | None = None,
        sensitivity: float = 1.0,
        smooth_window: int = 2,
        min_chunk_sentences: int = 2,
        max_chunk_sentences: int = 30,
    ) -> IngestResult:
        """Ingest a document."""
        return ingest_document(
            self.scorer,
            self.storage,
            title=title,
            content=content,
            fmt=format,
            tags=tags,
            source_path=source_path,
            metadata=metadata,
            sensitivity=sensitivity,
            smooth_window=smooth_window,
            min_chunk_sentences=min_chunk_sentences,
            max_chunk_sentences=max_chunk_sentences,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        tags: list[str] | None = None,
        min_confidence: float | None = None,
        include_scores: bool = False,
    ) -> SearchResponse:
        """Search for relevant chunks."""
        return search(
            self.scorer,
            self.storage,
            query,
            top_k=top_k,
            tags=tags,
            min_confidence=min_confidence,
            include_scores=include_scores,
        )

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all its chunks."""
        return self.storage.delete_document(doc_id)

    def status(self) -> dict:
        """Return system status."""
        return {
            "status": "ok",
            "n_documents": self.storage.count_documents(),
            "n_chunks": self.storage.count_chunks(),
            "lagoon_version": lagoon.__version__,
            "db_path": self._db_path,
        }

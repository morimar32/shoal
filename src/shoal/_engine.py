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
        max_chunk_sentences: int = 12,
        min_chunk_chars: int = 175,
        max_section_depth: int = 2,
        min_section_length: int = 200,
        min_reef_z: float = 2.0,
        reef_top_k: int = 38,
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
            min_chunk_chars=min_chunk_chars,
            max_section_depth=max_section_depth,
            min_section_length=min_section_length,
            min_reef_z=min_reef_z,
            reef_top_k=reef_top_k,
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

    def explain_query(self, query: str) -> dict:
        """Analyze a query and return detailed diagnostic information.

        Returns per-word scoring breakdown, overall confidence assessment,
        and warnings about poorly-understood terms.
        """
        # Score the full query
        full_result = self.scorer.score(query, top_k=10)

        # Score each word individually
        words = [w for w in query.split() if any(c.isalnum() for c in w)]
        word_details = []
        for word in words:
            w_result = self.scorer.score(word, top_k=3)
            word_details.append({
                "word": word,
                "confidence": w_result.confidence,
                "matched": w_result.matched_words > 0,
                "top_reefs": [
                    {"reef_name": r.name, "z_score": r.z_score}
                    for r in w_result.top_reefs[:3]
                ],
            })

        # Classify words by signal strength
        weak_words = [w for w in word_details if w["confidence"] < 0.1]
        strong_words = [w for w in word_details if w["confidence"] >= 0.5]

        # Build warnings
        warnings = []
        if full_result.confidence < 0.1:
            warnings.append(
                "Very low query confidence. The scorer has almost no semantic "
                "signal for this query. Results will be driven by incidental "
                "reef overlap rather than genuine topical matching."
            )
        elif full_result.confidence < 0.5:
            warnings.append(
                "Low query confidence. Results may not accurately reflect "
                "the intended topic."
            )

        if weak_words:
            weak_names = [w["word"] for w in weak_words]
            warnings.append(
                f"Weak signal words: {', '.join(weak_names)}. "
                f"These terms produce little or no reef signal. "
                f"Try adding more context words that describe the topic."
            )

        max_z = full_result.top_reefs[0].z_score if full_result.top_reefs else 0
        if max_z <= 1.0 and len(words) > 1:
            warnings.append(
                f"Top reef z-score is only {max_z:.2f}. "
                f"Combined query terms aren't producing strong reef activation. "
                f"The search will match on weak/generic reef overlap."
            )

        return {
            "query": query,
            "confidence": full_result.confidence,
            "coverage": full_result.coverage,
            "matched_words": full_result.matched_words,
            "total_words": len(words),
            "top_reefs": [
                {"reef_name": r.name, "z_score": r.z_score,
                 "n_contributing_words": r.n_contributing_words}
                for r in full_result.top_reefs[:10]
            ],
            "top_islands": [
                {"island_name": isl.name, "aggregate_z": isl.aggregate_z}
                for isl in full_result.top_islands[:5]
            ],
            "word_details": word_details,
            "warnings": warnings,
        }

    def status(self) -> dict:
        """Return system status."""
        return {
            "status": "ok",
            "n_documents": self.storage.count_documents(),
            "n_sections": self.storage.count_sections(),
            "n_chunks": self.storage.count_chunks(),
            "lagoon_version": lagoon.__version__,
            "db_path": self._db_path,
        }

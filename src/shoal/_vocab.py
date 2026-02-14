"""Vocabulary extension stub for Phase 1."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lagoon import ReefScorer

    from ._storage import Storage


def inject_custom_vocabulary_at_startup(scorer: ReefScorer, storage: Storage) -> int:
    """Load custom vocabulary from SQLite into the scorer.

    Returns the number of custom words injected.
    Phase 1: returns 0 (no custom vocabulary yet).
    Phase 2 will implement full injection.
    """
    return 0

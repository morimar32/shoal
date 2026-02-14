"""Test fixtures for shoal."""

import tempfile
from pathlib import Path

import lagoon
import pytest

from shoal._storage import Storage
from shoal import Engine


@pytest.fixture(scope="session")
def scorer():
    """Session-scoped lagoon scorer (expensive to load)."""
    return lagoon.load()


@pytest.fixture
def storage(tmp_path):
    """Per-test SQLite storage."""
    db_path = tmp_path / "test.db"
    s = Storage(db_path)
    s.connect()
    yield s
    s.close()


@pytest.fixture
def engine(tmp_path):
    """Per-test Engine instance."""
    db_path = tmp_path / "test.db"
    eng = Engine(db_path=db_path)
    eng.start()
    yield eng
    eng.close()

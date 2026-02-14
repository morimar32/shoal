"""SQLite storage layer for shoal."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from ._models import ChunkData, SearchResult, SharedReef

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    source_path   TEXT,
    content_hash  TEXT NOT NULL,
    format        TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata      TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS document_tags (
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag           TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    start_char      INTEGER NOT NULL,
    end_char        INTEGER NOT NULL,
    sentence_count  INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    coverage        REAL NOT NULL,
    matched_words   INTEGER NOT NULL,
    top_reef_id     INTEGER NOT NULL,
    top_reef_name   TEXT NOT NULL,
    arch_score_0    REAL NOT NULL,
    arch_score_1    REAL NOT NULL,
    arch_score_2    REAL NOT NULL,
    arch_score_3    REAL NOT NULL,
    metadata        TEXT DEFAULT '{}',
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_top_reef_id ON chunks(top_reef_id);
CREATE INDEX IF NOT EXISTS idx_chunks_confidence ON chunks(confidence);

CREATE TABLE IF NOT EXISTS chunk_reefs (
    chunk_id              INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    reef_id               INTEGER NOT NULL,
    reef_name             TEXT NOT NULL,
    z_score               REAL NOT NULL,
    raw_bm25              REAL NOT NULL,
    n_contributing_words  INTEGER NOT NULL,
    rank                  INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, reef_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_reefs_reef_id ON chunk_reefs(reef_id);
CREATE INDEX IF NOT EXISTS idx_chunk_reefs_z_score ON chunk_reefs(z_score);

CREATE TABLE IF NOT EXISTS chunk_islands (
    chunk_id              INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    island_id             INTEGER NOT NULL,
    island_name           TEXT NOT NULL,
    aggregate_z           REAL NOT NULL,
    n_contributing_reefs  INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, island_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_islands_island_id ON chunk_islands(island_id);

CREATE TABLE IF NOT EXISTS custom_words (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    word            TEXT NOT NULL UNIQUE,
    word_hash       INTEGER NOT NULL UNIQUE,
    word_id         INTEGER NOT NULL UNIQUE,
    is_compound     INTEGER NOT NULL DEFAULT 0,
    specificity     INTEGER NOT NULL,
    idf_q           INTEGER NOT NULL,
    n_occurrences   INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS custom_word_reefs (
    word_id     INTEGER NOT NULL REFERENCES custom_words(word_id) ON DELETE CASCADE,
    reef_id     INTEGER NOT NULL,
    bm25_q      INTEGER NOT NULL,
    association_strength REAL NOT NULL,
    PRIMARY KEY (word_id, reef_id)
);

CREATE TABLE IF NOT EXISTS word_observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    word          TEXT NOT NULL,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sentence_idx  INTEGER NOT NULL,
    context_reefs TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_word_observations_word ON word_observations(word);
"""


class Storage:
    """SQLite storage backend for shoal."""

    def __init__(self, db_path: str | Path = "shoal.db"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def db_path(self) -> str:
        return self._db_path

    def connect(self) -> None:
        """Open SQLite connection and create schema."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Storage not connected. Call connect() first.")
        return self._conn

    # ── Document CRUD ──

    def insert_document(
        self,
        title: str,
        content: str,
        fmt: str,
        source_path: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Insert a document and return its id."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        meta_json = json.dumps(metadata or {})
        cur = self.conn.execute(
            "INSERT INTO documents (title, source_path, content_hash, format, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, source_path, content_hash, fmt, meta_json),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def set_tags(self, doc_id: int, tags: list[str]) -> None:
        """Replace all tags for a document."""
        self.conn.execute("DELETE FROM document_tags WHERE document_id = ?", (doc_id,))
        for tag in tags:
            self.conn.execute(
                "INSERT INTO document_tags (document_id, tag) VALUES (?, ?)",
                (doc_id, tag),
            )
        self.conn.commit()

    def get_tags(self, doc_id: int) -> list[str]:
        """Get all tags for a document."""
        rows = self.conn.execute(
            "SELECT tag FROM document_tags WHERE document_id = ? ORDER BY tag",
            (doc_id,),
        ).fetchall()
        return [r["tag"] for r in rows]

    def get_document(self, doc_id: int) -> dict | None:
        """Get a document by id."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        doc = dict(row)
        doc["tags"] = self.get_tags(doc_id)
        return doc

    def list_documents(
        self,
        tag: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List documents, optionally filtered by tag."""
        if tag:
            rows = self.conn.execute(
                "SELECT d.* FROM documents d "
                "JOIN document_tags dt ON d.id = dt.document_id "
                "WHERE dt.tag = ? ORDER BY d.id LIMIT ? OFFSET ?",
                (tag, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM documents ORDER BY id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        result = []
        for row in rows:
            doc = dict(row)
            doc["tags"] = self.get_tags(doc["id"])
            result.append(doc)
        return result

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all associated data. Returns True if found."""
        cur = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def count_documents(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return row["n"]

    def count_chunks(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return row["n"]

    def count_chunks_for_document(self, doc_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchone()
        return row["n"]

    # ── Chunk insertion ──

    def insert_chunks(self, doc_id: int, chunks: list[ChunkData]) -> list[int]:
        """Insert chunks with their reef and island scores. Returns list of chunk ids."""
        chunk_ids = []
        for chunk in chunks:
            arch = chunk.arch_scores
            cur = self.conn.execute(
                "INSERT INTO chunks "
                "(document_id, chunk_index, text, start_char, end_char, sentence_count, "
                "confidence, coverage, matched_words, top_reef_id, top_reef_name, "
                "arch_score_0, arch_score_1, arch_score_2, arch_score_3, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.sentence_count,
                    chunk.confidence,
                    chunk.coverage,
                    chunk.matched_words,
                    chunk.top_reef_id,
                    chunk.top_reef_name,
                    arch[0] if len(arch) > 0 else 0.0,
                    arch[1] if len(arch) > 1 else 0.0,
                    arch[2] if len(arch) > 2 else 0.0,
                    arch[3] if len(arch) > 3 else 0.0,
                    json.dumps(chunk.metadata),
                ),
            )
            chunk_id = cur.lastrowid
            chunk_ids.append(chunk_id)  # type: ignore[arg-type]

            # Insert chunk_reefs rows
            for rank, (reef_id, reef_name, z_score, raw_bm25, n_words) in enumerate(chunk.top_reefs):
                self.conn.execute(
                    "INSERT INTO chunk_reefs "
                    "(chunk_id, reef_id, reef_name, z_score, raw_bm25, n_contributing_words, rank) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, reef_id, reef_name, z_score, raw_bm25, n_words, rank),
                )

            # Insert chunk_islands rows
            for island_id, island_name, agg_z, n_reefs in chunk.top_islands:
                self.conn.execute(
                    "INSERT INTO chunk_islands "
                    "(chunk_id, island_id, island_name, aggregate_z, n_contributing_reefs) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, island_id, island_name, agg_z, n_reefs),
                )

        self.conn.commit()
        return chunk_ids

    # ── Retrieval ──

    def search_by_reef_overlap(
        self,
        query_reefs: list[tuple[int, float]],  # [(reef_id, z_score), ...]
        *,
        top_k: int = 10,
        tags: list[str] | None = None,
        min_confidence: float | None = None,
        include_scores: bool = False,
    ) -> list[SearchResult]:
        """Find chunks sharing query reefs, ranked by weighted z-score overlap."""
        if not query_reefs:
            return []

        # Build the CTE VALUES clause for query reefs
        value_placeholders = ", ".join(["(?, ?)"] * len(query_reefs))
        params: list = []
        for reef_id, z_score in query_reefs:
            params.extend([reef_id, z_score])

        # Build SQL dynamically based on filters
        sql = f"""
        WITH qr(reef_id, query_z) AS (VALUES {value_placeholders})
        SELECT c.id, c.text, c.document_id, d.title AS document_title,
               c.chunk_index, c.start_char, c.end_char,
               c.confidence, c.coverage, c.top_reef_name,
               c.arch_score_0, c.arch_score_1, c.arch_score_2, c.arch_score_3,
               SUM(cr.z_score * qr.query_z) AS match_score,
               COUNT(cr.reef_id) AS n_shared_reefs
        FROM chunks c
        JOIN chunk_reefs cr ON c.id = cr.chunk_id
        JOIN qr ON cr.reef_id = qr.reef_id
        JOIN documents d ON c.document_id = d.id
        """

        where_clauses = []

        if tags:
            # Add tag filter — require all tags
            for i, tag in enumerate(tags):
                alias = f"dt{i}"
                sql += f" JOIN document_tags {alias} ON d.id = {alias}.document_id AND {alias}.tag = ?"
                params.append(tag)

        if min_confidence is not None:
            where_clauses.append("c.confidence > ?")
            params.append(min_confidence)

        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " GROUP BY c.id ORDER BY match_score DESC LIMIT ?"
        params.append(top_k)

        rows = self.conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            shared = []
            if include_scores:
                # Fetch individual shared reef details
                shared = self._get_shared_reefs(row["id"], query_reefs)

            results.append(SearchResult(
                text=row["text"],
                match_score=row["match_score"],
                n_shared_reefs=row["n_shared_reefs"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                chunk_index=row["chunk_index"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                confidence=row["confidence"],
                coverage=row["coverage"],
                top_reef_name=row["top_reef_name"],
                arch_scores=[
                    row["arch_score_0"], row["arch_score_1"],
                    row["arch_score_2"], row["arch_score_3"],
                ],
                shared_reefs=shared,
            ))

        return results

    def _get_shared_reefs(
        self, chunk_id: int, query_reefs: list[tuple[int, float]]
    ) -> list[SharedReef]:
        """Get the shared reef details between a chunk and query."""
        query_reef_map = {rid: z for rid, z in query_reefs}
        rows = self.conn.execute(
            "SELECT reef_id, reef_name, z_score FROM chunk_reefs WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchall()

        shared = []
        for row in rows:
            rid = row["reef_id"]
            if rid in query_reef_map:
                shared.append(SharedReef(
                    reef_name=row["reef_name"],
                    chunk_z=row["z_score"],
                    query_z=query_reef_map[rid],
                ))
        # Sort by contribution (chunk_z * query_z) descending
        shared.sort(key=lambda s: s.chunk_z * s.query_z, reverse=True)
        return shared

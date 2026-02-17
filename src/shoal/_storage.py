"""SQLite storage layer for shoal."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

from ._models import ChunkData, SearchResult, SectionData, SharedReef, WordObservation

# Default number of reefs to keep per chunk after IDF-sorted filtering.
# Reefs are sorted by z_score * idf (selectivity), and only the top K
# are stored. This makes reef overlap discriminating: topically relevant
# chunks share more of their distinctive reefs with the query.
_DEFAULT_REEF_TOP_K = 38

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

CREATE TABLE IF NOT EXISTS sections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id   INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    depth       INTEGER NOT NULL DEFAULT 0,
    position    INTEGER NOT NULL,
    start_char  INTEGER NOT NULL,
    end_char    INTEGER NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_parent_id ON sections(parent_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id      INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
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
    arch_score_4    REAL NOT NULL DEFAULT 0.0,
    reef_l2_norm    REAL NOT NULL DEFAULT 0.0,
    metadata        TEXT DEFAULT '{}',
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id);
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

CREATE TABLE IF NOT EXISTS chunk_custom_words (
    chunk_id        INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    custom_word_id  INTEGER NOT NULL REFERENCES custom_words(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, custom_word_id)
);
CREATE INDEX IF NOT EXISTS idx_ccw_custom_word_id ON chunk_custom_words(custom_word_id);
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

    def count_sections(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM sections").fetchone()
        return row["n"]

    # ── Section operations ──

    def insert_sections(self, doc_id: int, sections: list[SectionData]) -> list[int]:
        """Insert a section tree in order, resolving parent_index to parent_id.

        Returns list of section IDs matching the input list order.
        """
        section_ids: list[int] = []
        for section in sections:
            parent_id = None
            if section.parent_index is not None:
                parent_id = section_ids[section.parent_index]

            cur = self.conn.execute(
                "INSERT INTO sections "
                "(document_id, parent_id, title, depth, position, start_char, end_char, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    parent_id,
                    section.title,
                    section.depth,
                    section.position,
                    section.start_char,
                    section.end_char,
                    json.dumps(section.metadata),
                ),
            )
            section_ids.append(cur.lastrowid)  # type: ignore[arg-type]

        self.conn.commit()
        return section_ids

    def update_section_chunk_count(self, section_id: int, n_chunks: int) -> None:
        """Update the n_chunks count for a section."""
        self.conn.execute(
            "UPDATE sections SET n_chunks = ? WHERE id = ?",
            (n_chunks, section_id),
        )
        self.conn.commit()

    def get_sections_for_document(self, doc_id: int) -> list[dict]:
        """Return all sections for a document, ordered by depth then position."""
        rows = self.conn.execute(
            "SELECT * FROM sections WHERE document_id = ? ORDER BY depth, position",
            (doc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_section_path(self, section_id: int) -> list[str]:
        """Walk the parent_id chain to build a section path.

        Returns list like ["Global temperature rise", "Warming since..."].
        """
        path: list[str] = []
        current_id: int | None = section_id
        while current_id is not None:
            row = self.conn.execute(
                "SELECT id, parent_id, title FROM sections WHERE id = ?",
                (current_id,),
            ).fetchone()
            if row is None:
                break
            path.append(row["title"])
            current_id = row["parent_id"]
        path.reverse()
        return path

    # ── Reef IDF ──

    def get_reef_idf(self) -> dict[int, float]:
        """Compute IDF for each reef from current chunk_reefs data.

        IDF = log(N / df) where N = total chunks, df = chunks containing this reef.
        Returns empty dict if no chunks exist yet (first document).
        """
        n_chunks = self.count_chunks()
        if n_chunks == 0:
            return {}

        rows = self.conn.execute(
            "SELECT reef_id, COUNT(DISTINCT chunk_id) AS df FROM chunk_reefs GROUP BY reef_id"
        ).fetchall()
        return {row["reef_id"]: math.log(n_chunks / row["df"]) for row in rows}

    # ── Chunk insertion ──

    def insert_chunks(
        self,
        doc_id: int,
        chunks: list[ChunkData],
        section_ids: list[int] | None = None,
        reef_top_k: int = _DEFAULT_REEF_TOP_K,
    ) -> list[int]:
        """Insert chunks with their reef and island scores. Returns list of chunk ids.

        If section_ids is provided, each chunk's section_index is used to look up
        the actual section_id from the list.

        Reef filtering: each chunk's reefs are sorted by z_score * IDF
        (selectivity), and only the top reef_top_k are stored. This keeps
        the most corpus-distinctive reefs per chunk, making reef overlap
        discriminating for retrieval.
        """
        # Compute current reef IDF for selectivity-based filtering
        reef_idf = self.get_reef_idf()

        chunk_ids = []
        for chunk in chunks:
            arch = chunk.arch_scores

            # Resolve section_id
            if section_ids is not None:
                section_id = section_ids[chunk.section_index]
            else:
                section_id = chunk.section_index  # fallback: treat as direct id

            # IDF-sorted reef filtering: keep top-K by z * idf
            filtered_reefs = self._filter_reefs_by_selectivity(
                chunk.top_reefs, reef_idf, reef_top_k,
            )

            # Compute L2 norm from kept reefs
            reef_l2_norm = math.sqrt(
                sum(z ** 2 for _, _, z, _, _ in filtered_reefs)
            ) if filtered_reefs else 0.0

            cur = self.conn.execute(
                "INSERT INTO chunks "
                "(document_id, section_id, chunk_index, text, start_char, end_char, sentence_count, "
                "confidence, coverage, matched_words, top_reef_id, top_reef_name, "
                "arch_score_0, arch_score_1, arch_score_2, arch_score_3, arch_score_4, reef_l2_norm, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    section_id,
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
                    arch[4] if len(arch) > 4 else 0.0,
                    reef_l2_norm,
                    json.dumps(chunk.metadata),
                ),
            )
            chunk_id = cur.lastrowid
            chunk_ids.append(chunk_id)  # type: ignore[arg-type]

            # Insert filtered chunk_reefs rows
            for rank, (reef_id, reef_name, z_score, raw_bm25, n_words) in enumerate(filtered_reefs):
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

    @staticmethod
    def _filter_reefs_by_selectivity(
        top_reefs: list[tuple[int, str, float, float, int]],
        reef_idf: dict[int, float],
        top_k: int,
    ) -> list[tuple[int, str, float, float, int]]:
        """Filter chunk reefs to the top-K most corpus-distinctive.

        Sorts reefs by z_score * IDF (selectivity), keeping those that are
        both strong in this chunk AND rare across the corpus. Falls back to
        raw z_score sorting when IDF data is unavailable (first document).
        """
        if len(top_reefs) <= top_k:
            return top_reefs

        if reef_idf:
            # Unseen reefs get max IDF * 1.5: they're rarer than the rarest
            # known reef, so they should be FAVORED for storage, not excluded.
            default_idf = max(reef_idf.values()) * 1.5
            # Sort by selectivity = z_score * idf, descending
            scored = [
                (reef_id, name, z, bm25, nw, z * reef_idf.get(reef_id, default_idf))
                for reef_id, name, z, bm25, nw in top_reefs
            ]
            scored.sort(key=lambda x: x[5], reverse=True)
            return [(rid, name, z, bm25, nw) for rid, name, z, bm25, nw, _ in scored[:top_k]]
        else:
            # No IDF data yet — fall back to raw z-score ordering
            sorted_reefs = sorted(top_reefs, key=lambda x: x[2], reverse=True)
            return sorted_reefs[:top_k]

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
        """Find chunks sharing query reefs, ranked by normalized reef overlap.

        Scoring: SUM(chunk_z * query_z) * sqrt(n_shared) / reef_l2_norm

        The sqrt(n_shared) factor rewards chunks sharing multiple reefs with
        the query. Dividing by reef_l2_norm (precomputed at ingestion) normalizes
        away text-length bias without any query-time subqueries.
        """
        if not query_reefs:
            return []

        # Build the CTE VALUES clause for query reefs
        value_placeholders = ", ".join(["(?, ?)"] * len(query_reefs))
        params: list = []
        for reef_id, z_score in query_reefs:
            params.extend([reef_id, z_score])

        # Build SQL dynamically based on filters
        # Scoring: dot_product * sqrt(n_shared) / reef_l2_norm
        sql = f"""
        WITH qr(reef_id, query_z) AS (VALUES {value_placeholders})
        SELECT c.id, c.text, c.document_id, d.title AS document_title,
               c.section_id, s.title AS section_title,
               c.chunk_index, c.start_char, c.end_char,
               c.confidence, c.coverage, c.top_reef_name,
               c.arch_score_0, c.arch_score_1, c.arch_score_2, c.arch_score_3, c.arch_score_4,
               CASE WHEN c.reef_l2_norm > 0
                    THEN SUM(cr.z_score * qr.query_z) * sqrt(COUNT(cr.reef_id)) / c.reef_l2_norm
                    ELSE 0.0
               END AS match_score,
               COUNT(cr.reef_id) AS n_shared_reefs
        FROM chunks c
        JOIN chunk_reefs cr ON c.id = cr.chunk_id
        JOIN qr ON cr.reef_id = qr.reef_id
        JOIN documents d ON c.document_id = d.id
        JOIN sections s ON c.section_id = s.id
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

            # Get section path
            section_path = self.get_section_path(row["section_id"])

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
                chunk_id=row["id"],
                section_title=row["section_title"],
                section_path=section_path,
                arch_scores=[
                    row["arch_score_0"], row["arch_score_1"],
                    row["arch_score_2"], row["arch_score_3"],
                    row["arch_score_4"],
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

    # ── Word observations ──

    def insert_word_observations(self, observations: list[WordObservation]) -> None:
        """Bulk insert Pass 1 word observations."""
        self.conn.executemany(
            "INSERT INTO word_observations (word, document_id, sentence_idx, context_reefs) "
            "VALUES (?, ?, ?, ?)",
            [
                (obs.word, obs.document_id, obs.sentence_idx, json.dumps(obs.context_reefs))
                for obs in observations
            ],
        )
        self.conn.commit()

    def get_word_observations_for_document(self, doc_id: int) -> dict[str, list[list[dict]]]:
        """Get word observations grouped by word for a document.

        Returns {word: [context_reefs_list, ...]} where each entry is
        the context_reefs from one observation.
        """
        rows = self.conn.execute(
            "SELECT word, context_reefs FROM word_observations "
            "WHERE document_id = ? ORDER BY word, sentence_idx",
            (doc_id,),
        ).fetchall()

        grouped: dict[str, list[list[dict]]] = {}
        for row in rows:
            word = row["word"]
            reefs = json.loads(row["context_reefs"])
            grouped.setdefault(word, []).append(reefs)
        return grouped

    # ── Custom words ──

    def insert_custom_word(
        self,
        word: str,
        word_hash: int,
        word_id: int,
        specificity: int,
        idf_q: int,
        n_occurrences: int,
        reef_entries: list[tuple[int, int, float]],
    ) -> int:
        """Insert or update a custom word and its reef associations.

        reef_entries: [(reef_id, bm25_q, association_strength), ...]
        Returns the custom_words.id (autoincrement).
        """
        # Delete old reef associations first (before changing word_id via upsert)
        existing = self.conn.execute(
            "SELECT word_id FROM custom_words WHERE word = ?", (word,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "DELETE FROM custom_word_reefs WHERE word_id = ?",
                (existing["word_id"],),
            )

        self.conn.execute(
            "INSERT INTO custom_words (word, word_hash, word_id, specificity, idf_q, n_occurrences) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(word) DO UPDATE SET "
            "word_id = excluded.word_id, specificity = excluded.specificity, "
            "idf_q = excluded.idf_q, n_occurrences = excluded.n_occurrences, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            (word, word_hash, word_id, specificity, idf_q, n_occurrences),
        )

        # Get the id (works for both insert and update)
        row = self.conn.execute(
            "SELECT id FROM custom_words WHERE word = ?", (word,)
        ).fetchone()
        cw_id = row["id"]

        # Insert new reef associations
        for reef_id, bm25_q, strength in reef_entries:
            self.conn.execute(
                "INSERT INTO custom_word_reefs (word_id, reef_id, bm25_q, association_strength) "
                "VALUES (?, ?, ?, ?)",
                (word_id, reef_id, bm25_q, strength),
            )
        self.conn.commit()
        return cw_id

    def load_custom_vocabulary(self) -> list[dict]:
        """Load all custom words with their reef associations for startup injection."""
        words = self.conn.execute(
            "SELECT id, word, word_hash, word_id, specificity, idf_q, n_occurrences "
            "FROM custom_words ORDER BY id"
        ).fetchall()

        result = []
        for w in words:
            reefs = self.conn.execute(
                "SELECT reef_id, association_strength FROM custom_word_reefs WHERE word_id = ?",
                (w["word_id"],),
            ).fetchall()
            result.append({
                "id": w["id"],
                "word": w["word"],
                "word_hash": w["word_hash"],
                "word_id": w["word_id"],
                "specificity": w["specificity"],
                "idf_q": w["idf_q"],
                "n_occurrences": w["n_occurrences"],
                "reef_associations": [(r["reef_id"], r["association_strength"]) for r in reefs],
            })
        return result

    # ── Chunk custom words ──

    def insert_chunk_custom_words(self, chunk_id: int, custom_word_ids: list[int]) -> None:
        """Record which custom words appear in a chunk."""
        if not custom_word_ids:
            return
        self.conn.executemany(
            "INSERT OR IGNORE INTO chunk_custom_words (chunk_id, custom_word_id) VALUES (?, ?)",
            [(chunk_id, cw_id) for cw_id in custom_word_ids],
        )
        self.conn.commit()

    def get_chunks_by_custom_words(
        self, custom_word_ids: list[int], *, limit: int = 50
    ) -> list[int]:
        """Get chunk IDs containing any given custom word, ordered by match count desc."""
        if not custom_word_ids:
            return []
        placeholders = ", ".join(["?"] * len(custom_word_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, COUNT(*) AS n_matches "
            f"FROM chunk_custom_words "
            f"WHERE custom_word_id IN ({placeholders}) "
            f"GROUP BY chunk_id ORDER BY n_matches DESC LIMIT ?",
            [*custom_word_ids, limit],
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def score_chunks_by_reef_overlap(
        self,
        chunk_ids: list[int],
        query_reefs: list[tuple[int, float]],
    ) -> list[SearchResult]:
        """Score specific chunks by reef overlap (used by lightning rod).

        Same scoring formula as search_by_reef_overlap but pre-filtered
        to the given chunk IDs.
        """
        if not chunk_ids or not query_reefs:
            return []

        # Build query reef CTE
        value_placeholders = ", ".join(["(?, ?)"] * len(query_reefs))
        params: list = []
        for reef_id, z_score in query_reefs:
            params.extend([reef_id, z_score])

        # Build chunk_id filter
        chunk_placeholders = ", ".join(["?"] * len(chunk_ids))
        params.extend(chunk_ids)

        sql = f"""
        WITH qr(reef_id, query_z) AS (VALUES {value_placeholders})
        SELECT c.id, c.text, c.document_id, d.title AS document_title,
               c.section_id, s.title AS section_title,
               c.chunk_index, c.start_char, c.end_char,
               c.confidence, c.coverage, c.top_reef_name,
               c.arch_score_0, c.arch_score_1, c.arch_score_2, c.arch_score_3, c.arch_score_4,
               c.reef_l2_norm,
               CASE WHEN c.reef_l2_norm > 0
                    THEN SUM(cr.z_score * qr.query_z) * sqrt(COUNT(cr.reef_id)) / c.reef_l2_norm
                    ELSE 0.0
               END AS match_score,
               COUNT(cr.reef_id) AS n_shared_reefs
        FROM chunks c
        JOIN chunk_reefs cr ON c.id = cr.chunk_id
        JOIN qr ON cr.reef_id = qr.reef_id
        JOIN documents d ON c.document_id = d.id
        JOIN sections s ON c.section_id = s.id
        WHERE c.id IN ({chunk_placeholders})
        GROUP BY c.id
        ORDER BY match_score DESC
        """

        rows = self.conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            section_path = self.get_section_path(row["section_id"])
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
                chunk_id=row["id"],
                section_title=row["section_title"],
                section_path=section_path,
                arch_scores=[
                    row["arch_score_0"], row["arch_score_1"],
                    row["arch_score_2"], row["arch_score_3"],
                    row["arch_score_4"],
                ],
            ))
        return results

    def count_custom_words(self) -> int:
        """Count the number of custom words in the vocabulary."""
        row = self.conn.execute("SELECT COUNT(*) AS n FROM custom_words").fetchone()
        return row["n"]

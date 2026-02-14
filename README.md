# Shoal

Shoal is a retrieval engine — the "R" in RAG — built on [lagoon](https://github.com/morimar32/lagoon). It ingests documents, chunks them using lagoon's topic-shift detection, scores each chunk against 207 semantic reefs, and stores structured reef metadata in SQLite. Retrieval is standard SQL: score the query, find chunks whose reef profiles overlap via joins and filters. No vectors, no embeddings, no cosine similarity.

**Naming:** `shoal` is the library. `shoald` is the HTTP REST daemon (Phase 3).

---

## Quick Start

### Installation

```bash
# Install lagoon from sibling directory (not yet on PyPI)
pip install -e ../lagoon

# Install shoal
pip install -e .
```

### CLI Usage

```bash
# Ingest a single file
shoal ingest docs/photosynthesis.md

# Ingest an entire directory of .md and .txt files
shoal ingest-dir docs/

# Search
shoal search "photosynthesis in plants"

# Search with shared reef details
shoal search "volcanic eruption geology" --scores

# Filter by tag
shoal ingest docs/photosynthesis.md --tags biology,plants
shoal search "chlorophyll" --tags biology

# Check status
shoal status

# Use a custom database path (default: shoal.db)
shoal --db my_corpus.db ingest-dir docs/
```

### Python API

```python
from shoal import Engine, DocFormat

with Engine(db_path="shoal.db") as engine:
    # Ingest a document
    result = engine.ingest(
        title="Photosynthesis",
        content=open("docs/photosynthesis.md").read(),
        format=DocFormat.markdown,
        tags=["biology"],
    )
    print(f"Ingested: {result.n_chunks} chunks")

    # Search
    response = engine.search("photosynthesis in plants", top_k=10)
    for r in response.results:
        print(f"  [{r.document_title}] score={r.match_score:.1f} reef={r.top_reef_name}")

    # Search with detailed reef overlap
    response = engine.search("chlorophyll", include_scores=True)
    for r in response.results:
        for sr in r.shared_reefs:
            print(f"    {sr.reef_name}: chunk_z={sr.chunk_z:.1f}, query_z={sr.query_z:.1f}")

    # Status
    print(engine.status())
```

---

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | **Done** | Core library: ingest, chunk, store, retrieve via CLI and Python API. Single-pass ingestion (no vocabulary extension). |
| **Phase 2** | Planned | Vocabulary extension: two-pass ingestion pipeline, context-based reef learning for unknown words, compound detection. |
| **Phase 3** | Planned | `shoald` REST API: FastAPI HTTP daemon wrapping the library. |

### What's built (Phase 1)

- Document ingestion with lagoon topic-shift chunking (markdown + plaintext)
- Full SQLite schema (all 8 tables, including vocabulary tables ready for Phase 2)
- SQL reef-overlap retrieval with tag filtering, confidence thresholds, and shared reef detail
- `Engine` class with context manager support
- CLI: `ingest`, `ingest-dir`, `search`, `status`
- 37 tests covering parsers, storage, ingestion, retrieval, and engine integration
- Test corpus: 28 documents (26 Wikipedia + 2 Gutenberg books) producing 3,400+ chunks

---

## Package Structure

```
src/shoal/
├── __init__.py       # Public API: Engine, DocFormat, result types
├── _models.py        # Dataclasses: DocFormat, ChunkData, SearchResult, etc.
├── _parsers.py       # Markdown section splitter, plaintext passthrough
├── _storage.py       # SQLite schema, CRUD, reef overlap retrieval query
├── _ingest.py        # Ingestion pipeline: parse → analyze → store
├── _retrieve.py      # Score query → SQL reef overlap → ranked results
├── _vocab.py         # Stub for Phase 2 vocabulary extension
├── _engine.py        # Engine class: top-level orchestrator
└── _cli.py           # CLI: ingest, ingest-dir, search, status
tests/
├── conftest.py       # Fixtures: scorer (session-scoped), storage, engine
├── test_parsers.py   # Markdown/plaintext parsing tests
├── test_storage.py   # CRUD + reef overlap search tests
├── test_ingest.py    # Ingestion pipeline tests
├── test_retrieve.py  # Retrieval tests
└── test_engine.py    # Engine integration tests
pyproject.toml        # Hatch build, src/ layout
```

---

## CLI Reference

All commands accept `--db <path>` (default: `shoal.db`).

### `shoal ingest <path>`

Ingest a single `.md` or `.txt` file. Format is auto-detected from extension. Title defaults to the filename stem, titlecased.

| Flag | Description |
|------|-------------|
| `--title` | Override document title |
| `--tags` | Comma-separated tags (e.g., `biology,plants`) |

### `shoal ingest-dir <dir>`

Ingest all `.md` and `.txt` files in a directory. Shows progress per file.

| Flag | Description |
|------|-------------|
| `--tags` | Comma-separated tags applied to all files |

### `shoal search <query>`

Search and display ranked results with match score, shared reef count, confidence, top reef name, and text preview.

| Flag | Description |
|------|-------------|
| `--top-k` | Number of results (default: 10) |
| `--tags` | Filter by comma-separated tags |
| `--min-confidence` | Minimum chunk confidence threshold |
| `--scores` | Show individual shared reef z-scores |

### `shoal status`

Display document count, chunk count, lagoon version, and database path.

---

## Library API

### `Engine`

The main entry point. Loads lagoon, connects to SQLite, and provides ingest/search/status methods.

```python
Engine(db_path="shoal.db", lagoon_data_dir=None)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Load lagoon scorer and connect to database |
| `close()` | `None` | Close database connection |
| `ingest(*, title, content, format, tags, ...)` | `IngestResult` | Ingest a document |
| `search(query, *, top_k, tags, min_confidence, include_scores)` | `SearchResponse` | Search for matching chunks |
| `delete_document(doc_id)` | `bool` | Delete a document and its chunks |
| `status()` | `dict` | System status (counts, lagoon version, db path) |

Supports context manager (`with Engine(...) as engine:`).

### Key Types

| Type | Fields |
|------|--------|
| `DocFormat` | Enum: `markdown`, `plaintext` |
| `IngestResult` | `id`, `title`, `n_chunks`, `tags` |
| `SearchResponse` | `results: list[SearchResult]`, `query_info: QueryInfo` |
| `SearchResult` | `text`, `match_score`, `n_shared_reefs`, `document_id`, `document_title`, `chunk_index`, `start_char`, `end_char`, `confidence`, `coverage`, `top_reef_name`, `shared_reefs` |
| `SharedReef` | `reef_name`, `chunk_z`, `query_z` |
| `QueryInfo` | `top_reef`, `confidence`, `coverage`, `top_reefs` |

---

## 1. Architecture

```
                    ┌──────────────────────────┐
                    │         shoald           │
                    │   (FastAPI HTTP layer)   │
                    │                          │
                    │  POST /api/v1/search     │
                    │  POST /api/v1/documents  │
                    │  GET  /api/v1/status     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │         shoal            │
                    │       (library)          │
                    │                          │
                    │  Vocabulary extension    │
                    │  Ingestion pipeline      │
                    │  Chunking engine         │
                    │  Storage layer           │
                    │  Retrieval engine        │
                    └──────┬─────────┬─────────┘
                           │         │
              ┌────────────▼──┐  ┌───▼───────────┐
              │    lagoon     │  │   SQLite3     │
              │  (scorer,     │  │  (standard,   │
              │   analyzer)   │  │   no ext.)    │
              └───────▲───────┘  └───────────────┘
                      │                  │
                      └──────────────────┘
                        custom vocab
                        injected at
                        startup
```

**shoal** (library) handles:
- Document ingestion and format parsing (markdown, plain text)
- Chunking via lagoon's topic-shift detection with size constraints
- Per-chunk scoring via lagoon to produce structured reef metadata
- SQLite storage and SQL-based retrieval over reef scores
- Vocabulary extension: learning reef associations for unknown words from document context *(Phase 2)*

**shoald** (daemon) handles *(Phase 3)*:
- FastAPI HTTP REST layer exposing the library
- Request validation, serialization, error responses
- OpenAPI documentation at `/docs`

**lagoon** is loaded once at startup via `lagoon.load()`. In Phase 2, it will be extended with custom vocabulary from shoal's database using lagoon's public vocabulary extension API: `add_custom_word()` injects individual words, `compute_custom_word_scores()` computes BM25 entries, and `rebuild_compounds()` merges custom compounds into the Aho-Corasick automaton. After extension, the scorer operates on the superset vocabulary transparently — all downstream code (scoring, analysis) benefits from the extended vocabulary without modification.

**SQLite3** is the storage backend. Standard library `sqlite3` — no extensions required. Reef scores are stored as structured SQL rows, queried with standard joins and filters. Custom vocabulary is also stored in SQLite, feeding back into the scorer at startup.

---

## 2. How Lagoon Is Used

Shoal depends on two lagoon capabilities: **scoring** and **document analysis**.

### Scoring: `score()`

Lagoon's `ReefScorer.score(text, top_k=10)` runs a 5-phase pipeline (compound scan, tokenization, BM25 accumulation, background subtraction, result extraction) and returns a `TopicResult`. See lagoon README sections 5 and 7 for the full pipeline.

This is the primary lagoon API for shoal. Every chunk gets scored at ingestion time, and every query gets scored at retrieval time. The structured output — top reefs with z-scores, confidence, coverage — is what gets stored and queried.

**`TopicResult` fields relevant to shoal:**

| Field | Type | Used For |
|-------|------|----------|
| `top_reefs` | `list[ScoredReef]` | Stored per-chunk as normalized rows. Matched against query reefs at retrieval. |
| `confidence` | `float` | Stored per-chunk. Filterable at query time (e.g., "only return high-confidence chunks"). |
| `coverage` | `float` | Stored per-chunk. Indicates dictionary coverage quality. |
| `matched_words` | `int` | Stored per-chunk. |
| `unknown_words` | `list[str]` | Collected during Pass 1 of ingestion. Used to build the custom vocabulary (see [Section 3](#3-vocabulary-extension)). Stop words are excluded automatically. |
| `matched_word_ids` | `frozenset[int]` | Set of lagoon word IDs that matched. Enables deduplication and cross-referencing with custom vocabulary `word_id` values. |
| `top_islands` | `list[ScoredIsland]` | Stored per-chunk for island-level queries. |
| `arch_scores` | `list[float]` (len=4) | Stored per-chunk as 4 columns. Enables archipelago-level filtering. |

Each `ScoredReef` contains `reef_id`, `z_score`, `raw_bm25`, `n_contributing_words`, and `name`. These are the rows in the `chunk_reefs` table.

### Document Analysis: `analyze()`

Lagoon's `ReefScorer.analyze(text, *, sensitivity=1.0, smooth_window=2, min_chunk_sentences=2, max_chunk_sentences=30)` segments a document by topic shifts. Internally it scores each sentence to get z-score vectors and per-sentence `TopicResult` objects in a single pass, computes cosine similarity between consecutive smoothed vectors, detects boundaries at similarity valleys, and enforces chunk size constraints (splitting oversized segments at weakest internal similarities, merging undersized segments with predecessors). The vector math is internal to lagoon's analysis — shoal never stores or queries vectors.

Returns a `DocumentAnalysis` with:
- `segments: list[TopicSegment]` — each containing `sentences`, `start_idx`, `end_idx`, `topic` (a `TopicResult`), and `sentence_results` (a `list[TopicResult]` with per-sentence scoring data)
- `boundaries: list[int]` — sentence indices where topic shifts occur
- `n_sentences`, `n_segments`

Each `TopicSegment.sentence_results` provides per-sentence `TopicResult` objects including `matched_word_ids`, `unknown_words`, `top_reefs`, and `coverage`. This is the primary data source for Pass 1 vocabulary discovery — shoal reads `sentence_results` directly rather than re-scoring sentences individually.

Chunk size enforcement is handled entirely by lagoon's `analyze()`:
- `min_chunk_sentences` (default 2): Merge undersized segments with predecessors
- `max_chunk_sentences` (default 30): Split oversized segments at weakest internal similarity boundaries
- Set `max_chunk_sentences=0` to disable maximum size enforcement

### Vocabulary Extension at Startup

At startup, shoal extends the base lagoon scorer with custom vocabulary using lagoon's public API:

1. **Load base scorer:** `scorer = lagoon.load()` → `ReefScorer` with ~147K words, ~64K compounds, 207 reefs
2. **Load custom vocabulary from SQLite:**
   ```sql
   SELECT word, word_id, specificity, idf_q FROM custom_words;
   SELECT word_id, reef_id, bm25_q, association_strength FROM custom_word_reefs;
   ```
3. **Inject each custom word via `scorer.add_custom_word()`:**
   ```python
   for word_row in custom_words:
       reef_assocs = [(r.reef_id, r.association_strength) for r in word_row.reefs]
       scorer.add_custom_word(
           word_row.word,
           reef_associations=reef_assocs,
           specificity=word_row.specificity,
       )
   ```
   Each call normalizes the word, computes BM25 scores internally via `compute_custom_word_scores()`, allocates a `word_id`, and injects the word into the scorer's lookup structures. Word IDs are assigned sequentially starting after lagoon's last base word_id.
4. **Rebuild Aho-Corasick automaton for custom compounds:**
   ```python
   custom_compounds = [
       (word, word_id) for word, word_id, is_compound in custom_words
       if is_compound
   ]
   scorer.rebuild_compounds(custom_compounds)
   ```
   This additively merges custom compounds with lagoon's ~64K base compounds and rebuilds the automaton.
5. **Result:** The scorer operates on the superset vocabulary. `score()`, `analyze()`, and all downstream code benefit transparently. No special scoring paths.

### Additional Lagoon API for Vocabulary Extension

Lagoon exposes several methods specifically designed for shoal's vocabulary extension workflow:

| Method | Purpose |
|--------|---------|
| `scorer.filter_unknown(words)` | Batch-filter a list of words, returning only those unknown to lagoon (with stop words excluded). Used in Pass 1 to efficiently identify vocabulary gaps. |
| `scorer.compute_custom_word_scores(n_associated_reefs, associations)` | Compute quantized IDF (u8) and BM25 (u16) scores for a custom word given its reef associations. Lagoon owns the BM25 formula — shoal never computes BM25 directly. |
| `scorer.add_custom_word(word, reef_associations, specificity=2)` | Full injection pipeline: normalize, hash (FNV-1a), validate, compute scores, allocate word_id, inject into scorer. Returns a `WordInfo`. |
| `scorer.rebuild_compounds(additional_compounds)` | Additively merge custom compounds into the Aho-Corasick automaton alongside base compounds. |
| `scorer.next_word_id` | Property returning the next available word_id (current length of the word_reefs list). |
| `scorer.lookup_word(word)` | Look up a word (case-insensitive) and return its `WordInfo` or `None`. |
| `lagoon.STOP_WORDS` | `frozenset[str]` of ~130 English stop words. Used by `filter_unknown()` internally, also available for shoal's own tokenization logic. |

### Why Not Vectors?

Lagoon's `score_raw()` returns a 207-element z-score vector, and lagoon uses these vectors internally for topic-shift detection (cosine similarity between consecutive sentences in `analyze()`). But shoal does not store or query these vectors. Here's why:

1. **The structured output is the point.** Lagoon's `score()` already extracts the meaningful signal: which reefs activated, how strongly, and how confident the result is. Storing the full 207-dim vector and doing cosine similarity at query time would throw away this structure and treat lagoon's output as an opaque embedding — exactly what lagoon is designed to avoid.

2. **SQL is the right query language.** "Find chunks about botanical classification with confidence > 2.0 tagged as biology" is a SQL WHERE clause, not a nearest-neighbor search. The structured reef data enables precise, interpretable, composable queries.

3. **No extension dependencies.** Standard SQLite, standard SQL. No sqlite-vec, no FAISS, no ANN index. The retrieval engine is as portable and simple as the storage.

### Reef Hierarchy Reference

Lagoon organizes 207 reefs into a three-level hierarchy:
- **4 archipelagos**: natural sciences & taxonomy, physical world & materiality, abstract processes & systems, social order & assessment
- **52 islands**: mid-level semantic communities (e.g., "perception and attributes", "behavior and emotional states")
- **207 reefs**: finest-grained semantic clusters (e.g., "archaic literary terms", "neural and structural", "coastal landscapes and frontiers")

See lagoon README section 2 for full concept definitions.

---

## 3. Vocabulary Extension (Phase 2 — not yet implemented)

> The schema tables (`custom_words`, `custom_word_reefs`, `word_observations`) are created in Phase 1 but remain empty. The `_vocab.py` module is a stub that returns 0. Phase 2 will implement the full two-pass pipeline described below.

### Problem

Lagoon's vocabulary is frozen at build time: ~147K base words (plus ~27K Snowball stemmer mappings) derived from WordNet. Named entities, domain jargon, neologisms, and other out-of-vocabulary words are reported in `TopicResult.unknown_words` but do not contribute to BM25 scoring. For general-purpose text this is acceptable — coverage is typically 70-85%. But for domain-specific corpora (medical literature, legal documents, technical manuals), coverage drops and significant signal is lost.

The vocabulary extension system learns reef associations for unknown words from their document context, stores them in the database, and merges them into the scorer at startup. This is purely structured SQL — no vectors, no embeddings, no model inference.

### Two-Pass Ingestion Pipeline

Vocabulary extension requires two passes over each document:

**Pass 1 — Discovery:**
- Run `scorer.analyze(text, min_chunk_sentences=2, max_chunk_sentences=30)` for preliminary chunking and per-sentence scoring in a single pass. Each `TopicSegment` in the result contains `sentence_results` — a list of per-sentence `TopicResult` objects with `unknown_words`, `matched_word_ids`, `top_reefs`, and `coverage`.
- Extract unknown words from each sentence's `TopicResult.unknown_words`. Stop words are already excluded by lagoon. For batch identification across the full document, use `scorer.filter_unknown(all_unique_words)`.
- Capture reef context from surrounding known words at sentence-level and chunk-level granularity, weighted by confidence (see [Context Collection](#context-collection-algorithm) below). Sentence-level context comes directly from `sentence_results[i]`; chunk-level context comes from `segment.topic`.
- Do NOT store final chunk scores — this pass is for discovery only
- Store word observations in the `word_observations` table

**Vocabulary Build:**
- Aggregate statistics across all occurrences of each unknown word
- Apply minimum occurrence thresholds (see [Minimum Occurrence Threshold](#minimum-occurrence-threshold))
- Detect multi-word compounds via co-occurrence statistics (see [Multi-Word Compound Detection](#multi-word-compound-detection))
- Compute reef associations: determine associated reefs and normalize association strengths (see [Statistics Aggregation](#statistics-aggregation) and [Synthetic BM25 Score Computation](#synthetic-bm25-score-computation))
- Inject each word via `scorer.add_custom_word(word, reef_associations=[(reef_id, strength), ...])` — lagoon internally calls `compute_custom_word_scores()` to produce quantized IDF/BM25 values using its own BM25 formula. Shoal never computes BM25 directly.
- Persist the resulting `WordInfo` (word_id, idf_q, specificity) and reef entries to `custom_words` and `custom_word_reefs` tables for startup reload
- Rebuild the Aho-Corasick automaton with `scorer.rebuild_compounds(custom_compounds)` to additively merge custom compounds with lagoon's base compounds

**Pass 2 — Scoring:**
- Re-run `scorer.analyze(text, min_chunk_sentences=2, max_chunk_sentences=30)` with the extended vocabulary (lagoon base + custom words). Lagoon handles chunk size enforcement internally — oversized segments are split at weakest internal similarity boundaries, undersized segments are merged with predecessors.
- Topic-shift detection now benefits from the extended vocabulary — previously invisible words now contribute to per-sentence z-score vectors and `matched_word_ids`, improving boundary placement
- Each `TopicSegment` in the result includes `sentence_results` with per-sentence `TopicResult` objects and a `topic` (segment-level `TopicResult`). The segment-level `topic` is the primary scoring data stored per-chunk.
- Store final chunk data (text, reef scores, metadata) in `chunks`, `chunk_reefs`, and `chunk_islands`

### Context Collection Algorithm

For each unknown word W in sentence S within segment C (from Pass 1's `analyze()` results):

1. **Sentence-level context:** Read the per-sentence `TopicResult` from `segment.sentence_results[i]`. Extract the top reefs, z-scores, and `matched_word_ids`. No additional scoring call needed — `analyze()` provides this data.

2. **Chunk-level context:** Read the segment-level `TopicResult` from `segment.topic` (broader context). Extract its top reefs and z-scores.

3. **Confidence-weighted blending:** If the sentence-level `TopicResult` has high confidence (strong convergence on specific reefs), weight sentence context heavily. If sentence confidence is low (short sentence, generic words), rely more on chunk context.
   - High sentence confidence: sentence weight ~0.7, chunk weight ~0.3
   - Low sentence confidence: sentence weight ~0.3, chunk weight ~0.7

4. **Store as word observation:** `(word, document_id, sentence_idx, weighted_reef_profile)` where the reef profile is a JSON array of `{reef_id, z_score, weight}` entries representing the blended context.

### Statistics Aggregation

For each unknown word W with enough occurrences (see threshold below):

1. **Mean weighted z-score per reef:** Across all observations of W, compute the mean weighted z-score for each reef. Reefs that consistently appear in W's context are its associated reefs.

2. **Associated reef identification:** A reef is associated with W if the mean weighted z-score exceeds a threshold (indicating consistent co-occurrence, not noise).

3. **Synthetic IDF:** Computed from the number of associated reefs, using the same BM25 IDF formula as lagoon:

   ```
   IDF = ln((207 - n_reefs + 0.5) / (n_reefs + 0.5) + 1)
   ```

   Where `n_reefs` is the number of reefs associated with W. Words associated with fewer reefs get higher IDF (more discriminating), matching lagoon's semantics. Quantized to u8: `idf_q = round(IDF * 51)`.

4. **Specificity sigma band:** Derived from the reef spread. Words associated with very few reefs (1-3) get specificity +2 (highly specific). Words spread across many reefs get specificity -1 or -2 (generic). Same i8 scale as lagoon's `WordInfo.specificity`.

### Minimum Occurrence Threshold

Configurable, with two-tier defaults based on document length:

| Document Length | Default Threshold | Rationale |
|----------------|-------------------|-----------|
| < 500 words | `min_occurrences_short` = 2 | Short documents have fewer chances for a word to appear |
| >= 500 words | `min_occurrences_long` = 3 | Longer documents should provide more evidence |

The threshold is NOT proportional to document size. A 100K-word book still uses threshold 3, not 95. The purpose is to filter out typos and one-off mentions that don't carry enough context to learn reliable reef associations, not to scale with corpus size.

### Multi-Word Compound Detection

Lagoon's base vocabulary includes ~64K compound words (e.g., "heart attack", "black hole") matched via Aho-Corasick. The vocabulary extension system detects new compounds among unknown tokens:

1. **Co-occurrence tracking:** For each pair (and triple) of adjacent unknown tokens, track how often they appear together vs. separately across all documents.

2. **Significance testing:** Promote a bigram/trigram to compound when co-occurrence frequency is significantly above chance. Options:
   - Pointwise Mutual Information (PMI) threshold
   - Chi-squared test
   - Simple frequency ratio (co-occurrence / expected-if-independent)

3. **Compound registration:** Each promoted compound gets:
   - Its own `word_hash` (FNV-1a of the space-joined, lowercased string, matching lagoon's normalization)
   - Its own `word_id` (next available after singles)
   - Its own reef entries in `custom_word_reefs` (computed from the compound's context, not just the union of its parts)

4. **Automaton rebuild:** After all compounds are registered, rebuild the Aho-Corasick automaton with lagoon's base compounds + custom compounds. This ensures Pass 2 scoring matches compounds correctly.

### Synthetic BM25 Score Computation

Lagoon owns the BM25 formula. Shoal computes reef association strengths (normalized 0.0-1.0) and passes them to `scorer.compute_custom_word_scores()` or `scorer.add_custom_word()`, which handle BM25 quantization internally. The formula below documents what lagoon computes — shoal does not implement this directly.

For each custom word–reef association, lagoon computes a BM25 score compatible with its quantized format:

1. **Normalize association strength:**

   ```
   strength = mean_context_z[reef] / max(mean_context_z)
   ```

   This maps the strongest associated reef to 1.0, with others proportionally weaker.

2. **Use as tf proxy:** In lagoon's BM25 formula, `tf` is `n_dims / reef_total_dims` (depth-weighted term frequency). For custom words, use the normalized association strength as a synthetic tf proxy.

3. **Apply the BM25 formula:**

   ```
   bm25_q = round(IDF * (tf_proxy * (k1 + 1)) / (tf_proxy + k1 * (1 - b + b * reef_n_words[R] / avg_reef_words)) * 8192)
   ```

   Where:
   - `IDF` = synthetic IDF from step 3 of statistics aggregation (decoded: `idf_q / 51.0`)
   - `k1 = 1.2`, `b = 0.75` (lagoon's BM25 parameters)
   - `reef_n_words[R]` = word count for reef R (from lagoon's reef metadata)
   - `avg_reef_words` = ~5,282 (lagoon's average reef size)
   - `8192` = lagoon's BM25 quantization scale factor (u16)

This produces quantized u16 BM25 scores directly compatible with lagoon's accumulation in Phase 3 of the scoring pipeline. Custom words participate in `scores[reef_id] += bm25_q / 8192.0` identically to base vocabulary words.

### Startup Injection

See [Section 2 — Vocabulary Extension at Startup](#vocabulary-extension-at-startup) for the full startup injection flow using lagoon's public API (`add_custom_word()`, `rebuild_compounds()`). The startup flow is the same regardless of whether custom words were built during this session's ingestion or loaded from a previous session's database.

---

## 4. Document Ingestion Pipeline

### Current Implementation (Phase 1 — Single Pass)

Phase 1 uses a single-pass pipeline: parse the document into sections, run `scorer.analyze()` once per section, and store the resulting chunks. No vocabulary extension.

**Markdown (.md)**
1. Parse structure: extract header hierarchy (h1-h6), clean Wikipedia artifacts (`[edit]` lines, `Main article:` lines)
2. For each section (text between headers):
   - Run `scorer.analyze(section_text)` for topic-shift detection and chunk sizing
   - Compute character offsets by walking through the section text with a cursor
   - Store each segment as a chunk with the header hierarchy as metadata

**Plain text (.txt)**
1. Return the full text as a single section
2. Run `scorer.analyze(text)` for topic-shift detection
3. Store each segment as a chunk

### Full Pipeline (Phase 2 — Two-Pass with Vocabulary Extension)

Phase 2 will add a discovery pass before the scoring pass:

**Markdown (.md)**
1. Parse structure: extract header hierarchy (h1-h6)
2. For each section (text between headers):
   - **Pass 1:** Run `scorer.analyze(section_text, min_chunk_sentences=2, max_chunk_sentences=30)` for preliminary topic-shift detection within the section. Lagoon handles chunk size enforcement internally. Collect unknown words from each segment's `sentence_results[i].unknown_words`. Capture sentence-level reef context from `sentence_results[i]` and chunk-level from `segment.topic`. Store word observations.
   - **Vocabulary build:** Aggregate word observations across the section (or across the full document — see below). Apply occurrence thresholds. Detect compounds. Inject via `scorer.add_custom_word()`. Persist to `custom_words` and `custom_word_reefs`. Rebuild automaton via `scorer.rebuild_compounds()`.
   - **Pass 2:** Re-run `scorer.analyze()` with extended vocabulary for final topic-shift detection and chunk sizing. Carry the header hierarchy as chunk metadata (e.g., `["# Introduction", "## Background"]`). Each segment's `topic` provides the structured reef metadata for storage.

**Plain text (.txt)**
1. **Pass 1:** Run `scorer.analyze(text, min_chunk_sentences=2, max_chunk_sentences=30)` for preliminary topic-shift detection and per-sentence scoring. Collect unknown words from `sentence_results` across all segments. Capture reef context. Store word observations.
2. **Vocabulary build:** Aggregate observations. Apply thresholds. Detect compounds. Inject via `scorer.add_custom_word()`. Persist to custom vocabulary tables. Rebuild automaton via `scorer.rebuild_compounds()`.
3. **Pass 2:** Re-run `scorer.analyze()` with extended vocabulary for final topic-shift detection and chunk sizing. Each segment's `topic` provides structured reef metadata for storage.

**Note:** Vocabulary build can operate per-document or across a batch of documents. Batch mode provides more observations per unknown word, improving reef association quality. The occurrence threshold (see [Section 3](#3-vocabulary-extension-phase-2--not-yet-implemented)) ensures sufficient evidence regardless of mode.

### Chunking Parameters

All chunking parameters are passed directly to `scorer.analyze()` — lagoon handles boundary detection, size enforcement, and segment construction internally.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitivity` | 1.0 | Boundary detection threshold. Lower = more boundaries, higher = fewer. |
| `smooth_window` | 2 | Sliding window size for z-score vector smoothing. |
| `min_chunk_sentences` | 2 | Minimum sentences per chunk. Undersized segments are merged with predecessors. |
| `max_chunk_sentences` | 30 | Maximum sentences per chunk. Oversized segments are split at weakest internal similarity boundaries. Set to 0 to disable. |

### Per-Chunk Storage

Each chunk produces the following stored data, all derived from `scorer.score()`:

**Scalar fields (columns on `chunks` table):**

| Field | Type | Source |
|-------|------|--------|
| text | string | The chunk's raw text |
| confidence | f32 | `TopicResult.confidence` — z-score gap between #1 and #2 reef |
| coverage | f32 | `TopicResult.coverage` — fraction of words matched in lagoon's dictionary |
| matched_words | int | `TopicResult.matched_words` |
| top_reef_id | int | `TopicResult.top_reefs[0].reef_id` |
| top_reef_name | string | `TopicResult.top_reefs[0].name` |
| arch_score_0..3 | f32 | `TopicResult.arch_scores[0..3]` — one column per archipelago |
| start_char | int | Character offset of chunk start in the source document |
| end_char | int | Character offset of chunk end in the source document |
| sentence_count | int | Number of sentences in the chunk |
| metadata | JSON | Format-specific metadata (e.g., markdown header hierarchy) |

**Normalized reef scores (rows in `chunk_reefs` table):**

For each of the top-K reefs in `TopicResult.top_reefs`:

| Field | Type | Source |
|-------|------|--------|
| chunk_id | int | Foreign key to `chunks` |
| reef_id | int | `ScoredReef.reef_id` |
| reef_name | string | `ScoredReef.name` |
| z_score | f32 | `ScoredReef.z_score` — background-subtracted score |
| raw_bm25 | f32 | `ScoredReef.raw_bm25` — pre-subtraction score |
| n_contributing_words | int | `ScoredReef.n_contributing_words` |
| rank | int | Position in the top-K list (0 = top reef) |

**Normalized island scores (rows in `chunk_islands` table):**

For each island in `TopicResult.top_islands`:

| Field | Type | Source |
|-------|------|--------|
| chunk_id | int | Foreign key to `chunks` |
| island_id | int | `ScoredIsland.island_id` |
| island_name | string | `ScoredIsland.name` |
| aggregate_z | f32 | `ScoredIsland.aggregate_z` |
| n_contributing_reefs | int | `ScoredIsland.n_contributing_reefs` |

---

## 5. SQLite Schema

### `documents` table

```sql
CREATE TABLE documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    source_path   TEXT,
    content_hash  TEXT NOT NULL,          -- SHA-256 of raw content for dedup
    format        TEXT NOT NULL,          -- 'markdown' or 'plaintext'
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    metadata      TEXT DEFAULT '{}'       -- JSON for extensible fields
);
```

### `document_tags` table

```sql
CREATE TABLE document_tags (
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag           TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);

CREATE INDEX idx_document_tags_tag ON document_tags(tag);
```

Flat tag model — each document has zero or more string tags. No tag hierarchy, no tag table. Tags are denormalized for query simplicity: filter by tag with a simple join or subquery.

### `chunks` table

```sql
CREATE TABLE chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,       -- 0-indexed position within document
    text            TEXT NOT NULL,
    start_char      INTEGER NOT NULL,       -- character offset in source document
    end_char        INTEGER NOT NULL,
    sentence_count  INTEGER NOT NULL,
    confidence      REAL NOT NULL,          -- z-score gap: #1 - #2 reef
    coverage        REAL NOT NULL,          -- matched_words / total_words
    matched_words   INTEGER NOT NULL,
    top_reef_id     INTEGER NOT NULL,       -- reef_id of highest-scoring reef
    top_reef_name   TEXT NOT NULL,          -- human-readable name of top reef
    arch_score_0    REAL NOT NULL,          -- natural sciences & taxonomy
    arch_score_1    REAL NOT NULL,          -- physical world & materiality
    arch_score_2    REAL NOT NULL,          -- abstract processes & systems
    arch_score_3    REAL NOT NULL,          -- social order & assessment
    metadata        TEXT DEFAULT '{}',      -- JSON (header hierarchy, etc.)
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_top_reef_id ON chunks(top_reef_id);
CREATE INDEX idx_chunks_confidence ON chunks(confidence);
```

### `chunk_reefs` table

```sql
CREATE TABLE chunk_reefs (
    chunk_id              INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    reef_id               INTEGER NOT NULL,       -- 0..206
    reef_name             TEXT NOT NULL,
    z_score               REAL NOT NULL,          -- background-subtracted
    raw_bm25              REAL NOT NULL,          -- pre-subtraction
    n_contributing_words  INTEGER NOT NULL,
    rank                  INTEGER NOT NULL,       -- 0 = top reef
    PRIMARY KEY (chunk_id, reef_id)
);

CREATE INDEX idx_chunk_reefs_reef_id ON chunk_reefs(reef_id);
CREATE INDEX idx_chunk_reefs_z_score ON chunk_reefs(z_score);
```

This is the core retrieval table. Each chunk has ~10 rows (top-K reefs from lagoon's `score()`). Retrieval joins on `reef_id` to match query reefs against chunk reefs.

### `chunk_islands` table

```sql
CREATE TABLE chunk_islands (
    chunk_id              INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    island_id             INTEGER NOT NULL,       -- 0..51
    island_name           TEXT NOT NULL,
    aggregate_z           REAL NOT NULL,
    n_contributing_reefs  INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, island_id)
);

CREATE INDEX idx_chunk_islands_island_id ON chunk_islands(island_id);
```

Island-level aggregation for coarser-grained queries. Optional for V1 but cheap to populate since `score()` already returns it.

### `custom_words` table

```sql
CREATE TABLE custom_words (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    word            TEXT NOT NULL UNIQUE,
    word_hash       INTEGER NOT NULL UNIQUE,    -- FNV-1a u64 hash
    word_id         INTEGER NOT NULL UNIQUE,    -- index into extended _word_reefs
    is_compound     INTEGER NOT NULL DEFAULT 0, -- 1 if multi-word compound
    specificity     INTEGER NOT NULL,           -- i8 sigma band: +2...-2
    idf_q           INTEGER NOT NULL,           -- u8 quantized IDF (scale 51)
    n_occurrences   INTEGER NOT NULL,           -- total observations used
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
```

Custom words learned from document context. `word_hash` and `word_id` use the same encoding as lagoon's base vocabulary (FNV-1a u64 and sequential u32 respectively). IDF is quantized to u8 with scale factor 51, matching lagoon's scheme.

### `custom_word_reefs` table

```sql
CREATE TABLE custom_word_reefs (
    word_id     INTEGER NOT NULL REFERENCES custom_words(word_id) ON DELETE CASCADE,
    reef_id     INTEGER NOT NULL,               -- 0..206
    bm25_q      INTEGER NOT NULL,               -- u16 quantized BM25 (scale 8192)
    association_strength REAL NOT NULL,          -- 0.0-1.0 normalized strength
    PRIMARY KEY (word_id, reef_id)
);
```

Reef associations for custom words. `bm25_q` is quantized to u16 with scale factor 8192, matching lagoon's scheme. At startup, these entries are re-injected into the scorer via `add_custom_word()`, which allocates word IDs and reef entries internally.

### `word_observations` table

```sql
CREATE TABLE word_observations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    word          TEXT NOT NULL,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sentence_idx  INTEGER NOT NULL,
    context_reefs TEXT NOT NULL    -- JSON: [{"reef_id": int, "z_score": float, "weight": float}, ...]
);

CREATE INDEX idx_word_observations_word ON word_observations(word);
```

Raw observations from Pass 1 of ingestion. Each row records one occurrence of an unknown word with its confidence-weighted reef context. Used to aggregate statistics during the vocabulary build phase.

### Row Counts

For a corpus of N chunks with top_k=10:
- `chunks`: N rows
- `chunk_reefs`: ~10N rows (10 reefs per chunk)
- `chunk_islands`: ~5N rows (variable, typically 3-7 islands per chunk)
- `custom_words`: typically 100s-1000s per corpus (domain jargon, names)
- `custom_word_reefs`: ~10-20 per custom word (associated reefs)
- `word_observations`: temporary — potentially large during ingestion, can be pruned after vocabulary build

For the test corpus (~500-1000 chunks), this is well within SQLite's comfort zone.

---

## 6. Retrieval Pipeline

```
Query text
    │
    ▼
Score with lagoon: scorer.score(query) → TopicResult
    │
    ▼
Extract query's top reefs (reef_ids + z_scores)
    │
    ▼
SQL: find chunks sharing those reefs, weighted by z_score overlap
    │
    ▼
Apply filters: tags, confidence threshold, archipelago, reef constraints
    │
    ▼
Join with documents table for citation metadata
    │
    ▼
Return ranked results
```

### Query Flow

1. **Score the query:** `scorer.score(query_text)` → `TopicResult` with top reefs, confidence, coverage
2. **Reef-based retrieval:** Find chunks whose scored reefs overlap with the query's scored reefs
3. **Filter:** Apply optional constraints (tags, min confidence, specific reefs/islands/archipelagos)
4. **Rank:** Order by match quality (weighted reef overlap)
5. **Hydrate:** Join with `documents` and `document_tags` for full citation metadata

### Core Retrieval Query

The fundamental retrieval operation: find chunks that share the query's top reefs, weighted by the product of query z-score and chunk z-score for each shared reef.

```sql
-- Given: query scored with top reefs [(reef_42, 5.12), (reef_17, 3.87), (reef_103, 2.91)]

SELECT
    c.id,
    c.text,
    c.document_id,
    d.title AS document_title,
    c.chunk_index,
    c.start_char,
    c.end_char,
    c.confidence,
    c.coverage,
    c.top_reef_name,
    SUM(cr.z_score * qr.query_z) AS match_score,
    COUNT(cr.reef_id) AS n_shared_reefs
FROM chunks c
JOIN chunk_reefs cr ON c.id = cr.chunk_id
JOIN (
    -- Query reef scores, passed as parameters
    VALUES (42, 5.12), (17, 3.87), (103, 2.91)
) AS qr(reef_id, query_z) ON cr.reef_id = qr.reef_id
JOIN documents d ON c.document_id = d.id
GROUP BY c.id
ORDER BY match_score DESC
LIMIT ?;  -- top_k
```

This query:
- Joins `chunk_reefs` against the query's reef scores on `reef_id`
- Weights each shared reef by `chunk_z_score * query_z_score` — a chunk that strongly matches the query's strong reefs ranks higher
- Groups by chunk to aggregate across shared reefs
- Returns `match_score` (sum of weighted overlaps) and `n_shared_reefs` (how many reefs overlap)

### Filtered Retrieval

Filters compose naturally with SQL:

```sql
-- Add tag filter
JOIN document_tags dt ON d.id = dt.document_id AND dt.tag = 'biology'

-- Add confidence threshold
WHERE c.confidence > 1.5

-- Add archipelago filter (e.g., natural sciences only)
WHERE c.arch_score_0 > c.arch_score_1
  AND c.arch_score_0 > c.arch_score_2
  AND c.arch_score_0 > c.arch_score_3

-- Add specific reef filter
WHERE c.top_reef_id = 42
```

### Alternative Retrieval Strategies

**Top-reef match:** Simplest approach — find chunks whose `top_reef_id` matches the query's top reef. Fast (indexed), but misses multi-reef overlap.

```sql
SELECT c.*, d.title
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.top_reef_id = ?  -- query's top reef_id
ORDER BY c.confidence DESC
LIMIT ?;
```

**Island-level retrieval:** Coarser matching via the `chunk_islands` table. Useful when the query is broad or when reef-level matching produces too few results.

**Archipelago-level filtering:** Use the 4 `arch_score_*` columns to restrict results to a semantic quadrant (e.g., only natural science chunks).

### What "Match Score" Means

The `match_score = SUM(cr.z_score * qr.query_z)` is interpretable:
- It rewards chunks that strongly activate the same reefs the query strongly activates
- A chunk matching the query's top reef with high z-scores will score much higher than one matching a weak query reef weakly
- `n_shared_reefs` tells you how broad the overlap is — a chunk sharing 5 reefs is a more robust match than one sharing 1

This is fundamentally different from cosine similarity on opaque vectors. You can inspect *which reefs* contributed to the match score and explain the result in human-readable terms.

---

## 7. REST API — shoald (Phase 3 — not yet implemented)

> Phase 3 will wrap the `Engine` class in a FastAPI HTTP daemon. The library does all the work — `shoald` is a thin HTTP layer.

### Document Management

**POST /api/v1/documents** — Ingest a document

```json
{
    "title": "On the Origin of Species - Chapter 1",
    "content": "When we look to the individuals of the same variety...",
    "format": "plaintext",
    "tags": ["biology", "darwin", "gutenberg"],
    "metadata": {}
}
```

Response `201 Created`:
```json
{
    "id": 42,
    "title": "On the Origin of Species - Chapter 1",
    "n_chunks": 17,
    "tags": ["biology", "darwin", "gutenberg"]
}
```

**GET /api/v1/documents** — List documents

Query parameters: `tag` (filter by tag), `limit`, `offset`

Response `200 OK`:
```json
{
    "documents": [
        {
            "id": 42,
            "title": "On the Origin of Species - Chapter 1",
            "format": "plaintext",
            "n_chunks": 17,
            "tags": ["biology", "darwin", "gutenberg"],
            "created_at": "2026-02-13T12:00:00.000Z"
        }
    ],
    "total": 1
}
```

**GET /api/v1/documents/{id}** — Get document details (including chunk summaries)

**DELETE /api/v1/documents/{id}** — Delete document and its chunks

**PUT /api/v1/documents/{id}/tags** — Replace document tags

```json
{
    "tags": ["biology", "darwin", "gutenberg", "evolution"]
}
```

### Search

**POST /api/v1/search** — Search for relevant chunks

```json
{
    "query": "natural selection and variation in domestic animals",
    "top_k": 10,
    "filters": {
        "tags": ["biology"],
        "min_confidence": 1.5,
        "reef_ids": null,
        "archipelago": null
    },
    "include_scores": true
}
```

Response `200 OK`:
```json
{
    "results": [
        {
            "text": "When we look to the individuals of the same variety or sub-variety...",
            "match_score": 19.82,
            "n_shared_reefs": 4,
            "document_id": 42,
            "document_title": "On the Origin of Species - Chapter 1",
            "chunk_index": 3,
            "start_char": 1847,
            "end_char": 2391,
            "confidence": 3.21,
            "coverage": 0.82,
            "top_reef_name": "botanical classification systems",
            "tags": ["biology", "darwin", "gutenberg"],
            "shared_reefs": [
                {"reef_name": "botanical classification systems", "chunk_z": 5.12, "query_z": 4.87},
                {"reef_name": "animal body structures", "chunk_z": 3.87, "query_z": 3.41}
            ]
        }
    ],
    "query_info": {
        "top_reef": "botanical classification systems",
        "confidence": 2.84,
        "coverage": 0.78,
        "top_reefs": [
            {"reef_name": "botanical classification systems", "z_score": 4.87},
            {"reef_name": "animal body structures", "z_score": 3.41},
            {"reef_name": "selective breeding processes", "z_score": 2.95}
        ]
    }
}
```

**Key design choices:**
- `match_score` is the weighted reef overlap (interpretable, not an opaque distance)
- `n_shared_reefs` tells the caller how broad the match is
- `shared_reefs` (present when `include_scores: true`) shows exactly *which reefs* the query and chunk share, with both z-scores — full interpretability
- `query_info` returns the query's own reef scores, so the caller understands the query's semantic profile

### Status

**GET /api/v1/status** — System status

```json
{
    "status": "ok",
    "n_documents": 27,
    "n_chunks": 412,
    "lagoon_version": "0.1.0",
    "lagoon_n_reefs": 207,
    "db_path": "/data/shoal.db"
}
```

**GET /health** — Health check (returns `200 OK` with `{"status": "ok"}`)

### API Documentation

FastAPI auto-generates OpenAPI (Swagger) documentation at `/docs` and a ReDoc view at `/redoc`.

---

## 8. RAG Standards Discussion

No single formal RAG API standard exists. The closest reference points:

**OpenAI Chat Completions** is the de facto standard for generation endpoints. Not applicable — shoal is retrieval-only. Shoal's output is designed to be *consumed by* a Chat Completions call (the retrieved chunks become context in the prompt), but shoal itself does not call any LLM.

**Weaviate, Chroma, Qdrant** are vector databases that store dense embeddings and retrieve via ANN (approximate nearest-neighbor) search. Shoal's API surface follows similar document CRUD + search patterns, but the retrieval mechanism is fundamentally different: structured SQL queries over reef metadata, not cosine similarity over opaque vectors. This is a deliberate design choice, not a limitation.

**OpenAPI/Swagger** for documentation. FastAPI provides this out of the box at `/docs`. Clients can generate SDK code from the OpenAPI spec.

**Shoal's differentiator:** Every result explains *why* it matched. The `shared_reefs` field shows which reefs the query and chunk have in common, with both z-scores. A downstream consumer (or a human debugging retrieval quality) can read "this chunk matched because both the query and the chunk score highly on botanical classification systems (query z=4.87, chunk z=5.12)" — not just "cosine similarity = 0.847". This interpretability is a direct consequence of treating lagoon's output as structured data rather than flattening it into a vector.

---

## 9. Technology Choices & Rust Portability

### Python Stack (V1)

| Component | Library | Notes |
|-----------|---------|-------|
| HTTP server | FastAPI + Uvicorn | Async, auto-generates OpenAPI |
| Database | sqlite3 (stdlib) | No external DB dependency, no extensions |
| Scoring | lagoon | Loaded once at startup, extended with custom vocabulary |
| Validation | Pydantic | Request/response models |

### Rust Stack (future)

| Component | Crate | Notes |
|-----------|-------|-------|
| HTTP server | Axum | Tokio-based, similar to FastAPI |
| Database | rusqlite | SQLite bindings |
| Scoring | lagoon-rs | Port of lagoon (planned) |
| Validation | serde | Serialization/deserialization |

### Portability Constraints

These constraints ensure the Python prototype can be rewritten in Rust without architectural changes:

- **No pickle.** All storage uses SQLite + standard SQL types. JSON for extensible metadata.
- **No Python-specific serialization.** All data is standard SQL types (INTEGER, REAL, TEXT). No binary blobs, no format-specific encoding.
- **No ML model dependencies.** Lagoon's reef scores require no GPU, no PyTorch, no ONNX. Scoring is pure BM25 accumulation.
- **Standard SQL.** All queries are portable SQLite SQL. No ORM magic, no Python-specific query builders. The core retrieval query uses standard JOINs, GROUP BY, and ORDER BY.
- **Pydantic models map to serde structs.** Field names, types, and validation rules translate directly.
- **No extensions.** Standard SQLite3 only — no sqlite-vec, no FTS5, no custom C extensions. The Rust port uses the same SQL against rusqlite.
- **Quantization compatibility.** Custom word storage uses the same quantization scheme as lagoon's base vocabulary: u8 IDF (scale factor 51) and u16 BM25 (scale factor 8192). The `custom_words` and `custom_word_reefs` tables use INTEGER columns for these quantized values — same SQL schema works identically in Rust with rusqlite. No floating-point round-trip issues across languages.

---

## 10. V1 Scope

### In Scope

- **Formats:** Markdown (.md) and plain text (.txt)
- **Storage:** Flat document store with string tags, structured reef metadata in normalized SQL tables
- **Retrieval:** SQL-based reef overlap matching — join query reefs against chunk reefs
- **API:** REST endpoints for document CRUD, search, status
- **Chunking:** Lagoon topic-shift detection with configurable sensitivity and size limits
- **Vocabulary extension:** Two-pass ingestion with context-based reef learning for unknown words
- **No auth:** V1 runs on localhost or behind a reverse proxy. No API keys, no user accounts.

### Deferred to Future Versions

| Feature | Rationale |
|---------|-----------|
| PDF/DOCX ingestion | Requires external parsing libraries; focus V1 on text-native formats |
| Collections / namespaces | Flat store is sufficient for the test corpus; add when multi-tenant use cases arise |
| Authentication | V1 is single-user, local. Add API key auth when deploying as a shared service |
| Traditional dense embeddings | Evaluate adding sentence-transformers as a complementary retrieval path later |
| Hybrid search (FTS5 + reef) | Full-text keyword search via FTS5 is a natural complement; defer to V2 |
| Rate limiting | Not needed for single-user local deployment |

---

## 11. Test Corpus

The test corpus is designed to exercise lagoon's 4 archipelagos with cross-domain edge cases. All sources are public domain or freely licensed.

### Wikipedia Articles (~25 articles)

**Natural sciences and taxonomy (Archipelago 0):**

| Article | Rationale |
|---------|-----------|
| Photosynthesis | Core biology, high reef specificity expected |
| Tyrannosaurus | Paleontology, should activate "dinosaurs and fossils" reef |
| Corvidae (crow family) | Animal taxonomy, tests zoological classification reefs |
| Penicillin | Medical/biochemistry crossover |
| Mitochondrion | Cell biology, highly specific domain vocabulary |
| Great Barrier Reef | Marine biology + geography crossover |

**Physical world and materiality (Archipelago 1):**

| Article | Rationale |
|---------|-----------|
| Earthquake | Geology, physical processes |
| Jupiter | Astronomy, planetary science |
| Granite | Materials science, mineralogy |
| Atmospheric river | Meteorology, weather phenomena |
| Volcanic eruption | Geology + physical processes |

**Abstract processes and systems (Archipelago 2):**

| Article | Rationale |
|---------|-----------|
| Fourier transform | Mathematics, abstract and technical |
| Game theory | Abstract systems, strategic reasoning |
| Algorithm | Computer science, abstract processes |
| Natural selection | Evolutionary biology, abstract mechanism |
| Supply and demand | Economics, abstract systems |
| Thermodynamics | Physics, abstract laws governing physical systems |

**Social order and assessment (Archipelago 3):**

| Article | Rationale |
|---------|-----------|
| French Revolution | History, political events |
| United Nations | International governance, institutions |
| Supreme Court of the United States | Legal systems, constitutional law |
| Olympic Games | Sports, international culture |
| Renaissance | Cultural history, arts and sciences |

**Cross-domain edge cases:**

| Article | Rationale |
|---------|-----------|
| Coffee | Botany + culture + commerce — spans multiple archipelagos |
| Silk Road | Trade + geography + history — tests multi-domain text |
| DNA | Biology + chemistry + technology — high specificity vocabulary |
| Climate change | Science + policy + society — deliberately cross-domain |

### Project Gutenberg Books (2 books)

| Book | Author | Rationale |
|------|--------|-----------|
| Adventures of Huckleberry Finn | Mark Twain | Literary fiction. Should strongly activate "archaic literary terms" reef. Tests long-document chunking (multi-chapter). Lagoon's canonical test case. |
| On the Origin of Species | Charles Darwin | Scientific prose. Should activate natural science reefs. Tests academic writing style and domain-specific vocabulary. Contrasts strongly with Huck Finn. |

### Corpus Validation

After ingesting the full corpus with `shoal ingest-dir docs/`:

- **28 documents**, **3,463 chunks** total
- Largest documents: Origin of Species (476 chunks), Huckleberry Finn (414 chunks)
- Smallest documents: Atmospheric River (33 chunks), Supply and Demand (34 chunks)

Validation criteria:
- Each archipelago is represented by chunks with high-confidence scores in that domain
- Huck Finn chunks consistently show "archaic literary terms" as top reef
- Origin of Species chunks consistently show natural science reefs (botanical classification, animal body structures)
- Cross-domain articles produce lower confidence, spread across multiple archipelagos
- Retrieval queries for domain-specific topics return relevant chunks ranked above cross-domain noise

---

## 12. Open Questions & Future Directions

### Tuning Parameters
- What are optimal `sensitivity` and `smooth_window` values for each format? May need per-format defaults.
- Default `top_k` for search — 10 is a reasonable starting point but may need tuning.
- Chunk size limits (`min_chunk_sentences`, `max_chunk_sentences`) — need empirical testing on the test corpus.
- How many reefs to store per chunk (top_k in `score()`) — 10 is the default, but storing more improves multi-reef overlap at the cost of more rows.

### Vocabulary Extension
- **Confidence threshold for context weighting:** What sentence-level confidence value separates "high" (sentence-weighted 0.7) from "low" (chunk-weighted 0.7)? Likely needs empirical calibration on the test corpus.
- **Co-occurrence significance threshold for compound detection:** PMI, chi-squared, or simple frequency ratio? Each has trade-offs — PMI favors rare pairs, chi-squared accounts for sample size, simple ratio is easiest to tune. Needs experimentation.
- **Word observation retention:** Should word observations be permanent (for auditing and re-computation) or pruned after vocabulary build? Permanent storage enables re-running the vocabulary build with different thresholds but grows linearly with corpus size.
- **Vocabulary conflicts:** How to handle a custom word that later appears in a lagoon base vocabulary update? Options: prefer base (drop custom entry), prefer custom (keep learned associations), merge (combine base and custom reef profiles). Base-preferred is simplest and safest.
- **Association z-score threshold:** What mean weighted z-score qualifies a reef as "associated" with a custom word? Too low → noise reefs included, too high → too few associations, low IDF.

### Retrieval Quality
- How well does weighted reef overlap (`SUM(cr.z_score * qr.query_z)`) perform as a ranking signal compared to cosine similarity on the full 207-dim vector? Empirical evaluation needed on the test corpus.
- Should `n_shared_reefs` be a factor in the ranking (prefer broader matches), or is pure weighted overlap sufficient?
- For low-confidence queries (generic text that doesn't converge on specific reefs), should retrieval fall back to a different strategy?

### Hybrid Search (V2)
SQLite's FTS5 extension supports full-text search with BM25 ranking. A hybrid approach would combine:
1. Reef overlap matching (semantic matching via lagoon scores)
2. FTS5 keyword matching (exact term matching)
3. Score fusion to produce a final ranking

This would handle cases where exact keyword matches matter (e.g., searching for a specific proper noun that lagoon's vocabulary doesn't contain).

### Traditional Embeddings (V2+)
Adding sentence-transformer or similar dense embedding models as an additional retrieval path. This would provide:
- A complementary signal to lagoon's interpretable reef scores
- Better coverage for domains where lagoon's vocabulary is weak (abstract, technical)
- A comparison baseline for evaluating lagoon's retrieval quality

Trade-off: adds a model dependency (GPU beneficial, larger vectors, slower scoring).

### Collections / Namespaces
Allow grouping documents into collections for multi-tenant or multi-project use. Schema change: add a `collections` table and `collection_id` foreign key on `documents`.

### Security
- API key authentication for shoald
- Per-collection access control
- Rate limiting per API key
- Input size limits to prevent resource exhaustion

### Incremental Re-ingestion
When a document is updated, detect which chunks changed (via content hashing) and only re-score and re-index the modified chunks rather than re-ingesting the entire document.

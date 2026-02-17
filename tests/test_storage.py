"""Tests for _storage.py."""

import pytest

from shoal._models import ChunkData, SectionData, WordObservation
from shoal._storage import Storage


class TestDocumentCRUD:
    def test_insert_and_get(self, storage):
        doc_id = storage.insert_document("Test Doc", "content here", "plaintext")
        doc = storage.get_document(doc_id)
        assert doc is not None
        assert doc["title"] == "Test Doc"
        assert doc["format"] == "plaintext"
        assert doc["content_hash"]  # SHA-256 should be set

    def test_tags(self, storage):
        doc_id = storage.insert_document("Tagged", "content", "markdown")
        storage.set_tags(doc_id, ["bio", "sci"])
        tags = storage.get_tags(doc_id)
        assert tags == ["bio", "sci"]

        # Replace tags
        storage.set_tags(doc_id, ["new"])
        assert storage.get_tags(doc_id) == ["new"]

    def test_list_documents(self, storage):
        id1 = storage.insert_document("Doc A", "a", "plaintext")
        id2 = storage.insert_document("Doc B", "b", "markdown")
        storage.set_tags(id1, ["tag1"])
        storage.set_tags(id2, ["tag2"])

        all_docs = storage.list_documents()
        assert len(all_docs) == 2

        filtered = storage.list_documents(tag="tag1")
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Doc A"

    def test_delete_document(self, storage):
        doc_id = storage.insert_document("Delete Me", "bye", "plaintext")
        assert storage.delete_document(doc_id) is True
        assert storage.get_document(doc_id) is None
        assert storage.delete_document(999) is False

    def test_count_documents(self, storage):
        assert storage.count_documents() == 0
        storage.insert_document("A", "a", "plaintext")
        assert storage.count_documents() == 1


def _make_section(title="Test Section", depth=0, position=0, parent_index=None) -> SectionData:
    return SectionData(
        title=title,
        depth=depth,
        position=position,
        start_char=0,
        end_char=100,
        parent_index=parent_index,
        analysis_text="Some analysis text.",
    )


class TestSectionStorage:
    def test_insert_and_get_sections(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        sections = [
            _make_section("Intro", depth=0, position=0),
            _make_section("Body", depth=0, position=1),
        ]
        section_ids = storage.insert_sections(doc_id, sections)
        assert len(section_ids) == 2

        retrieved = storage.get_sections_for_document(doc_id)
        assert len(retrieved) == 2
        assert retrieved[0]["title"] == "Intro"
        assert retrieved[1]["title"] == "Body"

    def test_section_parent_child(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        sections = [
            _make_section("Parent", depth=0, position=0),
            _make_section("Child", depth=1, position=0, parent_index=0),
        ]
        section_ids = storage.insert_sections(doc_id, sections)
        assert len(section_ids) == 2

        retrieved = storage.get_sections_for_document(doc_id)
        child = next(s for s in retrieved if s["title"] == "Child")
        assert child["parent_id"] == section_ids[0]

    def test_section_path(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        sections = [
            _make_section("Root", depth=0, position=0),
            _make_section("Child", depth=1, position=0, parent_index=0),
        ]
        section_ids = storage.insert_sections(doc_id, sections)

        path = storage.get_section_path(section_ids[1])
        assert path == ["Root", "Child"]

        root_path = storage.get_section_path(section_ids[0])
        assert root_path == ["Root"]

    def test_section_chunk_count(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        sections = [_make_section("Section")]
        section_ids = storage.insert_sections(doc_id, sections)

        storage.update_section_chunk_count(section_ids[0], 5)
        retrieved = storage.get_sections_for_document(doc_id)
        assert retrieved[0]["n_chunks"] == 5

    def test_count_sections(self, storage):
        assert storage.count_sections() == 0
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        storage.insert_sections(doc_id, [_make_section("A"), _make_section("B", position=1)])
        assert storage.count_sections() == 2

    def test_cascade_delete_sections(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        storage.insert_sections(doc_id, [_make_section("A")])
        assert storage.count_sections() == 1
        storage.delete_document(doc_id)
        assert storage.count_sections() == 0


class TestChunkStorage:
    def _make_chunk(self, index=0, section_index=0) -> ChunkData:
        return ChunkData(
            text="Sample chunk text.",
            chunk_index=index,
            start_char=0,
            end_char=18,
            sentence_count=1,
            confidence=3.5,
            coverage=0.85,
            matched_words=5,
            top_reef_id=42,
            top_reef_name="botanical classification systems",
            arch_scores=[1.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[
                (42, "botanical classification systems", 5.0, 3.0, 4),
                (17, "animal body structures", 3.0, 2.0, 3),
            ],
            top_islands=[
                (10, "biological taxonomy", 8.0, 3),
            ],
            section_index=section_index,
            metadata={"headers": ["# Test"]},
        )

    def test_insert_and_count(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        section_ids = storage.insert_sections(doc_id, [_make_section("Section")])
        chunks = [self._make_chunk(0), self._make_chunk(1)]
        chunk_ids = storage.insert_chunks(doc_id, chunks, section_ids=section_ids)
        assert len(chunk_ids) == 2
        assert storage.count_chunks() == 2
        assert storage.count_chunks_for_document(doc_id) == 2

    def test_cascade_delete(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        section_ids = storage.insert_sections(doc_id, [_make_section("Section")])
        storage.insert_chunks(doc_id, [self._make_chunk()], section_ids=section_ids)
        assert storage.count_chunks() == 1
        storage.delete_document(doc_id)
        assert storage.count_chunks() == 0


class TestReefOverlapSearch:
    def _setup_doc_with_section(self, storage) -> tuple[int, list[int]]:
        """Helper to create a document with a section."""
        doc_id = storage.insert_document("Bio Doc", "text", "plaintext")
        section_ids = storage.insert_sections(doc_id, [_make_section("Main")])
        return doc_id, section_ids

    def test_basic_search(self, storage):
        doc_id, section_ids = self._setup_doc_with_section(storage)
        chunk = ChunkData(
            text="Plants use photosynthesis.",
            chunk_index=0,
            start_char=0,
            end_char=25,
            sentence_count=1,
            confidence=3.5,
            coverage=0.9,
            matched_words=4,
            top_reef_id=42,
            top_reef_name="botanical classification systems",
            arch_scores=[2.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[
                (42, "botanical classification systems", 5.0, 3.0, 4),
                (17, "animal body structures", 3.0, 2.0, 3),
            ],
            top_islands=[(10, "bio taxonomy", 8.0, 3)],
            section_index=0,
        )
        storage.insert_chunks(doc_id, [chunk], section_ids=section_ids)

        # Search with overlapping reef
        results = storage.search_by_reef_overlap(
            [(42, 4.0), (99, 2.0)],
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].document_title == "Bio Doc"
        # Scoring: SUM(chunk_z * query_z * reef_idf) * sqrt(n_shared) / reef_l2_norm
        # IDF = log(1 + N/df): N=1, reef 42 df=1 → idf=log(2)
        # dot = 5.0 * 4.0 * log(2) = 20 * log(2), n_shared = 1, l2_norm = sqrt(5^2 + 3^2) = sqrt(34)
        import math
        idf_42 = math.log(2)  # log(1 + 1/1)
        expected = (5.0 * 4.0 * idf_42) * math.sqrt(1) / math.sqrt(34)
        assert results[0].match_score == pytest.approx(expected)
        assert results[0].n_shared_reefs == 1
        assert results[0].section_title == "Main"
        assert results[0].section_path == ["Main"]

    def test_search_with_scores(self, storage):
        doc_id, section_ids = self._setup_doc_with_section(storage)
        chunk = ChunkData(
            text="Test chunk.",
            chunk_index=0,
            start_char=0,
            end_char=11,
            sentence_count=1,
            confidence=2.0,
            coverage=0.8,
            matched_words=3,
            top_reef_id=42,
            top_reef_name="reef A",
            arch_scores=[1.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[
                (42, "reef A", 5.0, 3.0, 4),
                (17, "reef B", 3.0, 2.0, 3),
            ],
            top_islands=[],
            section_index=0,
        )
        storage.insert_chunks(doc_id, [chunk], section_ids=section_ids)

        results = storage.search_by_reef_overlap(
            [(42, 4.0), (17, 2.0)],
            top_k=5,
            include_scores=True,
        )
        assert len(results) == 1
        assert len(results[0].shared_reefs) == 2
        # Scoring: SUM(chunk_z * query_z * reef_idf) * sqrt(n_shared) / reef_l2_norm
        # IDF = log(1 + N/df): N=1, both reefs df=1 → idf=log(2)
        # dot = (5.0*4.0 + 3.0*2.0) * log(2) = 26.0 * log(2), n_shared = 2, l2_norm = sqrt(34)
        import math
        idf = math.log(2)
        expected = (5.0 * 4.0 * idf + 3.0 * 2.0 * idf) * math.sqrt(2) / math.sqrt(34)
        assert results[0].match_score == pytest.approx(expected)

    def test_search_no_overlap(self, storage):
        doc_id, section_ids = self._setup_doc_with_section(storage)
        chunk = ChunkData(
            text="Test.",
            chunk_index=0,
            start_char=0,
            end_char=5,
            sentence_count=1,
            confidence=2.0,
            coverage=0.8,
            matched_words=1,
            top_reef_id=42,
            top_reef_name="reef A",
            arch_scores=[1.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[(42, "reef A", 5.0, 3.0, 4)],
            top_islands=[],
            section_index=0,
        )
        storage.insert_chunks(doc_id, [chunk], section_ids=section_ids)

        results = storage.search_by_reef_overlap([(99, 4.0)], top_k=5)
        assert len(results) == 0

    def test_search_with_tag_filter(self, storage):
        id1 = storage.insert_document("Bio", "text", "plaintext")
        storage.set_tags(id1, ["biology"])
        sec_ids1 = storage.insert_sections(id1, [_make_section("Bio Section")])

        id2 = storage.insert_document("Phys", "text", "plaintext")
        storage.set_tags(id2, ["physics"])
        sec_ids2 = storage.insert_sections(id2, [_make_section("Phys Section")])

        chunk_template = ChunkData(
            text="Chunk.",
            chunk_index=0,
            start_char=0,
            end_char=6,
            sentence_count=1,
            confidence=2.0,
            coverage=0.8,
            matched_words=1,
            top_reef_id=42,
            top_reef_name="reef",
            arch_scores=[1.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[(42, "reef", 5.0, 3.0, 4)],
            top_islands=[],
            section_index=0,
        )
        storage.insert_chunks(id1, [chunk_template], section_ids=sec_ids1)

        chunk2 = ChunkData(
            text="Chunk 2.",
            chunk_index=0,
            start_char=0,
            end_char=8,
            sentence_count=1,
            confidence=2.0,
            coverage=0.8,
            matched_words=1,
            top_reef_id=42,
            top_reef_name="reef",
            arch_scores=[1.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[(42, "reef", 5.0, 3.0, 4)],
            top_islands=[],
            section_index=0,
        )
        storage.insert_chunks(id2, [chunk2], section_ids=sec_ids2)

        # Filter by biology tag
        results = storage.search_by_reef_overlap(
            [(42, 4.0)], top_k=5, tags=["biology"]
        )
        assert len(results) == 1
        assert results[0].document_title == "Bio"

    def test_search_empty_reefs(self, storage):
        results = storage.search_by_reef_overlap([], top_k=5)
        assert results == []


class TestWordObservations:
    def test_insert_and_get(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        obs = [
            WordObservation(
                word="testword", document_id=doc_id, sentence_idx=0,
                context_reefs=[{"reef_id": 42, "z_score": 3.0, "weight": 0.7}],
            ),
            WordObservation(
                word="testword", document_id=doc_id, sentence_idx=1,
                context_reefs=[{"reef_id": 17, "z_score": 2.0, "weight": 0.3}],
            ),
            WordObservation(
                word="other", document_id=doc_id, sentence_idx=0,
                context_reefs=[{"reef_id": 99, "z_score": 1.5, "weight": 0.5}],
            ),
        ]
        storage.insert_word_observations(obs)

        grouped = storage.get_word_observations_for_document(doc_id)
        assert "testword" in grouped
        assert "other" in grouped
        assert len(grouped["testword"]) == 2
        assert len(grouped["other"]) == 1
        # Check context_reefs are deserialized
        assert grouped["testword"][0][0]["reef_id"] == 42

    def test_empty_document(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        grouped = storage.get_word_observations_for_document(doc_id)
        assert grouped == {}


class TestCustomWords:
    def test_insert_and_count(self, storage):
        assert storage.count_custom_words() == 0
        cw_id = storage.insert_custom_word(
            word="photovoltaic", word_hash=12345, word_id=100,
            specificity=2, idf_q=10, n_occurrences=3,
            reef_entries=[(42, 0, 0.9), (17, 0, 0.5)],
        )
        assert cw_id > 0
        assert storage.count_custom_words() == 1

    def test_idempotent_insert(self, storage):
        storage.insert_custom_word(
            word="term", word_hash=111, word_id=50,
            specificity=1, idf_q=5, n_occurrences=2,
            reef_entries=[(10, 0, 1.0)],
        )
        # Insert again — should update
        cw_id = storage.insert_custom_word(
            word="term", word_hash=111, word_id=51,
            specificity=2, idf_q=8, n_occurrences=5,
            reef_entries=[(10, 0, 0.8), (20, 0, 0.6)],
        )
        assert storage.count_custom_words() == 1
        vocab = storage.load_custom_vocabulary()
        assert len(vocab) == 1
        assert vocab[0]["specificity"] == 2
        assert vocab[0]["n_occurrences"] == 5
        assert len(vocab[0]["reef_associations"]) == 2

    def test_load_custom_vocabulary(self, storage):
        storage.insert_custom_word(
            word="alpha", word_hash=100, word_id=10,
            specificity=2, idf_q=5, n_occurrences=3,
            reef_entries=[(42, 0, 0.9)],
        )
        storage.insert_custom_word(
            word="beta", word_hash=200, word_id=20,
            specificity=1, idf_q=8, n_occurrences=4,
            reef_entries=[(17, 0, 0.7), (99, 0, 0.3)],
        )
        vocab = storage.load_custom_vocabulary()
        assert len(vocab) == 2
        assert vocab[0]["word"] == "alpha"
        assert vocab[1]["word"] == "beta"
        assert len(vocab[0]["reef_associations"]) == 1
        assert len(vocab[1]["reef_associations"]) == 2


class TestChunkCustomWords:
    def _setup(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        section_ids = storage.insert_sections(doc_id, [_make_section("Section")])
        chunk = ChunkData(
            text="Sample chunk text about a specialized topic.",
            chunk_index=0, start_char=0, end_char=45,
            sentence_count=1, confidence=3.0, coverage=0.8,
            matched_words=4, top_reef_id=42,
            top_reef_name="reef A",
            arch_scores=[1.0, 0.5, 0.3, 0.1, 0.0],
            top_reefs=[(42, "reef A", 5.0, 3.0, 4)],
            top_islands=[], section_index=0,
        )
        chunk2 = ChunkData(
            text="Another chunk about different topics entirely.",
            chunk_index=1, start_char=46, end_char=92,
            sentence_count=1, confidence=2.5, coverage=0.7,
            matched_words=3, top_reef_id=17,
            top_reef_name="reef B",
            arch_scores=[0.5, 1.0, 0.3, 0.1, 0.0],
            top_reefs=[(17, "reef B", 4.0, 2.0, 3)],
            top_islands=[], section_index=0,
        )
        chunk_ids = storage.insert_chunks(doc_id, [chunk, chunk2], section_ids=section_ids)
        return doc_id, chunk_ids

    def test_insert_and_get(self, storage):
        _, chunk_ids = self._setup(storage)
        # Insert custom word association
        cw_id = storage.insert_custom_word(
            word="termA", word_hash=100, word_id=10,
            specificity=2, idf_q=5, n_occurrences=3,
            reef_entries=[(42, 0, 0.9)],
        )
        storage.insert_chunk_custom_words(chunk_ids[0], [cw_id])
        storage.insert_chunk_custom_words(chunk_ids[1], [cw_id])

        result = storage.get_chunks_by_custom_words([cw_id])
        assert len(result) == 2
        assert set(result) == set(chunk_ids)

    def test_multiple_custom_words_ordering(self, storage):
        _, chunk_ids = self._setup(storage)
        cw1 = storage.insert_custom_word(
            word="termA", word_hash=100, word_id=10,
            specificity=2, idf_q=5, n_occurrences=3,
            reef_entries=[(42, 0, 0.9)],
        )
        cw2 = storage.insert_custom_word(
            word="termB", word_hash=200, word_id=20,
            specificity=1, idf_q=8, n_occurrences=4,
            reef_entries=[(17, 0, 0.7)],
        )
        # chunk 0 has both words, chunk 1 has only one
        storage.insert_chunk_custom_words(chunk_ids[0], [cw1, cw2])
        storage.insert_chunk_custom_words(chunk_ids[1], [cw1])

        result = storage.get_chunks_by_custom_words([cw1, cw2])
        # chunk 0 should come first (2 matches vs 1)
        assert result[0] == chunk_ids[0]

    def test_empty_custom_words(self, storage):
        result = storage.get_chunks_by_custom_words([])
        assert result == []

    def test_score_chunks_by_reef_overlap(self, storage):
        _, chunk_ids = self._setup(storage)
        # Score specific chunks with query reefs
        results = storage.score_chunks_by_reef_overlap(
            chunk_ids, [(42, 4.0)],
        )
        # Only chunk 0 has reef 42
        assert len(results) >= 1
        assert results[0].chunk_id == chunk_ids[0]

    def test_score_empty_chunks(self, storage):
        results = storage.score_chunks_by_reef_overlap([], [(42, 4.0)])
        assert results == []

    def test_score_empty_reefs(self, storage):
        results = storage.score_chunks_by_reef_overlap([1, 2], [])
        assert results == []

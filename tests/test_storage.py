"""Tests for _storage.py."""

import pytest

from shoal._models import ChunkData
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


class TestChunkStorage:
    def _make_chunk(self, index=0) -> ChunkData:
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
            arch_scores=[1.0, 0.5, 0.3, 0.1],
            top_reefs=[
                (42, "botanical classification systems", 5.0, 3.0, 4),
                (17, "animal body structures", 3.0, 2.0, 3),
            ],
            top_islands=[
                (10, "biological taxonomy", 8.0, 3),
            ],
            metadata={"headers": ["# Test"]},
        )

    def test_insert_and_count(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        chunks = [self._make_chunk(0), self._make_chunk(1)]
        chunk_ids = storage.insert_chunks(doc_id, chunks)
        assert len(chunk_ids) == 2
        assert storage.count_chunks() == 2
        assert storage.count_chunks_for_document(doc_id) == 2

    def test_cascade_delete(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        storage.insert_chunks(doc_id, [self._make_chunk()])
        assert storage.count_chunks() == 1
        storage.delete_document(doc_id)
        assert storage.count_chunks() == 0


class TestReefOverlapSearch:
    def test_basic_search(self, storage):
        doc_id = storage.insert_document("Bio Doc", "text", "plaintext")
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
            arch_scores=[2.0, 0.5, 0.3, 0.1],
            top_reefs=[
                (42, "botanical classification systems", 5.0, 3.0, 4),
                (17, "animal body structures", 3.0, 2.0, 3),
            ],
            top_islands=[(10, "bio taxonomy", 8.0, 3)],
        )
        storage.insert_chunks(doc_id, [chunk])

        # Search with overlapping reef
        results = storage.search_by_reef_overlap(
            [(42, 4.0), (99, 2.0)],
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].document_title == "Bio Doc"
        assert results[0].match_score == pytest.approx(20.0)  # 5.0 * 4.0
        assert results[0].n_shared_reefs == 1

    def test_search_with_scores(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
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
            arch_scores=[1.0, 0.5, 0.3, 0.1],
            top_reefs=[
                (42, "reef A", 5.0, 3.0, 4),
                (17, "reef B", 3.0, 2.0, 3),
            ],
            top_islands=[],
        )
        storage.insert_chunks(doc_id, [chunk])

        results = storage.search_by_reef_overlap(
            [(42, 4.0), (17, 2.0)],
            top_k=5,
            include_scores=True,
        )
        assert len(results) == 1
        assert len(results[0].shared_reefs) == 2
        # match_score = 5.0*4.0 + 3.0*2.0 = 26.0
        assert results[0].match_score == pytest.approx(26.0)

    def test_search_no_overlap(self, storage):
        doc_id = storage.insert_document("Doc", "text", "plaintext")
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
            arch_scores=[1.0, 0.5, 0.3, 0.1],
            top_reefs=[(42, "reef A", 5.0, 3.0, 4)],
            top_islands=[],
        )
        storage.insert_chunks(doc_id, [chunk])

        results = storage.search_by_reef_overlap([(99, 4.0)], top_k=5)
        assert len(results) == 0

    def test_search_with_tag_filter(self, storage):
        id1 = storage.insert_document("Bio", "text", "plaintext")
        storage.set_tags(id1, ["biology"])
        id2 = storage.insert_document("Phys", "text", "plaintext")
        storage.set_tags(id2, ["physics"])

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
            arch_scores=[1.0, 0.5, 0.3, 0.1],
            top_reefs=[(42, "reef", 5.0, 3.0, 4)],
            top_islands=[],
        )
        storage.insert_chunks(id1, [chunk_template])

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
            arch_scores=[1.0, 0.5, 0.3, 0.1],
            top_reefs=[(42, "reef", 5.0, 3.0, 4)],
            top_islands=[],
        )
        storage.insert_chunks(id2, [chunk2])

        # Filter by biology tag
        results = storage.search_by_reef_overlap(
            [(42, 4.0)], top_k=5, tags=["biology"]
        )
        assert len(results) == 1
        assert results[0].document_title == "Bio"

    def test_search_empty_reefs(self, storage):
        results = storage.search_by_reef_overlap([], top_k=5)
        assert results == []

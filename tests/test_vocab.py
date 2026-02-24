"""Tests for _vocab.py — startup injection, observation collection, vocabulary build."""

from unittest.mock import MagicMock

import pytest

from shoal._models import WordObservation
from shoal._vocab import (
    build_vocabulary,
    collect_word_observations,
    get_custom_word_ids_for_chunk,
    inject_custom_vocabulary_at_startup,
)


def _make_mock_scorer():
    """Create a mock scorer with the methods we need."""
    scorer = MagicMock()
    scorer.lookup_word.return_value = None  # word not known
    scorer.get_word_tags.return_value = {}
    scorer.calc_custom_idf.return_value = 148
    scorer.calc_custom_weight.side_effect = lambda rid, strength: round(5000 * strength)
    scorer.reef_word_counts = [4000] * 444  # uniform reef sizes for tests
    scorer.avg_reef_words = 4000.0
    return scorer


class TestInjectCustomVocabularyAtStartup:
    def test_empty_vocabulary(self, storage):
        scorer = _make_mock_scorer()
        count = inject_custom_vocabulary_at_startup(scorer, storage)
        assert count == 0
        scorer.add_custom_word.assert_not_called()

    def test_inject_words_with_stored_weights(self, storage):
        scorer = _make_mock_scorer()
        # Pre-populate storage with a custom word that has bm25_q values
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        storage.insert_custom_word(
            word="photovoltaic",
            word_hash=12345,
            word_id=100,
            specificity=2,
            idf_q=10,
            n_occurrences=3,
            reef_entries=[(42, 4500, 0.9), (17, 2500, 0.5)],
        )
        count = inject_custom_vocabulary_at_startup(scorer, storage)
        assert count == 1
        scorer.add_custom_word.assert_called_once()
        call_args = scorer.add_custom_word.call_args
        assert call_args[0][0] == "photovoltaic"
        # reef_weights should be the stored (reef_id, bm25_q) tuples
        assert set(call_args[0][1]) == {(42, 4500), (17, 2500)}
        assert call_args[1]["idf_q"] == 10
        assert call_args[1]["specificity"] == 2
        # calc_custom_weight should NOT have been called (stored weights used)
        scorer.calc_custom_weight.assert_not_called()

    def test_inject_words_migration_zero_weights(self, storage):
        scorer = _make_mock_scorer()
        # Old DB format: bm25_q = 0 for all entries
        doc_id = storage.insert_document("Doc", "text", "plaintext")
        storage.insert_custom_word(
            word="photovoltaic",
            word_hash=12345,
            word_id=100,
            specificity=2,
            idf_q=10,
            n_occurrences=3,
            reef_entries=[(42, 0, 0.9), (17, 0, 0.5)],
        )
        count = inject_custom_vocabulary_at_startup(scorer, storage)
        assert count == 1
        # calc_custom_weight should have been called (migration path)
        assert scorer.calc_custom_weight.call_count == 2

    def test_skip_on_value_error(self, storage):
        scorer = _make_mock_scorer()
        scorer.add_custom_word.side_effect = ValueError("already exists")
        storage.insert_custom_word(
            word="test", word_hash=99, word_id=50,
            specificity=1, idf_q=5, n_occurrences=2,
            reef_entries=[(10, 0, 1.0)],
        )
        count = inject_custom_vocabulary_at_startup(scorer, storage)
        assert count == 0


class TestCollectWordObservations:
    def _make_mock_analysis(self, unknown_words=None, confidence=2.0):
        """Build a minimal mock DocumentAnalysis."""
        analysis = MagicMock()

        sent_result = MagicMock()
        sent_result.unknown_words = unknown_words if unknown_words is not None else ["xyzterm"]
        sent_result.confidence = confidence
        sent_result.matched_word_ids = frozenset([1, 2])

        reef1 = MagicMock()
        reef1.reef_id = 42
        reef1.z_score = 3.5
        reef2 = MagicMock()
        reef2.reef_id = 17
        reef2.z_score = 2.0
        sent_result.top_reefs = [reef1, reef2]

        segment = MagicMock()
        segment.start_idx = 0
        segment.sentence_results = [sent_result]

        chunk_reef = MagicMock()
        chunk_reef.reef_id = 42
        chunk_reef.z_score = 4.0
        chunk_reef2 = MagicMock()
        chunk_reef2.reef_id = 99
        chunk_reef2.z_score = 1.5
        segment.topic.top_reefs = [chunk_reef, chunk_reef2]

        analysis.segments = [segment]
        return analysis

    def test_collect_basic(self):
        scorer = _make_mock_scorer()
        analysis = self._make_mock_analysis()
        obs = collect_word_observations(scorer, analysis, doc_id=1)
        assert len(obs) == 1
        assert obs[0].word == "xyzterm"
        assert obs[0].document_id == 1
        assert len(obs[0].context_reefs) > 0

    def test_high_confidence_weights(self):
        scorer = _make_mock_scorer()
        analysis = self._make_mock_analysis(confidence=2.0)  # > 1.5 threshold
        obs = collect_word_observations(scorer, analysis, doc_id=1)
        # With high confidence, sentence weight is 0.7
        reef_42 = next(r for r in obs[0].context_reefs if r["reef_id"] == 42)
        # sentence z=3.5 * 0.7 + chunk z=4.0 * 0.3 = 2.45 + 1.2 = 3.65
        assert abs(reef_42["z_score"] - 3.65) < 0.01

    def test_low_confidence_weights(self):
        scorer = _make_mock_scorer()
        analysis = self._make_mock_analysis(confidence=0.5)  # < 1.5 threshold
        obs = collect_word_observations(scorer, analysis, doc_id=1, confidence_threshold=1.5)
        reef_42 = next(r for r in obs[0].context_reefs if r["reef_id"] == 42)
        # sentence z=3.5 * 0.3 + chunk z=4.0 * 0.7 = 1.05 + 2.8 = 3.85
        assert abs(reef_42["z_score"] - 3.85) < 0.01

    def test_no_unknown_words(self):
        scorer = _make_mock_scorer()
        analysis = self._make_mock_analysis(unknown_words=[])
        obs = collect_word_observations(scorer, analysis, doc_id=1)
        assert len(obs) == 0

    def test_multiple_unknown_words(self):
        scorer = _make_mock_scorer()
        analysis = self._make_mock_analysis(unknown_words=["term1", "term2"])
        obs = collect_word_observations(scorer, analysis, doc_id=1)
        assert len(obs) == 2
        words = {o.word for o in obs}
        assert words == {"term1", "term2"}


class TestBuildVocabulary:
    def test_build_from_observations(self, storage):
        scorer = _make_mock_scorer()
        doc_id = storage.insert_document("Doc", "text", "plaintext")

        # Create mock WordInfo for add_custom_word
        mock_word_info = MagicMock()
        mock_word_info.word_hash = 12345
        mock_word_info.word_id = 100
        mock_word_info.idf_q = 148
        scorer.add_custom_word.return_value = mock_word_info

        # Insert enough observations (>= 2 for short doc)
        obs = [
            WordObservation(
                word="xyzterm", document_id=doc_id, sentence_idx=0,
                context_reefs=[{"reef_id": 42, "z_score": 3.0, "weight": 0.7}],
            ),
            WordObservation(
                word="xyzterm", document_id=doc_id, sentence_idx=1,
                context_reefs=[{"reef_id": 42, "z_score": 4.0, "weight": 0.7}],
            ),
        ]
        storage.insert_word_observations(obs)

        n_new = build_vocabulary(scorer, storage, doc_id, total_word_count=100)
        assert n_new == 1
        scorer.add_custom_word.assert_called_once()
        # Verify helpers were called
        scorer.calc_custom_idf.assert_called_once()
        scorer.calc_custom_weight.assert_called()
        # Verify reef_weights (not reef_associations) passed
        call_args = scorer.add_custom_word.call_args
        assert "idf_q" in call_args[1]
        # Verify stored in DB
        assert storage.count_custom_words() == 1

    def test_below_occurrence_threshold(self, storage):
        scorer = _make_mock_scorer()
        doc_id = storage.insert_document("Doc", "text", "plaintext")

        # Only 1 observation — below threshold of 2
        obs = [
            WordObservation(
                word="rareword", document_id=doc_id, sentence_idx=0,
                context_reefs=[{"reef_id": 42, "z_score": 3.0, "weight": 0.7}],
            ),
        ]
        storage.insert_word_observations(obs)

        n_new = build_vocabulary(scorer, storage, doc_id, total_word_count=100)
        assert n_new == 0

    def test_skip_known_word(self, storage):
        scorer = _make_mock_scorer()
        scorer.lookup_word.return_value = MagicMock()  # word is known
        doc_id = storage.insert_document("Doc", "text", "plaintext")

        obs = [
            WordObservation(
                word="knownword", document_id=doc_id, sentence_idx=i,
                context_reefs=[{"reef_id": 42, "z_score": 3.0, "weight": 0.7}],
            )
            for i in range(3)
        ]
        storage.insert_word_observations(obs)

        n_new = build_vocabulary(scorer, storage, doc_id, total_word_count=100)
        assert n_new == 0

    def test_long_doc_threshold(self, storage):
        scorer = _make_mock_scorer()
        doc_id = storage.insert_document("Doc", "text", "plaintext")

        mock_word_info = MagicMock()
        mock_word_info.word_hash = 12345
        mock_word_info.word_id = 100
        mock_word_info.idf_q = 10
        scorer.add_custom_word.return_value = mock_word_info

        # 2 observations for a long doc — needs 3
        obs = [
            WordObservation(
                word="term", document_id=doc_id, sentence_idx=i,
                context_reefs=[{"reef_id": 42, "z_score": 3.0, "weight": 0.7}],
            )
            for i in range(2)
        ]
        storage.insert_word_observations(obs)

        n_new = build_vocabulary(scorer, storage, doc_id, total_word_count=600)
        assert n_new == 0  # below threshold of 3

    def test_low_z_reefs_filtered(self, storage):
        scorer = _make_mock_scorer()
        doc_id = storage.insert_document("Doc", "text", "plaintext")

        # Observations with z_score below default 0.5 threshold
        obs = [
            WordObservation(
                word="weakterm", document_id=doc_id, sentence_idx=i,
                context_reefs=[{"reef_id": 42, "z_score": 0.2, "weight": 0.7}],
            )
            for i in range(3)
        ]
        storage.insert_word_observations(obs)

        n_new = build_vocabulary(scorer, storage, doc_id, total_word_count=100)
        assert n_new == 0  # all reefs filtered out


class TestGetCustomWordIdsForChunk:
    def test_no_custom_words(self):
        scorer = _make_mock_scorer()
        scorer.get_word_tags.return_value = {}
        result = get_custom_word_ids_for_chunk(scorer, frozenset([1, 2, 3]))
        assert result == {}

    def test_with_custom_words(self):
        scorer = _make_mock_scorer()
        scorer.get_word_tags.return_value = {5: 10, 7: 20}
        result = get_custom_word_ids_for_chunk(scorer, frozenset([1, 5, 7]))
        assert result == {5: 10, 7: 20}

    def test_empty_word_ids(self):
        scorer = _make_mock_scorer()
        result = get_custom_word_ids_for_chunk(scorer, frozenset())
        assert result == {}

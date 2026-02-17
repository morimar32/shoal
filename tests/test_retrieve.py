"""Tests for _retrieve.py (requires lagoon scorer)."""

from shoal._ingest import ingest_document
from shoal._models import DocFormat
from shoal._retrieve import _is_stopword_query, search


class TestRetrieve:
    def _ingest_bio_doc(self, scorer, storage):
        text = (
            "Photosynthesis is the process by which plants convert sunlight into energy. "
            "Chlorophyll in the chloroplasts absorbs light energy from the sun. "
            "The light reactions split water molecules and produce oxygen. "
            "Carbon dioxide is fixed into sugar molecules during the Calvin cycle. "
            "Plants are the primary producers in most terrestrial ecosystems."
        )
        return ingest_document(
            scorer, storage,
            title="Photosynthesis Basics",
            content=text,
            fmt=DocFormat.plaintext,
            tags=["biology"],
        )

    def _ingest_geo_doc(self, scorer, storage):
        text = (
            "Earthquakes occur when tectonic plates move along fault lines. "
            "Seismic waves propagate through the earth during an earthquake. "
            "The Richter scale measures the magnitude of seismic events. "
            "Volcanic eruptions often accompany tectonic plate movements. "
            "Geology is the study of the earth and its physical processes."
        )
        return ingest_document(
            scorer, storage,
            title="Earthquakes and Geology",
            content=text,
            fmt=DocFormat.plaintext,
            tags=["geology"],
        )

    def test_basic_search(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        self._ingest_geo_doc(scorer, storage)

        response = search(scorer, storage, "photosynthesis in plants", top_k=5)
        assert len(response.results) > 0
        assert response.query_info.top_reef != ""
        assert response.query_info.confidence > 0

    def test_search_with_scores(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)

        response = search(
            scorer, storage, "photosynthesis chlorophyll",
            top_k=5, include_scores=True,
        )
        if response.results:
            assert len(response.results[0].shared_reefs) > 0

    def test_search_with_tag_filter(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        self._ingest_geo_doc(scorer, storage)

        response = search(
            scorer, storage, "photosynthesis",
            top_k=5, tags=["geology"],
        )
        for r in response.results:
            assert r.document_title == "Earthquakes and Geology"

    def test_search_no_results(self, scorer, storage):
        # Empty database
        response = search(scorer, storage, "photosynthesis", top_k=5)
        assert len(response.results) == 0

    def test_query_info_populated(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        response = search(scorer, storage, "photosynthesis", top_k=5)
        qi = response.query_info
        assert qi.top_reef != ""
        assert len(qi.top_reefs) > 0
        assert qi.coverage > 0

    def test_search_results_include_section_info(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        response = search(scorer, storage, "photosynthesis", top_k=5)
        if response.results:
            r = response.results[0]
            assert r.section_title != ""
            assert len(r.section_path) > 0

    def test_stopword_query_returns_empty(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        response = search(scorer, storage, "the", top_k=5)
        assert len(response.results) == 0
        assert response.query_info.confidence == 0.0
        assert response.query_info.top_reef == ""

    def test_stopword_multi_word_returns_empty(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        response = search(scorer, storage, "the and or but", top_k=5)
        assert len(response.results) == 0

    def test_stopword_mixed_query_proceeds(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        # "the photosynthesis" has a real word — should NOT be blocked
        response = search(scorer, storage, "the photosynthesis", top_k=5)
        assert len(response.results) > 0

    def test_top_k_is_maximum(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        self._ingest_geo_doc(scorer, storage)
        response = search(scorer, storage, "photosynthesis", top_k=100)
        # Should return fewer than 100 — top_k is a max, not a target
        assert len(response.results) <= 100


    def test_search_with_lightning_rod_disabled(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        response = search(
            scorer, storage, "photosynthesis in plants",
            top_k=5, enable_lightning_rod=False,
        )
        assert len(response.results) > 0
        # tagged_words should be empty when lightning rod is disabled
        assert response.query_info.tagged_words == []

    def test_query_info_has_tagged_words(self, scorer, storage):
        self._ingest_bio_doc(scorer, storage)
        response = search(scorer, storage, "photosynthesis", top_k=5)
        qi = response.query_info
        # tagged_words should be a list (may be empty if no custom words match)
        assert isinstance(qi.tagged_words, list)


class TestStopWordDetection:
    def test_single_stop_word(self):
        assert _is_stopword_query("the") is True

    def test_multiple_stop_words(self):
        assert _is_stopword_query("the and or but") is True

    def test_mixed_query(self):
        assert _is_stopword_query("the photosynthesis") is False

    def test_real_query(self):
        assert _is_stopword_query("photosynthesis in plants") is False

    def test_empty_query(self):
        assert _is_stopword_query("") is True

    def test_punctuation_only(self):
        assert _is_stopword_query("... !!!") is True

    def test_case_insensitive(self):
        assert _is_stopword_query("The AND Or") is True

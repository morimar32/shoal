"""Tests for _retrieve.py (requires lagoon scorer)."""

from shoal._ingest import ingest_document
from shoal._models import DocFormat
from shoal._retrieve import search


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

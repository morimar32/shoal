"""Tests for _engine.py (integration tests)."""

from shoal import Engine, DocFormat


class TestEngine:
    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "test.db"
        with Engine(db_path=db_path) as engine:
            s = engine.status()
            assert s["status"] == "ok"
            assert s["n_documents"] == 0

    def test_ingest_and_search(self, engine):
        text = (
            "Photosynthesis converts sunlight into chemical energy in plants. "
            "Chlorophyll pigments absorb light in the chloroplasts. "
            "The Calvin cycle fixes carbon dioxide into glucose. "
            "Oxygen is released as a byproduct of water splitting."
        )
        result = engine.ingest(
            title="Photosynthesis",
            content=text,
            format=DocFormat.plaintext,
            tags=["biology"],
        )
        assert result.n_chunks >= 1

        response = engine.search("photosynthesis plants", top_k=5)
        assert len(response.results) > 0
        assert response.query_info.top_reef != ""

    def test_delete_document(self, engine):
        result = engine.ingest(
            title="To Delete",
            content=(
                "Some text about geology and rocks. "
                "Earthquakes happen along fault lines. "
                "Seismic waves travel through the earth."
            ),
            format=DocFormat.plaintext,
        )
        assert engine.status()["n_documents"] == 1
        assert engine.delete_document(result.id)
        assert engine.status()["n_documents"] == 0

    def test_status(self, engine):
        s = engine.status()
        assert s["status"] == "ok"
        assert s["n_documents"] == 0
        assert s["n_chunks"] == 0
        assert "lagoon_version" in s
        assert "db_path" in s

    def test_multiple_documents(self, engine):
        engine.ingest(
            title="Doc A",
            content=(
                "Photosynthesis converts light into energy. "
                "Plants use chlorophyll for this process."
            ),
            format=DocFormat.plaintext,
        )
        engine.ingest(
            title="Doc B",
            content=(
                "Earthquakes shake the ground violently. "
                "Seismic activity is measured by seismographs."
            ),
            format=DocFormat.plaintext,
        )
        assert engine.status()["n_documents"] == 2

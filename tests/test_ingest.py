"""Tests for _ingest.py (requires lagoon scorer)."""

from shoal._ingest import ingest_document
from shoal._models import DocFormat


class TestIngest:
    def test_ingest_plaintext(self, scorer, storage):
        text = (
            "The neuron transmits electrical signals through the axon. "
            "Dendrites receive signals from neighboring neurons. "
            "Synaptic transmission occurs at the junction between neurons. "
            "The brain processes information through complex neural networks. "
            "Neurotransmitters carry chemical signals across synapses."
        )
        result = ingest_document(
            scorer, storage,
            title="Neural Signals",
            content=text,
            fmt=DocFormat.plaintext,
            tags=["neuroscience"],
        )
        assert result.id > 0
        assert result.title == "Neural Signals"
        assert result.n_chunks >= 1
        assert result.n_sections >= 1
        assert result.tags == ["neuroscience"]

        # Verify data was stored
        assert storage.count_documents() == 1
        assert storage.count_chunks() == result.n_chunks
        assert storage.count_sections() == result.n_sections
        assert storage.get_tags(result.id) == ["neuroscience"]

    def test_ingest_markdown(self, scorer, storage):
        md = (
            "# Photosynthesis\n\n"
            "Plants convert light energy into chemical energy through photosynthesis. "
            "Chlorophyll absorbs sunlight in the chloroplasts of plant cells. "
            "The light reactions produce ATP and NADPH from water molecules. "
            "Carbon dioxide is fixed into glucose during the Calvin cycle.\n\n"
            "## Light Reactions\n\n"
            "The light-dependent reactions occur in the thylakoid membranes. "
            "Photosystem II splits water molecules to release oxygen gas. "
            "Electrons travel through the electron transport chain. "
            "ATP synthase produces ATP using the proton gradient.\n"
        )
        result = ingest_document(
            scorer, storage,
            title="Photosynthesis",
            content=md,
            fmt=DocFormat.markdown,
        )
        assert result.n_chunks >= 1
        assert result.n_sections >= 1
        assert result.title == "Photosynthesis"

        # Verify sections were created
        sections = storage.get_sections_for_document(result.id)
        assert len(sections) >= 1

    def test_ingest_empty_text(self, scorer, storage):
        result = ingest_document(
            scorer, storage,
            title="Empty",
            content="",
            fmt=DocFormat.plaintext,
        )
        assert result.n_chunks == 0
        assert result.n_sections == 0
        assert storage.count_documents() == 1

    def test_ingest_with_metadata(self, scorer, storage):
        text = (
            "Earthquakes occur when tectonic plates shift along fault lines. "
            "Seismic waves travel through the earth after a quake. "
            "The Richter scale measures earthquake magnitude. "
            "Aftershocks can continue for weeks following a major seismic event."
        )
        result = ingest_document(
            scorer, storage,
            title="Earthquakes",
            content=text,
            fmt=DocFormat.plaintext,
            source_path="/docs/earthquake.txt",
            metadata={"source": "wikipedia"},
        )
        doc = storage.get_document(result.id)
        assert doc is not None
        assert doc["source_path"] == "/docs/earthquake.txt"

    def test_ingest_creates_sections_for_markdown(self, scorer, storage):
        md = (
            "# Biology Overview\n\n"
            "Biology is the study of living organisms.\n\n"
            "## Cell Structure\n\n"
            "Cells are the basic building blocks of life. "
            "They contain organelles that perform specific functions. "
            "The cell membrane controls what enters and exits the cell.\n\n"
            "## Genetics\n\n"
            "DNA carries genetic information in all living organisms. "
            "Genes are segments of DNA that encode proteins. "
            "Mutations can alter the function of proteins.\n"
        )
        result = ingest_document(
            scorer, storage,
            title="Biology",
            content=md,
            fmt=DocFormat.markdown,
        )
        assert result.n_sections >= 2
        sections = storage.get_sections_for_document(result.id)
        titles = [s["title"] for s in sections]
        assert "Cell Structure" in titles
        assert "Genetics" in titles

    def test_section_chunk_counts_updated(self, scorer, storage):
        text = (
            "The neuron transmits electrical signals through the axon. "
            "Dendrites receive signals from neighboring neurons. "
            "Synaptic transmission occurs at the junction between neurons. "
            "The brain processes information through complex neural networks."
        )
        result = ingest_document(
            scorer, storage,
            title="Neurons",
            content=text,
            fmt=DocFormat.plaintext,
        )
        sections = storage.get_sections_for_document(result.id)
        total_chunks_in_sections = sum(s["n_chunks"] for s in sections)
        assert total_chunks_in_sections == result.n_chunks

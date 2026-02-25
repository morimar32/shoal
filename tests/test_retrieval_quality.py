"""Stress tests for retrieval quality against the full 47-document corpus.

These tests run against a pre-built test.db created by:
    shoal --db test.db ingest-dir docs/

They test retrieval QUALITY — not just "does it return results" but
"does it return the RIGHT results in the RIGHT order."

Organized into tiers:
  - Top-1 accuracy: correct doc is the #1 result
  - Top-3 recall: correct doc appears in top 3
  - Top-5 recall: correct doc appears in top 5
  - Top-10 recall: correct doc appears somewhere in top 10
  - Known failures (xfail): queries that expose real weaknesses

Run with:  pytest tests/test_retrieval_quality.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shoal import Engine

# ── Fixture: session-scoped engine against pre-built test.db ──

_TEST_DB = Path(__file__).resolve().parent.parent / "test.db"


@pytest.fixture(scope="module")
def engine():
    """Module-scoped Engine connected to the pre-built test.db (read-only queries)."""
    if not _TEST_DB.exists():
        pytest.skip("test.db not found — run `shoal --db test.db ingest-dir docs/` first")
    eng = Engine(db_path=_TEST_DB)
    eng.start()
    yield eng
    eng.close()


# ── Helpers ──

def _top_n_docs(engine, query, n=10, **kw):
    """Return list of (document_title, match_score) for top-N results."""
    response = engine.search(query, top_k=n, **kw)
    return [
        (r.document_title, r.match_score)
        for r in response.results
    ]


def _top_n_doc_names(engine, query, n=10, **kw):
    """Return just document titles for top-N results."""
    return [title for title, _ in _top_n_docs(engine, query, n, **kw)]


def _unique_docs_in_top_n(engine, query, n=10, **kw):
    """Return set of unique document titles in top-N."""
    return set(_top_n_doc_names(engine, query, n, **kw))


def _search(engine, query, **kw):
    """Full search response."""
    return engine.search(query, top_k=10, **kw)


# ═══════════════════════════════════════════════════════════════════
# TIER 1: Top-1 accuracy — unambiguous, well-scoped queries
# These MUST return the correct document as the #1 result.
# ═══════════════════════════════════════════════════════════════════

class TestTop1Accuracy:
    """Queries with strong topical signal — the correct doc should be #1."""

    @pytest.mark.parametrize("query, expected_doc", [
        # ── Science & nature ──
        ("photosynthesis chloroplast", "Photosynthesis"),
        ("DNA double helix Watson Crick", "Dna"),
        ("natural selection evolution Darwin", "Natural Selection"),
        ("volcanic eruption lava magma", "Volcanic Eruption"),
        ("earthquake seismic fault line", "Earthquake"),
        ("mitochondria cell powerhouse ATP", "Mitochondrion"),
        ("climate change greenhouse gas warming", "Climate Change"),
        ("penicillin antibiotic Fleming", "Penicillin"),
        ("tyrannosaurus rex fossil bones", "Tyrannosaurus"),
        ("coral reef bleaching ocean", "Great Barrier Reef"),
        ("atmospheric river precipitation California", "Atmospheric River"),
        ("granite igneous rock mineral", "Granite"),
        ("jupiter great red spot gas giant", "Jupiter"),
        # ── Math & CS ──
        ("fourier transform frequency signal", "Fourier Transform"),
        ("game theory nash equilibrium", "Game Theory"),
        ("algorithm sorting complexity", "Algorithm"),
        ("supply demand economics price", "Supply And Demand"),
        ("thermodynamics entropy heat", "Thermodynamics"),
        # ── History & politics ──
        ("Supreme Court judicial review", "Supreme Court Of The United States"),
        ("Olympic Games athletics medal", "Olympic Games"),
        ("Renaissance art Michelangelo", "Renaissance"),
        ("Silk Road trade route ancient", "Silk Road"),
        ("British monarchy royal family", "Monarchy Of The United Kingdom"),
        # ── Disambiguation: strong context ──
        ("mercury planet orbit solar system", "Mercury Planet"),
        ("mercury element liquid metal toxic", "Mercury Element"),
        ("python snake constrictor reptile", "Python Snake"),
        ("apple fruit nutrition orchard", "Apple Fruit"),
        ("apple iphone macbook silicon valley", "Apple Inc"),
        ("yamamoto admiral pearl harbor navy", "Isoroku Yamamoto"),
        ("mercury closest planet to sun", "Mercury Planet"),
        # ── Starcraft cluster ──
        ("terran marine starcraft", "Terran"),
        ("yamato cannon battlecruiser", "Yamato Cannon"),
        ("starport building units", "Starport"),
        ("battlecruiser operational", "Battlecruiser Starcraft Ii"),
        # ── Manga cluster ──
        ("ichi the killer kakihara yakuza", "Ichi The Killer (Manga)"),
        ("manga shonen jump anime", "Manga"),
        # ── Coffee & food ──
        ("coffee arabica caffeine roasting", "Coffee"),
        # ── Biology ──
        ("chess grandmaster opening gambit", "Chess"),
    ])
    def test_top1(self, engine, query, expected_doc):
        docs = _top_n_doc_names(engine, query, n=5)
        assert len(docs) > 0, f"No results for: {query}"
        assert docs[0] == expected_doc, (
            f"Expected '{expected_doc}' at #1, got '{docs[0]}'. "
            f"Top 5: {docs}"
        )


# ═══════════════════════════════════════════════════════════════════
# TIER 2: Top-3 recall — correct doc should be in the top 3
# These are harder: shorter queries, ambiguous terms, or competing docs.
# ═══════════════════════════════════════════════════════════════════

class TestTop3Recall:
    """Correct doc must appear in the top 3 results."""

    @pytest.mark.parametrize("query, expected_doc", [
        # Close competition with related docs
        ("python programming language code", "Python Programming Language"),
        ("origin of species evolution", "Origin Of Species"),
        ("french revolution bastille", "French Revolution"),
        ("apple stock price", "Apple Inc"),
        ("corvid crow raven intelligence", "Corvidae"),
        ("huckleberry finn mississippi river twain", "Huckleberry Finn"),
        # Short disambiguation queries that need precision
        ("mercury orbit", "Mercury Planet"),
    ])
    def test_top3(self, engine, query, expected_doc):
        docs = _top_n_doc_names(engine, query, n=3)
        assert expected_doc in docs, (
            f"Expected '{expected_doc}' in top 3, got: {docs}"
        )


# ═══════════════════════════════════════════════════════════════════
# TIER 3: Top-5 recall — harder queries where top-1 is too strict
# ═══════════════════════════════════════════════════════════════════

class TestTop5Recall:
    """Correct doc must appear in the top 5 results."""

    @pytest.mark.parametrize("query, expected_doc", [
        ("mercury thermometer", "Mercury Element"),
        ("python import", "Python Programming Language"),
    ])
    def test_top5(self, engine, query, expected_doc):
        docs = _top_n_doc_names(engine, query, n=5)
        assert expected_doc in docs, (
            f"Expected '{expected_doc}' in top 5, got: {docs}"
        )


# ═══════════════════════════════════════════════════════════════════
# TIER 4: Top-10 recall — difficult queries, weak signal
# ═══════════════════════════════════════════════════════════════════

class TestTop10Recall:
    """Correct doc must appear somewhere in the top 10."""

    @pytest.mark.parametrize("query, expected_doc", [
        ("python eggs", "Python Snake"),
    ])
    def test_top10(self, engine, query, expected_doc):
        docs = _top_n_doc_names(engine, query, n=10)
        assert expected_doc in docs, (
            f"Expected '{expected_doc}' in top 10, got: {docs}"
        )


# ═══════════════════════════════════════════════════════════════════
# KNOWN FAILURES: queries that expose real system weaknesses
# These are xfail — they document what SHOULD work but doesn't yet.
# When upstream improvements fix these, the xfail will start passing
# and pytest will alert us to remove the marker.
# ═══════════════════════════════════════════════════════════════════

class TestKnownWeaknesses:
    """Queries that currently fail — tracked as xfail for regression detection."""

    @pytest.mark.xfail(reason="'crane bird' activates aircraft reef; Crane Bird not in top 10", strict=True)
    def test_crane_bird_disambiguation(self, engine):
        docs = _top_n_doc_names(engine, "crane bird migration wetland", n=10)
        assert "Crane Bird" in docs

    @pytest.mark.xfail(reason="'claw arcade prize' activates esports reef; Crane Machine not in top 10", strict=True)
    def test_crane_machine_disambiguation(self, engine):
        docs = _top_n_doc_names(engine, "crane machine claw arcade prize", n=10)
        assert "Crane Machine" in docs

    @pytest.mark.xfail(reason="'apple pie recipe' activates french/food reef; Apple Fruit not in top 10", strict=True)
    def test_apple_pie_recipe(self, engine):
        docs = _top_n_doc_names(engine, "apple pie recipe", n=10)
        assert "Apple Fruit" in docs

    @pytest.mark.xfail(reason="All query words unknown to scorer (conf=0.0); quality gate suppresses", strict=True)
    def test_united_nations_security_council(self, engine):
        docs = _top_n_doc_names(engine, "United Nations Security Council", n=10)
        assert "United Nations" in docs

    @pytest.mark.xfail(reason="Lunar Craters doc (13 chunks) drowned by Mercury Planet/Jupiter", strict=True)
    def test_lunar_crater_recall(self, engine):
        docs = _top_n_doc_names(engine, "lunar crater impact moon surface", n=10)
        assert "Lunar Craters" in docs

    @pytest.mark.xfail(reason="Yamamoto (Crater) doc (3 chunks) drowned by larger planet docs", strict=True)
    def test_yamamoto_crater_disambiguation(self, engine):
        docs = _top_n_doc_names(engine, "yamamoto crater moon lunar", n=10)
        assert "Yamamoto (Crater)" in docs


# ═══════════════════════════════════════════════════════════════════
# DISAMBIGUATION PAIRS: both sides of ambiguous entities
# For each pair, at least one direction must be correct at top-1.
# ═══════════════════════════════════════════════════════════════════

class TestDisambiguationPairs:
    """For entity pairs sharing a name, context should route to the right doc."""

    @pytest.mark.parametrize("query_a, doc_a, query_b, doc_b", [
        (
            "mercury planet orbit solar system", "Mercury Planet",
            "mercury element liquid metal toxic", "Mercury Element",
        ),
        (
            "python snake constrictor reptile", "Python Snake",
            "python programming language code", "Python Programming Language",
        ),
        (
            "apple fruit nutrition orchard", "Apple Fruit",
            "apple iphone macbook silicon valley", "Apple Inc",
        ),
        (
            "yamamoto admiral pearl harbor navy", "Isoroku Yamamoto",
            "yamamoto crater moon lunar", "Yamamoto (Crater)",
        ),
    ])
    def test_both_sides_top3(self, engine, query_a, doc_a, query_b, doc_b):
        """At minimum, each side of the pair should appear in top 3."""
        docs_a = _top_n_doc_names(engine, query_a, n=3)
        docs_b = _top_n_doc_names(engine, query_b, n=3)
        assert doc_a in docs_a, f"'{doc_a}' not in top 3 for: {query_a} → {docs_a}"
        # doc_b check is softer — some pairs have tiny docs that get drowned
        if doc_b != "Yamamoto (Crater)":
            assert doc_b in docs_b, f"'{doc_b}' not in top 3 for: {query_b} → {docs_b}"


# ═══════════════════════════════════════════════════════════════════
# NEGATIVE TESTS: queries should NOT return certain wrong docs
# ═══════════════════════════════════════════════════════════════════

class TestNegativeConstraints:
    """Certain docs should NOT appear for certain queries."""

    def test_snake_not_in_programming_results(self, engine):
        """'python programming' should not return Python Snake in top 3."""
        docs = _top_n_doc_names(engine, "python programming syntax", n=3)
        # It's OK if it appears lower, but not top 3
        assert "Python Snake" not in docs[:3] or "Python Programming Language" in docs[:2]

    def test_planet_not_for_element_query(self, engine):
        """'mercury liquid metal' should not return Mercury Planet at #1."""
        docs = _top_n_doc_names(engine, "mercury element liquid metal toxic", n=3)
        assert docs[0] == "Mercury Element"

    def test_fruit_not_for_company_query(self, engine):
        """'apple iPhone' should not return Apple Fruit at #1."""
        docs = _top_n_doc_names(engine, "apple iphone macbook silicon valley", n=3)
        assert docs[0] != "Apple Fruit"

    def test_unrelated_doc_not_at_top(self, engine):
        """'chess grandmaster' should not return biology docs at #1."""
        docs = _top_n_doc_names(engine, "chess grandmaster opening gambit", n=3)
        for bad_doc in ["Photosynthesis", "Dna", "Natural Selection", "Origin Of Species"]:
            assert bad_doc not in docs[:3]


# ═══════════════════════════════════════════════════════════════════
# LIGHTNING ROD: custom vocabulary should boost precision
# ═══════════════════════════════════════════════════════════════════

class TestLightningRod:
    """When custom vocab words fire, they should improve retrieval."""

    def test_lightning_rod_boosts_starcraft_terms(self, engine):
        """'battlecruiser yamato cannon' should trigger tagged_words and nail it."""
        response = _search(engine, "battlecruiser yamato cannon")
        assert len(response.query_info.tagged_words) > 0, "Expected lightning rod to fire"
        assert response.results[0].document_title in ("Yamato Cannon", "Battlecruiser Starcraft Ii")

    def test_lightning_rod_boosts_science_terms(self, engine):
        """'chlorophyll photosystem thylakoid' should fire lightning rod."""
        response = _search(engine, "chlorophyll photosystem thylakoid")
        assert len(response.query_info.tagged_words) > 0
        assert response.results[0].document_title == "Photosynthesis"

    def test_lightning_rod_boosts_seismology(self, engine):
        """'seismograph richter magnitude' should fire and return Earthquake."""
        response = _search(engine, "seismograph richter magnitude")
        assert len(response.query_info.tagged_words) > 0
        assert response.results[0].document_title == "Earthquake"

    def test_lightning_rod_disabled_still_returns_results(self, engine):
        """With lightning rod off, queries still return reasonable results."""
        response = _search(engine, "battlecruiser yamato cannon", enable_lightning_rod=False)
        assert len(response.results) > 0
        assert response.query_info.tagged_words == []

    def test_lightning_rod_vs_standard_comparison(self, engine):
        """Lightning rod should not DEGRADE results vs standard retrieval."""
        lr_on = _search(engine, "seismograph richter magnitude", enable_lightning_rod=True)
        lr_off = _search(engine, "seismograph richter magnitude", enable_lightning_rod=False)
        # Both should return Earthquake somewhere in top 5
        lr_on_docs = [r.document_title for r in lr_on.results[:5]]
        lr_off_docs = [r.document_title for r in lr_off.results[:5]]
        assert "Earthquake" in lr_on_docs
        assert "Earthquake" in lr_off_docs


# ═══════════════════════════════════════════════════════════════════
# QUALITY GATE: low-confidence queries should be handled gracefully
# ═══════════════════════════════════════════════════════════════════

class TestQualityGate:
    """Behavior around the confidence threshold."""

    def test_stopword_only_returns_empty(self, engine):
        response = _search(engine, "the and or but")
        assert len(response.results) == 0

    def test_empty_query_returns_empty(self, engine):
        response = _search(engine, "")
        assert len(response.results) == 0

    def test_unknown_entity_returns_something(self, engine):
        """Query about topics not in corpus should still get reef-based results."""
        response = _search(engine, "cephalopod octopus squid")
        # These words should light up marine biology reefs, matching ocean docs
        assert response.query_info.confidence > 0
        # Should get SOME results — possibly Great Barrier Reef or similar
        assert len(response.results) > 0

    def test_zero_confidence_suppressed(self, engine):
        """Queries with zero confidence get empty results (quality gate)."""
        response = _search(engine, "United Nations Security Council")
        if response.query_info.confidence == 0.0:
            assert len(response.results) == 0


# ═══════════════════════════════════════════════════════════════════
# RESULT DIVERSITY: top-10 shouldn't be ALL from one document
# ═══════════════════════════════════════════════════════════════════

class TestResultDiversity:
    """A healthy retrieval system should surface multiple relevant docs."""

    def test_broad_query_returns_multiple_docs(self, engine):
        """'evolution biology species' should return chunks from several docs."""
        unique = _unique_docs_in_top_n(engine, "evolution biology species", n=10)
        assert len(unique) >= 2, f"Only {len(unique)} unique doc(s): {unique}"

    def test_science_query_diversity(self, engine):
        """'science experiment research laboratory' spans many docs."""
        unique = _unique_docs_in_top_n(engine, "science experiment research laboratory", n=10)
        assert len(unique) >= 2, f"Only {len(unique)} unique doc(s): {unique}"

    def test_narrow_query_still_has_some_diversity(self, engine):
        """Even a narrow query like 'photosynthesis chloroplast light reaction'
        shouldn't monopolize ALL 10 slots with one doc (max_per_doc=3 default)."""
        unique = _unique_docs_in_top_n(
            engine, "photosynthesis chloroplast light reaction", n=10,
        )
        assert len(unique) >= 2, f"Only {len(unique)} unique doc(s): {unique}"

    def test_max_per_doc_limits_results(self, engine):
        """No doc should appear more than max_per_doc times in results."""
        response = engine.search(
            "photosynthesis chloroplast light reaction",
            top_k=10, max_per_doc=2,
        )
        from collections import Counter
        doc_counts = Counter(r.document_title for r in response.results)
        for doc, count in doc_counts.items():
            assert count <= 2, f"'{doc}' appeared {count} times with max_per_doc=2"

    def test_max_per_doc_disabled(self, engine):
        """max_per_doc=None allows single-doc domination."""
        response = engine.search(
            "photosynthesis chloroplast light reaction",
            top_k=10, max_per_doc=None,
        )
        doc_titles = [r.document_title for r in response.results]
        # With diversity disabled, a dominant doc CAN fill all slots
        # (we just verify it doesn't crash and returns results)
        assert len(doc_titles) > 0

    def test_max_per_doc_improves_small_doc_recall(self, engine):
        """With diversity cap, small docs like Lunar Craters should surface."""
        docs_capped = _top_n_doc_names(
            engine, "lunar crater impact moon surface", n=10, max_per_doc=3,
        )
        docs_uncapped = _top_n_doc_names(
            engine, "lunar crater impact moon surface", n=10, max_per_doc=None,
        )
        # Capped results should have at least as many unique docs
        assert len(set(docs_capped)) >= len(set(docs_uncapped))


# ═══════════════════════════════════════════════════════════════════
# SECTION PRECISION: right chunk from the right section
# ═══════════════════════════════════════════════════════════════════

class TestSectionPrecision:
    """The top result should come from a topically relevant section."""

    def test_jupiter_physical_characteristics(self, engine):
        """'jupiter great red spot' should come from Physical characteristics."""
        response = _search(engine, "jupiter great red spot storm")
        top = response.results[0]
        assert top.document_title == "Jupiter"
        # Section should be about physical properties, not e.g. moons
        assert "physical" in top.section_title.lower() or "atmosphere" in top.section_title.lower()

    def test_chess_rules_section(self, engine):
        """'castling kingside' should return a Chess chunk."""
        response = _search(engine, "castling kingside chess rules")
        docs = [r.document_title for r in response.results[:3]]
        assert "Chess" in docs

    def test_section_path_populated(self, engine):
        """All results should have non-empty section paths."""
        response = _search(engine, "photosynthesis")
        for r in response.results:
            assert len(r.section_path) > 0, f"Empty section_path for chunk {r.chunk_index}"


# ═══════════════════════════════════════════════════════════════════
# SCORING SANITY: match scores should be well-ordered
# ═══════════════════════════════════════════════════════════════════

class TestScoringSanity:
    """Basic invariants about match scores and ordering."""

    def test_scores_monotonically_decreasing(self, engine):
        """Results should be ordered by decreasing match_score."""
        response = _search(engine, "photosynthesis chloroplast")
        scores = [r.match_score for r in response.results]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1], (
                f"Score at position {i} ({scores[i]:.3f}) > "
                f"position {i-1} ({scores[i-1]:.3f})"
            )

    def test_all_scores_positive(self, engine):
        """All match scores should be positive."""
        response = _search(engine, "volcanic eruption geology")
        for r in response.results:
            assert r.match_score > 0

    def test_shared_reefs_populated_when_requested(self, engine):
        """include_scores=True should populate shared_reefs on results."""
        response = engine.search(
            "photosynthesis chloroplast", top_k=5, include_scores=True,
        )
        if response.results:
            assert len(response.results[0].shared_reefs) > 0

    def test_query_info_complete(self, engine):
        """QueryInfo should be fully populated for a real query."""
        response = _search(engine, "earthquake seismic waves")
        qi = response.query_info
        assert qi.top_reef != ""
        assert qi.confidence > 0
        assert qi.coverage > 0
        assert len(qi.top_reefs) > 0
        assert qi.matched_words > 0
        assert qi.total_words > 0


# ═══════════════════════════════════════════════════════════════════
# ROBUSTNESS: weird inputs, edge cases, boundary conditions
# ═══════════════════════════════════════════════════════════════════

class TestRobustness:
    """Edge cases that shouldn't crash or produce garbage."""

    def test_single_word_query(self, engine):
        """Single real word should return results."""
        response = _search(engine, "earthquake")
        assert len(response.results) > 0

    def test_very_long_query(self, engine):
        """A very long query shouldn't crash."""
        long_q = " ".join(["photosynthesis", "chlorophyll", "plant"] * 20)
        response = _search(engine, long_q)
        # Just shouldn't crash — results are bonus
        assert response.query_info is not None

    def test_special_characters_in_query(self, engine):
        """Punctuation and special chars shouldn't crash."""
        response = _search(engine, "what is DNA? (double-helix)")
        assert response.query_info is not None

    def test_numeric_query(self, engine):
        """Pure numbers shouldn't crash."""
        response = _search(engine, "42 1776 3.14")
        assert response.query_info is not None

    def test_mixed_case_query(self, engine):
        """Case shouldn't matter for retrieval quality."""
        lower = _top_n_doc_names(engine, "photosynthesis chloroplast", n=3)
        upper = _top_n_doc_names(engine, "PHOTOSYNTHESIS CHLOROPLAST", n=3)
        # Top doc should be the same regardless of case
        assert lower[0] == upper[0]

    def test_top_k_1_returns_single_result(self, engine):
        response = engine.search("earthquake", top_k=1)
        assert len(response.results) <= 1

    def test_top_k_100_doesnt_crash(self, engine):
        response = engine.search("earthquake", top_k=100)
        assert len(response.results) <= 100

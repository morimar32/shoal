"""Vocabulary extension for shoal — custom word discovery, injection, and tracking."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ._models import WordObservation

if TYPE_CHECKING:
    from lagoon import DocumentAnalysis, ReefScorer

    from ._storage import Storage

# ── Constants ──

_HIGH_CONFIDENCE_THRESHOLD = 1.5  # sentence confidence → sentence-heavy blend
_HIGH_SENTENCE_WEIGHT = 0.7
_LOW_SENTENCE_WEIGHT = 0.3
_MIN_OCCURRENCES_SHORT = 2   # docs < 500 words
_MIN_OCCURRENCES_LONG = 3    # docs >= 500 words
_DOCUMENT_LENGTH_THRESHOLD = 500
_ASSOCIATION_Z_THRESHOLD = 1.5
_RELATIVE_Z_CUTOFF = 0.3  # keep reefs with mean_z >= 30% of max mean_z
_MAX_ASSOCIATED_REEFS = 15
_MAX_REEFS_PER_OBSERVATION = 20
_REEF_SIZE_ADJUSTMENT_CAP = 3.0  # max boost for small reefs in size adjustment


def inject_custom_vocabulary_at_startup(scorer: ReefScorer, storage: Storage) -> int:
    """Load custom vocabulary from SQLite and inject into the scorer.

    Uses stored reef_weights (pre-computed weight_q values) directly,
    avoiding recalculation on every startup. Falls back to recomputing
    via calc_custom_weight if stored weights are all zero (old DB format).

    Returns the number of custom words injected.
    """
    custom_words = storage.load_custom_vocabulary()
    injected = 0
    for cw in custom_words:
        reef_weights = cw["reef_weights"]  # [(reef_id, bm25_q), ...]

        # Migration: if all weight_q are 0, recompute from associations
        if reef_weights and all(wq == 0 for _, wq in reef_weights):
            reef_weights = [
                (rid, scorer.calc_custom_weight(rid, strength))
                for rid, strength in cw["reef_associations"]
            ]

        try:
            scorer.add_custom_word(
                cw["word"],
                reef_weights,
                idf_q=cw["idf_q"],
                specificity=cw["specificity"],
                tag=cw["id"],
            )
            injected += 1
        except ValueError:
            # Word already in base vocab (hash collision or re-injection)
            pass
    return injected


def collect_word_observations(
    scorer: ReefScorer,
    analysis: DocumentAnalysis,
    doc_id: int,
    confidence_threshold: float = _HIGH_CONFIDENCE_THRESHOLD,
) -> list[WordObservation]:
    """Walk sentence results to collect observations of unknown words.

    For each unknown word in each sentence, blends sentence-level and
    chunk-level reef context to create a weighted observation.
    """
    observations: list[WordObservation] = []

    for segment in analysis.segments:
        # Chunk-level reefs from the segment topic
        chunk_reefs: dict[int, float] = {}
        for reef in segment.topic.top_reefs:
            if reef.z_score > 0:
                chunk_reefs[reef.reef_id] = reef.z_score

        for sent_result in segment.sentence_results:
            if not sent_result.unknown_words:
                continue

            # Sentence-level reefs
            sent_reefs: dict[int, float] = {}
            for reef in sent_result.top_reefs:
                if reef.z_score > 0:
                    sent_reefs[reef.reef_id] = reef.z_score

            # Blend weights based on sentence confidence
            if sent_result.confidence >= confidence_threshold:
                s_weight = _HIGH_SENTENCE_WEIGHT
                c_weight = _LOW_SENTENCE_WEIGHT
            else:
                s_weight = _LOW_SENTENCE_WEIGHT
                c_weight = _HIGH_SENTENCE_WEIGHT

            # Union all reef_ids from both levels
            all_reef_ids = set(sent_reefs.keys()) | set(chunk_reefs.keys())

            context_reefs = []
            for reef_id in all_reef_ids:
                sent_z = sent_reefs.get(reef_id, 0.0)
                chunk_z = chunk_reefs.get(reef_id, 0.0)
                weighted_z = s_weight * sent_z + c_weight * chunk_z
                if weighted_z > 0:
                    context_reefs.append({
                        "reef_id": reef_id,
                        "z_score": round(weighted_z, 4),
                        "weight": round(s_weight if sent_z > 0 else c_weight, 2),
                    })

            # Sort by z_score desc and cap
            context_reefs.sort(key=lambda r: r["z_score"], reverse=True)
            context_reefs = context_reefs[:_MAX_REEFS_PER_OBSERVATION]

            if not context_reefs:
                continue

            # Determine sentence_idx from segment position
            # sentence_results align with segment.sentences by index
            sent_idx = segment.start_idx + segment.sentence_results.index(sent_result)

            for word in sent_result.unknown_words:
                observations.append(WordObservation(
                    word=word,
                    document_id=doc_id,
                    sentence_idx=sent_idx,
                    context_reefs=context_reefs,
                ))

    return observations


def build_vocabulary(
    scorer: ReefScorer,
    storage: Storage,
    doc_id: int,
    total_word_count: int,
    association_z_threshold: float = _ASSOCIATION_Z_THRESHOLD,
) -> int:
    """Build vocabulary from word observations for a document.

    Algorithm:
    1. Determine occurrence threshold based on document length
    2. Load observations grouped by word
    3. For each word meeting threshold: compute mean reef z-scores,
       filter by threshold, compute specificity, inject into scorer
    4. Return count of new words

    Returns the number of new words added.
    """
    # 1. Occurrence threshold
    if total_word_count < _DOCUMENT_LENGTH_THRESHOLD:
        min_occurrences = _MIN_OCCURRENCES_SHORT
    else:
        min_occurrences = _MIN_OCCURRENCES_LONG

    # 2. Load observations grouped by word
    grouped = storage.get_word_observations_for_document(doc_id)

    n_new = 0
    for word, obs_list in grouped.items():
        # Check occurrence threshold
        if len(obs_list) < min_occurrences:
            continue

        # Skip if already known
        if scorer.lookup_word(word) is not None:
            continue

        # 3. Compute mean_z[reef_id] across all observations
        reef_z_sums: dict[int, float] = defaultdict(float)
        reef_z_counts: dict[int, int] = defaultdict(int)

        for context_reefs in obs_list:
            for entry in context_reefs:
                rid = entry["reef_id"]
                reef_z_sums[rid] += entry["z_score"]
                reef_z_counts[rid] += 1

        mean_z: dict[int, float] = {
            rid: reef_z_sums[rid] / reef_z_counts[rid]
            for rid in reef_z_sums
        }

        # Adjust for reef size: large reefs fire on everything,
        # small domain-specific reefs are more informative
        reef_wc = scorer.reef_word_counts
        avg_rw = scorer.avg_reef_words
        for rid in mean_z:
            nw = reef_wc[rid] if rid < len(reef_wc) else avg_rw
            adjustment = min(avg_rw / max(nw, 1), _REEF_SIZE_ADJUSTMENT_CAP)
            mean_z[rid] *= adjustment

        # Keep reefs where mean_z >= absolute threshold
        qualifying = {
            rid: mz for rid, mz in mean_z.items()
            if mz >= association_z_threshold
        }

        # Relative filter: also require mean_z >= 30% of the max
        if qualifying:
            peak_z = max(qualifying.values())
            relative_cutoff = peak_z * _RELATIVE_Z_CUTOFF
            qualifying = {
                rid: mz for rid, mz in qualifying.items()
                if mz >= relative_cutoff
            }

        # Cap at max reefs
        if len(qualifying) > _MAX_ASSOCIATED_REEFS:
            # Keep top by mean_z
            sorted_reefs = sorted(qualifying.items(), key=lambda x: x[1], reverse=True)
            qualifying = dict(sorted_reefs[:_MAX_ASSOCIATED_REEFS])

        # Must have at least 1 reef
        if len(qualifying) < 1:
            continue

        # Normalize strengths: strength = mean_z / max(mean_z)
        max_z = max(qualifying.values())
        reef_associations = [
            (rid, round(mz / max_z, 4))
            for rid, mz in qualifying.items()
        ]

        # Compute specificity from reef count
        n_reefs = len(reef_associations)
        if n_reefs <= 3:
            specificity = 2
        elif n_reefs <= 8:
            specificity = 1
        elif n_reefs <= 15:
            specificity = 0
        elif n_reefs <= 25:
            specificity = -1
        else:
            specificity = -2

        # Pre-compute IDF and per-reef calibrated weights
        idf_q = scorer.calc_custom_idf(n_reefs)
        reef_weights = [
            (rid, scorer.calc_custom_weight(rid, strength))
            for rid, strength in reef_associations
        ]

        # Inject into scorer with tag=0 (will update after persisting)
        try:
            word_info = scorer.add_custom_word(
                word, reef_weights,
                idf_q=idf_q, specificity=specificity, tag=0,
            )
        except ValueError:
            # Hash collision or word somehow became known
            continue

        # Build reef entries for storage: (reef_id, bm25_q, strength)
        reef_entries = [
            (rid, weight_q, strength)
            for (rid, weight_q), (_, strength)
            in zip(reef_weights, reef_associations)
        ]

        cw_id = storage.insert_custom_word(
            word=word,
            word_hash=word_info.word_hash,
            word_id=word_info.word_id,
            specificity=specificity,
            idf_q=word_info.idf_q,
            n_occurrences=len(obs_list),
            reef_entries=reef_entries,
        )

        # Update tag to the stable DB id
        scorer._word_tags[word_info.word_id] = cw_id

        n_new += 1

    return n_new


def get_custom_word_ids_for_chunk(
    scorer: ReefScorer,
    matched_word_ids: frozenset[int],
) -> dict[int, int]:
    """Get custom word DB ids for matched words in a chunk.

    Returns {word_id: custom_words_id} for words that have non-zero tags
    (i.e., custom vocabulary words, not base vocabulary).
    """
    if not matched_word_ids:
        return {}
    return scorer.get_word_tags(matched_word_ids)

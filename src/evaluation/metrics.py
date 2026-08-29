"""Ranking metrics implemented from first principles.

Evaluation is query-level: we score the ordered candidate list for one query,
then later average those query scores. Treating query-product rows as i.i.d.
classification examples would hide whether relevant products appear near the top.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.constants import BINARY_RELEVANT_LABELS, DEFAULT_K, RELEVANCE_GAIN

# DCG@k = sum_{i=1..k} gain_i / log2(i + 1)
# with gain_i taken from this project's RELEVANCE_GAIN mapping.
# Rank positions are 1-based, so the first item is discounted by log2(2) = 1.


def dcg_at_k(gains: Sequence[float], k: int = DEFAULT_K) -> float:
    """Discounted cumulative gain at cutoff k using logarithmic discount."""
    if k < 1:
        return 0.0
    values = np.asarray(list(gains)[:k], dtype=float)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum(values / discounts))


def ndcg_at_k(gains: Sequence[float], k: int = DEFAULT_K) -> float:
    """NDCG@k with linear gains and log2(rank + 1) discount.

    If the ideal DCG is 0 (no positive-gain items), the score is 0 rather than
    NaN. That treats an all-irrelevant candidate list as contributing nothing
    instead of pretending every ranking is perfect.
    """
    actual = dcg_at_k(gains, k)
    ideal = dcg_at_k(sorted(gains, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return float(actual / ideal)


def recall_at_k(
    labels: Sequence[str],
    k: int = DEFAULT_K,
    relevant_labels: frozenset[str] = BINARY_RELEVANT_LABELS,
) -> float:
    """Binary Recall@k.

    Relevant means E or S under this project's convention. If a query has no
    relevant candidates, recall is 0: there is nothing to retrieve.
    """
    if k < 1:
        return 0.0
    values = np.asarray(list(labels))
    n_relevant = int(np.isin(values, list(relevant_labels)).sum())
    if n_relevant == 0:
        return 0.0
    n_relevant_at_k = int(np.isin(values[:k], list(relevant_labels)).sum())
    return float(n_relevant_at_k / n_relevant)


def reciprocal_rank(
    labels: Sequence[str],
    relevant_labels: frozenset[str] = BINARY_RELEVANT_LABELS,
) -> float:
    """Reciprocal rank of the first E/S item. 0 if none exist."""
    for rank, label in enumerate(labels, start=1):
        if label in relevant_labels:
            return 1.0 / rank
    return 0.0


def gains_from_labels(labels: Sequence[str]) -> list[int]:
    """Map ESCI labels to this project's graded gains."""
    unknown = sorted({label for label in labels if label not in RELEVANCE_GAIN})
    if unknown:
        raise ValueError(f"Unknown ESCI labels: {unknown}.")
    return [RELEVANCE_GAIN[label] for label in labels]

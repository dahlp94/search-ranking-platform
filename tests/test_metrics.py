"""Hand-checked ranking metric tests on tiny lists."""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import dcg_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking_has_ndcg_one():
    gains = [3, 2, 1, 0]
    assert ndcg_at_k(gains, k=4) == 1.0
    assert ndcg_at_k(gains, k=10) == 1.0


def test_reversed_ranking_lowers_ndcg():
    gains = [3, 2, 1]
    reversed_gains = [1, 2, 3]
    expected_dcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    expected_idcg = expected_dcg
    expected_rev = 1 / math.log2(2) + 2 / math.log2(3) + 3 / math.log2(4)
    assert dcg_at_k(gains, k=3) == pytest.approx(expected_dcg)
    assert ndcg_at_k(reversed_gains, k=3) == pytest.approx(expected_rev / expected_idcg)
    assert ndcg_at_k(reversed_gains, k=3) < ndcg_at_k(gains, k=3)


def test_ndcg_zero_ideal_gain_is_zero_not_nan():
    assert ndcg_at_k([0, 0, 0], k=10) == 0.0
    assert ndcg_at_k([], k=10) == 0.0


def test_recall_at_k_hand_calculation():
    # Relevant = E or S. Labels [E, I, S, I]; two relevant items, one in top 2.
    labels = ["E", "I", "S", "I"]
    assert recall_at_k(labels, k=2) == 0.5
    assert recall_at_k(labels, k=3) == 1.0
    assert recall_at_k(labels, k=10) == 1.0


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k(["I", "C"], k=10) == 0.0


def test_mrr_hand_calculation():
    assert reciprocal_rank(["E", "I"]) == 1.0
    assert reciprocal_rank(["I", "S", "E"]) == 0.5
    assert reciprocal_rank(["I", "I", "C", "E"]) == 0.25


def test_mrr_with_no_relevant_items_is_zero():
    assert reciprocal_rank(["I", "C"]) == 0.0
    assert reciprocal_rank([]) == 0.0

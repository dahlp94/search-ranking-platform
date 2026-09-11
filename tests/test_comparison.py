"""Tests for deterministic ranking and paired query-level comparison."""

import pandas as pd
import pytest

from src.evaluation.comparison import paired_summary, rank_scored_candidates


def test_deterministic_within_query_ranking():
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "product_id": ["b", "a", "c"],
            "example_id": [2, 1, 3],
            "xgb_score": [1.0, 1.0, 0.5],
        }
    )

    ranked = rank_scored_candidates(
        frame.sample(frac=1, random_state=1)
    )

    assert list(ranked["product_id"]) == ["a", "b", "c"]
    assert list(ranked["predicted_rank"]) == [1, 2, 3]


def test_ranking_preserves_candidates():
    frame = pd.DataFrame(
        {
            "query_id": [7, 7, 8, 8],
            "product_id": ["p2", "p1", "p4", "p3"],
            "example_id": [2, 1, 4, 3],
            "xgb_score": [0.2, 0.9, 0.4, 0.4],
        }
    )

    ranked = rank_scored_candidates(frame)

    assert len(ranked) == len(frame)
    assert set(ranked["example_id"]) == set(frame["example_id"])

    query_8 = ranked[ranked["query_id"] == 8]
    assert list(query_8["product_id"]) == ["p3", "p4"]


def test_paired_summary_requires_matching_queries():
    left = pd.DataFrame(
        {
            "query_id": [1, 2],
            "ndcg@10": [0.8, 0.7],
        }
    )
    right = pd.DataFrame(
        {
            "query_id": [1, 3],
            "ndcg@10_tfidf": [0.6, 0.5],
        }
    )

    with pytest.raises(ValueError, match="Query ID sets do not match"):
        paired_summary(
            left,
            right,
            "ndcg@10",
            "ndcg@10_tfidf",
        )


def test_paired_summary_on_hand_checked_example():
    left = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "ndcg@10": [0.9, 0.5, 0.4],
        }
    )
    right = pd.DataFrame(
        {
            "query_id": [3, 1, 2],
            "ndcg@10_tfidf": [0.7, 0.8, 0.5],
        }
    )

    paired, summary = paired_summary(
        left,
        right,
        "ndcg@10",
        "ndcg@10_tfidf",
    )

    assert paired["delta"].tolist() == pytest.approx(
        [0.1, 0.0, -0.3]
    )

    assert summary["n_queries"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["ties"] == 1

    assert summary["mean_delta"] == pytest.approx(
        (0.1 + 0.0 - 0.3) / 3
    )
    assert summary["median_delta"] == pytest.approx(0.0)

    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["loss_rate"] == pytest.approx(1 / 3)
    assert summary["tie_rate"] == pytest.approx(1 / 3)

    assert (
        summary["wins"]
        + summary["losses"]
        + summary["ties"]
        == summary["n_queries"]
    )
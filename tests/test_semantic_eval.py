"""Tests for saved-ranker comparison."""

import pandas as pd
import pytest

from src.constants import SEED
from src.evaluation.bootstrap import paired_bootstrap_mean_ci
from src.evaluation.ranker_scoring import (
    RankerEvalError,
    assert_same_candidates,
    assert_same_queries,
    paired_metric_table,
    score_saved_ranker,
)
from src.features.build import FEATURES
from src.ranking.groups import prepare_grouped_data
from src.ranking.ranker import (
    build_ranker,
    save_ranker,
    train_ranker,
)


def ranking_frame(query_ids, start_example=1):
    rows = []
    example_id = start_example

    for query_id in query_ids:
        for rank in range(4):
            row = {
                "example_id": example_id,
                "query_id": query_id,
                "product_id": f"p{example_id}",
                "query": f"query {query_id}",
                "relevance_gain": 3 - rank,
            }

            for i, feature in enumerate(FEATURES):
                row[feature] = float(4 - rank + i * 0.1)

            rows.append(row)
            example_id += 1

    return pd.DataFrame(rows)


def save_tiny_ranker(tmp_path):
    train = prepare_grouped_data(
        ranking_frame([1, 2]),
        features=FEATURES,
    )

    validation = prepare_grouped_data(
        ranking_frame([3], start_example=100),
        features=FEATURES,
    )

    ranker = train_ranker(
        train,
        validation,
        ranker=build_ranker(n_estimators=5),
        verbose=False,
    )

    path = tmp_path / "ranker.json"
    save_ranker(ranker, path)

    return path, validation.frame


def test_saved_ranker_scores_without_retraining(tmp_path, monkeypatch):
    path, frame = save_tiny_ranker(tmp_path)

    def fail_fit(*args, **kwargs):
        raise AssertionError("Evaluation must not retrain the model.")

    monkeypatch.setattr(
        "xgboost.XGBRanker.fit",
        fail_fit,
    )

    ranked = score_saved_ranker(
        path,
        frame,
        FEATURES,
    )

    assert len(ranked) == len(frame)


def test_candidate_and_query_sets_must_match():
    left = ranking_frame([1, 2])
    right = ranking_frame([1, 2], start_example=100)

    assert_same_queries(left, right)

    with pytest.raises(RankerEvalError):
        assert_same_queries(
            left,
            ranking_frame([1, 3]),
        )

    same_candidates = left.copy()
    assert_same_candidates(left, same_candidates)

    changed = same_candidates.copy()
    changed.loc[0, "example_id"] = 999

    with pytest.raises(RankerEvalError):
        assert_same_candidates(left, changed)


def test_paired_metric_differences():
    semantic = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": ["a", "b", "c"],
            "n_candidates": [4, 4, 4],
            "ndcg@10": [0.9, 0.5, 0.4],
            "recall@10": [1.0, 0.5, 0.5],
            "mrr": [1.0, 0.5, 0.5],
        }
    )

    reference = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": ["a", "b", "c"],
            "n_candidates": [4, 4, 4],
            "ndcg@10": [0.8, 0.5, 0.7],
            "recall@10": [1.0, 0.5, 0.5],
            "mrr": [1.0, 0.5, 0.5],
        }
    )

    paired = paired_metric_table(
        semantic,
        reference,
        "semantic",
        "reference",
        k=10,
    )

    assert paired["delta_ndcg_at_10"].tolist() == pytest.approx(
        [0.1, 0.0, -0.3]
    )


def test_paired_bootstrap_is_reproducible():
    deltas = [0.1, 0.0, -0.1, 0.05]

    first = paired_bootstrap_mean_ci(
        deltas,
        n_replicates=200,
        seed=SEED,
    )

    second = paired_bootstrap_mean_ci(
        deltas,
        n_replicates=200,
        seed=SEED,
    )

    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]
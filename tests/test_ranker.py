"""Tests for the XGBoost learning-to-rank model."""

import pandas as pd

from src.constants import SEED
from src.features.build import FEATURES
from src.ranking.groups import prepare_grouped_data
from src.ranking.ranker import (
    RANKER_PARAMS,
    build_ranker,
    load_ranker,
    predict_scores,
    predictions_equivalent,
    save_ranker,
    train_ranker,
)


def make_frame(query_ids):
    rows = []
    example_id = 1

    for query_id in query_ids:
        for rank in range(4):
            row = {
                "example_id": example_id,
                "query_id": query_id,
                "relevance_gain": 3 - rank,
            }

            for i, feature in enumerate(FEATURES):
                row[feature] = float(4 - rank + i * 0.1)

            rows.append(row)
            example_id += 1

    return pd.DataFrame(rows)


def test_ranker_trains_saves_and_reloads(tmp_path):
    train = prepare_grouped_data(make_frame([1, 2]))
    validation = prepare_grouped_data(make_frame([3]))

    ranker = train_ranker(
        train,
        validation,
        ranker=build_ranker(n_estimators=5),
        verbose=False,
    )

    original_scores = predict_scores(ranker, validation)

    assert len(original_scores) == validation.n_rows

    model_path = tmp_path / "ranker.json"
    save_ranker(ranker, model_path)

    reloaded = load_ranker(model_path)
    reloaded_scores = predict_scores(reloaded, validation)

    assert model_path.exists()
    assert predictions_equivalent(original_scores, reloaded_scores)


def test_ranker_configuration():
    ranker = build_ranker()
    params = ranker.get_params()

    assert params["objective"] == "rank:ndcg"
    assert params["eval_metric"] == "ndcg@10"
    assert params["random_state"] == SEED
    assert params["n_estimators"] == RANKER_PARAMS["n_estimators"] == 200

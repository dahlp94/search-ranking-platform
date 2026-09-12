"""Tests for semantic ranking features and training."""

import pandas as pd
import pytest

from src.features.build import FEATURES
from src.features.semantic import (
    SEMANTIC_FEATURE,
    SEMANTIC_FEATURES,
    SemanticJoinError,
    join_semantic_similarity,
)
from src.ranking.groups import prepare_grouped_data
from src.ranking.ranker import (
    build_ranker,
    load_ranker,
    predict_scores,
    predictions_equivalent,
    save_ranker,
    train_ranker,
)


def feature_frame(query_ids, start_example=1, include_semantic=True):
    rows = []
    example_id = start_example

    for query_id in query_ids:
        for rank in range(4):
            row = {
                "example_id": example_id,
                "query_id": query_id,
                "product_id": f"p{example_id}",
                "relevance_gain": 3 - rank,
            }

            for i, feature in enumerate(FEATURES):
                row[feature] = float(4 - rank + i * 0.1)

            if include_semantic:
                row[SEMANTIC_FEATURE] = 0.9 - 0.15 * rank

            rows.append(row)
            example_id += 1

    return pd.DataFrame(rows)


def semantic_table(frame):
    return frame[
        ["example_id", "query_id", "product_id", SEMANTIC_FEATURE]
    ].copy()


def test_semantic_feature_set():
    assert len(FEATURES) == 10
    assert SEMANTIC_FEATURES == [*FEATURES, SEMANTIC_FEATURE]
    assert len(SEMANTIC_FEATURES) == 11


def test_semantic_join_preserves_rows_and_identity():
    features = feature_frame([1, 2], include_semantic=False)
    semantic = semantic_table(feature_frame([1, 2]))

    joined, _ = join_semantic_similarity(features, semantic)

    assert len(joined) == len(features)
    assert joined["example_id"].tolist() == features["example_id"].tolist()
    assert SEMANTIC_FEATURE in joined.columns


def test_semantic_join_rejects_missing_or_duplicate_matches():
    features = feature_frame([1], include_semantic=False)
    semantic = semantic_table(feature_frame([1]))

    with pytest.raises(SemanticJoinError):
        join_semantic_similarity(
            features,
            semantic.iloc[:-1],
        )

    duplicated = pd.concat(
        [semantic, semantic.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(SemanticJoinError):
        join_semantic_similarity(
            features,
            duplicated,
        )


def test_grouped_data_uses_semantic_features():
    grouped = prepare_grouped_data(
        feature_frame([2, 1]),
        features=SEMANTIC_FEATURES,
    )

    assert grouped.n_rows == 8
    assert grouped.n_groups == 2
    assert grouped.group_sizes.tolist() == [4, 4]
    assert list(grouped.X.columns) == SEMANTIC_FEATURES


def test_semantic_ranker_trains_and_predicts():
    train = prepare_grouped_data(
        feature_frame([1, 2]),
        features=SEMANTIC_FEATURES,
    )

    validation = prepare_grouped_data(
        feature_frame([3], start_example=100),
        features=SEMANTIC_FEATURES,
    )

    ranker = train_ranker(
        train,
        validation,
        ranker=build_ranker(n_estimators=5),
        verbose=False,
    )

    scores = predict_scores(ranker, validation)

    assert len(scores) == validation.n_rows


def test_semantic_ranker_save_reload(tmp_path):
    train = prepare_grouped_data(
        feature_frame([1, 2]),
        features=SEMANTIC_FEATURES,
    )

    validation = prepare_grouped_data(
        feature_frame([3], start_example=100),
        features=SEMANTIC_FEATURES,
    )

    ranker = train_ranker(
        train,
        validation,
        ranker=build_ranker(n_estimators=5),
        verbose=False,
    )

    original = predict_scores(ranker, validation)

    path = tmp_path / "xgb_ranker_semantic.json"
    save_ranker(ranker, path)

    reloaded = load_ranker(path)

    assert predictions_equivalent(
        original,
        predict_scores(reloaded, validation),
    )
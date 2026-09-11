"""Tests for query grouping used by XGBRanker."""

import pandas as pd
import pytest

from src.features.build import FEATURES
from src.ranking.groups import (
    RankingGroupError,
    assert_queries_disjoint,
    prepare_grouped_data,
)


def make_frame(query_ids, candidates_per_query=3):
    rows = []
    example_id = 1

    for query_id in query_ids:
        for rank in range(candidates_per_query):
            row = {
                "example_id": example_id,
                "query_id": query_id,
                "relevance_gain": 3 - rank,
            }

            for i, feature in enumerate(FEATURES):
                row[feature] = float(rank + i)

            rows.append(row)
            example_id += 1

    return pd.DataFrame(rows)


def test_prepare_grouped_data():
    frame = make_frame([2, 1], candidates_per_query=3)
    grouped = prepare_grouped_data(frame)

    assert grouped.query_ids.tolist() == [1, 1, 1, 2, 2, 2]
    assert grouped.group_sizes.tolist() == [3, 3]
    assert grouped.n_rows == 6
    assert grouped.n_groups == 2
    assert list(grouped.X.columns) == list(FEATURES)


def test_grouping_is_independent_of_input_order():
    frame = make_frame([1, 2])
    shuffled = frame.sample(frac=1, random_state=0)

    first = prepare_grouped_data(frame)
    second = prepare_grouped_data(shuffled)

    assert first.example_ids.tolist() == second.example_ids.tolist()
    assert first.X.equals(second.X)


def test_train_validation_queries_must_be_disjoint():
    train = make_frame([1, 2])
    validation = make_frame([3, 4])

    assert assert_queries_disjoint(train, validation) == 0

    with pytest.raises(RankingGroupError):
        assert_queries_disjoint(train, make_frame([2, 3]))


def test_only_approved_features_are_allowed():
    frame = make_frame([1])

    with pytest.raises(RankingGroupError):
        prepare_grouped_data(frame, features=[*FEATURES, "query_id"])

    with pytest.raises(RankingGroupError):
        prepare_grouped_data(frame, features=FEATURES[:-1])


def test_required_columns_must_exist():
    frame = make_frame([1]).drop(columns=["relevance_gain"])

    with pytest.raises(RankingGroupError):
        prepare_grouped_data(frame)

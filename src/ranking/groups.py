"""Prepare query-grouped data for learning-to-rank."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build import FEATURES
from src.features.semantic import SEMANTIC_FEATURES


class RankingGroupError(ValueError):
    """Raised when ranking data is invalid."""


@dataclass(frozen=True)
class GroupedRankingData:
    frame: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    group_sizes: np.ndarray
    features: tuple[str, ...]

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_groups(self) -> int:
        return len(self.group_sizes)

    @property
    def example_ids(self) -> np.ndarray:
        return self.frame["example_id"].to_numpy()

    @property
    def query_ids(self) -> np.ndarray:
        return self.frame["query_id"].to_numpy()


APPROVED_FEATURE_SETS = {
    tuple(FEATURES),
    tuple(SEMANTIC_FEATURES),
}


def assert_queries_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Fail if a query appears in both partitions."""
    overlap = set(train["query_id"]) & set(validation["query_id"])

    if overlap:
        raise RankingGroupError(
            f"Train and validation overlap on {len(overlap)} queries."
        )


def validate_model_features(features) -> list[str]:
    """Require one of the approved full ranking feature sets."""
    features = tuple(features)

    if features not in APPROVED_FEATURE_SETS:
        raise RankingGroupError(
            "Features must match the approved lexical or semantic feature set."
        )

    return list(features)


def validate_feature_subset(features) -> list[str]:
    """Validate a non-empty subset of lexical ranking features."""
    features = list(features)

    if not features:
        raise RankingGroupError("Feature subset is empty.")

    unknown = [feature for feature in features if feature not in FEATURES]

    if unknown:
        raise RankingGroupError(
            f"Unknown or disallowed ranking features: {unknown}."
        )

    return [feature for feature in FEATURES if feature in features]


def _prepare(
    df: pd.DataFrame,
    features: list[str],
) -> GroupedRankingData:
    """Sort by query and build XGBoost ranking groups."""
    required = [
        "query_id",
        "example_id",
        "relevance_gain",
        *features,
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RankingGroupError(
            f"Missing required columns: {missing}"
        )

    ordered = (
        df.sort_values(
            ["query_id", "example_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    group_sizes = (
        ordered.groupby("query_id", sort=False)
        .size()
        .to_numpy(dtype=np.int64)
    )

    return GroupedRankingData(
        frame=ordered,
        X=ordered[features].copy(),
        y=ordered["relevance_gain"].to_numpy(dtype=float),
        group_sizes=group_sizes,
        features=tuple(features),
    )


def prepare_grouped_data(
    df: pd.DataFrame,
    features=None,
) -> GroupedRankingData:
    """Prepare data using an approved full feature set."""
    features = validate_model_features(
        FEATURES if features is None else features
    )

    return _prepare(df, features)


def prepare_grouped_subset(
    df: pd.DataFrame,
    features,
) -> GroupedRankingData:
    """Prepare data using a lexical feature subset."""
    return _prepare(
        df,
        validate_feature_subset(features),
    )
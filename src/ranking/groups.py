"""Prepare query-grouped data for learning-to-rank."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build import FEATURES


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


def validate_model_features(features) -> list[str]:
    """Require exactly the approved ranking features."""
    features = list(features)

    if features != list(FEATURES):
        raise RankingGroupError(
            f"Expected features {list(FEATURES)}, got {features}."
        )

    return features


def assert_queries_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> int:
    """Fail if a query appears in both partitions."""
    overlap = set(train["query_id"]) & set(validation["query_id"])

    if overlap:
        raise RankingGroupError(
            f"Train and validation overlap on {len(overlap)} queries."
        )

    return 0


def prepare_grouped_data(
    df: pd.DataFrame,
    features=None,
) -> GroupedRankingData:
    """Sort rows by query and prepare XGBoost ranking groups."""
    features = validate_model_features(
        FEATURES if features is None else features
    )

    required = ["query_id", "example_id", "relevance_gain", *features]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise RankingGroupError(f"Missing required columns: {missing}")

    ordered = (
        df.sort_values(["query_id", "example_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    X = ordered[features].copy()
    y = ordered["relevance_gain"].to_numpy(dtype=float)

    group_sizes = (
        ordered.groupby("query_id", sort=False)
        .size()
        .to_numpy(dtype=np.int64)
    )

    return GroupedRankingData(
        frame=ordered,
        X=X,
        y=y,
        group_sizes=group_sizes,
        features=tuple(features),
    )
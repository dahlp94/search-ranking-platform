"""Build ranking feature tables."""

import numpy as np
import pandas as pd

from src.features.features import add_ranking_features


FEATURES = [
    "bm25_score",
    "tfidf_score",
    "shared_token_count",
    "query_token_coverage",
    "title_token_coverage",
    "exact_query_in_title",
    "brand_match",
    "title_token_count",
    "token_length_difference",
    "query_bullet_coverage",
]


KEEP_COLUMNS = [
    "example_id",
    "query_id",
    "product_id",
    "query",
    "product_title",
    "esci_label",
    "relevance_gain",
    *FEATURES,
]


def build_feature_table(
    df: pd.DataFrame,
    cached_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one ranking-feature table."""

    featured = add_ranking_features(
        df,
        cached_scores=cached_scores,
    )

    missing = [
        col
        for col in KEEP_COLUMNS
        if col not in featured.columns
    ]

    if missing:
        raise KeyError(
            f"Missing feature-table columns: {missing}"
        )

    table = featured[KEEP_COLUMNS].copy()

    if len(table) != len(df):
        raise ValueError(
            "Feature construction changed the row count."
        )

    if (
        table["example_id"].tolist()
        != df["example_id"].tolist()
    ):
        raise ValueError(
            "Feature construction changed row identity."
        )

    values = table[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Feature table contains missing or non-finite values."
        )

    return table


def build_feature_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_scores: pd.DataFrame | None = None,
    validation_scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train and validation ranking features."""

    overlap = (
        set(train["query_id"])
        & set(validation["query_id"])
    )

    if overlap:
        raise ValueError(
            "Train and validation queries overlap."
        )

    train_features = build_feature_table(
        train,
        cached_scores=train_scores,
    )

    validation_features = build_feature_table(
        validation,
        cached_scores=validation_scores,
    )

    return train_features, validation_features
"""Join semantic similarity onto ranking features."""

import numpy as np
import pandas as pd

from src.features.build import FEATURES


SEMANTIC_FEATURE = "semantic_similarity"
SEMANTIC_FEATURES = [*FEATURES, SEMANTIC_FEATURE]

JOIN_KEY = "example_id"
IDENTITY_COLUMNS = ["example_id", "query_id", "product_id"]


class SemanticJoinError(ValueError):
    """Raised when semantic features cannot be joined safely."""


def join_semantic_similarity(
    features: pd.DataFrame,
    semantic: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Join semantic_similarity to the ranking feature table."""

    required_features = IDENTITY_COLUMNS
    required_semantic = [*IDENTITY_COLUMNS, SEMANTIC_FEATURE]

    missing_features = [
        col for col in required_features
        if col not in features.columns
    ]
    missing_semantic = [
        col for col in required_semantic
        if col not in semantic.columns
    ]

    if missing_features:
        raise SemanticJoinError(
            f"Feature table missing columns: {missing_features}"
        )

    if missing_semantic:
        raise SemanticJoinError(
            f"Semantic table missing columns: {missing_semantic}"
        )

    if features[JOIN_KEY].duplicated().any():
        raise SemanticJoinError(
            "Feature table contains duplicate example_id values."
        )

    if semantic[JOIN_KEY].duplicated().any():
        raise SemanticJoinError(
            "Semantic table contains duplicate example_id values."
        )

    semantic = semantic[
        [*IDENTITY_COLUMNS, SEMANTIC_FEATURE]
    ].copy()

    merged = features.merge(
        semantic,
        on=JOIN_KEY,
        how="left",
        suffixes=("", "_semantic"),
        validate="one_to_one",
    )

    if len(merged) != len(features):
        raise SemanticJoinError(
            "Semantic join changed the row count."
        )

    if merged[SEMANTIC_FEATURE].isna().any():
        raise SemanticJoinError(
            "Some candidates are missing semantic_similarity."
        )

    values = merged[SEMANTIC_FEATURE].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise SemanticJoinError(
            "semantic_similarity contains non-finite values."
        )

    for column in ("query_id", "product_id"):
        semantic_column = f"{column}_semantic"

        if not merged[column].equals(merged[semantic_column]):
            raise SemanticJoinError(
                f"{column} mismatch after semantic join."
            )

        merged = merged.drop(columns=semantic_column)

    report = {
        "rows": len(merged),
        "missing_matches": 0,
        "join_key": JOIN_KEY,
    }

    return merged, report
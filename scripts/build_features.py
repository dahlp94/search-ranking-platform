#!/usr/bin/env python3
"""Build train and validation ranking features."""

from pathlib import Path

import pandas as pd

from src.data.load_data import PROCESSED_DIR
from src.features.build import build_feature_datasets


OUTPUT_DIR = Path("artifacts/features")

TRAIN_FEATURES = OUTPUT_DIR / "train_features.parquet"
VALIDATION_FEATURES = OUTPUT_DIR / "validation_features.parquet"


def load_cached_scores(path: Path):
    """Load existing BM25/TF-IDF scores if available."""

    if not path.exists():
        return None

    print(f"Using cached lexical scores from {path}")

    return pd.read_parquet(
        path,
        columns=[
            "example_id",
            "bm25_score",
            "tfidf_score",
        ],
    )


def main():
    train = pd.read_parquet(
        PROCESSED_DIR / "project_train.parquet"
    )

    validation = pd.read_parquet(
        PROCESSED_DIR / "project_validation.parquet"
    )

    # Reuse lexical scores from a previous feature build.
    train_scores = load_cached_scores(
        TRAIN_FEATURES
    )

    validation_scores = load_cached_scores(
        VALIDATION_FEATURES
    )

    train_features, validation_features = (
        build_feature_datasets(
            train,
            validation,
            train_scores=train_scores,
            validation_scores=validation_scores,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_features.to_parquet(
        TRAIN_FEATURES,
        index=False,
    )

    validation_features.to_parquet(
        VALIDATION_FEATURES,
        index=False,
    )

    print(
        f"Train: {len(train_features):,} rows, "
        f"{train_features['query_id'].nunique():,} queries"
    )

    print(
        f"Validation: {len(validation_features):,} rows, "
        f"{validation_features['query_id'].nunique():,} queries"
    )


if __name__ == "__main__":
    main()
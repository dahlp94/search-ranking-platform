#!/usr/bin/env python3
"""Add semantic similarity to ranking feature tables."""

from pathlib import Path

import pandas as pd

from src.features.semantic import (
    SEMANTIC_FEATURE,
    join_semantic_similarity,
)
from src.ranking.groups import assert_queries_disjoint


ROOT = Path(__file__).resolve().parents[1]

TRAIN_FEATURES = ROOT / "artifacts/features/train_features.parquet"
VALIDATION_FEATURES = ROOT / "artifacts/features/validation_features.parquet"

TRAIN_SEMANTIC = ROOT / "artifacts/semantic/train_semantic.parquet"
VALIDATION_SEMANTIC = ROOT / "artifacts/semantic/validation_semantic.parquet"

OUTPUT_TRAIN = ROOT / "artifacts/features/train_features_semantic.parquet"
OUTPUT_VALIDATION = ROOT / "artifacts/features/validation_features_semantic.parquet"


def load_table(path: Path) -> pd.DataFrame:
    """Load a development artifact."""
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError("Official ESCI test rows are not allowed.")

    return df


def build_table(
    feature_path: Path,
    semantic_path: Path,
) -> pd.DataFrame:
    """Join semantic similarity onto one feature table."""

    features = load_table(feature_path)
    semantic = load_table(semantic_path)

    if SEMANTIC_FEATURE in features.columns:
        raise ValueError(
            f"{SEMANTIC_FEATURE} already exists in feature table."
        )

    joined, _ = join_semantic_similarity(
        features,
        semantic,
    )

    return joined


def main() -> None:
    train = build_table(
        TRAIN_FEATURES,
        TRAIN_SEMANTIC,
    )

    validation = build_table(
        VALIDATION_FEATURES,
        VALIDATION_SEMANTIC,
    )

    assert_queries_disjoint(train, validation)

    OUTPUT_TRAIN.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        OUTPUT_TRAIN,
        index=False,
    )

    validation.to_parquet(
        OUTPUT_VALIDATION,
        index=False,
    )

    print(
        f"Train: {len(train):,} rows / "
        f"{train['query_id'].nunique():,} queries"
    )

    print(
        f"Validation: {len(validation):,} rows / "
        f"{validation['query_id'].nunique():,} queries"
    )

    print(f"Wrote {OUTPUT_TRAIN}")
    print(f"Wrote {OUTPUT_VALIDATION}")
    print("Official ESCI test used: NO")


if __name__ == "__main__":
    main()
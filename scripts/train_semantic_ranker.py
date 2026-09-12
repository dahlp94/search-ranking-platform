#!/usr/bin/env python3
"""Train the semantic-augmented XGBRanker."""

from pathlib import Path

import pandas as pd

from src.features.semantic import SEMANTIC_FEATURES
from src.ranking.groups import assert_queries_disjoint, prepare_grouped_data
from src.ranking.ranker import (
    RANKER_PARAMS,
    load_ranker,
    predict_scores,
    predictions_equivalent,
    save_ranker,
    train_ranker,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "artifacts/features/train_features_semantic.parquet"
VALIDATION_PATH = ROOT / "artifacts/features/validation_features_semantic.parquet"

MODEL_PATH = ROOT / "artifacts/models/xgb_ranker_semantic.json"


def load_features(path: Path) -> pd.DataFrame:
    """Load a semantic ranking feature table."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature table: {path}\n"
            "Run `python -m scripts.build_semantic_ltr_features` first."
        )

    df = pd.read_parquet(path)

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError("Official ESCI test rows are not allowed.")

    missing = [
        feature
        for feature in SEMANTIC_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ranking features: {missing}"
        )

    return df


def main() -> None:
    train = load_features(TRAIN_PATH)
    validation = load_features(VALIDATION_PATH)

    assert_queries_disjoint(train, validation)

    train_data = prepare_grouped_data(
        train,
        features=SEMANTIC_FEATURES,
    )

    validation_data = prepare_grouped_data(
        validation,
        features=SEMANTIC_FEATURES,
    )

    print(
        f"Train: {train_data.n_rows:,} rows / "
        f"{train_data.n_groups:,} queries"
    )

    print(
        f"Validation: {validation_data.n_rows:,} rows / "
        f"{validation_data.n_groups:,} queries"
    )

    print(
        f"Features ({len(SEMANTIC_FEATURES)}): "
        f"{SEMANTIC_FEATURES}"
    )

    ranker = train_ranker(
        train_data,
        validation_data,
    )

    save_ranker(
        ranker,
        MODEL_PATH,
    )

    reloaded = load_ranker(MODEL_PATH)

    reload_ok = predictions_equivalent(
        predict_scores(ranker, validation_data),
        predict_scores(reloaded, validation_data),
    )

    if not reload_ok:
        raise RuntimeError(
            "Reloaded model predictions do not match."
        )

    print(f"Saved model: {MODEL_PATH}")
    print(f"Objective: {RANKER_PARAMS['objective']}")
    print("Reload check: PASS")
    print("Official ESCI test used: NO")


if __name__ == "__main__":
    main()
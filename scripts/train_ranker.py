#!/usr/bin/env python3
"""Train and save the query-grouped XGBRanker."""

import sys
from pathlib import Path

import pandas as pd
import xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import write_json
from src.features.build import FEATURES
from src.ranking.groups import assert_queries_disjoint, prepare_grouped_data
from src.ranking.ranker import (
    RANKER_PARAMS,
    eval_history_final,
    load_ranker,
    predict_scores,
    predictions_equivalent,
    save_ranker,
    train_ranker,
)


TRAIN_PATH = PROJECT_ROOT / "artifacts/features/train_features.parquet"
VALIDATION_PATH = PROJECT_ROOT / "artifacts/features/validation_features.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts/models/xgb_ranker.json"
SUMMARY_PATH = PROJECT_ROOT / "artifacts/metrics/ranker_training_summary.json"


def load_features(path: Path) -> pd.DataFrame:
    """Load an approved development feature table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {path}\n"
            "Run `python scripts/build_features.py` first."
        )

    df = pd.read_parquet(path)

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError("Official ESCI test rows are not allowed during training.")

    return df


def main() -> None:
    train = load_features(TRAIN_PATH)
    validation = load_features(VALIDATION_PATH)

    assert_queries_disjoint(train, validation)

    train_data = prepare_grouped_data(train)
    validation_data = prepare_grouped_data(validation)

    print(
        f"Train: {train_data.n_rows:,} rows / {train_data.n_groups:,} queries"
    )
    print(
        f"Validation: {validation_data.n_rows:,} rows / "
        f"{validation_data.n_groups:,} queries"
    )

    ranker = train_ranker(train_data, validation_data)
    diagnostics = eval_history_final(ranker)

    save_ranker(ranker, MODEL_PATH)

    reloaded = load_ranker(MODEL_PATH)
    reload_ok = predictions_equivalent(
        predict_scores(ranker, validation_data),
        predict_scores(reloaded, validation_data),
    )

    if not reload_ok:
        raise RuntimeError("Reloaded model predictions do not match.")

    summary = {
        "model_type": "XGBRanker",
        "xgboost_version": xgboost.__version__,
        "objective": RANKER_PARAMS["objective"],
        "eval_metric": RANKER_PARAMS["eval_metric"],
        "hyperparameters": RANKER_PARAMS,
        "features": list(FEATURES),
        "train_rows": train_data.n_rows,
        "train_queries": train_data.n_groups,
        "validation_rows": validation_data.n_rows,
        "validation_queries": validation_data.n_groups,
        **diagnostics,
        "model_path": str(MODEL_PATH),
        "reload_check": reload_ok,
        "official_test_used": False,
    }

    write_json(SUMMARY_PATH, summary)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
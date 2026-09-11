"""Train and persist the XGBoost learning-to-rank model."""

from pathlib import Path

import numpy as np
from xgboost import XGBRanker

from src.constants import SEED
from src.ranking.groups import GroupedRankingData


RANKER_PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": SEED,
}


def build_ranker(**overrides) -> XGBRanker:
    """Create the project ranker."""
    return XGBRanker(**{**RANKER_PARAMS, **overrides})


def train_ranker(
    train: GroupedRankingData,
    validation: GroupedRankingData,
    ranker: XGBRanker | None = None,
    verbose: bool | int = 25,
) -> XGBRanker:
    """Train XGBRanker using query-group sizes."""
    ranker = ranker or build_ranker()

    ranker.fit(
        train.X,
        train.y,
        group=train.group_sizes,
        eval_set=[
            (train.X, train.y),
            (validation.X, validation.y),
        ],
        eval_group=[
            train.group_sizes,
            validation.group_sizes,
        ],
        verbose=verbose,
    )

    return ranker


def predict_scores(
    ranker: XGBRanker,
    data: GroupedRankingData,
) -> np.ndarray:
    """Return one ranking score per candidate."""
    return np.asarray(ranker.predict(data.X), dtype=float)


def save_ranker(ranker: XGBRanker, path: Path) -> Path:
    """Save a trained ranker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(path)
    return path


def load_ranker(path: Path) -> XGBRanker:
    """Load a saved ranker."""
    ranker = XGBRanker()
    ranker.load_model(path)
    return ranker


def predictions_equivalent(
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    """Check that saved/reloaded predictions match."""
    return bool(np.allclose(left, right))


def eval_history_final(ranker: XGBRanker) -> dict[str, float | None]:
    """Return the final XGBoost training diagnostics."""
    history = ranker.evals_result()

    train_values = history.get("validation_0", {}).get("ndcg@10", [])
    validation_values = history.get("validation_1", {}).get("ndcg@10", [])

    return {
        "train_ndcg@10": float(train_values[-1]) if train_values else None,
        "validation_ndcg@10": (
            float(validation_values[-1]) if validation_values else None
        ),
    }
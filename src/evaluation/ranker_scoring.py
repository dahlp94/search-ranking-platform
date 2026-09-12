"""Score saved rankers and compare per-query results."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.comparison import rank_scored_candidates
from src.ranking.groups import prepare_grouped_data
from src.ranking.ranker import load_ranker, predict_scores


class RankerEvalError(ValueError):
    """Raised when ranker evaluation inputs are inconsistent."""


def assert_same_candidates(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    """Require the same candidate rows."""

    required = {"example_id"}

    if not required.issubset(left.columns) or not required.issubset(right.columns):
        raise RankerEvalError("Both tables must contain example_id.")

    if len(left) != len(right):
        raise RankerEvalError("Candidate row counts do not match.")

    if set(left["example_id"]) != set(right["example_id"]):
        raise RankerEvalError("Candidate sets do not match.")


def assert_same_queries(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    """Require the same query set."""

    if "query_id" not in left.columns or "query_id" not in right.columns:
        raise RankerEvalError("Both tables must contain query_id.")

    if set(left["query_id"]) != set(right["query_id"]):
        raise RankerEvalError("Query sets do not match.")


def score_saved_ranker(
    model_path: Path,
    frame: pd.DataFrame,
    features,
    score_col: str = "model_score",
) -> pd.DataFrame:
    """Load a saved ranker and score candidates."""

    grouped = prepare_grouped_data(
        frame,
        features=features,
    )

    ranker = load_ranker(Path(model_path))
    scores = predict_scores(ranker, grouped)

    if len(scores) != grouped.n_rows:
        raise RankerEvalError(
            "Prediction count does not match candidate rows."
        )

    scored = grouped.frame.copy()
    scored[score_col] = np.asarray(scores, dtype=float)

    return rank_scored_candidates(
        scored,
        score_col=score_col,
    )


def paired_metric_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_prefix: str,
    right_prefix: str,
    k: int = 10,
) -> pd.DataFrame:
    """Combine per-query metrics and calculate paired differences."""

    assert_same_queries(left, right)

    metrics = {
        f"ndcg@{k}": f"ndcg_at_{k}",
        f"recall@{k}": f"recall_at_{k}",
        "mrr": "mrr",
    }

    required_left = [
        "query_id",
        "query",
        "n_candidates",
        *metrics,
    ]

    required_right = [
        "query_id",
        *metrics,
    ]

    missing_left = [
        column for column in required_left
        if column not in left.columns
    ]

    missing_right = [
        column for column in required_right
        if column not in right.columns
    ]

    if missing_left or missing_right:
        raise RankerEvalError(
            f"Missing columns: left={missing_left}, right={missing_right}"
        )

    left_table = left[required_left].rename(
        columns={
            metric: f"{left_prefix}_{name}"
            for metric, name in metrics.items()
        }
    )

    right_table = right[required_right].rename(
        columns={
            metric: f"{right_prefix}_{name}"
            for metric, name in metrics.items()
        }
    )

    paired = left_table.merge(
        right_table,
        on="query_id",
        validate="one_to_one",
    )

    for name in metrics.values():
        paired[f"delta_{name}"] = (
            paired[f"{left_prefix}_{name}"]
            - paired[f"{right_prefix}_{name}"]
        )

    return paired.sort_values("query_id").reset_index(drop=True)

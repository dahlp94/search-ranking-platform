"""Utilities for deterministic ranking and paired query-level comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.retrieval.ranking import sort_and_rank


TIE_ATOL = 1e-12


def rank_scored_candidates(
    df: pd.DataFrame,
    score_col: str = "xgb_score",
) -> pd.DataFrame:
    """Rank candidates within each query using deterministic tie-breaking."""
    if "query_id" not in df.columns:
        raise ValueError("Missing required column: query_id")
    if score_col not in df.columns:
        raise ValueError(f"Missing required column: {score_col}")

    ranked_groups = []

    for _, group in df.groupby("query_id", sort=True, dropna=False):
        ranked_groups.append(
            sort_and_rank(group, score_col=score_col)
        )

    if not ranked_groups:
        return df.iloc[0:0].copy()

    ranked = pd.concat(ranked_groups, ignore_index=True)

    if len(ranked) != len(df):
        raise ValueError("Ranking changed the number of candidate rows.")

    if "example_id" in df.columns:
        if set(ranked["example_id"]) != set(df["example_id"]):
            raise ValueError("Ranking changed candidate identity.")

    return ranked

def paired_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_metric: str,
    right_metric: str,
    query_col: str = "query_id",
    tie_atol: float = TIE_ATOL,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compare two models query by query using left - right metric differences."""
    left_ids = set(left[query_col])
    right_ids = set(right[query_col])

    if left_ids != right_ids:
        raise ValueError("Query ID sets do not match.")

    paired = left[[query_col, left_metric]].merge(
        right[[query_col, right_metric]],
        on=query_col,
        validate="one_to_one",
    )

    paired = paired.rename(
        columns={
            left_metric: "left_value",
            right_metric: "right_value",
        }
    )
    paired["delta"] = paired["left_value"] - paired["right_value"]
    paired = paired.sort_values(query_col).reset_index(drop=True)

    deltas = paired["delta"].to_numpy(dtype=float)
    n = len(deltas)

    wins = int(np.sum(deltas > tie_atol))
    losses = int(np.sum(deltas < -tie_atol))
    ties = n - wins - losses

    summary = {
        "n_queries": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta": float(np.mean(deltas)) if n else 0.0,
        "median_delta": float(np.median(deltas)) if n else 0.0,
        "win_rate": wins / n if n else 0.0,
        "loss_rate": losses / n if n else 0.0,
        "tie_rate": ties / n if n else 0.0,
        "tie_atol": float(tie_atol),
    }

    return paired, summary
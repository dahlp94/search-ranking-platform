"""Query-level evaluation, aggregation, and baseline comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import DEFAULT_K
from src.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def per_query_metrics(
    ranked: pd.DataFrame,
    k: int = DEFAULT_K,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Compute NDCG@k, Recall@k, and RR for each query.

    `ranked` must already be ordered by `predicted_rank` within query_id.
    """
    required = ["query_id", "query", "predicted_rank", "esci_label", "relevance_gain"]
    missing = [col for col in required if col not in ranked.columns]
    if missing:
        raise KeyError(f"per_query_metrics missing columns {missing}.")

    rows: list[dict[str, Any]] = []
    for query_id, group in ranked.groupby("query_id", sort=True, dropna=False):
        ordered = group.sort_values("predicted_rank", kind="mergesort")
        labels = ordered["esci_label"].tolist()
        gains = ordered["relevance_gain"].tolist()
        rows.append(
            {
                "query_id": query_id,
                "query": ordered["query"].iloc[0],
                "n_candidates": int(len(ordered)),
                f"ndcg@{k}": ndcg_at_k(gains, k),
                f"recall@{k}": recall_at_k(labels, k),
                "mrr": reciprocal_rank(labels),
            }
        )
    out = pd.DataFrame(rows)
    if model_name is not None:
        out.insert(0, "model", model_name)
    return out


def summarize_query_metrics(per_query: pd.DataFrame, k: int = DEFAULT_K) -> dict[str, Any]:
    """Aggregate query-level metrics. Rows are queries, not query-product pairs."""
    metric_cols = [f"ndcg@{k}", f"recall@{k}", "mrr"]
    summary: dict[str, Any] = {"n_queries": int(len(per_query))}
    if "model" in per_query.columns and per_query["model"].nunique() == 1:
        summary["model"] = str(per_query["model"].iloc[0])
    for col in metric_cols:
        values = per_query[col].to_numpy(dtype=float)
        summary[col] = {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "p25": float(np.percentile(values, 25)) if len(values) else 0.0,
            "p75": float(np.percentile(values, 75)) if len(values) else 0.0,
        }
    return summary


def compare_models(
    per_query_by_model: Mapping[str, pd.DataFrame],
    metric: str = f"ndcg@{DEFAULT_K}",
    left: str = "bm25",
    right: str = "tfidf",
) -> dict[str, Any]:
    """Descriptive query-level comparison. No hypothesis test in Week 1."""
    if left not in per_query_by_model or right not in per_query_by_model:
        raise KeyError(f"Need '{left}' and '{right}' in per_query_by_model.")
    left_df = per_query_by_model[left][["query_id", "query", metric]].rename(
        columns={metric: f"{metric}_{left}"}
    )
    right_df = per_query_by_model[right][["query_id", metric]].rename(
        columns={metric: f"{metric}_{right}"}
    )
    merged = left_df.merge(right_df, on="query_id", how="inner", validate="one_to_one")
    merged["diff"] = merged[f"{metric}_{left}"] - merged[f"{metric}_{right}"]
    return {
        "metric": metric,
        "left_model": left,
        "right_model": right,
        "n_queries_compared": int(len(merged)),
        f"{left}_beats_{right}": int((merged["diff"] > 0).sum()),
        f"{right}_beats_{left}": int((merged["diff"] < 0).sum()),
        "ties": int((merged["diff"] == 0).sum()),
        "mean_diff_left_minus_right": float(merged["diff"].mean()) if len(merged) else 0.0,
        "largest_positive_diffs": merged.sort_values("diff", ascending=False)
        .head(10)[["query_id", "query", f"{metric}_{left}", f"{metric}_{right}", "diff"]]
        .to_dict(orient="records"),
        "largest_negative_diffs": merged.sort_values("diff", ascending=True)
        .head(10)[["query_id", "query", f"{metric}_{left}", f"{metric}_{right}", "diff"]]
        .to_dict(orient="records"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))

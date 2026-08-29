"""Deterministic ranking utilities.

Ranking scores can tie. We never use the inherited DataFrame row order as a
tie-breaker, because that order is not a documented ranking signal and would
make metrics depend on parquet read order.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

RankFn = Callable[..., pd.DataFrame]


def sort_and_rank(
    df: pd.DataFrame,
    score_col: str,
    k: int | None = None,
    id_col: str = "product_id",
) -> pd.DataFrame:
    """Sort by score descending, then product_id ascending, and assign ranks.

    `predicted_rank` is 1-based within this candidate list.
    """
    if score_col not in df.columns:
        raise KeyError(f"Score column '{score_col}' is not in the candidate frame.")
    if id_col not in df.columns:
        raise KeyError(f"Tie-break column '{id_col}' is not in the candidate frame.")

    out = df.copy()
    tie_break = out[id_col].astype(str).to_numpy()
    scores = -out[score_col].to_numpy(dtype=float)
    if "example_id" in out.columns:
        tertiary = out["example_id"].astype(str).to_numpy()
        order = np.lexsort((tertiary, tie_break, scores))
    else:
        order = np.lexsort((tie_break, scores))
    out = out.iloc[order]
    out["predicted_rank"] = np.arange(1, len(out) + 1)
    if k is not None:
        if k < 0:
            raise ValueError(f"k must be >= 0; got {k}.")
        out = out.iloc[:k]
    return out.reset_index(drop=True)


def rerank_dataset(
    df: pd.DataFrame,
    rank_fn: RankFn,
    k: int | None = None,
) -> pd.DataFrame:
    """Apply a per-query ranker while preserving each query's candidate set.

    Queries are processed in sorted query_id order so full-dataset runs are
    deterministic even if a ranker has internal randomness keyed by visit order.
    """
    if "query_id" not in df.columns or "query" not in df.columns:
        raise KeyError("rerank_dataset requires 'query_id' and 'query' columns.")

    pieces: list[pd.DataFrame] = []
    for query_id, group in df.groupby("query_id", sort=True, dropna=False):
        query = group["query"].iloc[0]
        ranked = rank_fn(str(query), group, k=k)
        n_in = len(group)
        n_out = len(ranked)
        if k is None and n_out != n_in:
            raise ValueError(
                f"Ranker changed the candidate set for query_id={query_id}: "
                f"{n_in} in, {n_out} out. Re-ranking must not add or drop candidates."
            )
        if k is not None and n_out != min(k, n_in):
            raise ValueError(
                f"Ranker returned {n_out} rows for query_id={query_id}; "
                f"expected min(k={k}, n_candidates={n_in})."
            )
        pieces.append(ranked)
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)

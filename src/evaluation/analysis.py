"""Select validation queries for error analysis.

These heuristics are descriptive. They do not look at the official test holdout.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.constants import BINARY_RELEVANT_LABELS, DEFAULT_K
from src.retrieval.text import tokenize

_DIGIT_RE = re.compile(r"\d")


def _token_overlap(query: str, title: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(tokenize(title))) / len(query_tokens)


def select_inspection_queries(
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
    bm25_ranked: pd.DataFrame,
    k: int = DEFAULT_K,
) -> list[dict[str, Any]]:
    """Pick a small set of project-validation queries illustrating lexical behavior."""
    ndcg_col = f"ndcg@{k}_bm25"
    tfidf_col = f"ndcg@{k}_tfidf"
    needed = {"query_id", "query", ndcg_col, tfidf_col}
    missing = needed - set(metrics.columns)
    if missing:
        raise KeyError(f"Inspection metrics table missing {missing}.")

    label_sets = validation.groupby("query_id")["esci_label"].apply(lambda s: set(s))
    has_relevant = validation.groupby("query_id")["esci_label"].apply(
        lambda labels: bool(labels.isin(BINARY_RELEVANT_LABELS).any())
    )
    metrics = metrics.copy()
    metrics["n_query_tokens"] = metrics["query"].map(lambda q: len(tokenize(q)))
    metrics["has_relevant"] = metrics["query_id"].map(has_relevant).fillna(False)
    metrics["has_digit"] = metrics["query"].map(lambda q: bool(_DIGIT_RE.search(str(q))))
    metrics["ndcg_diff_bm25_tfidf"] = metrics[ndcg_col] - metrics[tfidf_col]

    top1 = (
        bm25_ranked.loc[bm25_ranked["predicted_rank"] == 1, ["query_id", "query", "product_title", "esci_label"]]
        .drop_duplicates("query_id")
        .copy()
    )
    top1["overlap"] = [
        _token_overlap(row.query, row.product_title) for row in top1.itertuples(index=False)
    ]

    selected: list[dict[str, Any]] = []
    used: set[Any] = set()

    def _take(query_id: Any, kind: str, reason: str) -> None:
        if query_id in used or pd.isna(query_id):
            return
        row = metrics.loc[metrics["query_id"] == query_id].iloc[0]
        selected.append(
            {
                "kind": kind,
                "reason": reason,
                "query_id": row["query_id"],
                "query": row["query"],
                "ndcg@10_bm25": float(row[ndcg_col]),
                "ndcg@10_tfidf": float(row[tfidf_col]),
                "ndcg@10_random": float(row.get(f"ndcg@{k}_random", float("nan"))),
            }
        )
        used.add(query_id)

    mixed = metrics.loc[metrics["query_id"].map(lambda qid: len(label_sets.get(qid, set())) >= 2)]
    if not mixed.empty:
        _take(
            mixed.sort_values(ndcg_col, ascending=False)["query_id"].iloc[0],
            "bm25_strong",
            "Highest BM25 NDCG among queries with mixed ESCI labels.",
        )

    hard = metrics.loc[metrics["has_relevant"]]
    if not hard.empty:
        _take(
            hard.sort_values(ndcg_col, ascending=True)["query_id"].iloc[0],
            "bm25_weak",
            "Lowest BM25 NDCG among queries that have at least one E/S item.",
        )

    specific = metrics.loc[metrics["has_digit"]]
    if not specific.empty:
        _take(
            specific.sort_values(ndcg_col, ascending=False)["query_id"].iloc[0],
            "brand_or_model",
            "Query contains a digit, typical of brand/model-number lookups.",
        )

    generic = metrics.loc[metrics["n_query_tokens"] <= 2]
    if not generic.empty:
        _take(
            generic.sort_values("n_query_tokens", ascending=True)["query_id"].iloc[0],
            "broad_generic",
            "Short query, more likely to be broad/generic.",
        )

    misleading = top1.loc[top1["esci_label"].eq("I") & (top1["overlap"] >= 0.5)]
    if not misleading.empty:
        _take(
            misleading.sort_values("overlap", ascending=False)["query_id"].iloc[0],
            "lexical_overlap_misleading",
            "BM25 top result has high title overlap but is labeled Irrelevant.",
        )

    disagree = metrics.loc[metrics["query_id"].map(lambda qid: qid not in used)]
    if not disagree.empty:
        idx = disagree["ndcg_diff_bm25_tfidf"].abs().sort_values(ascending=False).index[0]
        _take(
            disagree.loc[idx, "query_id"],
            "tfidf_bm25_disagree",
            "Largest absolute NDCG@10 gap between BM25 and TF-IDF.",
        )

    # Fill remaining slots with distinctive unused queries if a heuristic missed.
    leftovers = metrics.loc[~metrics["query_id"].isin(used)].sort_values(ndcg_col, ascending=False)
    for _, row in leftovers.iterrows():
        if len(selected) >= 6:
            break
        _take(row["query_id"], "additional", "Additional validation query for inspection.")

    return selected

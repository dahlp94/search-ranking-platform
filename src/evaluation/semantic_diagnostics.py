"""Diagnostics for semantic-vs-reference ranking performance."""

import numpy as np
import pandas as pd

from src.constants import BINARY_RELEVANT_LABELS
from src.evaluation.comparison import rank_scored_candidates
from src.evaluation.ranker_analysis import add_query_length_segment
from src.retrieval.text import tokenize


OVERLAP_LOW = 0.40
OVERLAP_HIGH = 0.70

SPEARMAN_LOW = 0.20
SPEARMAN_HIGH = 0.60

MIN_SEGMENT_QUERIES = 50


class SemanticDiagnosticError(ValueError):
    """Raised when diagnostic inputs are invalid."""


def gain_importance_table(ranker, features) -> pd.DataFrame:
    """Return normalized XGBoost gain importance."""

    gain = ranker.get_booster().get_score(importance_type="gain")

    table = pd.DataFrame(
        {
            "feature": features,
            "gain": [gain.get(feature, 0.0) for feature in features],
        }
    )

    total = table["gain"].sum()
    table["gain_importance"] = (
        table["gain"] / total if total else 0.0
    )

    return (
        table.sort_values("gain_importance", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda x: np.arange(1, len(x) + 1))
    )


def relevant_overlap_by_query(candidates: pd.DataFrame) -> pd.Series:
    """Mean lexical overlap for judged-relevant candidates."""

    relevant = candidates[
        candidates["esci_label"].isin(BINARY_RELEVANT_LABELS)
    ]

    return (
        relevant.groupby("query_id")["query_token_coverage"]
        .mean()
        .rename("relevant_overlap")
    )


def overlap_segment(value: float) -> str:
    if pd.isna(value):
        return "no relevant candidates"
    if value < OVERLAP_LOW:
        return "low"
    if value < OVERLAP_HIGH:
        return "medium"
    return "high"


def semantic_tfidf_spearman_by_query(
    candidates: pd.DataFrame,
) -> pd.Series:
    """Per-query semantic-vs-TF-IDF rank correlation."""

    def correlation(group):
        if (
            group["semantic_similarity"].nunique() < 2
            or group["tfidf_score"].nunique() < 2
        ):
            return np.nan

        return group["semantic_similarity"].corr(
            group["tfidf_score"],
            method="spearman",
        )

    return (
        candidates.groupby("query_id")
        .apply(correlation, include_groups=False)
        .rename("spearman_sem_tfidf")
    )


def disagreement_segment(value: float) -> str:
    if pd.isna(value):
        return "undefined"
    if value < SPEARMAN_LOW:
        return "high disagreement"
    if value < SPEARMAN_HIGH:
        return "mixed"
    return "high agreement"


def identifier_segment(query: object) -> str:
    """Identify queries containing numbers/model-like tokens."""

    has_digit = any(
        any(char.isdigit() for char in token)
        for token in tokenize(query)
    )

    return "identifier-style" if has_digit else "other"


def annotate_diagnostic_queries(
    per_query: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Add diagnostic query characteristics."""

    if set(per_query["query_id"]) != set(candidates["query_id"]):
        raise SemanticDiagnosticError(
            "Candidate and per-query query sets do not match."
        )

    out = add_query_length_segment(per_query)

    out = out.merge(
        relevant_overlap_by_query(candidates),
        on="query_id",
        how="left",
    )

    out = out.merge(
        semantic_tfidf_spearman_by_query(candidates),
        on="query_id",
        how="left",
    )

    out["overlap_segment"] = out["relevant_overlap"].map(
        overlap_segment
    )

    out["disagreement_segment"] = out["spearman_sem_tfidf"].map(
        disagreement_segment
    )

    out["identifier_segment"] = out["query"].map(
        identifier_segment
    )

    return out.sort_values("query_id").reset_index(drop=True)


def summarize_segments(
    annotated: pd.DataFrame,
    segment_col: str,
    min_queries: int = MIN_SEGMENT_QUERIES,
) -> pd.DataFrame:
    """Summarize ranking performance by query segment."""

    summary = (
        annotated.groupby(segment_col, dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            reference_ndcg_at_10=(
                "reference_ndcg_at_10",
                "mean",
            ),
            semantic_ndcg_at_10=(
                "semantic_ndcg_at_10",
                "mean",
            ),
            delta_ndcg_at_10=(
                "delta_ndcg_at_10",
                "mean",
            ),
        )
        .reset_index()
        .rename(columns={segment_col: "segment"})
    )

    return summary[
        summary["n_queries"] >= min_queries
    ].reset_index(drop=True)


def select_diagnostic_queries(
    annotated: pd.DataFrame,
    n: int = 3,
) -> pd.DataFrame:
    """Select representative improvements, regressions, and ties."""

    wins = (
        annotated[annotated["delta_ndcg_at_10"] > 0]
        .nlargest(n, "delta_ndcg_at_10")
        .assign(kind="large_improvement")
    )

    losses = (
        annotated[annotated["delta_ndcg_at_10"] < 0]
        .nsmallest(n, "delta_ndcg_at_10")
        .assign(kind="large_regression")
    )

    ties = (
        annotated[np.isclose(annotated["delta_ndcg_at_10"], 0)]
        .nlargest(n, "reference_ndcg_at_10")
        .assign(kind="tie")
    )

    selected = pd.concat(
        [wins, losses, ties],
        ignore_index=True,
    )

    return selected[
        [
            "kind",
            "query_id",
            "query",
            "delta_ndcg_at_10",
            "reference_ndcg_at_10",
            "semantic_ndcg_at_10",
            "overlap_segment",
            "disagreement_segment",
            "identifier_segment",
        ]
    ]


def add_comparison_ranks(scored: pd.DataFrame) -> pd.DataFrame:
    """Add ranks for available ranking signals."""

    out = scored.copy()

    score_columns = {
        "reference": "reference_score",
        "semantic": "semantic_score",
        "tfidf": "tfidf_score",
        "bm25": "bm25_score",
    }

    for name, score_col in score_columns.items():
        if score_col not in scored.columns:
            continue

        ranked = rank_scored_candidates(
            scored,
            score_col=score_col,
        )[
            ["example_id", "predicted_rank"]
        ].rename(
            columns={"predicted_rank": f"{name}_rank"}
        )

        out = out.merge(
            ranked,
            on="example_id",
            validate="one_to_one",
        )

    return out


def extract_semantic_examples(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Return candidate rankings for selected queries."""

    if selected.empty:
        return pd.DataFrame()

    examples = ranked[
        ranked["query_id"].isin(selected["query_id"])
    ].copy()

    meta = selected[
        [
            "query_id",
            "kind",
            "delta_ndcg_at_10",
        ]
    ]

    examples = examples.merge(
        meta,
        on="query_id",
        how="left",
    )

    useful_columns = [
        "kind",
        "query_id",
        "query",
        "product_title",
        "esci_label",
        "relevance_gain",
        "bm25_score",
        "tfidf_score",
        "semantic_similarity",
        "reference_rank",
        "semantic_rank",
        "tfidf_rank",
        "bm25_rank",
        "delta_ndcg_at_10",
    ]

    useful_columns = [
        column
        for column in useful_columns
        if column in examples.columns
    ]

    return (
        examples[useful_columns]
        .sort_values(["kind", "query_id", "semantic_rank"])
        .reset_index(drop=True)
    )
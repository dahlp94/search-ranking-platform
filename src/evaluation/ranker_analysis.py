"""Interpretation helpers for the approved XGBRanker."""

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.constants import DEFAULT_K
from src.evaluation.comparison import rank_scored_candidates
from src.evaluation.evaluate import per_query_metrics, summarize_query_metrics
from src.features.build import FEATURES
from src.ranking.groups import prepare_grouped_data, prepare_grouped_subset
from src.ranking.ranker import build_ranker, predict_scores
from src.retrieval.text import tokenize


OVERLAP_FEATURES = [
    "shared_token_count",
    "query_token_coverage",
    "title_token_coverage",
    "exact_query_in_title",
    "query_bullet_coverage",
]

STRUCTURED_FEATURES = [
    "brand_match",
    "title_token_count",
    "token_length_difference",
]

ABLATIONS = {
    "full": list(FEATURES),
    "lexical_scores": ["bm25_score", "tfidf_score"],
    "minus_tfidf": [f for f in FEATURES if f != "tfidf_score"],
    "minus_bm25": [f for f in FEATURES if f != "bm25_score"],
    "minus_overlap": [f for f in FEATURES if f not in OVERLAP_FEATURES],
    "minus_structured": [f for f in FEATURES if f not in STRUCTURED_FEATURES],
}


def validate_ablations() -> None:
    """Ensure every ablation uses only approved model features."""
    approved = set(FEATURES)

    if ABLATIONS["full"] != list(FEATURES):
        raise ValueError("Full ablation must match the approved feature list.")

    for name, features in ABLATIONS.items():
        if not features:
            raise ValueError(f"Ablation '{name}' has no features.")

        unknown = set(features) - approved
        if unknown:
            raise ValueError(
                f"Ablation '{name}' contains unknown features: {sorted(unknown)}"
            )


def gain_importance_table(ranker: XGBRanker) -> pd.DataFrame:
    """Return normalized XGBoost gain importance."""
    raw = ranker.get_booster().get_score(importance_type="gain")

    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "gain": [float(raw.get(feature, 0.0)) for feature in FEATURES],
        }
    )

    total = table["gain"].sum()
    table["gain_importance"] = (
        table["gain"] / total if total else 0.0
    )

    table = table.sort_values(
        "gain_importance",
        ascending=False,
    ).reset_index(drop=True)

    table["rank"] = np.arange(1, len(table) + 1)

    return table


def evaluate_ranker(
    ranker: XGBRanker,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Evaluate one ranker using the project's ranking metrics."""
    data = prepare_grouped_subset(validation, features)

    scored = data.frame.copy()
    scored["xgb_score"] = predict_scores(ranker, data)

    ranked = rank_scored_candidates(scored, "xgb_score")
    per_query = per_query_metrics(
        ranked,
        k=DEFAULT_K,
    )
    summary = summarize_query_metrics(
        per_query,
        k=DEFAULT_K,
    )

    return {
        "ndcg@10": float(summary["ndcg@10"]["mean"]),
        "recall@10": float(summary["recall@10"]["mean"]),
        "mrr": float(summary["mrr"]["mean"]),
    }


def run_ablation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    """Train and evaluate one predefined feature ablation."""
    train_data = prepare_grouped_subset(train, features)
    val_data = prepare_grouped_subset(validation, features)

    ranker = build_ranker()
    ranker.fit(
        train_data.X,
        train_data.y,
        group=train_data.group_sizes,
        eval_set=[(val_data.X, val_data.y)],
        eval_group=[val_data.group_sizes],
        verbose=False,
    )

    return evaluate_ranker(
        ranker,
        validation,
        features,
    )


def score_full_model(
    ranker: XGBRanker,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Attach saved-model scores to the validation candidates."""
    data = prepare_grouped_data(validation)

    scored = data.frame.copy()
    scored["xgb_score"] = predict_scores(ranker, data)

    return scored


def add_model_ranks(scored: pd.DataFrame) -> pd.DataFrame:
    """Add XGBRanker, TF-IDF, and BM25 candidate ranks."""
    ranks = {}

    for name, score_col in {
        "xgb": "xgb_score",
        "tfidf": "tfidf_score",
        "bm25": "bm25_score",
    }.items():
        ranked = rank_scored_candidates(
            scored,
            score_col,
        )

        ranks[name] = ranked[
            ["example_id", "predicted_rank"]
        ].rename(
            columns={"predicted_rank": f"{name}_rank"}
        )

    out = scored.copy()

    for ranked in ranks.values():
        out = out.merge(
            ranked,
            on="example_id",
            validate="one_to_one",
        )

    return out


def extract_query_examples(
    ranked: pd.DataFrame,
    query_ids: list,
    per_query: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return candidate rankings for selected inspection queries."""
    available = set(ranked["query_id"])
    missing = set(query_ids) - available

    if missing:
        raise ValueError(
            f"Missing query IDs: {sorted(missing)}"
        )

    selected = ranked[
        ranked["query_id"].isin(query_ids)
    ].copy()

    columns = [
        "query_id",
        "query",
        "example_id",
        "product_id",
        "product_title",
        "esci_label",
        "relevance_gain",
        "bm25_score",
        "tfidf_score",
        "xgb_score",
        "bm25_rank",
        "tfidf_rank",
        "xgb_rank",
    ]

    columns = [
        column
        for column in columns
        if column in selected.columns
    ]

    selected = selected[columns]

    if per_query is not None:
        selected = selected.merge(
            per_query[
                ["query_id", "delta_ndcg_at_10"]
            ],
            on="query_id",
            how="left",
        )

    return selected.sort_values(
        ["query_id", "xgb_rank"]
    ).reset_index(drop=True)


def add_query_length_segment(
    per_query: pd.DataFrame,
) -> pd.DataFrame:
    """Add a simple query-length segment for diagnostic analysis."""

    def bucket(query: object) -> str:
        n_tokens = len(tokenize(query))

        if n_tokens <= 1:
            return "1 token"
        if n_tokens <= 3:
            return "2–3 tokens"
        return "4+ tokens"

    out = per_query.copy()
    out["query_length_segment"] = out["query"].map(bucket)

    return out

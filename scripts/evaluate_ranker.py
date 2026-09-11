"""Evaluate the saved XGBRanker on project validation."""

import json
from pathlib import Path

import pandas as pd

from src.constants import DEFAULT_K, SEED
from src.evaluation.bootstrap import paired_bootstrap_mean_ci
from src.evaluation.comparison import paired_summary, rank_scored_candidates
from src.evaluation.evaluate import per_query_metrics, summarize_query_metrics, write_json
from src.features.build import FEATURES
from src.ranking.groups import prepare_grouped_data
from src.ranking.ranker import load_ranker, predict_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FEATURES = PROJECT_ROOT / "artifacts/features/validation_features.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts/models/xgb_ranker.json"

BASELINE_PER_QUERY = PROJECT_ROOT / "artifacts/metrics/baseline_per_query.csv"
BASELINE_SUMMARY = PROJECT_ROOT / "artifacts/metrics/baseline_summary.json"

RANKER_SUMMARY = PROJECT_ROOT / "artifacts/metrics/ranker_summary.json"
RANKER_PER_QUERY = PROJECT_ROOT / "artifacts/metrics/ranker_per_query.csv"
RANKER_VS_TFIDF = PROJECT_ROOT / "artifacts/metrics/ranker_vs_tfidf.json"

EXPECTED_VAL_ROWS = 63_038
EXPECTED_VAL_QUERIES = 3_134
BOOTSTRAP_REPLICATES = 2000


def load_validation() -> pd.DataFrame:
    """Load and validate the frozen project-validation feature table."""
    if not VALIDATION_FEATURES.is_file():
        raise FileNotFoundError(VALIDATION_FEATURES)

    df = pd.read_parquet(VALIDATION_FEATURES)

    if len(df) != EXPECTED_VAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_VAL_ROWS:,} validation rows, got {len(df):,}.")

    if df["query_id"].nunique() != EXPECTED_VAL_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_VAL_QUERIES:,} validation queries, "
            f"got {df['query_id'].nunique():,}."
        )

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError("Official test rows found in validation features.")

    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing approved features: {missing}")

    return df


def load_baselines() -> tuple[pd.DataFrame, dict]:
    """Load trusted baseline artifacts and verify the TF-IDF per-query table."""
    per_query = pd.read_csv(BASELINE_PER_QUERY)
    summary = json.loads(BASELINE_SUMMARY.read_text())["mean_table"]

    required = {
        "query_id",
        "ndcg@10_tfidf",
        "recall@10_tfidf",
        "mrr_tfidf",
    }
    missing = required - set(per_query.columns)
    if missing:
        raise ValueError(f"Baseline per-query file missing columns: {sorted(missing)}")

    if per_query["query_id"].nunique() != EXPECTED_VAL_QUERIES:
        raise ValueError("Baseline table does not contain all validation queries.")

    observed = float(per_query["ndcg@10_tfidf"].mean())
    expected = float(summary["tfidf"]["ndcg@10"])

    if abs(observed - expected) > 1e-12:
        raise ValueError(
            f"TF-IDF per-query mean ({observed}) does not match "
            f"baseline summary ({expected})."
        )

    return per_query, summary


def main() -> None:
    validation = load_validation()
    tfidf, baseline_means = load_baselines()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    ranker = load_ranker(MODEL_PATH)

    grouped = prepare_grouped_data(validation)
    scores = predict_scores(ranker, grouped)

    if len(scores) != len(validation):
        raise ValueError("Prediction count does not match validation rows.")

    scored = grouped.frame.copy()
    scored["xgb_score"] = scores

    ranked = rank_scored_candidates(scored, score_col="xgb_score")

    xgb_per_query = per_query_metrics(
        ranked,
        k=DEFAULT_K,
        model_name="xgb",
    )
    xgb_summary = summarize_query_metrics(
        xgb_per_query,
        k=DEFAULT_K,
    )

    paired, comparison = paired_summary(
        xgb_per_query,
        tfidf,
        left_metric=f"ndcg@{DEFAULT_K}",
        right_metric=f"ndcg@{DEFAULT_K}_tfidf",
    )

    if comparison["n_queries"] != EXPECTED_VAL_QUERIES:
        raise ValueError("Paired comparison does not contain all validation queries.")

    bootstrap = paired_bootstrap_mean_ci(
        paired["delta"].to_numpy(),
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )

    per_query = (
        xgb_per_query.merge(
            tfidf[
                [
                    "query_id",
                    "ndcg@10_tfidf",
                    "recall@10_tfidf",
                    "mrr_tfidf",
                ]
            ],
            on="query_id",
            validate="one_to_one",
        )
        .rename(
            columns={
                f"ndcg@{DEFAULT_K}": "xgb_ndcg_at_10",
                f"recall@{DEFAULT_K}": "xgb_recall_at_10",
                "mrr": "xgb_mrr",
                "ndcg@10_tfidf": "tfidf_ndcg_at_10",
                "recall@10_tfidf": "tfidf_recall_at_10",
                "mrr_tfidf": "tfidf_mrr",
            }
        )
    )

    per_query["delta_ndcg_at_10"] = (
        per_query["xgb_ndcg_at_10"] - per_query["tfidf_ndcg_at_10"]
    )
    per_query["delta_recall_at_10"] = (
        per_query["xgb_recall_at_10"] - per_query["tfidf_recall_at_10"]
    )
    per_query["delta_mrr"] = (
        per_query["xgb_mrr"] - per_query["tfidf_mrr"]
    )

    top_5 = (
        per_query.nlargest(5, "delta_ndcg_at_10")[
            ["query_id", "query", "delta_ndcg_at_10"]
        ]
        .to_dict("records")
    )
    bottom_5 = (
        per_query.nsmallest(5, "delta_ndcg_at_10")[
            ["query_id", "query", "delta_ndcg_at_10"]
        ]
        .to_dict("records")
    )

    ranker_means = {
        f"ndcg@{DEFAULT_K}": xgb_summary[f"ndcg@{DEFAULT_K}"]["mean"],
        f"recall@{DEFAULT_K}": xgb_summary[f"recall@{DEFAULT_K}"]["mean"],
        "mrr": xgb_summary["mrr"]["mean"],
    }

    write_json(
        RANKER_SUMMARY,
        {
            "partition": "project_validation",
            "rows": len(validation),
            "queries": validation["query_id"].nunique(),
            "model_path": str(MODEL_PATH),
            "model_retrained": False,
            "official_test_used": False,
            "XGBRanker": ranker_means,
            "baseline_references": baseline_means,
        },
    )

    per_query.to_csv(RANKER_PER_QUERY, index=False)

    write_json(
        RANKER_VS_TFIDF,
        {
            "metric": f"ndcg@{DEFAULT_K}",
            **comparison,
            "bootstrap_replicates": bootstrap["n_replicates"],
            "bootstrap_seed": bootstrap["seed"],
            "bootstrap_resampling_unit": bootstrap["resampling_unit"],
            "bootstrap_ci_95": [
                bootstrap["ci_lower"],
                bootstrap["ci_upper"],
            ],
            "top_5_positive_delta": top_5,
            "bottom_5_delta": bottom_5,
            "official_test_used": False,
        },
    )

    print("\nProject-validation results")
    print(f"{'Model':<10} {'NDCG@10':>10} {'Recall@10':>10} {'MRR':>10}")

    for model in ("random", "bm25", "tfidf"):
        metrics = baseline_means[model]
        print(
            f"{model:<10} "
            f"{metrics['ndcg@10']:10.4f} "
            f"{metrics['recall@10']:10.4f} "
            f"{metrics['mrr']:10.4f}"
        )

    print(
        f"{'xgb':<10} "
        f"{ranker_means['ndcg@10']:10.4f} "
        f"{ranker_means['recall@10']:10.4f} "
        f"{ranker_means['mrr']:10.4f}"
    )

    print(
        f"\nXGB - TF-IDF ΔNDCG@10: "
        f"mean={comparison['mean_delta']:.6f}, "
        f"median={comparison['median_delta']:.6f}, "
        f"wins={comparison['wins']}, "
        f"losses={comparison['losses']}, "
        f"ties={comparison['ties']}"
    )

    print(
        "Bootstrap 95% CI: "
        f"[{bootstrap['ci_lower']:.6f}, {bootstrap['ci_upper']:.6f}]"
    )


if __name__ == "__main__":
    main()
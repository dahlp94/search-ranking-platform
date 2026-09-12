#!/usr/bin/env python3
"""Compare reference and semantic rankers on validation data."""

import json
from pathlib import Path

import pandas as pd

from src.constants import DEFAULT_K, SEED
from src.evaluation.bootstrap import paired_bootstrap_mean_ci
from src.evaluation.comparison import paired_summary
from src.evaluation.evaluate import (
    per_query_metrics,
    summarize_query_metrics,
    write_json,
)
from src.evaluation.ranker_scoring import (
    assert_same_candidates,
    paired_metric_table,
    score_saved_ranker,
)
from src.features.build import FEATURES
from src.features.semantic import SEMANTIC_FEATURES


ROOT = Path(__file__).resolve().parents[1]

REFERENCE_MODEL = ROOT / "artifacts/models/xgb_ranker.json"
SEMANTIC_MODEL = ROOT / "artifacts/models/xgb_ranker_semantic.json"

REFERENCE_VALIDATION = ROOT / "artifacts/features/validation_features.parquet"
SEMANTIC_VALIDATION = (
    ROOT / "artifacts/features/validation_features_semantic.parquet"
)

SUMMARY_PATH = ROOT / "artifacts/metrics/semantic_ranker_summary.json"
PER_QUERY_PATH = ROOT / "artifacts/metrics/semantic_ranker_per_query.csv"
COMPARISON_PATH = ROOT / "artifacts/metrics/semantic_vs_reference.json"

BOOTSTRAP_REPLICATES = 2000


def load_validation(path: Path) -> pd.DataFrame:
    """Load validation data only."""
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError("Official ESCI test rows are not allowed.")

    return df


def mean_metrics(per_query: pd.DataFrame) -> dict[str, float]:
    summary = summarize_query_metrics(per_query, k=DEFAULT_K)

    return {
        f"ndcg@{DEFAULT_K}": summary[f"ndcg@{DEFAULT_K}"]["mean"],
        f"recall@{DEFAULT_K}": summary[f"recall@{DEFAULT_K}"]["mean"],
        "mrr": summary["mrr"]["mean"],
    }


def main() -> None:
    reference = load_validation(REFERENCE_VALIDATION)
    semantic = load_validation(SEMANTIC_VALIDATION)

    assert_same_candidates(reference, semantic)

    reference_ranked = score_saved_ranker(
        REFERENCE_MODEL,
        reference,
        FEATURES,
        score_col="reference_score",
    )

    semantic_ranked = score_saved_ranker(
        SEMANTIC_MODEL,
        semantic,
        SEMANTIC_FEATURES,
        score_col="semantic_score",
    )

    reference_query = per_query_metrics(
        reference_ranked,
        k=DEFAULT_K,
        model_name="reference_ltr",
    )

    semantic_query = per_query_metrics(
        semantic_ranked,
        k=DEFAULT_K,
        model_name="semantic_ltr",
    )

    paired = paired_metric_table(
        semantic_query,
        reference_query,
        left_prefix="semantic",
        right_prefix="reference",
        k=DEFAULT_K,
    )

    _, comparison = paired_summary(
        semantic_query.rename(
            columns={f"ndcg@{DEFAULT_K}": "semantic_ndcg"}
        ),
        reference_query.rename(
            columns={f"ndcg@{DEFAULT_K}": "reference_ndcg"}
        ),
        left_metric="semantic_ndcg",
        right_metric="reference_ndcg",
    )

    bootstrap = paired_bootstrap_mean_ci(
        paired[f"delta_ndcg_at_{DEFAULT_K}"].to_numpy(),
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=SEED,
    )

    reference_means = mean_metrics(reference_query)
    semantic_means = mean_metrics(semantic_query)

    summary = {
        "Reference LTR": reference_means,
        "Semantic LTR": semantic_means,
        "official_test_used": False,
    }

    comparison_summary = {
        "metric": f"ndcg@{DEFAULT_K}",
        "comparison": "semantic_ltr - reference_ltr",
        **comparison,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_ci_95": [
            bootstrap["ci_lower"],
            bootstrap["ci_upper"],
        ],
        "official_test_used": False,
    }

    write_json(SUMMARY_PATH, summary)
    write_json(COMPARISON_PATH, comparison_summary)
    paired.to_csv(PER_QUERY_PATH, index=False)

    print("\nValidation results")
    print(f"{'Model':<18} {'NDCG@10':>10} {'Recall@10':>10} {'MRR':>10}")

    for name, metrics in [
        ("Reference LTR", reference_means),
        ("Semantic LTR", semantic_means),
    ]:
        print(
            f"{name:<18} "
            f"{metrics['ndcg@10']:10.4f} "
            f"{metrics['recall@10']:10.4f} "
            f"{metrics['mrr']:10.4f}"
        )

    print(
        f"\nΔNDCG@10: "
        f"mean={comparison['mean_delta']:.6f}, "
        f"median={comparison['median_delta']:.6f}, "
        f"wins={comparison['wins']}, "
        f"losses={comparison['losses']}, "
        f"ties={comparison['ties']}"
    )

    print(
        f"Bootstrap 95% CI: "
        f"[{bootstrap['ci_lower']:.6f}, "
        f"{bootstrap['ci_upper']:.6f}]"
    )

    print("Official ESCI test used: NO")


if __name__ == "__main__":
    main()
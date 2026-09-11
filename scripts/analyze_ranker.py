"""Analyze the approved XGBRanker with importance, ablation, and examples."""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import write_json
from src.evaluation.ranker_analysis import (
    ABLATIONS,
    add_model_ranks,
    add_query_length_segment,
    extract_query_examples,
    gain_importance_table,
    run_ablation,
    score_full_model,
    validate_ablations,
)
from src.ranking.groups import assert_queries_disjoint
from src.ranking.ranker import load_ranker


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FEATURES = PROJECT_ROOT / "artifacts/features/train_features.parquet"
VALIDATION_FEATURES = PROJECT_ROOT / "artifacts/features/validation_features.parquet"
MODEL_PATH = PROJECT_ROOT / "artifacts/models/xgb_ranker.json"

RANKER_SUMMARY = PROJECT_ROOT / "artifacts/metrics/ranker_summary.json"
RANKER_PER_QUERY = PROJECT_ROOT / "artifacts/metrics/ranker_per_query.csv"

IMPORTANCE_PATH = PROJECT_ROOT / "artifacts/metrics/ranker_feature_importance.csv"
ABLATION_PATH = PROJECT_ROOT / "artifacts/metrics/ranker_ablation.csv"
SEGMENTS_PATH = PROJECT_ROOT / "artifacts/metrics/ranker_segments.csv"
EXAMPLES_PATH = PROJECT_ROOT / "artifacts/examples/ranker_query_examples.csv"
ANALYSIS_SUMMARY = PROJECT_ROOT / "artifacts/metrics/ranker_analysis_summary.json"


INSPECTION_QUERY_IDS = [
    100955,
    107914,
    18994,
    67057,
    78865,
    102658,
]


def load_features(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError(f"{path.name} contains official test rows.")

    return df


def main() -> None:
    validate_ablations()

    train = load_features(TRAIN_FEATURES)
    validation = load_features(VALIDATION_FEATURES)
    assert_queries_disjoint(train, validation)

    ranker = load_ranker(MODEL_PATH)

    # 1. Feature importance
    importance = gain_importance_table(ranker)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    # 2. Ablation
    approved = json.loads(RANKER_SUMMARY.read_text())

    full_metrics = approved["XGBRanker"]
    full_ndcg = full_metrics["ndcg@10"]
    tfidf_ndcg = approved["baseline_references"]["tfidf"]["ndcg@10"]

    rows = [
        {
            "variant": "full",
            "n_features": len(ABLATIONS["full"]),
            **full_metrics,
            "delta_vs_full": 0.0,
            "delta_vs_tfidf": full_ndcg - tfidf_ndcg,
        }
    ]

    for name, features in ABLATIONS.items():
        if name == "full":
            continue

        print(f"[analysis] running {name} ...", flush=True)

        metrics = run_ablation(
            train,
            validation,
            features,
        )

        rows.append(
            {
                "variant": name,
                "n_features": len(features),
                **metrics,
                "delta_vs_full": metrics["ndcg@10"] - full_ndcg,
                "delta_vs_tfidf": metrics["ndcg@10"] - tfidf_ndcg,
            }
        )

    ablation = pd.DataFrame(rows)
    ablation.to_csv(ABLATION_PATH, index=False)

    # 3. Simple query-length segmentation
    per_query = pd.read_csv(RANKER_PER_QUERY)

    segmented = add_query_length_segment(per_query)

    segments = (
        segmented.groupby("query_length_segment")["delta_ndcg_at_10"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(
            columns={
                "count": "n_queries",
                "mean": "mean_delta_ndcg_at_10",
            }
        )
    )

    segments.to_csv(SEGMENTS_PATH, index=False)

    # 4. Representative query examples
    scored = score_full_model(
        ranker,
        validation,
    )
    ranked = add_model_ranks(scored)

    examples = extract_query_examples(
        ranked,
        INSPECTION_QUERY_IDS,
        per_query=per_query,
    )
    examples.to_csv(EXAMPLES_PATH, index=False)

    # Compact machine-readable summary
    write_json(
        ANALYSIS_SUMMARY,
        {
            "primary_model_retrained": False,
            "official_test_used": False,
            "feature_importance": importance.to_dict("records"),
            "ablation": ablation.to_dict("records"),
            "segments": segments.to_dict("records"),
            "inspection_query_ids": INSPECTION_QUERY_IDS,
        },
    )

    print("\nFeature importance")
    print(
        importance[
            ["rank", "feature", "gain_importance"]
        ].to_string(index=False)
    )

    print("\nAblation")
    print(
        ablation[
            [
                "variant",
                "ndcg@10",
                "delta_vs_full",
                "delta_vs_tfidf",
            ]
        ].to_string(index=False)
    )

    print(f"\nWrote {IMPORTANCE_PATH}")
    print(f"Wrote {ABLATION_PATH}")
    print(f"Wrote {SEGMENTS_PATH}")
    print(f"Wrote {EXAMPLES_PATH}")
    print(f"Wrote {ANALYSIS_SUMMARY}")


if __name__ == "__main__":
    main()

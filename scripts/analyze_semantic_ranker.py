#!/usr/bin/env python3
"""Analyze where semantic ranking helps or hurts."""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import write_json
from src.evaluation.ranker_scoring import score_saved_ranker
from src.evaluation.semantic_diagnostics import (
    add_comparison_ranks,
    annotate_diagnostic_queries,
    extract_semantic_examples,
    gain_importance_table,
    select_diagnostic_queries,
    summarize_segments,
)
from src.features.build import FEATURES
from src.features.semantic import SEMANTIC_FEATURE, SEMANTIC_FEATURES
from src.ranking.ranker import load_ranker


ROOT = Path(__file__).resolve().parents[1]

METRICS = ROOT / "artifacts/metrics"
FEATURES_DIR = ROOT / "artifacts/features"
MODELS = ROOT / "artifacts/models"
EXAMPLES = ROOT / "artifacts/examples"

PER_QUERY = METRICS / "semantic_ranker_per_query.csv"
COMPARISON = METRICS / "semantic_vs_reference.json"

REFERENCE_FEATURES = FEATURES_DIR / "validation_features.parquet"
SEMANTIC_FEATURES_PATH = FEATURES_DIR / "validation_features_semantic.parquet"

REFERENCE_MODEL = MODELS / "xgb_ranker.json"
SEMANTIC_MODEL = MODELS / "xgb_ranker_semantic.json"

IMPORTANCE_OUT = METRICS / "semantic_feature_importance.csv"
SEGMENTS_OUT = METRICS / "semantic_segments.csv"
EXAMPLES_OUT = EXAMPLES / "semantic_query_examples.csv"
SUMMARY_OUT = METRICS / "semantic_diagnostics_summary.json"


SEGMENTS = [
    "query_length_segment",
    "overlap_segment",
    "disagreement_segment",
    "identifier_segment",
]


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError("Official ESCI test rows are not allowed.")

    return df


def main() -> None:
    per_query = pd.read_csv(PER_QUERY)
    candidates = load_parquet(SEMANTIC_FEATURES_PATH)
    comparison = json.loads(COMPARISON.read_text())

    # Query-level diagnostics
    annotated = annotate_diagnostic_queries(
        per_query,
        candidates,
    )

    segments = pd.concat(
        [
            summarize_segments(annotated, column)
            for column in SEGMENTS
        ],
        ignore_index=True,
    )

    # Feature importance
    ranker = load_ranker(SEMANTIC_MODEL)

    importance = gain_importance_table(
        ranker,
        SEMANTIC_FEATURES,
    )

    semantic_rank = int(
        importance.loc[
            importance["feature"] == SEMANTIC_FEATURE,
            "rank",
        ].iloc[0]
    )

    # Representative query examples
    selected = select_diagnostic_queries(annotated)
    query_ids = selected["query_id"]

    reference = load_parquet(REFERENCE_FEATURES)
    reference = reference[
        reference["query_id"].isin(query_ids)
    ]

    semantic = candidates[
        candidates["query_id"].isin(query_ids)
    ]

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

    scored = semantic_ranked.merge(
        reference_ranked[
            ["example_id", "reference_score"]
        ],
        on="example_id",
        validate="one_to_one",
    )

    ranked = add_comparison_ranks(scored)

    examples = extract_semantic_examples(
        ranked,
        selected,
    )

    # Save results
    importance.to_csv(IMPORTANCE_OUT, index=False)
    segments.to_csv(SEGMENTS_OUT, index=False)
    examples.to_csv(EXAMPLES_OUT, index=False)

    write_json(
        SUMMARY_OUT,
        {
            "delta_ndcg@10": comparison["mean_delta"],
            "semantic_similarity_rank": semantic_rank,
            "segments": segments.to_dict("records"),
            "official_test_used": False,
        },
    )

    print("\nFeature importance")
    print(
        importance[
            ["rank", "feature", "gain_importance"]
        ].to_string(index=False)
    )

    print("\nSegments")
    print(segments.to_string(index=False))

    print("\nIllustrative extreme queries")
    print(
        selected[
            ["kind", "query", "delta_ndcg_at_10"]
        ].to_string(index=False)
    )

    print("\nOfficial ESCI test used: NO")


if __name__ == "__main__":
    main()

"""Tests for semantic diagnostic helpers."""

import pandas as pd
import pytest

from src.evaluation.semantic_diagnostics import (
    SemanticDiagnosticError,
    annotate_diagnostic_queries,
    extract_semantic_examples,
    gain_importance_table,
    relevant_overlap_by_query,
    select_diagnostic_queries,
    summarize_segments,
)
from src.features.semantic import SEMANTIC_FEATURES


def candidates():
    return pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2, 3, 3],
            "query": [
                "red running shoes",
                "red running shoes",
                "sony rx100",
                "sony rx100",
                "laptop for work",
                "laptop for work",
            ],
            "example_id": [11, 12, 21, 22, 31, 32],
            "product_id": ["a", "b", "c", "d", "e", "f"],
            "product_title": [
                "red running shoes",
                "blue boots",
                "sony camera",
                "canon lens",
                "office laptop",
                "desktop warranty",
            ],
            "esci_label": ["E", "I", "E", "I", "E", "I"],
            "relevance_gain": [3, 0, 3, 0, 3, 0],
            "query_token_coverage": [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            "tfidf_score": [0.9, 0.1, 0.8, 0.2, 0.1, 0.9],
            "semantic_similarity": [0.9, 0.2, 0.7, 0.6, 0.8, 0.1],
            "bm25_score": [2.0, 0.2, 1.5, 0.3, 0.2, 1.8],
        }
    )


def per_query():
    return pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": [
                "red running shoes",
                "sony rx100",
                "laptop for work",
            ],
            "n_candidates": [2, 2, 2],
            "semantic_ndcg_at_10": [0.9, 0.6, 0.8],
            "reference_ndcg_at_10": [0.7, 0.8, 0.4],
            "delta_ndcg_at_10": [0.2, -0.2, 0.4],
        }
    )


def test_relevant_overlap_by_query():
    overlap = relevant_overlap_by_query(candidates())

    assert overlap.loc[1] == pytest.approx(1.0)
    assert overlap.loc[2] == pytest.approx(0.5)
    assert overlap.loc[3] == pytest.approx(0.0)


def test_annotate_and_summarize_segments():
    annotated = annotate_diagnostic_queries(
        per_query(),
        candidates(),
    )

    assert "overlap_segment" in annotated.columns
    assert "disagreement_segment" in annotated.columns
    assert "identifier_segment" in annotated.columns

    assert annotated.loc[
        annotated["query_id"] == 2,
        "identifier_segment",
    ].iloc[0] == "identifier-style"

    summary = summarize_segments(
        annotated,
        "identifier_segment",
        min_queries=1,
    )

    identifier = summary[
        summary["segment"] == "identifier-style"
    ].iloc[0]

    assert identifier["n_queries"] == 1
    assert identifier["delta_ndcg_at_10"] == pytest.approx(-0.2)


def test_annotation_requires_matching_queries():
    smaller_candidates = candidates()
    smaller_candidates = smaller_candidates[
        smaller_candidates["query_id"] != 3
    ]

    with pytest.raises(SemanticDiagnosticError):
        annotate_diagnostic_queries(
            per_query(),
            smaller_candidates,
        )


def test_select_diagnostic_queries():
    annotated = annotate_diagnostic_queries(
        per_query(),
        candidates(),
    )

    selected = select_diagnostic_queries(
        annotated,
        n=1,
    )

    improvement = selected[
        selected["kind"] == "large_improvement"
    ]

    regression = selected[
        selected["kind"] == "large_regression"
    ]

    assert improvement["query_id"].iloc[0] == 3
    assert regression["query_id"].iloc[0] == 2


def test_gain_importance():
    class FakeBooster:
        def get_score(self, importance_type="gain"):
            return {
                "tfidf_score": 2.0,
                "semantic_similarity": 3.0,
            }

    class FakeRanker:
        def get_booster(self):
            return FakeBooster()

    table = gain_importance_table(
        FakeRanker(),
        SEMANTIC_FEATURES,
    )

    assert table.iloc[0]["feature"] == "semantic_similarity"
    assert table.iloc[0]["rank"] == 1
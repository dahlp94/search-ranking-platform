"""Tests for ranker interpretation helpers."""

import pandas as pd
import pytest

from src.evaluation.ranker_analysis import (
    ABLATIONS,
    add_query_length_segment,
    extract_query_examples,
    validate_ablations,
)
from src.features.build import FEATURES
from src.ranking.groups import RankingGroupError, prepare_grouped_subset


def test_ablation_feature_sets_are_valid():
    validate_ablations()

    assert ABLATIONS["full"] == list(FEATURES)
    assert ABLATIONS["lexical_scores"] == [
        "bm25_score",
        "tfidf_score",
    ]

    assert "tfidf_score" not in ABLATIONS["minus_tfidf"]
    assert "bm25_score" not in ABLATIONS["minus_bm25"]
    assert "query_token_coverage" not in ABLATIONS["minus_overlap"]
    assert "brand_match" not in ABLATIONS["minus_structured"]


def test_grouped_subset_rejects_identifiers():
    frame = pd.DataFrame(
        {
            "example_id": [1, 2],
            "query_id": [10, 10],
            "relevance_gain": [3, 0],
            "bm25_score": [1.0, 0.1],
            "tfidf_score": [0.8, 0.2],
        }
    )

    with pytest.raises(
        RankingGroupError,
        match="Identifier or target",
    ):
        prepare_grouped_subset(
            frame,
            ["bm25_score", "query_id"],
        )


def test_query_example_extraction_preserves_identity():
    ranked = pd.DataFrame(
        {
            "query_id": [5, 5, 8],
            "query": ["q5", "q5", "q8"],
            "example_id": [11, 12, 21],
            "product_id": ["p1", "p2", "p3"],
            "product_title": ["a", "b", "c"],
            "esci_label": ["E", "I", "S"],
            "relevance_gain": [3, 0, 2],
            "bm25_score": [1.0, 0.1, 0.4],
            "tfidf_score": [0.9, 0.2, 0.3],
            "xgb_score": [0.8, 0.1, 0.5],
            "bm25_rank": [1, 2, 1],
            "tfidf_rank": [1, 2, 1],
            "xgb_rank": [1, 2, 1],
        }
    )

    examples = extract_query_examples(
        ranked,
        [5, 8],
    )

    assert set(examples["query_id"]) == {5, 8}
    assert set(examples["example_id"]) == {11, 12, 21}
    assert set(examples["product_id"]) == {"p1", "p2", "p3"}
    assert len(examples) == 3


def test_query_length_segmentation():
    per_query = pd.DataFrame(
        {
            "query_id": [1, 2, 3],
            "query": [
                "nike",
                "running shoes",
                "red nike running shoes extra",
            ],
        }
    )

    segmented = add_query_length_segment(per_query)

    assert segmented["query_length_segment"].tolist() == [
        "1 token",
        "2–3 tokens",
        "4+ tokens",
    ]

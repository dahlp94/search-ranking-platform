"""Tests for ranking features."""

import math

import pytest

from src.features.build import (
    FEATURES,
    build_feature_datasets,
    build_feature_table,
)
from src.features.features import add_ranking_features
from src.retrieval.baselines import rank_tfidf
from src.retrieval.bm25 import rank_candidates
from tests.fixtures.make_frames import ranking_candidates


def make_data():
    df = ranking_candidates()

    df["product_brand"] = [
        "Nike",
        None,
        "Adidas",
    ]

    df["product_bullet_point"] = [
        "Lightweight running trainers",
        None,
        "Rubber sole running shoes",
    ]

    return df


def test_overlap_features():
    featured = add_ranking_features(make_data())

    nike = featured[
        featured["product_id"] == "p_nike"
    ].iloc[0]

    assert nike["shared_token_count"] == 3
    assert nike["query_token_coverage"] == pytest.approx(1.0)
    assert nike["exact_query_in_title"] == 1
    assert nike["title_token_count"] == 6
    assert nike["token_length_difference"] == -3


def test_brand_and_bullet_features():
    featured = add_ranking_features(make_data())

    nike = featured[
        featured["product_id"] == "p_nike"
    ].iloc[0]

    blender = featured[
        featured["product_id"] == "p_blender"
    ].iloc[0]

    assert nike["brand_match"] == 1
    assert nike["query_bullet_coverage"] == pytest.approx(1 / 3)

    assert blender["brand_match"] == 0


def test_missing_brand_and_bullets_become_zero():
    featured = add_ranking_features(
        ranking_candidates()
    )

    assert (featured["brand_match"] == 0).all()

    assert (
        featured["query_bullet_coverage"] == 0
    ).all()


def test_feature_ranges():
    featured = add_ranking_features(make_data())

    assert set(
        featured["brand_match"].unique()
    ) <= {0, 1}

    assert set(
        featured["exact_query_in_title"].unique()
    ) <= {0, 1}

    coverage_features = [
        "query_token_coverage",
        "title_token_coverage",
        "query_bullet_coverage",
    ]

    for column in coverage_features:
        assert featured[column].between(0, 1).all()


def test_feature_table_has_no_missing_values():
    featured = build_feature_table(make_data())

    assert not featured[FEATURES].isna().any().any()


def test_feature_table_preserves_rows():
    source = make_data()

    featured = build_feature_table(source)

    assert len(featured) == len(source)

    assert (
        featured["example_id"].tolist()
        == source["example_id"].tolist()
    )


def test_model_features_exclude_ids_and_target():
    forbidden = {
        "example_id",
        "query_id",
        "product_id",
        "query",
        "esci_label",
        "relevance_gain",
    }

    assert forbidden.isdisjoint(FEATURES)


def test_lexical_scores_match_existing_baselines():
    source = make_data()

    featured = add_ranking_features(source)

    bm25 = rank_candidates(
        "nike running shoes",
        source,
        text_col="product_title",
    )

    tfidf = rank_tfidf(
        "nike running shoes",
        source,
        text_col="product_title",
    )

    bm25_scores = dict(
        zip(
            bm25["example_id"],
            bm25["bm25_score"],
        )
    )

    tfidf_scores = dict(
        zip(
            tfidf["example_id"],
            tfidf["tfidf_score"],
        )
    )

    for _, row in featured.iterrows():
        example_id = row["example_id"]

        assert math.isclose(
            row["bm25_score"],
            bm25_scores[example_id],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

        assert math.isclose(
            row["tfidf_score"],
            tfidf_scores[example_id],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_train_validation_queries_must_be_disjoint():
    train = make_data()

    validation = make_data().assign(
        query_id=202
    )

    build_feature_datasets(
        train,
        validation,
    )

    overlapping = make_data()

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        build_feature_datasets(
            train,
            overlapping,
        )
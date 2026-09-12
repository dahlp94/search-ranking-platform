"""Tests for semantic query-title similarity."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.semantic.embeddings import (
    INPUT_COLUMNS,
    assert_queries_disjoint,
    build_partition_semantic_features,
    clean_text,
    encode_unique_texts,
    extract_semantic_input,
)


class FakeEncoder:
    """Small deterministic encoder for unit tests."""

    def __init__(self, dimension: int = 16):
        self.embedding_dimension = dimension
        self.encoded_texts = []

    def encode(self, texts, *, batch_size: int, normalize: bool = True):
        texts = list(texts)
        self.encoded_texts.extend(texts)

        vectors = np.stack([self._vector(text) for text in texts])

        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        return vectors.astype(np.float32)

    def _vector(self, text: str):
        vector = np.zeros(self.embedding_dimension)

        if not text:
            vector[0] = 1.0
            return vector

        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.embedding_dimension
            vector[index] += 1.0

        return vector


def train_rows():
    return pd.DataFrame(
        {
            "example_id": [1, 2, 3],
            "query_id": [101, 101, 101],
            "product_id": ["p1", "p2", "p3"],
            "query": ["nike running shoes"] * 3,
            "product_title": [
                "Nike Running Shoes",
                "Kitchen Blender",
                "Adidas Running Shoes",
            ],
            "relevance_gain": [3, 0, 2],
            "bm25_score": [1.2, 0.1, 0.8],
        }
    )


def validation_rows():
    return pd.DataFrame(
        {
            "example_id": [10, 11],
            "query_id": [202, 202],
            "product_id": ["p4", "p5"],
            "query": ["kitchen blender"] * 2,
            "product_title": [
                "Kitchen Blender",
                "Nike Running Shoes",
            ],
        }
    )


def test_extract_semantic_input():
    source = train_rows()

    result = extract_semantic_input(source)

    assert len(result) == len(source)
    assert list(result.columns) == INPUT_COLUMNS
    assert result["example_id"].tolist() == source["example_id"].tolist()
    assert "relevance_gain" not in result.columns
    assert "bm25_score" not in result.columns


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""
    assert clean_text("   ") == ""
    assert clean_text(" Nike Shoes ") == "Nike Shoes"


def test_unique_texts_are_encoded_once():
    encoder = FakeEncoder()

    vectors, stats = encode_unique_texts(
        ["nike shoes", "nike shoes", "blender"],
        encoder,
        batch_size=8,
    )

    assert vectors.shape[0] == 3
    assert stats["n_unique_texts"] == 2
    assert stats["n_reused"] == 1
    assert encoder.encoded_texts == ["nike shoes", "blender"]


def test_semantic_features_preserve_rows_and_identity():
    train = train_rows()
    validation = validation_rows()

    train_out, validation_out, _ = build_partition_semantic_features(
        train,
        validation,
        FakeEncoder(),
        batch_size=8,
    )

    assert len(train_out) == len(train)
    assert len(validation_out) == len(validation)

    assert train_out["example_id"].tolist() == train["example_id"].tolist()
    assert validation_out["example_id"].tolist() == validation["example_id"].tolist()


def test_semantic_similarity_is_finite_and_bounded():
    train_out, validation_out, _ = build_partition_semantic_features(
        train_rows(),
        validation_rows(),
        FakeEncoder(),
        batch_size=8,
    )

    values = pd.concat(
        [
            train_out["semantic_similarity"],
            validation_out["semantic_similarity"],
        ]
    ).to_numpy()

    assert np.isfinite(values).all()
    assert values.min() >= -1.0
    assert values.max() <= 1.0


def test_related_title_scores_above_unrelated_title():
    train_out, _, _ = build_partition_semantic_features(
        train_rows(),
        validation_rows(),
        FakeEncoder(),
        batch_size=8,
    )

    related = train_out.loc[0, "semantic_similarity"]
    unrelated = train_out.loc[1, "semantic_similarity"]

    assert related > unrelated


def test_encoding_is_deterministic():
    first, _, _ = build_partition_semantic_features(
        train_rows(),
        validation_rows(),
        FakeEncoder(),
        batch_size=8,
    )

    second, _, _ = build_partition_semantic_features(
        train_rows(),
        validation_rows(),
        FakeEncoder(),
        batch_size=8,
    )

    np.testing.assert_allclose(
        first["semantic_similarity"],
        second["semantic_similarity"],
    )


def test_train_validation_queries_must_be_disjoint():
    train = train_rows()
    validation = validation_rows()

    assert_queries_disjoint(train, validation)

    with pytest.raises(ValueError, match="overlap"):
        assert_queries_disjoint(train, train)


def test_official_test_split_is_rejected():
    source = train_rows()
    source["split"] = "test"

    with pytest.raises(ValueError, match="Official ESCI test"):
        extract_semantic_input(source)
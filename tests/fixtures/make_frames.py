"""Shared synthetic ESCI-like frames for unit tests.

These fixtures are tiny and invented. They are not Amazon data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import RELEVANCE_GAIN
from src.retrieval.text import add_product_text_columns


def tiny_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["p_nike", "p_adidas", "p_blender", "p_jp"],
            "product_title": [
                "Red Nike Running Shoes Size 10",
                "Blue Adidas Running Shoes",
                "Kitchen Blender 500W",
                "Japanese Only Product",
            ],
            "product_description": ["Lightweight trainers", None, "For smoothies", "desc"],
            "product_bullet_point": ["Mesh upper", "Rubber sole", "Glass jar", None],
            "product_brand": ["Nike", "Adidas", "BlendCo", "JPBrand"],
            "product_color": ["red", "blue", None, "black"],
            "product_locale": ["us", "us", "us", "jp"],
        }
    )


def tiny_examples() -> pd.DataFrame:
    rows = [
        # query 1: nike shoes, official train
        (1, "nike running shoes", 101, "p_nike", "us", "E", 1, 1, "train"),
        (2, "nike running shoes", 101, "p_adidas", "us", "S", 1, 1, "train"),
        (3, "nike running shoes", 101, "p_blender", "us", "I", 1, 1, "train"),
        # query 2: blender, official train
        (4, "kitchen blender", 102, "p_blender", "us", "E", 1, 1, "train"),
        (5, "kitchen blender", 102, "p_nike", "us", "I", 1, 1, "train"),
        # query 3: official test holdout
        (6, "adidas shoes", 201, "p_adidas", "us", "E", 1, 1, "test"),
        (7, "adidas shoes", 201, "p_nike", "us", "S", 1, 1, "test"),
        # rows that must be filtered out
        (8, "japanese query", 301, "p_jp", "jp", "E", 1, 1, "train"),
        (9, "large only", 401, "p_nike", "us", "E", 0, 1, "train"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
            "split",
        ],
    )


def ranking_candidates() -> pd.DataFrame:
    """Single-query candidate set used by ranking tests."""
    df = pd.DataFrame(
        {
            "example_id": [1, 2, 3],
            "query": ["nike running shoes"] * 3,
            "query_id": [101] * 3,
            "product_id": ["p_nike", "p_blender", "p_adidas"],
            "product_title": [
                "Red Nike Running Shoes Size 10",
                "Kitchen Blender 500W",
                "Blue Adidas Running Shoes",
            ],
            "esci_label": ["E", "I", "S"],
        }
    )
    df["relevance_gain"] = df["esci_label"].map(RELEVANCE_GAIN)
    return add_product_text_columns(df)


def write_raw_parquet(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    examples_path = raw_dir / "shopping_queries_dataset_examples.parquet"
    products_path = raw_dir / "shopping_queries_dataset_products.parquet"
    tiny_examples().to_parquet(examples_path, index=False)
    tiny_products().to_parquet(products_path, index=False)
    return examples_path, products_path


def write_named_tiny_fixtures(fixture_dir: Path) -> tuple[Path, Path]:
    """Write clearly named synthetic parquets (never use official ESCI filenames)."""
    fixture_dir.mkdir(parents=True, exist_ok=True)
    examples_path = fixture_dir / "tiny_examples.parquet"
    products_path = fixture_dir / "tiny_products.parquet"
    tiny_examples().to_parquet(examples_path, index=False)
    tiny_products().to_parquet(products_path, index=False)
    return examples_path, products_path

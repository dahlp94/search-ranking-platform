"""Light text normalization shared by queries and product fields.

This is a lexical baseline. We lowercase and squeeze whitespace, but we do
not stem, lemmatize, or strip model numbers / SKUs. Those tokens are often
the exact signal a shopper typed.
"""

from __future__ import annotations

import re

import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
# Keep alphanumeric tokens and simple model-number punctuation such as "xbox-one" or "v1.2".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")

DEFAULT_TITLE_FIELDS = ("product_title",)
DEFAULT_COMBINED_FIELDS = ("product_title", "product_brand", "product_bullet_point")


def normalize_text(value: object) -> str:
    """Lowercase and collapse whitespace. Missing values become empty strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return _WHITESPACE_RE.sub(" ", text).lower()


def tokenize(value: object) -> list[str]:
    """Tokenize with the same rules used for queries and candidate documents."""
    return _TOKEN_RE.findall(normalize_text(value))


def build_product_text(row: pd.Series | dict, fields: tuple[str, ...] = DEFAULT_TITLE_FIELDS) -> str:
    """Join selected product fields into one lexical document."""
    parts: list[str] = []
    for field in fields:
        text = normalize_text(row.get(field, "") if hasattr(row, "get") else row[field])
        if text:
            parts.append(text)
    return " ".join(parts)


def add_product_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add title-only and title+brand+bullets text fields.

    Baselines use `product_text` (title only) unless a caller chooses otherwise.
    """
    df = df.copy()
    df["product_text"] = _combine_fields(df, DEFAULT_TITLE_FIELDS)
    df["product_text_combined"] = _combine_fields(df, DEFAULT_COMBINED_FIELDS)
    n_empty = int(df["product_text"].eq("").sum())
    print(
        f"[text] built product_text from title; "
        f"{n_empty:,} / {len(df):,} rows have empty title text."
    )
    return df


def _combine_fields(df: pd.DataFrame, fields: tuple[str, ...]) -> pd.Series:
    combined = pd.Series("", index=df.index, dtype="object")
    for field in fields:
        if field not in df.columns:
            continue
        part = df[field].map(normalize_text)
        combined = (combined + " " + part).map(normalize_text)
    return combined

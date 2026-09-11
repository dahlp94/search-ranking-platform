"""Ranking feature calculations."""

import numpy as np
import pandas as pd

from src.retrieval.baselines import candidate_tfidf_scores
from src.retrieval.bm25 import candidate_bm25_scores
from src.retrieval.text import tokenize


def _contains_phrase(needle: list[str], haystack: list[str]) -> bool:
    """Return True if one token sequence appears inside another."""
    if not needle:
        return False

    n = len(needle)

    return any(
        haystack[i : i + n] == needle
        for i in range(len(haystack) - n + 1)
    )


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the inexpensive query-product features."""

    required = {"query_id", "query", "product_title"}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)

    brands = (
        out["product_brand"]
        if "product_brand" in out.columns
        else pd.Series([None] * len(out))
    )

    bullets = (
        out["product_bullet_point"]
        if "product_bullet_point" in out.columns
        else pd.Series([None] * len(out))
    )

    shared_counts = []
    query_coverages = []
    title_coverages = []
    exact_matches = []
    brand_matches = []
    title_lengths = []
    length_differences = []
    bullet_coverages = []

    for query, title, brand, bullet in zip(
        out["query"],
        out["product_title"],
        brands,
        bullets,
    ):
        query_tokens = tokenize(query)
        title_tokens = tokenize(title)

        query_set = set(query_tokens)
        title_set = set(title_tokens)

        shared = len(query_set & title_set)

        shared_counts.append(shared)

        query_coverages.append(
            shared / len(query_set) if query_set else 0.0
        )

        title_coverages.append(
            shared / len(title_set) if title_set else 0.0
        )

        exact_matches.append(
            int(_contains_phrase(query_tokens, title_tokens))
        )

        brand_tokens = tokenize(brand)

        brand_matches.append(
            int(
                bool(brand_tokens)
                and _contains_phrase(brand_tokens, query_tokens)
            )
        )

        title_lengths.append(len(title_tokens))

        length_differences.append(
            len(query_tokens) - len(title_tokens)
        )

        bullet_tokens = set(tokenize(bullet))

        bullet_coverages.append(
            len(query_set & bullet_tokens) / len(query_set)
            if query_set
            else 0.0
        )

    out["shared_token_count"] = shared_counts
    out["query_token_coverage"] = query_coverages
    out["title_token_coverage"] = title_coverages
    out["exact_query_in_title"] = exact_matches
    out["brand_match"] = brand_matches
    out["title_token_count"] = title_lengths
    out["token_length_difference"] = length_differences
    out["query_bullet_coverage"] = bullet_coverages

    return out


def add_lexical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candidate-local BM25 and TF-IDF scores."""

    out = df.copy()

    bm25_scores = np.zeros(len(out))
    tfidf_scores = np.zeros(len(out))

    groups = out.groupby("query_id", sort=False).indices
    total_queries = len(groups)

    for i, indices in enumerate(groups.values(), start=1):
        query = out.iloc[indices[0]]["query"]
        documents = out.iloc[indices]["product_title"].tolist()

        bm25_scores[indices] = candidate_bm25_scores(
            query,
            documents,
        )

        tfidf_scores[indices] = candidate_tfidf_scores(
            query,
            documents,
        )

        if i % 2000 == 0 or i == total_queries:
            print(
                f"Scored {i:,}/{total_queries:,} queries",
                flush=True,
            )

    out["bm25_score"] = bm25_scores
    out["tfidf_score"] = tfidf_scores

    return out


def add_cached_lexical_scores(
    df: pd.DataFrame,
    cached_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Reuse previously computed BM25 and TF-IDF scores."""

    required = {
        "example_id",
        "bm25_score",
        "tfidf_score",
    }

    missing = required - set(cached_scores.columns)

    if missing:
        raise ValueError(
            f"Cached scores are missing columns: {sorted(missing)}"
        )

    if not df["example_id"].is_unique:
        raise ValueError("example_id must be unique.")

    if not cached_scores["example_id"].is_unique:
        raise ValueError("Cached example_id values must be unique.")

    if set(df["example_id"]) != set(cached_scores["example_id"]):
        raise ValueError(
            "Cached lexical scores do not match the current data."
        )

    scores = cached_scores.set_index("example_id")

    out = df.copy()

    out["bm25_score"] = out["example_id"].map(
        scores["bm25_score"]
    )

    out["tfidf_score"] = out["example_id"].map(
        scores["tfidf_score"]
    )

    print("Reused cached BM25 and TF-IDF scores.")

    return out


def add_ranking_features(
    df: pd.DataFrame,
    cached_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add all ranking features."""

    out = add_basic_features(df)

    if cached_scores is not None:
        out = add_cached_lexical_scores(
            out,
            cached_scores,
        )
    else:
        out = add_lexical_scores(out)

    return out
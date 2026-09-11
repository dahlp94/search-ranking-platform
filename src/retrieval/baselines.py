"""Simple lexical baselines for candidate re-ranking."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import SEED
from src.retrieval.ranking import sort_and_rank
from src.retrieval.text import normalize_text, tokenize


def _query_seed(query_id: object, seed: int) -> int:
    """Create a stable random seed for one query."""
    text = f"{seed}:{query_id}".encode()
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def rank_random(
    query: str,
    candidates: pd.DataFrame,
    k: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Randomly rank candidates using a reproducible per-query seed."""
    del query

    if candidates.empty:
        out = candidates.copy()
        out["random_score"] = []
        out["predicted_rank"] = []
        return out

    if "query_id" not in candidates.columns:
        raise KeyError("Missing query_id column.")

    query_ids = candidates["query_id"].unique()
    if len(query_ids) != 1:
        raise ValueError("Candidates must belong to one query.")

    out = candidates.sort_values("product_id", kind="mergesort").copy()

    rng = np.random.default_rng(_query_seed(query_ids[0], seed))
    out["random_score"] = rng.random(len(out))

    return sort_and_rank(out, score_col="random_score", k=k)


def candidate_tfidf_scores(
    query: str,
    documents: Sequence[object],
) -> np.ndarray:
    """Compute TF-IDF cosine similarity within one query's candidate set."""
    docs = [normalize_text(doc) for doc in documents]
    query = normalize_text(query)

    if not docs or not tokenize(query) or not any(tokenize(doc) for doc in docs):
        return np.zeros(len(docs))

    vectorizer = TfidfVectorizer(
        analyzer=tokenize,
        lowercase=False,
        norm="l2",
    )

    try:
        doc_matrix = vectorizer.fit_transform(docs)
        query_vector = vectorizer.transform([query])
        return cosine_similarity(query_vector, doc_matrix).ravel()
    except ValueError:
        return np.zeros(len(docs))


def rank_tfidf(
    query: str,
    candidates: pd.DataFrame,
    k: int | None = None,
    text_col: str = "product_text",
) -> pd.DataFrame:
    """Rank candidates by candidate-local TF-IDF similarity."""
    if text_col not in candidates.columns:
        raise KeyError(f"Missing text column: {text_col}")

    out = candidates.copy()
    out["tfidf_score"] = candidate_tfidf_scores(
        query,
        out[text_col],
    )

    return sort_and_rank(out, score_col="tfidf_score", k=k)
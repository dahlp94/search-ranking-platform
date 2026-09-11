"""BM25 candidate re-ranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.retrieval.ranking import sort_and_rank
from src.retrieval.text import tokenize


def candidate_bm25_scores(
    query: str,
    documents: Sequence[object],
) -> np.ndarray:
    """Compute BM25 scores within one query's candidate set."""
    docs = [tokenize(doc) for doc in documents]
    query_tokens = tokenize(query)

    if not docs:
        return np.array([], dtype=float)

    if not query_tokens or not any(docs):
        return np.zeros(len(docs), dtype=float)

    bm25 = BM25Okapi(docs)
    return np.asarray(bm25.get_scores(query_tokens), dtype=float)


def rank_candidates(
    query: str,
    candidates: pd.DataFrame,
    k: int | None = None,
    text_col: str = "product_text",
) -> pd.DataFrame:
    """Rank candidates by candidate-local BM25 score."""
    if text_col not in candidates.columns:
        raise KeyError(f"Missing text column: {text_col}")

    out = candidates.copy()

    if out.empty:
        out["bm25_score"] = pd.Series(dtype=float)
        out["predicted_rank"] = pd.Series(dtype=int)
        return out

    out["bm25_score"] = candidate_bm25_scores(
        query,
        out[text_col],
    )

    return sort_and_rank(out, score_col="bm25_score", k=k)
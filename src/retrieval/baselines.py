"""Simple lexical baselines: random permutation and TF-IDF cosine similarity.

Both methods re-rank the candidate products already associated with a query.
They do not retrieve from the full Amazon catalog.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import SEED
from src.retrieval.ranking import sort_and_rank
from src.retrieval.text import normalize_text, tokenize


def _query_seed(query_id: object, seed: int) -> int:
    """Stable per-query RNG seed that does not depend on PYTHONHASHSEED."""
    payload = f"{seed}:{query_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def rank_random(
    query: str,
    candidates: pd.DataFrame,
    k: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Randomly permute a query's candidates with a fixed seed.

    This baseline calibrates ranking metrics. It is not intended to be competitive.
    Random scores are assigned after sorting by product_id so the inherited
    DataFrame order cannot affect the permutation.
    """
    del query  # the query string is unused; randomness is keyed by query_id + seed
    if candidates.empty:
        out = candidates.copy()
        out["random_score"] = pd.Series(dtype=float)
        out["predicted_rank"] = pd.Series(dtype=int)
        return out

    out = candidates.copy()
    if "query_id" not in out.columns:
        raise KeyError("rank_random requires a query_id column.")
    query_ids = out["query_id"].unique()
    if len(query_ids) != 1:
        raise ValueError(
            "rank_random expects candidates from a single query; "
            f"received {len(query_ids)} query_ids."
        )
    rng = np.random.default_rng(_query_seed(query_ids[0], seed))
    out = out.sort_values("product_id", kind="mergesort")
    out["random_score"] = rng.random(len(out))
    return sort_and_rank(out, score_col="random_score", k=k)


def rank_tfidf(
    query: str,
    candidates: pd.DataFrame,
    k: int | None = None,
    text_col: str = "product_text",
) -> pd.DataFrame:
    """TF-IDF cosine similarity between the query and each candidate document.

    IDF is computed from this query's candidate texts only, matching the Week 1
    BM25 design: query-specific candidate re-ranking, not a global index.
    """
    out = candidates.copy()
    if text_col not in out.columns:
        raise KeyError(f"Missing text column '{text_col}'.")

    documents = out[text_col].map(normalize_text).tolist()
    query_text = normalize_text(query)
    n = len(documents)
    scores = np.zeros(n, dtype=float)

    tokenized_docs = [tokenize(text) for text in documents]
    has_vocab = any(tokenized_docs) and len(tokenize(query_text)) > 0
    if n > 0 and has_vocab:
        vectorizer = TfidfVectorizer(analyzer=tokenize, lowercase=False, norm="l2")
        try:
            doc_matrix = vectorizer.fit_transform(documents)
            query_vector = vectorizer.transform([query_text])
            scores = cosine_similarity(query_vector, doc_matrix).ravel()
        except ValueError:
            # Empty vocabulary after tokenization.
            scores = np.zeros(n, dtype=float)

    out["tfidf_score"] = scores
    return sort_and_rank(out, score_col="tfidf_score", k=k)

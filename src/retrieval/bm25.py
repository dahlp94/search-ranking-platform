"""BM25 candidate re-ranking for a single query's ESCI candidate set.

This baseline uses BM25 as a query-specific re-ranker. Corpus statistics (IDF, average
document length) are estimated from the candidate product texts associated with
THAT query. This is not a full-catalog retrieval index and does not search the
Amazon product catalog.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.retrieval.ranking import sort_and_rank
from src.retrieval.text import tokenize


def rank_candidates(
    query: str,
    candidates: pd.DataFrame,
    k: int | None = None,
    text_col: str = "product_text",
) -> pd.DataFrame:
    """Re-rank one query's candidate products with BM25.

    Parameters
    ----------
    query:
        Raw query string. Tokenized with the same rules as product text.
    candidates:
        Candidate rows already associated with this query in ESCI. The function
        does not add products from outside this set.
    k:
        Optional cutoff. If None, the full candidate set is returned, reordered.
    text_col:
        Column containing the lexical document for each product.
    """
    out = candidates.copy()
    if text_col not in out.columns:
        raise KeyError(f"Missing text column '{text_col}'.")

    tokenized_docs = [tokenize(text) for text in out[text_col].tolist()]
    query_tokens = tokenize(query)
    n = len(tokenized_docs)
    if n == 0:
        out["bm25_score"] = pd.Series(dtype=float)
        out["predicted_rank"] = pd.Series(dtype=int)
        return out

    if all(len(doc) == 0 for doc in tokenized_docs) or len(query_tokens) == 0:
        scores = np.zeros(n, dtype=float)
    else:
        bm25 = BM25Okapi(tokenized_docs)
        scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)

    out["bm25_score"] = scores
    return sort_and_rank(out, score_col="bm25_score", k=k)

"""BM25 / TF-IDF / random re-ranking tests on a tiny candidate set."""

from __future__ import annotations

import pandas as pd

from src.constants import SEED
from src.retrieval.baselines import rank_random, rank_tfidf
from src.retrieval.bm25 import rank_candidates
from src.retrieval.ranking import rerank_dataset, sort_and_rank
from tests.fixtures.make_frames import ranking_candidates


def test_bm25_sorts_descending_and_prefers_lexical_match():
    ranked = rank_candidates("nike running shoes", ranking_candidates())
    scores = ranked["bm25_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert ranked.iloc[0]["product_id"] == "p_nike"
    assert ranked.iloc[-1]["product_id"] == "p_blender"


def test_reranking_preserves_candidate_set():
    original = ranking_candidates()
    ranked = rank_candidates("nike running shoes", original)
    assert set(ranked["product_id"]) == set(original["product_id"])
    assert set(ranked["example_id"]) == set(original["example_id"])
    assert len(ranked) == len(original)


def test_k_cutoff_returns_requested_rows():
    ranked = rank_candidates("nike running shoes", ranking_candidates(), k=2)
    assert len(ranked) == 2
    assert list(ranked["predicted_rank"]) == [1, 2]


def test_bm25_is_deterministic_and_ignores_input_row_order():
    base = ranking_candidates()
    shuffled = base.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ranked_a = rank_candidates("nike running shoes", base)
    ranked_b = rank_candidates("nike running shoes", shuffled)
    assert list(ranked_a["product_id"]) == list(ranked_b["product_id"])
    assert list(ranked_a["predicted_rank"]) == list(ranked_b["predicted_rank"])
    assert list(ranked_a["bm25_score"]) == list(ranked_b["bm25_score"])


def test_tied_scores_break_ties_by_product_id():
    tied = pd.DataFrame(
        {
            "query_id": [1, 1],
            "query": ["nike", "nike"],
            "product_id": ["p_b", "p_a"],
            "product_title": ["nike shoes", "nike shoes"],
            "product_text": ["nike shoes", "nike shoes"],
            "esci_label": ["E", "E"],
            "relevance_gain": [3, 3],
        }
    )
    ranked = rank_candidates("nike shoes", tied)
    assert ranked["bm25_score"].nunique() == 1
    assert list(ranked["product_id"]) == ["p_a", "p_b"]
    assert list(ranked["predicted_rank"]) == [1, 2]


def test_sort_and_rank_does_not_use_frame_order_for_ties():
    frame = pd.DataFrame(
        {
            "product_id": ["z", "a", "m"],
            "score": [1.0, 1.0, 1.0],
        }
    )
    ranked = sort_and_rank(frame, score_col="score")
    assert list(ranked["product_id"]) == ["a", "m", "z"]


def test_random_ranking_is_reproducible_with_fixed_seed():
    candidates = ranking_candidates()
    first = rank_random("nike running shoes", candidates, seed=SEED)
    second = rank_random("nike running shoes", candidates, seed=SEED)
    shuffled = candidates.sample(frac=1.0, random_state=1).reset_index(drop=True)
    third = rank_random("nike running shoes", shuffled, seed=SEED)
    assert list(first["product_id"]) == list(second["product_id"]) == list(third["product_id"])
    assert list(first["predicted_rank"]) == list(second["predicted_rank"])


def test_tfidf_ranks_lexical_match_first():
    ranked = rank_tfidf("nike running shoes", ranking_candidates())
    assert ranked.iloc[0]["product_id"] == "p_nike"
    assert set(ranked["product_id"]) == set(ranking_candidates()["product_id"])


def test_rerank_dataset_does_not_add_or_drop_candidates():
    two_queries = pd.concat(
        [
            ranking_candidates(),
            ranking_candidates().assign(query_id=202, query="kitchen blender"),
        ],
        ignore_index=True,
    )
    ranked = rerank_dataset(two_queries, rank_candidates)
    assert len(ranked) == len(two_queries)
    assert set(ranked["example_id"]) == set(two_queries["example_id"])
    for query_id, group in ranked.groupby("query_id"):
        original_ids = set(two_queries.loc[two_queries["query_id"] == query_id, "product_id"])
        assert set(group["product_id"]) == original_ids

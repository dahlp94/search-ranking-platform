"""Project-wide conventions for Week 1.

Keep these mappings in one place so evaluation stays internally consistent.
They are this project's conventions, not Amazon's official benchmark gains.
"""

# Used for query-level train/validation splitting and the random ranking baseline.
SEED = 1234

# Official training queries are split by query_id (not by row).
PROJECT_TRAIN_QUERY_FRACTION = 0.85

# Graded relevance used by NDCG. Exact > Substitute > Complement > Irrelevant.
RELEVANCE_GAIN = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0,
}

# RELEVANCE_GAIN = {
#     "E": 1.0,
#     "S": 0.1,
#     "C": 0.01,
#     "I": 0.0,
# }

# Binary relevance used by Recall@K and MRR.
# Exact and Substitute count as relevant; Complement and Irrelevant do not.
BINARY_RELEVANT_LABELS = frozenset({"E", "S"})

DEFAULT_K = 10

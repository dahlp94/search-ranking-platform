"""Query-level splitting that avoids leakage into the official test set.

The unit held out is the QUERY, not the query-product row. If we split by row,
products for the same query could appear in both train and validation, and a
model could look stronger than it is because it already saw that query's
candidate set during development.

The official ESCI `split=='test'` partition stays untouched as a final holdout.
Week 1 baselines are evaluated on a project validation slice of official train.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.constants import PROJECT_TRAIN_QUERY_FRACTION, SEED


class SplitError(ValueError):
    """Raised when a split would leak queries across partitions."""


def _query_ids(df: pd.DataFrame) -> set[Any]:
    return set(df["query_id"].unique())


def assert_query_sets_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    official_test: pd.DataFrame,
) -> None:
    train_ids = _query_ids(train)
    val_ids = _query_ids(validation)
    test_ids = _query_ids(official_test)
    train_val = train_ids & val_ids
    train_test = train_ids & test_ids
    val_test = val_ids & test_ids
    if train_val or train_test or val_test:
        raise SplitError(
            "Query IDs leak across partitions: "
            f"train∩val={len(train_val)}, train∩test={len(train_test)}, "
            f"val∩test={len(val_test)}."
        )


def assert_query_rows_stay_together(*frames: pd.DataFrame) -> None:
    """Every candidate row for a query_id must live in exactly one partition."""
    locations: dict[Any, str] = {}
    for idx, df in enumerate(frames):
        name = df.attrs.get("partition_name", f"partition_{idx}")
        for query_id in df["query_id"].unique():
            previous = locations.get(query_id)
            if previous is not None and previous != name:
                raise SplitError(
                    f"query_id={query_id} has rows in both '{previous}' and '{name}'. "
                    "Candidate rows for a query must stay in one partition."
                )
            locations[query_id] = name


def split_official_train_by_query(
    df: pd.DataFrame,
    seed: int = SEED,
    train_fraction: float = PROJECT_TRAIN_QUERY_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split official training queries into project train / validation.

    Returns
    -------
    project_train, project_validation, official_test
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}.")

    official_test = df.loc[df["split"] == "test"].copy()
    official_train = df.loc[df["split"] == "train"].copy()
    unexpected = sorted(set(df["split"].unique()) - {"train", "test"})
    if unexpected:
        raise SplitError(f"Unexpected official split values: {unexpected}.")
    if official_train.empty:
        raise SplitError("No official training rows available to split.")

    split_nunique = df.groupby("query_id", sort=False)["split"].nunique()
    mixed = split_nunique[split_nunique > 1]
    if not mixed.empty:
        raise SplitError(
            "Some query_id values appear with more than one official split value, "
            f"e.g. {list(mixed.head(5).index)}. Candidate rows would leak across "
            "the official holdout if we split those queries."
        )

    query_ids = np.array(sorted(official_train["query_id"].unique()), dtype=object)
    n_queries = len(query_ids)
    n_train = int(n_queries * train_fraction)
    n_val = n_queries - n_train
    if n_train == 0 or n_val == 0:
        raise SplitError(
            f"Split of {n_queries} official-train queries with fraction "
            f"{train_fraction} produced an empty partition "
            f"(n_train={n_train}, n_val={n_val})."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(query_ids)
    train_ids = set(query_ids[:n_train])
    val_ids = set(query_ids[n_train:])

    project_train = official_train.loc[official_train["query_id"].isin(train_ids)].copy()
    project_validation = official_train.loc[official_train["query_id"].isin(val_ids)].copy()

    project_train.attrs["partition_name"] = "project_train"
    project_validation.attrs["partition_name"] = "project_validation"
    official_test.attrs["partition_name"] = "official_test"

    assert_query_sets_disjoint(project_train, project_validation, official_test)
    assert_query_rows_stay_together(project_train, project_validation, official_test)

    print(
        "[split] official train queries "
        f"({n_queries:,}) -> project_train={len(train_ids):,} queries / "
        f"{len(project_train):,} rows, project_validation={len(val_ids):,} queries / "
        f"{len(project_validation):,} rows; official_test={official_test['query_id'].nunique():,} "
        f"queries / {len(official_test):,} rows remain untouched."
    )
    return project_train, project_validation, official_test

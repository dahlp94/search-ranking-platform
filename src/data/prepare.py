"""Build the Week 1 processed tables from official ESCI parquet files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import PROJECT_TRAIN_QUERY_FRACTION, RELEVANCE_GAIN, SEED
from src.data.load_data import (
    PROCESSED_DIR,
    RAW_DIR,
    filter_examples_us_small,
    filter_products_us,
    load_examples,
    load_products,
    require_raw_files,
)
from src.data.split import split_official_train_by_query
from src.data.validate import (
    compute_dataset_stats,
    find_duplicate_query_products,
    merge_examples_products,
)
from src.retrieval.text import add_product_text_columns


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def add_relevance_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Attach this project's ordinal relevance mapping.

    Unknown ESCI labels fail loudly rather than becoming NaN in NDCG.
    """
    unknown = sorted(set(df["esci_label"].dropna().unique()) - set(RELEVANCE_GAIN))
    if unknown:
        raise ValueError(f"Unknown esci_label values: {unknown}.")
    n_missing_labels = int(df["esci_label"].isna().sum())
    if n_missing_labels:
        raise ValueError(f"{n_missing_labels} rows have missing esci_label.")
    df = df.copy()
    df["relevance_gain"] = df["esci_label"].map(RELEVANCE_GAIN).astype(int)
    return df


def prepare_dataset(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    seed: int = SEED,
    train_fraction: float = PROJECT_TRAIN_QUERY_FRACTION,
) -> dict[str, Any]:
    """Load, validate, merge, split, and write processed Week 1 tables."""
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    require_raw_files(raw_dir)

    examples = filter_examples_us_small(load_examples(raw_dir / "shopping_queries_dataset_examples.parquet"))
    n_query_ids = int(examples["query_id"].nunique())
    if Path(raw_dir).resolve() == RAW_DIR.resolve() and n_query_ids < 1000:
        raise ValueError(
            f"Only {n_query_ids} query IDs remain after US + small_version filters. "
            "Official ESCI Task 1 US data has tens of thousands of queries. "
            "Refusing to treat this as the real dataset. Do not copy the synthetic "
            "files under tests/fixtures/ into data/raw/."
        )
    products = filter_products_us(
        load_products(raw_dir / "shopping_queries_dataset_products.parquet", locale="us")
    )

    merged, merge_report = merge_examples_products(examples, products)
    duplicate_report = find_duplicate_query_products(merged)
    merged = add_relevance_gain(merged)
    merged = add_product_text_columns(merged)

    stats = compute_dataset_stats(merged)
    stats["merge"] = merge_report
    stats["duplicate_query_products"] = duplicate_report

    project_train, project_validation, official_test = split_official_train_by_query(
        merged,
        seed=seed,
        train_fraction=train_fraction,
    )
    stats["partitions"] = {
        "project_train": {
            "n_rows": int(len(project_train)),
            "n_query_ids": int(project_train["query_id"].nunique()),
        },
        "project_validation": {
            "n_rows": int(len(project_validation)),
            "n_query_ids": int(project_validation["query_id"].nunique()),
        },
        "official_test_holdout": {
            "n_rows": int(len(official_test)),
            "n_query_ids": int(official_test["query_id"].nunique()),
        },
        "seed": seed,
        "train_fraction": train_fraction,
    }

    train_path = processed_dir / "project_train.parquet"
    val_path = processed_dir / "project_validation.parquet"
    test_path = processed_dir / "official_test_holdout.parquet"
    stats_path = processed_dir / "dataset_stats.json"

    project_train.to_parquet(train_path, index=False)
    project_validation.to_parquet(val_path, index=False)
    official_test.to_parquet(test_path, index=False)
    stats_path.write_text(json.dumps(_json_ready(stats), indent=2))

    if merge_report["unmatched_rows"]:
        unmatched_cols = [
            col
            for col in ["example_id", "query_id", "product_id", "product_locale", "esci_label"]
            if col in merged.columns
        ]
        unmatched = merged.loc[~merged["product_matched"], unmatched_cols]
        unmatched.head(50).to_csv(processed_dir / "unmatched_examples_sample.csv", index=False)
        print(
            f"[prepare] wrote unmatched sample ({merge_report['unmatched_rows']:,} unmatched rows kept)."
        )
    if duplicate_report["n_duplicate_pairs"]:
        pair_cols = ["query_id", "product_id"]
        dup_mask = merged.duplicated(pair_cols, keep=False)
        merged.loc[dup_mask].head(50).to_csv(
            processed_dir / "duplicate_query_product_sample.csv",
            index=False,
        )
        print(
            "[prepare] duplicate (query_id, product_id) pairs were kept and sampled; "
            "they were not silently dropped."
        )

    print(f"[prepare] wrote {train_path}")
    print(f"[prepare] wrote {val_path}")
    print(f"[prepare] wrote {test_path} (holdout; do not use for Week 1 iteration)")
    print(f"[prepare] wrote {stats_path}")
    return stats

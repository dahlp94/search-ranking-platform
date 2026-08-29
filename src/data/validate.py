"""Data-integrity checks for ESCI candidate re-ranking.

These checks exist because ranking metrics are only trustworthy if:
  * each product-locale key maps to one product record (merge is many-to-one)
  * unmatched example rows are visible rather than silently dropped
  * duplicate query-product candidates are reported rather than silently removed
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when a data-quality issue would make ranking evaluation ambiguous."""


def validate_required_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns {missing}.")


def validate_product_key_uniqueness(
    products: pd.DataFrame,
    sample_size: int = 10,
) -> dict[str, Any]:
    """Require unique (product_id, product_locale) keys before merging.

    Duplicate keys would make a many-to-one merge ambiguous: one example row
    could match multiple product records and silently multiply the candidate set.
    """
    validate_required_columns(products, ["product_id", "product_locale"], "products")
    key_cols = ["product_id", "product_locale"]
    duplicate_mask = products.duplicated(key_cols, keep=False)
    n_duplicate_rows = int(duplicate_mask.sum())
    n_duplicate_keys = (
        int(products.loc[duplicate_mask, key_cols].drop_duplicates().shape[0])
        if n_duplicate_rows
        else 0
    )
    sample = (
        products.loc[duplicate_mask, key_cols]
        .drop_duplicates()
        .head(sample_size)
        .to_dict(orient="records")
        if n_duplicate_rows
        else []
    )
    report = {
        "n_product_rows": int(len(products)),
        "n_unique_product_keys": int(products[key_cols].drop_duplicates().shape[0]),
        "n_duplicate_key_rows": n_duplicate_rows,
        "n_duplicate_keys": n_duplicate_keys,
        "duplicate_key_sample": sample,
    }
    print(
        "[integrity] product keys (product_id, product_locale): "
        f"{report['n_unique_product_keys']:,} unique keys / "
        f"{report['n_product_rows']:,} rows; "
        f"duplicate keys={n_duplicate_keys:,}"
    )
    if n_duplicate_keys:
        raise DataIntegrityError(
            "Product table has duplicate (product_id, product_locale) keys, "
            "so a many-to-one merge would be ambiguous. "
            f"duplicate_keys={n_duplicate_keys}, duplicate_rows={n_duplicate_rows}. "
            f"Sample: {sample}. Refusing to silently deduplicate."
        )
    return report


def merge_examples_products(
    examples: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-merge examples to products with an explicit many-to-one check.

    Unmatched example rows are kept and reported. They are not dropped.
    """
    validate_required_columns(examples, ["product_id", "product_locale"], "examples")
    product_key_report = validate_product_key_uniqueness(products)

    rows_before = int(len(examples))
    merged = examples.merge(
        products,
        on=["product_id", "product_locale"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    rows_after = int(len(merged))
    matched_rows = int((merged["_merge"] == "both").sum())
    unmatched_rows = int((merged["_merge"] == "left_only").sum())
    unmatched_sample = (
        merged.loc[merged["_merge"] == "left_only", ["query_id", "product_id", "product_locale"]]
        .head(10)
        .to_dict(orient="records")
    )

    report = {
        "rows_before_merge": rows_before,
        "rows_after_merge": rows_after,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "unmatched_sample": unmatched_sample,
        "product_key_report": product_key_report,
    }
    print(
        "[merge] left join on (product_id, product_locale), validate='many_to_one': "
        f"before={rows_before:,}, after={rows_after:,}, "
        f"matched={matched_rows:,}, unmatched={unmatched_rows:,}"
    )
    if rows_after != rows_before:
        raise DataIntegrityError(
            "Merge changed the example row count unexpectedly "
            f"({rows_before} -> {rows_after}). This usually means the product "
            "key was not unique. Refusing to continue."
        )
    merged["product_matched"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])
    return merged, report


def find_duplicate_query_products(
    df: pd.DataFrame,
    sample_size: int = 10,
) -> dict[str, Any]:
    """Report duplicate (query_id, product_id) candidate rows.

    Duplicate candidates are not dropped here. Conflicting ESCI labels for the
    same pair make relevance ambiguous, so we fail in that case.
    """
    validate_required_columns(df, ["query_id", "product_id", "esci_label"], "merged examples")
    pair_cols = ["query_id", "product_id"]
    duplicate_mask = df.duplicated(pair_cols, keep=False)
    n_duplicate_rows = int(duplicate_mask.sum())
    n_duplicate_pairs = (
        int(df.loc[duplicate_mask, pair_cols].drop_duplicates().shape[0])
        if n_duplicate_rows
        else 0
    )
    sample = (
        df.loc[duplicate_mask, ["query_id", "product_id", "esci_label", "example_id"]]
        .head(sample_size)
        .to_dict(orient="records")
        if n_duplicate_rows and "example_id" in df.columns
        else (
            df.loc[duplicate_mask, ["query_id", "product_id", "esci_label"]]
            .head(sample_size)
            .to_dict(orient="records")
            if n_duplicate_rows
            else []
        )
    )

    label_nunique = df.groupby(pair_cols, sort=False)["esci_label"].nunique()
    n_conflicting_pairs = int((label_nunique > 1).sum())
    conflict_sample: list[dict[str, Any]] = []
    if n_conflicting_pairs:
        conflict_keys = label_nunique[label_nunique > 1].head(sample_size).index
        conflict_sample = (
            df.set_index(pair_cols)
            .loc[conflict_keys, ["esci_label"]]
            .reset_index()
            .head(sample_size)
            .to_dict(orient="records")
        )

    report = {
        "n_duplicate_pair_rows": n_duplicate_rows,
        "n_duplicate_pairs": n_duplicate_pairs,
        "n_conflicting_label_pairs": n_conflicting_pairs,
        "duplicate_sample": sample,
        "conflict_sample": conflict_sample,
        "decision": (
            "fail_on_conflicting_labels; keep non-conflicting duplicates and report them"
        ),
    }
    print(
        "[integrity] duplicate (query_id, product_id) pairs: "
        f"{n_duplicate_pairs:,} pairs / {n_duplicate_rows:,} rows; "
        f"conflicting labels={n_conflicting_pairs:,}"
    )
    if n_conflicting_pairs:
        raise DataIntegrityError(
            "Found duplicate (query_id, product_id) candidates with conflicting "
            f"ESCI labels ({n_conflicting_pairs} pairs). Relevance would be "
            f"ambiguous. Sample: {conflict_sample}. Refusing to silently drop or "
            "reconcile them."
        )
    return report


def _series_missingness(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if df[col].dtype == object or str(df[col].dtype) == "string":
            n_blank = int(df[col].fillna("").astype(str).str.strip().eq("").sum())
        else:
            n_blank = n_missing
        out[col] = {"n_missing": n_missing, "n_missing_or_blank": n_blank}
    return out


def compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Summaries needed to understand the candidate re-ranking problem."""
    candidate_counts = df.groupby("query_id", sort=False).size()
    query_id_to_query = df.groupby("query_id", sort=False)["query"].nunique()
    stats: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_unique_query_ids": int(df["query_id"].nunique()),
        "n_unique_query_strings": int(df["query"].nunique()),
        "n_query_ids_with_multiple_query_strings": int((query_id_to_query > 1).sum()),
        "n_unique_products": int(df["product_id"].nunique()),
        "esci_label_counts": {
            str(k): int(v) for k, v in df["esci_label"].value_counts(dropna=False).items()
        },
        "split_counts": {
            str(k): int(v) for k, v in df["split"].value_counts(dropna=False).items()
        },
        "missingness": _series_missingness(
            df,
            [
                "query",
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
            ],
        ),
        "candidate_count_per_query": {
            "min": int(candidate_counts.min()),
            "max": int(candidate_counts.max()),
            "mean": float(candidate_counts.mean()),
            "median": float(candidate_counts.median()),
            "p25": float(np.percentile(candidate_counts, 25)),
            "p75": float(np.percentile(candidate_counts, 75)),
        },
    }
    print(
        "[stats] "
        f"rows={stats['n_rows']:,}, "
        f"query_ids={stats['n_unique_query_ids']:,}, "
        f"query_strings={stats['n_unique_query_strings']:,}, "
        f"products={stats['n_unique_products']:,}"
    )
    print(f"[stats] ESCI labels: {stats['esci_label_counts']}")
    print(f"[stats] official splits: {stats['split_counts']}")
    print(
        "[stats] candidates/query: "
        f"min={stats['candidate_count_per_query']['min']}, "
        f"median={stats['candidate_count_per_query']['median']:.1f}, "
        f"max={stats['candidate_count_per_query']['max']}"
    )
    return stats

"""Data loading, filtering, merge integrity, and query-split tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.constants import SEED
from src.data.load_data import (
    MissingRawDataError,
    SchemaError,
    filter_examples_us_small,
    filter_products_us,
    load_examples,
    load_products,
    require_raw_files,
)
from src.data.prepare import prepare_dataset
from src.data.split import SplitError, split_official_train_by_query
from src.data.validate import (
    DataIntegrityError,
    find_duplicate_query_products,
    merge_examples_products,
    validate_product_key_uniqueness,
    validate_required_columns,
)
from tests.fixtures.make_frames import tiny_examples, tiny_products, write_raw_parquet


def test_require_raw_files_explains_missing_paths(tmp_path):
    with pytest.raises(MissingRawDataError, match="shopping_queries_dataset_examples.parquet"):
        require_raw_files(tmp_path)


def test_load_examples_rejects_missing_columns(tmp_path):
    path = tmp_path / "shopping_queries_dataset_examples.parquet"
    pd.DataFrame({"query": ["x"]}).to_parquet(path, index=False)
    with pytest.raises(SchemaError, match="missing required columns"):
        load_examples(path)


def test_us_and_small_version_filters():
    examples = filter_examples_us_small(tiny_examples())
    products = filter_products_us(tiny_products())
    assert set(examples["product_locale"].unique()) == {"us"}
    assert set(examples["small_version"].unique()) == {1}
    assert "jp" not in set(products["product_locale"].unique())
    assert 301 not in set(examples["query_id"])
    assert 401 not in set(examples["query_id"])


def test_load_parquet_applies_us_product_filter(tmp_path):
    write_raw_parquet(tmp_path)
    products = load_products(tmp_path / "shopping_queries_dataset_products.parquet", locale="us")
    assert set(products["product_locale"].unique()) == {"us"}
    assert "p_jp" not in set(products["product_id"])


def test_required_column_validation():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_columns(pd.DataFrame({"a": [1]}), ["a", "b"], "frame")


def test_product_key_uniqueness_passes_on_clean_table():
    report = validate_product_key_uniqueness(filter_products_us(tiny_products()))
    assert report["n_duplicate_keys"] == 0


def test_product_key_uniqueness_fails_on_duplicates():
    products = filter_products_us(tiny_products())
    dup = pd.concat([products, products.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataIntegrityError, match="duplicate"):
        validate_product_key_uniqueness(dup)


def test_many_to_one_merge_and_row_counts():
    examples = filter_examples_us_small(tiny_examples())
    products = filter_products_us(tiny_products())
    merged, report = merge_examples_products(examples, products)
    assert report["rows_before_merge"] == len(examples)
    assert report["rows_after_merge"] == len(examples)
    assert report["matched_rows"] == len(examples)
    assert report["unmatched_rows"] == 0
    assert len(merged) == len(examples)
    assert merged["product_matched"].all()


def test_unmatched_example_rows_are_kept():
    examples = filter_examples_us_small(tiny_examples())
    products = filter_products_us(tiny_products()).iloc[0:0]
    merged, report = merge_examples_products(examples, products)
    assert report["unmatched_rows"] == len(examples)
    assert len(merged) == len(examples)
    assert not merged["product_matched"].any()


def test_duplicate_query_product_detection_keeps_nonconflicting_pairs():
    examples = filter_examples_us_small(tiny_examples())
    dup = pd.concat([examples, examples.iloc[[0]]], ignore_index=True)
    report = find_duplicate_query_products(dup)
    assert report["n_duplicate_pairs"] >= 1
    assert report["n_conflicting_label_pairs"] == 0


def test_duplicate_query_product_fails_on_conflicting_labels():
    examples = filter_examples_us_small(tiny_examples())
    conflict = examples.iloc[[0]].copy()
    conflict["esci_label"] = "I"
    conflict["example_id"] = 999
    with pytest.raises(DataIntegrityError, match="conflicting"):
        find_duplicate_query_products(pd.concat([examples, conflict], ignore_index=True))


def _many_query_frame(n_train_queries: int = 20, n_test_queries: int = 5) -> pd.DataFrame:
    rows = []
    example_id = 1
    for query_id in range(1, n_train_queries + 1):
        for product_id, label in (("p1", "E"), ("p2", "I")):
            rows.append(
                {
                    "example_id": example_id,
                    "query": f"query {query_id}",
                    "query_id": query_id,
                    "product_id": product_id,
                    "product_locale": "us",
                    "esci_label": label,
                    "split": "train",
                }
            )
            example_id += 1
    for query_id in range(100, 100 + n_test_queries):
        rows.append(
            {
                "example_id": example_id,
                "query": f"query {query_id}",
                "query_id": query_id,
                "product_id": "p1",
                "product_locale": "us",
                "esci_label": "E",
                "split": "test",
            }
        )
        example_id += 1
    return pd.DataFrame(rows)


def test_query_split_is_disjoint_and_keeps_rows_together():
    df = _many_query_frame()
    train, val, test = split_official_train_by_query(df, seed=SEED, train_fraction=0.85)
    train_ids = set(train["query_id"])
    val_ids = set(val["query_id"])
    test_ids = set(test["query_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert test_ids == set(range(100, 105))

    counts = df.groupby("query_id").size()
    for part in (train, val, test):
        part_counts = part.groupby("query_id").size()
        for query_id, n_rows in part_counts.items():
            assert n_rows == counts[query_id]


def test_split_is_reproducible():
    df = _many_query_frame()
    train_a, val_a, _ = split_official_train_by_query(df, seed=SEED)
    train_b, val_b, _ = split_official_train_by_query(df, seed=SEED)
    assert set(train_a["query_id"]) == set(train_b["query_id"])
    assert set(val_a["query_id"]) == set(val_b["query_id"])


def test_prepare_dataset_on_synthetic_raw_files(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    write_raw_parquet(raw_dir)
    stats = prepare_dataset(raw_dir=raw_dir, processed_dir=processed_dir)
    assert (processed_dir / "project_train.parquet").is_file()
    assert (processed_dir / "project_validation.parquet").is_file()
    assert (processed_dir / "official_test_holdout.parquet").is_file()
    assert stats["n_unique_query_ids"] == 3
    assert stats["merge"]["unmatched_rows"] == 0


def test_prepare_refuses_tiny_files_in_project_raw_dir(tmp_path, monkeypatch):
    import src.data.prepare as prepare_mod

    raw_dir = tmp_path / "raw"
    write_raw_parquet(raw_dir)
    monkeypatch.setattr(prepare_mod, "RAW_DIR", raw_dir)
    with pytest.raises(ValueError, match="Refusing to treat this as the real dataset"):
        prepare_dataset(raw_dir=raw_dir, processed_dir=tmp_path / "processed")


def test_split_rejects_query_ids_that_span_official_splits():
    df = _many_query_frame()
    leaked = df.iloc[[0]].copy()
    leaked["split"] = "test"
    leaked["example_id"] = 9999
    with pytest.raises(SplitError, match="more than one official split"):
        split_official_train_by_query(pd.concat([df, leaked], ignore_index=True))

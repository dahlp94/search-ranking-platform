"""Load ESCI parquet files and apply the Week 1 US / small-version filters.

This project does not download or redistribute the official Amazon dataset.
If the expected files are missing, we fail with setup instructions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EXAMPLES_FILENAME = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILENAME = "shopping_queries_dataset_products.parquet"

EXAMPLE_COLUMNS = [
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "split",
]

PRODUCT_COLUMNS = [
    "product_id",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "product_locale",
]

DATA_SETUP_INSTRUCTIONS = """
This repository does not redistribute Amazon's Shopping Queries / ESCI dataset.

Place the official files at:

  data/raw/shopping_queries_dataset_examples.parquet
  data/raw/shopping_queries_dataset_products.parquet

How to obtain them:

  1. Review https://github.com/amazon-science/esci-data
  2. Install Git LFS, then clone the official repo:
       git lfs install
       git clone https://github.com/amazon-science/esci-data.git
  3. Copy the parquet files from esci-data/shopping_queries_dataset/ into data/raw/

The products parquet is large (on the order of 1 GB). This project will not
download it automatically.
""".strip()


class MissingRawDataError(FileNotFoundError):
    """Raised when official ESCI parquet files are not in data/raw/."""


class SchemaError(ValueError):
    """Raised when a parquet file is missing required columns."""


def _parquet_column_names(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _validate_columns(path: Path, required: list[str], frame_name: str) -> None:
    present = _parquet_column_names(path)
    missing = [col for col in required if col not in present]
    if missing:
        raise SchemaError(
            f"{frame_name} file is missing required columns {missing}. "
            f"Found columns: {present}. Expected at least: {required}. "
            f"File: {path}"
        )


def require_raw_files(raw_dir: Path | None = None) -> tuple[Path, Path]:
    """Return paths to the official example and product parquet files.

    Fails with setup instructions if either file is absent.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    examples_path = raw_dir / EXAMPLES_FILENAME
    products_path = raw_dir / PRODUCTS_FILENAME
    missing = [str(path) for path in (examples_path, products_path) if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise MissingRawDataError(
            "Required ESCI parquet file(s) not found:\n"
            f"{missing_list}\n\n"
            f"{DATA_SETUP_INSTRUCTIONS}"
        )
    return examples_path, products_path


def load_examples(path: Path | None = None) -> pd.DataFrame:
    """Load the query-product example table without filtering."""
    if path is None:
        path, _ = require_raw_files()
    else:
        path = Path(path)
        if not path.is_file():
            raise MissingRawDataError(
                f"Examples parquet not found: {path}\n\n{DATA_SETUP_INSTRUCTIONS}"
            )
    _validate_columns(path, EXAMPLE_COLUMNS, "Examples")
    df = pd.read_parquet(path, columns=EXAMPLE_COLUMNS)
    print(f"[load] examples: {df.shape[0]:,} rows x {df.shape[1]} columns from {path}")
    return df


def load_products(path: Path | None = None, locale: str | None = "us") -> pd.DataFrame:
    """Load product metadata.

    If locale is set, push the filter into the parquet read so we do not
    materialize Japanese/Spanish catalog rows that Week 1 will not use.
    """
    if path is None:
        _, path = require_raw_files()
    else:
        path = Path(path)
        if not path.is_file():
            raise MissingRawDataError(
                f"Products parquet not found: {path}\n\n{DATA_SETUP_INSTRUCTIONS}"
            )
    _validate_columns(path, PRODUCT_COLUMNS, "Products")
    filters = [("product_locale", "==", locale)] if locale is not None else None
    try:
        df = pd.read_parquet(path, columns=PRODUCT_COLUMNS, filters=filters)
    except Exception as exc:
        print(f"[load] parquet filter failed ({exc}); loading columns and filtering in pandas")
        df = pd.read_parquet(path, columns=PRODUCT_COLUMNS)
        if locale is not None:
            df = df.loc[df["product_locale"] == locale]
    locale_note = f" (locale={locale})" if locale is not None else ""
    print(f"[load] products{locale_note}: {df.shape[0]:,} rows x {df.shape[1]} columns from {path}")
    return df


def filter_examples_us_small(examples: pd.DataFrame) -> pd.DataFrame:
    """Keep Task 1 / small-version English-US query-product judgments."""
    required = ["small_version", "product_locale"]
    missing = [col for col in required if col not in examples.columns]
    if missing:
        raise SchemaError(f"Cannot filter examples; missing columns {missing}.")

    n_before = len(examples)
    mask = (examples["small_version"] == 1) & (examples["product_locale"] == "us")
    filtered = examples.loc[mask]
    n_dropped = n_before - len(filtered)
    print(
        f"[filter] examples US + small_version==1: {len(filtered):,} rows "
        f"(dropped {n_dropped:,} of {n_before:,})"
    )
    if filtered.empty:
        raise ValueError(
            "No example rows remain after filtering to small_version==1 and "
            "product_locale=='us'. Check that you placed the official ESCI files."
        )
    return filtered


def filter_products_us(products: pd.DataFrame) -> pd.DataFrame:
    """Keep US-locale product records."""
    if "product_locale" not in products.columns:
        raise SchemaError("Cannot filter products; missing column 'product_locale'.")
    n_before = len(products)
    filtered = products.loc[products["product_locale"] == "us"]
    print(
        f"[filter] products US: {len(filtered):,} rows "
        f"(dropped {n_before - len(filtered):,} of {n_before:,})"
    )
    if filtered.empty:
        raise ValueError(
            "No product rows remain after filtering to product_locale=='us'."
        )
    return filtered

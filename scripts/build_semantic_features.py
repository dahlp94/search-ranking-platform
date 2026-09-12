#!/usr/bin/env python3
"""Build semantic query-title similarity features for ESCI development data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.semantic.embeddings import (
    DEFAULT_MODEL,
    SentenceTransformerEncoder,
    assert_expected_development_sizes,
    assert_queries_disjoint,
    build_partition_semantic_features,
    collect_runtime_metadata,
    default_batch_size,
    extract_semantic_input,
    load_development_semantic_sources,
    resolve_device,
    similarity_distribution,
)


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
INPUT_DIR = ROOT / "artifacts" / "semantic_input"
OUTPUT_DIR = ROOT / "artifacts" / "semantic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ESCI query-title semantic similarity features."
    )
    parser.add_argument(
        "--extract-inputs",
        action="store_true",
        help="Create semantic-input parquet files without encoding.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def extract_inputs() -> None:
    train, validation = load_development_semantic_sources(PROCESSED_DIR)

    assert_expected_development_sizes(train, validation)
    assert_queries_disjoint(train, validation)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(INPUT_DIR / "train_semantic_input.parquet", index=False)
    validation.to_parquet(
        INPUT_DIR / "validation_semantic_input.parquet",
        index=False,
    )

    print(
        f"Train: {len(train):,} rows / "
        f"{train['query_id'].nunique():,} queries"
    )
    print(
        f"Validation: {len(validation):,} rows / "
        f"{validation['query_id'].nunique():,} queries"
    )
    print("Official ESCI test used: NO")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = INPUT_DIR / "train_semantic_input.parquet"
    validation_path = INPUT_DIR / "validation_semantic_input.parquet"

    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Semantic inputs missing. Run:\n"
            "python -m scripts.build_semantic_features --extract-inputs"
        )

    train = extract_semantic_input(pd.read_parquet(train_path))
    validation = extract_semantic_input(pd.read_parquet(validation_path))

    assert_expected_development_sizes(train, validation)
    assert_queries_disjoint(train, validation)

    return train, validation


def encode(args: argparse.Namespace) -> None:
    train, validation = load_inputs()

    device = resolve_device(args.device)
    batch_size = args.batch_size or default_batch_size(device)

    print(
        f"Model: {args.model}\n"
        f"Device: {device}\n"
        f"Batch size: {batch_size}"
    )

    encoder = SentenceTransformerEncoder(
        model_identifier=args.model,
        device=device,
        show_progress=not args.no_progress,
    )

    started = time.perf_counter()

    train_features, validation_features, stats = (
        build_partition_semantic_features(
            train,
            validation,
            encoder,
            batch_size=batch_size,
            normalize=True,
        )
    )

    runtime = time.perf_counter() - started

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_features.to_parquet(
        OUTPUT_DIR / "train_semantic.parquet",
        index=False,
    )
    validation_features.to_parquet(
        OUTPUT_DIR / "validation_semantic.parquet",
        index=False,
    )

    summary = {
        **collect_runtime_metadata(
            encoder,
            device=device,
            batch_size=batch_size,
            normalize=True,
        ),
        "train_candidate_rows": len(train_features),
        "validation_candidate_rows": len(validation_features),
        "train_queries": train_features["query_id"].nunique(),
        "validation_queries": validation_features["query_id"].nunique(),
        "unique_query_texts": stats["query"]["n_unique_texts"],
        "unique_title_texts": stats["title"]["n_unique_texts"],
        "reused_query_rows": stats["query"]["n_reused"],
        "reused_title_rows": stats["title"]["n_reused"],
        "semantic_similarity": {
            "train": similarity_distribution(
                train_features["semantic_similarity"].to_numpy()
            ),
            "validation": similarity_distribution(
                validation_features["semantic_similarity"].to_numpy()
            ),
        },
        "runtime_seconds": runtime,
        "official_test_used": False,
    }

    save_json(OUTPUT_DIR / "semantic_summary.json", summary)

    print(
        f"Train semantic rows: {len(train_features):,}\n"
        f"Validation semantic rows: {len(validation_features):,}\n"
        f"Runtime: {runtime:.1f} seconds\n"
        f"Official ESCI test used: NO"
    )


def main() -> None:
    args = parse_args()

    if args.extract_inputs:
        extract_inputs()
    else:
        encode(args)


if __name__ == "__main__":
    main()
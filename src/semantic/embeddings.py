"""Semantic query-title similarity for ESCI candidate re-ranking."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INPUT_COLUMNS = [
    "example_id",
    "query_id",
    "product_id",
    "query",
    "product_title",
]

ID_COLUMNS = [
    "example_id",
    "query_id",
    "product_id",
]

EXPECTED_SIZES = {
    "train": {"rows": 356_615, "queries": 17_754},
    "validation": {"rows": 63_038, "queries": 3_134},
}


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------


def resolve_device(device: str | None = None) -> str:
    """Use requested device, otherwise CUDA when available."""
    if device:
        return device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


def default_batch_size(device: str) -> int:
    return 256 if device.startswith("cuda") else 32


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------


def clean_text(value: object) -> str:
    """Convert missing text to an empty string."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "<na>", "<nat>"}:
        return ""

    return text


def extract_semantic_input(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only candidate identity, query, and title."""

    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing semantic input columns: {missing}")

    if "split" in df.columns and (df["split"] == "test").any():
        raise ValueError(
            "Official ESCI test rows are not allowed in semantic inputs."
        )

    out = df[INPUT_COLUMNS].copy().reset_index(drop=True)

    if out["example_id"].isna().any():
        raise ValueError("example_id contains missing values.")

    if not out["example_id"].is_unique:
        raise ValueError("example_id must be unique.")

    out["query"] = out["query"].map(clean_text)
    out["product_title"] = out["product_title"].map(clean_text)

    return out


def load_development_semantic_sources(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the frozen train and validation partitions."""

    train_path = Path(processed_dir) / "project_train.parquet"
    validation_path = Path(processed_dir) / "project_validation.parquet"

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    train = pd.read_parquet(train_path, columns=INPUT_COLUMNS)
    validation = pd.read_parquet(validation_path, columns=INPUT_COLUMNS)

    train = extract_semantic_input(train)
    validation = extract_semantic_input(validation)

    assert_queries_disjoint(train, validation)
    assert_expected_development_sizes(train, validation)

    return train, validation


# ---------------------------------------------------------------------
# Development-partition validation
# ---------------------------------------------------------------------


def assert_queries_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    overlap = set(train["query_id"]) & set(validation["query_id"])

    if overlap:
        raise ValueError(
            f"Train/validation query overlap: {len(overlap)} queries."
        )


def assert_expected_development_sizes(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:

    for name, frame in [
        ("train", train),
        ("validation", validation),
    ]:
        expected = EXPECTED_SIZES[name]

        if len(frame) != expected["rows"]:
            raise ValueError(
                f"{name}: expected {expected['rows']:,} rows, "
                f"got {len(frame):,}."
            )

        queries = frame["query_id"].nunique()

        if queries != expected["queries"]:
            raise ValueError(
                f"{name}: expected {expected['queries']:,} queries, "
                f"got {queries:,}."
            )


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------


def encode_unique_texts(
    texts: Sequence[str],
    encoder: Any,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode each unique text once and map vectors back to rows."""

    texts = [clean_text(text) for text in texts]

    unique_texts = list(dict.fromkeys(texts))

    unique_vectors = encoder.encode(
        unique_texts,
        batch_size=batch_size,
        normalize=True,
    )

    lookup = {
        text: i
        for i, text in enumerate(unique_texts)
    }

    row_indices = np.fromiter(
        (lookup[text] for text in texts),
        dtype=np.intp,
        count=len(texts),
    )

    vectors = unique_vectors[row_indices]

    stats = {
        "n_input_texts": len(texts),
        "n_unique_texts": len(unique_texts),
        "n_reused": len(texts) - len(unique_texts),
    }

    return vectors, stats


# ---------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------


def cosine_similarity(
    query_vectors: np.ndarray,
    title_vectors: np.ndarray,
) -> np.ndarray:
    """Row-wise cosine similarity for normalized embeddings."""

    if query_vectors.shape != title_vectors.shape:
        raise ValueError(
            "Query/title embedding shapes differ: "
            f"{query_vectors.shape} vs {title_vectors.shape}"
        )

    similarity = np.einsum(
        "ij,ij->i",
        query_vectors,
        title_vectors,
    )

    similarity = np.clip(similarity, -1.0, 1.0)

    if not np.isfinite(similarity).all():
        raise ValueError("semantic_similarity contains non-finite values.")

    return similarity


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------


def build_partition_semantic_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    encoder: Any,
    *,
    batch_size: int,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build semantic similarity for train and validation."""

    del normalize  # embeddings are always normalized in this pipeline

    train = extract_semantic_input(train)
    validation = extract_semantic_input(validation)

    assert_queries_disjoint(train, validation)

    n_train = len(train)

    all_queries = pd.concat(
        [train["query"], validation["query"]],
        ignore_index=True,
    )

    all_titles = pd.concat(
        [train["product_title"], validation["product_title"]],
        ignore_index=True,
    )

    query_vectors, query_stats = encode_unique_texts(
        all_queries.tolist(),
        encoder,
        batch_size=batch_size,
    )

    title_vectors, title_stats = encode_unique_texts(
        all_titles.tolist(),
        encoder,
        batch_size=batch_size,
    )

    similarity = cosine_similarity(
        query_vectors,
        title_vectors,
    )

    train_out = train[ID_COLUMNS].copy()
    validation_out = validation[ID_COLUMNS].copy()

    train_out["semantic_similarity"] = similarity[:n_train]
    validation_out["semantic_similarity"] = similarity[n_train:]

    stats = {
        "query": query_stats,
        "title": title_stats,
        "unique_train_query_texts": int(train["query"].nunique()),
        "unique_validation_query_texts": int(
            validation["query"].nunique()
        ),
        "unique_train_title_texts": int(
            train["product_title"].nunique()
        ),
        "unique_validation_title_texts": int(
            validation["product_title"].nunique()
        ),
    }

    return train_out, validation_out, stats


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def similarity_distribution(
    values: np.ndarray,
) -> dict[str, float]:

    values = np.asarray(values, dtype=float)

    return {
        "min": float(values.min()),
        "p1": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def collect_runtime_metadata(
    encoder: Any,
    *,
    device: str,
    batch_size: int,
    normalize: bool,
) -> dict[str, Any]:

    metadata = {
        "model_identifier": encoder.model_identifier,
        "model_revision": encoder.model_revision,
        "embedding_dimension": encoder.embedding_dimension,
        "device": device,
        "batch_size": batch_size,
        "normalization": "l2" if normalize else "none",
    }

    try:
        import sentence_transformers

        metadata["sentence_transformers_version"] = (
            sentence_transformers.__version__
        )
    except ImportError:
        pass

    try:
        import torch

        metadata["torch_version"] = torch.__version__

        if device.startswith("cuda") and torch.cuda.is_available():
            metadata["cuda_device_name"] = torch.cuda.get_device_name(0)

    except ImportError:
        pass

    return metadata


# ---------------------------------------------------------------------
# Sentence Transformer wrapper
# ---------------------------------------------------------------------


class SentenceTransformerEncoder:
    """Thin wrapper around one pretrained sentence transformer."""

    def __init__(
        self,
        model_identifier: str = DEFAULT_MODEL,
        device: str | None = None,
        show_progress: bool = True,
    ) -> None:

        from sentence_transformers import SentenceTransformer

        self.model_identifier = model_identifier
        self.device = resolve_device(device)
        self.show_progress = show_progress

        self.model = SentenceTransformer(
            model_identifier,
            device=self.device,
        )

        self.embedding_dimension = int(
            self.model.get_sentence_embedding_dimension()
        )

        self.model_revision = self._model_revision()

    def _model_revision(self) -> str | None:
        try:
            return str(
                self.model[0].auto_model.config._commit_hash
            )
        except (AttributeError, IndexError, TypeError):
            return None

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        normalize: bool = True,
    ) -> np.ndarray:

        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=self.show_progress,
        )

        return np.asarray(vectors, dtype=np.float32)
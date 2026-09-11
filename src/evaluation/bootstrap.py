"""Query-level paired bootstrap for a mean difference."""

import numpy as np

from src.constants import SEED


def paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    n_replicates: int = 2000,
    seed: int = SEED,
    ci: float = 0.95,
) -> dict:
    """Return a percentile CI for the mean query-level paired difference."""
    values = np.asarray(deltas, dtype=float)

    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty difference vector.")
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_replicates, values.size),
        replace=True,
    )
    bootstrap_means = samples.mean(axis=1)

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha, 1 - alpha],
    )

    return {
        "n_queries": int(values.size),
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "ci_level": float(ci),
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "resampling_unit": "query",
    }

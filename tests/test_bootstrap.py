"""Tests for the paired query-level bootstrap."""

import numpy as np
import pytest

from src.constants import SEED
from src.evaluation.bootstrap import paired_bootstrap_mean_ci


def test_bootstrap_reports_query_level_resampling():
    deltas = np.array([1.0, -1.0])

    result = paired_bootstrap_mean_ci(
        deltas,
        n_replicates=200,
        seed=SEED,
    )

    assert result["resampling_unit"] == "query"
    assert result["n_queries"] == 2
    assert result["mean"] == pytest.approx(0.0)


def test_bootstrap_is_reproducible():
    deltas = np.array([0.1, -0.05, 0.02, 0.0, -0.03])

    first = paired_bootstrap_mean_ci(
        deltas,
        n_replicates=200,
        seed=SEED,
    )
    second = paired_bootstrap_mean_ci(
        deltas,
        n_replicates=200,
        seed=SEED,
    )

    assert first["ci_lower"] == second["ci_lower"]
    assert first["ci_upper"] == second["ci_upper"]


def test_bootstrap_returns_valid_interval():
    deltas = np.array([0.2, -0.1, 0.05, 0.0])

    result = paired_bootstrap_mean_ci(
        deltas,
        n_replicates=300,
        seed=SEED,
    )

    assert np.isfinite(result["ci_lower"])
    assert np.isfinite(result["ci_upper"])
    assert result["ci_lower"] <= result["ci_upper"]
"""End-to-end smoke test.

Builds a tiny synthetic SurvivalCohort where the true risk is a linear
function of a subset of genes, runs the full cross-validated training
harness, and asserts:

    1. The run completes without errors.
    2. The reported C-index is *meaningfully above 0.5* — if the harness
       cannot recover signal from a cohort where signal obviously exists,
       something is wrong end-to-end (loss sign flipped, metric inverted,
       preprocessing leaking test labels in a way that still hurts
       performance, etc.).
    3. The run log round-trips through write_run_log / read_run_log.

This is the one test that would actually catch a cross-module
regression (e.g. "loss optimizes the negative of what the metric
rewards"). It is slower than a unit test (~5-10 seconds on CPU) but
still inexpensive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathogems.config import ExperimentConfig
from pathogems.data import SurvivalCohort
from pathogems.run_log import read_run_log, write_run_log
from pathogems.train import cross_validate


@pytest.mark.slow
def test_baseline_recovers_signal_on_synthetic_cohort(
    tmp_path: Path, synthetic_cohort: SurvivalCohort
) -> None:
    """Can the full harness learn a linear Cox signal end-to-end?

    The cohort itself comes from `conftest.py::synthetic_cohort` — same
    builder as other tests, so "synthetic cohort" means the same thing
    everywhere.
    """
    cohort = synthetic_cohort
    cfg = ExperimentConfig(
        name="smoke_synthetic",
        cohort="synthetic",
        study_data_dir="",  # unused — we pass the cohort directly
        top_k_genes=50,
        hidden_dims=(32,),  # keep tiny; this is a smoke test
        dropout=0.0,
        epochs=30,
        early_stopping_patience=5,
        batch_size=64,
        lr=1e-3,
        n_folds=3,
        seed=123,
        notes="Synthetic end-to-end smoke test.",
    )

    result = cross_validate(cohort, cfg, verbose=False)

    # (1) All folds produced a finite C-index.
    finite = [c for c in result.per_fold_c_index() if np.isfinite(c)]
    assert len(finite) == cfg.n_folds, f"Got non-finite C-indices: {result.per_fold_c_index()}"

    # (2) Mean C-index is meaningfully above chance. On this generative
    # setup (linear Cox signal, 10 informative genes out of 200, 300
    # patients, 40% event rate) a correct harness reliably scores > 0.6.
    # We leave a conservative 0.55 floor so the test is not flaky on
    # unlucky seeds but still catches real breakage.
    assert result.c_index_mean > 0.55, (
        f"Baseline failed to recover synthetic signal: "
        f"mean C-index = {result.c_index_mean:.4f}. "
        "Check loss sign, metric orientation, and preprocessing."
    )

    # (3) Run-log round-trip.
    log_path = write_run_log(cfg, result, logs_dir=tmp_path, status="success", error=None)
    loaded = read_run_log(log_path)
    assert loaded["run_name"] == "smoke_synthetic"
    assert loaded["metrics"]["c_index_mean"] == pytest.approx(result.c_index_mean, abs=1e-6)

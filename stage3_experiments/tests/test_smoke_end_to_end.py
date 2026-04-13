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
import pandas as pd
import pytest

from pathogems.config import ExperimentConfig
from pathogems.data import SurvivalCohort
from pathogems.run_log import read_run_log, write_run_log
from pathogems.train import cross_validate


def _make_synthetic_cohort(
    n_patients: int = 300,
    n_genes: int = 200,
    n_signal: int = 10,
    event_rate: float = 0.4,
    seed: int = 7,
) -> SurvivalCohort:
    """Synthetic cohort with an injected linear risk signal.

    True risk = linear combination of the first `n_signal` genes. Time is
    drawn from an exponential with rate proportional to exp(true_risk),
    which is the exact generative assumption the Cox PH model makes —
    so the baseline should recover it reliably.
    """
    rng = np.random.default_rng(seed)

    # Raw expression: RSEM-like, non-negative.
    raw = rng.lognormal(mean=3.0, sigma=1.0, size=(n_patients, n_genes))
    # Signal genes carry the prognostic information.
    weights = rng.standard_normal(n_signal)
    signal_feats = np.log2(raw[:, :n_signal] + 1.0)
    signal_feats = (signal_feats - signal_feats.mean(0)) / (signal_feats.std(0) + 1e-8)
    true_risk = signal_feats @ weights  # higher => earlier event on average

    # Exponential times with rate = exp(true_risk). Scale so median ~24 months.
    baseline = 24.0
    true_time = rng.exponential(scale=baseline * np.exp(-true_risk))
    # Censor randomly at ~event_rate.
    cens_time = rng.uniform(0, baseline * 2, size=n_patients)
    observed_time = np.minimum(true_time, cens_time)
    event = (true_time <= cens_time).astype(int)
    # Trim to target event rate so the harness sees something realistic.
    if event.mean() > event_rate:
        # Force some events to be censored.
        to_flip = rng.choice(
            np.where(event == 1)[0],
            size=int((event.mean() - event_rate) * n_patients),
            replace=False,
        )
        event[to_flip] = 0

    patients = [f"P{i:04d}" for i in range(n_patients)]
    genes = [f"G{i:04d}" for i in range(n_genes)]
    expr = pd.DataFrame(raw, index=patients, columns=genes)
    time = pd.Series(observed_time.astype(float), index=patients)
    ev = pd.Series(event.astype(int), index=patients)
    return SurvivalCohort(expression=expr, time=time, event=ev, study_id="synthetic")


@pytest.mark.slow
def test_baseline_recovers_signal_on_synthetic_cohort(tmp_path: Path) -> None:
    """Can the full harness learn a linear Cox signal end-to-end?"""
    cohort = _make_synthetic_cohort()
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

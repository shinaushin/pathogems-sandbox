"""Shared pytest fixtures.

The synthetic `SurvivalCohort` here is the canonical fake used across
`test_data.py`, `test_smoke_end_to_end.py`, and anywhere else a test needs
a small but realistic-shaped cohort. Centralizing it prevents the two
almost-identical builders that previously lived in each test file from
drifting apart (e.g. one gets a signal, the other does not, and it
becomes unclear what "synthetic" means anywhere).

The fixture is parametrizable via `synthetic_cohort_factory` so callers
can tweak n_patients / n_genes / n_signal for their own needs while still
sharing the generative model (linear Cox signal + exponential times +
random censoring).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from pathogems.data import SurvivalCohort


def _make_synthetic_cohort(
    n_patients: int = 300,
    n_genes: int = 200,
    n_signal: int = 10,
    event_rate: float = 0.4,
    seed: int = 7,
) -> SurvivalCohort:
    """Build a synthetic cohort with an injected linear Cox signal.

    True risk is a linear combination of the first `n_signal` genes. Time
    is drawn from an exponential with rate proportional to exp(true_risk),
    matching the generative assumption of Cox PH — so a correctly wired
    harness should recover a C-index meaningfully above 0.5.
    """
    rng = np.random.default_rng(seed)

    # Raw expression: RSEM-like, non-negative.
    raw = rng.lognormal(mean=3.0, sigma=1.0, size=(n_patients, n_genes))
    weights = rng.standard_normal(n_signal)
    signal_feats = np.log2(raw[:, :n_signal] + 1.0)
    signal_feats = (signal_feats - signal_feats.mean(0)) / (signal_feats.std(0) + 1e-8)
    true_risk = signal_feats @ weights  # higher => earlier event on average

    baseline = 24.0
    true_time = rng.exponential(scale=baseline * np.exp(-true_risk))
    cens_time = rng.uniform(0, baseline * 2, size=n_patients)
    observed_time = np.minimum(true_time, cens_time)
    event = (true_time <= cens_time).astype(int)
    # Trim to target event rate so tests see a realistic censoring level.
    if event.mean() > event_rate:
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


@pytest.fixture
def synthetic_cohort_factory() -> Callable[..., SurvivalCohort]:
    """Return the `_make_synthetic_cohort` constructor for tests that want
    non-default sizes or seeds. Most callers should prefer `synthetic_cohort`.
    """
    return _make_synthetic_cohort


@pytest.fixture
def synthetic_cohort() -> SurvivalCohort:
    """Default synthetic cohort used by the smoke test and anywhere else
    that just needs "a realistic-looking cohort".
    """
    return _make_synthetic_cohort()

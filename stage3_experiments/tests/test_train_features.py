"""Tests for new training-loop features: SWA, LR warmup, AdamW, GELU.

These are end-to-end smoke tests that run the full ``cross_validate`` harness
with a tiny synthetic cohort and the feature under test enabled. They do not
assert a specific C-index — the goal is to exercise the code path and confirm
it completes without errors and produces finite outputs.

All tests are marked ``slow`` because they run actual training (even if only
for a handful of epochs).
"""

from __future__ import annotations

import numpy as np
import pytest

from pathogems.config import ExperimentConfig
from pathogems.data import SurvivalCohort
from pathogems.train import CVResult, cross_validate


def _fast_config(**overrides: object) -> ExperimentConfig:
    """Return a minimal ExperimentConfig suitable for a 2-fold, 8-epoch run."""
    defaults: dict[str, object] = dict(
        name="feature_test",
        cohort="synthetic",
        study_data_dir="",
        top_k_genes=50,
        hidden_dims=(32,),
        use_batchnorm=False,  # avoids update_bn issues with tiny batch sizes
        dropout=0.0,
        epochs=8,
        early_stopping_patience=0,  # disable so all code after the loop runs
        batch_size=64,
        lr=1e-3,
        weight_decay=0.0,
        n_folds=2,
        seed=42,
        notes="Feature smoke test.",
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)  # type: ignore[arg-type]


def _all_finite(result: CVResult) -> bool:
    return all(np.isfinite(c) for c in result.per_fold_c_index())


@pytest.mark.slow
def test_swa_runs_without_error(synthetic_cohort: SurvivalCohort) -> None:
    """SWA path: swa_start_fraction=0.5 with 8 epochs → SWA active from epoch 4."""
    cfg = _fast_config(
        name="swa_smoke",
        swa_start_fraction=0.5,
    )
    result = cross_validate(synthetic_cohort, cfg, verbose=False)
    assert _all_finite(result)


@pytest.mark.slow
def test_cosine_lr_warmup_runs_without_error(synthetic_cohort: SurvivalCohort) -> None:
    """Composable schedule: warmup for 2 epochs then cosine for 6 (SequentialLR path)."""
    cfg = _fast_config(
        name="cosine_warmup_smoke",
        lr_schedule="cosine",
        lr_warmup_epochs=2,
    )
    result = cross_validate(synthetic_cohort, cfg, verbose=False)
    assert _all_finite(result)


@pytest.mark.slow
def test_warmup_only_runs_without_error(synthetic_cohort: SurvivalCohort) -> None:
    """Warmup without a following cosine schedule (single LinearLR path)."""
    cfg = _fast_config(
        name="warmup_only_smoke",
        lr_warmup_epochs=3,
    )
    result = cross_validate(synthetic_cohort, cfg, verbose=False)
    assert _all_finite(result)


@pytest.mark.slow
def test_adamw_optimizer_runs_without_error(synthetic_cohort: SurvivalCohort) -> None:
    """AdamW optimizer factory is wired end-to-end through the training loop."""
    cfg = _fast_config(
        name="adamw_smoke",
        optimizer="adamw",
        weight_decay=1e-4,
    )
    result = cross_validate(synthetic_cohort, cfg, verbose=False)
    assert _all_finite(result)


@pytest.mark.slow
def test_gelu_activation_runs_without_error(synthetic_cohort: SurvivalCohort) -> None:
    """GELU activation end-to-end: Xavier init + GELU forward through cross_validate."""
    cfg = _fast_config(
        name="gelu_smoke",
        activation="gelu",
    )
    result = cross_validate(synthetic_cohort, cfg, verbose=False)
    assert _all_finite(result)

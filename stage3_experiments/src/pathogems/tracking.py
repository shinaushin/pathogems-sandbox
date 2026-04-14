"""MLflow experiment tracking — optional wrapper around the training loop.

Design
------
MLflow is an *observer* of the run, not a driver of it. The JSON run log
in `run_log.py` remains the source of truth for the Stage 3 -> Stage 4
contract, because:

1. It's versioned and has an explicit schema (ADR 0005). A downstream
   consumer never has to pay the cost of spinning up MLflow just to
   read what happened.
2. MLflow's storage format is opaque to diff tools. Committing a run
   log to git makes "what did I change and what did that do?" a
   `git diff` away; "open the MLflow UI and eyeball two runs" is
   slower and doesn't survive a repo move.

That said, MLflow is useful when you have more than a handful of runs
and want to plot C-index vs. learning rate across them, or tag a
family of experiments for later. So we support it *opt-in*: nothing
breaks if `enable_mlflow=False` (the default), and nothing breaks if
mlflow is not installed.

Context-manager API
-------------------
The CLI wraps training with `track_run(config)`:

    with track_run(config) as tracker:
        result = cross_validate(cohort, config, ...)
        tracker.log_cv_result(result)
        log_path = write_run_log(...)
        tracker.log_artifact(log_path)

When `enable_mlflow=False`, `track_run` returns a no-op tracker whose
methods are all no-ops, so the CLI has exactly one code path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .train import CVResult

log = logging.getLogger(__name__)


class _NullTracker:
    """No-op tracker used when mlflow is disabled or unavailable.

    Keeping this class as a stand-in lets the CLI stay linear — we never
    have to wrap tracker calls in `if tracker: ...`.
    """

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        pass

    def log_cv_result(self, result: "CVResult") -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass


class _MLflowTracker:
    """Thin adapter around an active `mlflow.ActiveRun`.

    This is the *only* place where mlflow imports are allowed so that the
    rest of the codebase stays free of a soft dependency — `import
    mlflow` happens only when `enable_mlflow=True`.
    """

    def __init__(self, mlflow_module: Any) -> None:
        self._ml = mlflow_module

    def log_params(self, params: dict[str, Any]) -> None:
        # MLflow casts all param values to str; none-like values would
        # round-trip as "None". Filter them out for cleaner UI.
        self._ml.log_params({k: v for k, v in params.items() if v is not None})

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        # mlflow silently drops non-finite floats; be explicit.
        if value != value or value in (float("inf"), float("-inf")):  # NaN or inf
            return
        self._ml.log_metric(name, float(value), step=step)

    def log_cv_result(self, result: "CVResult") -> None:
        """Log aggregate + per-fold metrics. Per-fold metrics use `step`
        so MLflow displays them as a small per-fold curve in the UI.

        Per-epoch loss curves are logged as ``fold{i}_train_loss`` and
        ``fold{i}_val_loss`` with ``step=epoch``, which MLflow renders
        as a line chart per fold.
        """
        self.log_metric("c_index_mean", result.c_index_mean)
        self.log_metric("c_index_std", result.c_index_std)
        self.log_metric("final_loss_mean", result.final_loss_mean)
        for fold in result.folds:
            self.log_metric("fold_c_index", fold.c_index, step=fold.fold_id)
            self.log_metric("fold_final_val_loss", fold.final_val_loss, step=fold.fold_id)
            self.log_metric("fold_epochs_trained", fold.epochs_trained, step=fold.fold_id)
            # Per-epoch loss curves for convergence diagnostics.
            for epoch, tl in enumerate(fold.train_losses, start=1):
                self.log_metric(f"fold{fold.fold_id}_train_loss", tl, step=epoch)
            for epoch, vl in enumerate(fold.val_losses, start=1):
                self.log_metric(f"fold{fold.fold_id}_val_loss", vl, step=epoch)

    def log_artifact(self, path: Path) -> None:
        self._ml.log_artifact(str(path))


@contextmanager
def track_run(config: "ExperimentConfig") -> Iterator[_NullTracker | _MLflowTracker]:
    """Context manager that yields a tracker matching the config.

    If `enable_mlflow` is False, yields a `_NullTracker` (all no-ops).
    If `enable_mlflow` is True but mlflow is not importable, yields a
    `_NullTracker` and prints a warning — we don't want to fail a real
    training run because observability is misconfigured.
    """
    if not config.enable_mlflow:
        yield _NullTracker()
        return

    try:
        import mlflow
    except ImportError:
        log.warning(
            "mlflow not installed — `pip install mlflow` to enable. "
            "Continuing without tracking."
        )
        yield _NullTracker()
        return

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.name):
        tracker = _MLflowTracker(mlflow)
        # Log the config up front so a failed run still has the hyperparams
        # visible. The CLI is expected to also call `tracker.log_cv_result`
        # once training completes.
        tracker.log_params(config.to_dict())
        yield tracker

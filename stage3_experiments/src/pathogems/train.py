"""Single-fold and cross-validated training loops.

Design:
    `train_one_fold` is a pure function over `(FoldTensors, ExperimentConfig)`
    returning a `FoldResult`. It does the full inner loop (train/val split,
    epoch loop, early stopping, final test C-index) with no side effects
    beyond console logging. Saving files and writing the run log happens
    one level up in `cross_validate`, which is where orchestration lives.

    Keeping the training logic side-effect-free makes it easy to:
      * unit-test on tiny synthetic data in under a second,
      * plug into a future hyperparameter search without rewriting I/O,
      * reason about reproducibility (the config is the only input).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .data import FoldTensors, SurvivalCohort, build_fold_tensors, cv_splits
from .loss import cox_ph_loss
from .metrics import concordance_index
from .model import OmicsMLP, OmicsMLPConfig


# --------------------------------------------------------------------------- #
# Per-fold result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FoldResult:
    """Outcome of training on a single CV fold."""

    fold_id: int
    c_index: float  # held-out test-fold C-index — this is the headline number
    final_train_loss: float
    final_val_loss: float
    epochs_trained: int  # may be < config.epochs if early stopping fired
    best_epoch: int  # epoch at which val loss was lowest
    wall_clock_sec: float


@dataclass(frozen=True, slots=True)
class CVResult:
    """Aggregate of all folds."""

    folds: list[FoldResult]

    @property
    def c_index_mean(self) -> float:
        vals = [f.c_index for f in self.folds if not math.isnan(f.c_index)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def c_index_std(self) -> float:
        vals = [f.c_index for f in self.folds if not math.isnan(f.c_index)]
        # ddof=1 because we're estimating population std from a sample of folds.
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    @property
    def final_loss_mean(self) -> float:
        return float(np.mean([f.final_val_loss for f in self.folds]))

    def per_fold_c_index(self) -> list[float]:
        return [f.c_index for f in self.folds]

    def per_fold_final_loss(self) -> list[float]:
        return [f.final_val_loss for f in self.folds]


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _build_model(config: ExperimentConfig, in_features: int) -> nn.Module:
    if config.model == "omics_mlp":
        return OmicsMLP(
            OmicsMLPConfig(
                in_features=in_features,
                hidden_dims=tuple(config.hidden_dims),
                dropout=config.dropout,
                use_batchnorm=config.use_batchnorm,
            )
        )
    raise ValueError(f"Unknown model: {config.model!r}. Add a case in _build_model.")


def _build_optimizer(config: ExperimentConfig, params: list[nn.Parameter]) -> optim.Optimizer:
    if config.optimizer == "adam":
        return optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {config.optimizer!r}.")


def _compute_loss(config: ExperimentConfig, risk: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if config.loss == "cox_ph":
        return cox_ph_loss(risk, t, e)
    raise ValueError(f"Unknown loss: {config.loss!r}.")


# --------------------------------------------------------------------------- #
# Inner train/val split
# --------------------------------------------------------------------------- #
def _inner_split(
    n_train: int,
    val_fraction: float,
    event_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training fold into inner train / val for early stopping.

    Stratified on event to keep val event rate reasonable when val_fraction
    is small. Deterministic from `seed`.
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(n_train)
    # train_test_split handles the stratify-by-event case and keeps the
    # resulting indices aligned with the input order.
    inner_train, inner_val = train_test_split(
        idx, test_size=val_fraction, random_state=seed, stratify=event_train
    )
    return inner_train, inner_val


# --------------------------------------------------------------------------- #
# Single-fold training
# --------------------------------------------------------------------------- #
def train_one_fold(
    fold: FoldTensors,
    config: ExperimentConfig,
    fold_id: int,
    device: torch.device | None = None,
) -> FoldResult:
    """Train one model on `fold`, return its held-out C-index and metadata."""
    device = device or torch.device("cpu")
    started = time.perf_counter()

    torch.manual_seed(config.seed + fold_id)  # fold-unique but deterministic
    np.random.seed(config.seed + fold_id)  # noqa: NPY002 — torch-style global seed

    # Inner val split for early stopping.
    inner_train_idx, inner_val_idx = _inner_split(
        n_train=fold.x_train.shape[0],
        val_fraction=config.val_fraction,
        event_train=fold.event_train,
        seed=config.seed + fold_id,
    )

    def _to_t(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=device)

    x_tr = _to_t(fold.x_train[inner_train_idx])
    t_tr = _to_t(fold.time_train[inner_train_idx])
    e_tr = _to_t(fold.event_train[inner_train_idx])
    x_va = _to_t(fold.x_train[inner_val_idx])
    t_va = _to_t(fold.time_train[inner_val_idx])
    e_va = _to_t(fold.event_train[inner_val_idx])
    x_te = _to_t(fold.x_test)
    t_te = _to_t(fold.time_test)
    e_te = _to_t(fold.event_test)

    # NB: Cox PH is a rank-based loss; we want each mini-batch to contain
    # enough events that the risk set is meaningful. TensorDataset + shuffle
    # works because our cohorts are small enough that big batches fit in
    # memory trivially.
    loader = DataLoader(
        TensorDataset(x_tr, t_tr, e_tr),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = _build_model(config, in_features=fold.x_train.shape[1]).to(device)
    optimizer = _build_optimizer(config, list(model.parameters()))

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    final_train_loss = float("nan")
    epochs_trained = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, tb, eb in loader:
            optimizer.zero_grad(set_to_none=True)
            risk = model(xb)
            loss = _compute_loss(config, risk, tb, eb)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
        final_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = _compute_loss(config, model(x_va), t_va, e_va).item()

        epochs_trained = epoch
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if config.early_stopping_patience > 0 and no_improve >= config.early_stopping_patience:
                break

    # Restore best weights for test evaluation.
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        risk_te = model(x_te).cpu().numpy()
    c = concordance_index(risk_te, fold.time_test, fold.event_test)

    return FoldResult(
        fold_id=fold_id,
        c_index=float(c),
        final_train_loss=final_train_loss,
        final_val_loss=float(best_val if math.isfinite(best_val) else float("nan")),
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        wall_clock_sec=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------- #
# Cross-validated training
# --------------------------------------------------------------------------- #
def cross_validate(
    cohort: SurvivalCohort,
    config: ExperimentConfig,
    device: torch.device | None = None,
    verbose: bool = True,
) -> CVResult:
    """Run the full 5-fold (or n_folds) CV. Returns per-fold results."""
    splits = cv_splits(cohort, n_folds=config.n_folds, seed=config.seed)
    folds: list[FoldResult] = []
    for fold_id, (train_idx, test_idx) in enumerate(splits):
        tensors = build_fold_tensors(
            cohort, train_idx, test_idx, top_k=config.top_k_genes
        )
        result = train_one_fold(tensors, config, fold_id=fold_id, device=device)
        if verbose:
            print(
                f"[fold {fold_id + 1}/{config.n_folds}] "
                f"C-index={result.c_index:.4f}  "
                f"val_loss={result.final_val_loss:.4f}  "
                f"epochs={result.epochs_trained} (best@{result.best_epoch})  "
                f"{result.wall_clock_sec:.1f}s"
            )
        folds.append(result)
    return CVResult(folds=folds)

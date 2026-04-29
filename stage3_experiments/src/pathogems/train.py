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
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .data import FoldTensors, SurvivalCohort, build_fold_tensors, cv_splits
from .loss import LOSS_REGISTRY
from .metrics import concordance_index
from .models import MODEL_REGISTRY
from .optimizers import OPTIMIZER_REGISTRY

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-fold result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FoldResult:
    """Outcome of training on a single CV fold.

    ``train_losses`` and ``val_losses`` store the per-epoch loss curve so
    convergence can be inspected after the fact — either plotted via MLflow
    or dumped into the run log for Stage 4 analysis.
    """

    fold_id: int
    c_index: float  # held-out test-fold C-index — this is the headline number
    final_train_loss: float
    final_val_loss: float
    epochs_trained: int  # may be < config.epochs if early stopping fired
    best_epoch: int  # epoch at which val loss was lowest
    wall_clock_sec: float
    train_losses: tuple[float, ...] = ()  # length == epochs_trained
    val_losses: tuple[float, ...] = ()  # length == epochs_trained


@dataclass(frozen=True, slots=True)
class CVResult:
    """Aggregate of all folds."""

    folds: list[FoldResult]

    @property
    def c_index_mean(self) -> float:
        """Mean C-index across folds, ignoring NaN (failed) folds."""
        vals = [f.c_index for f in self.folds if not math.isnan(f.c_index)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def c_index_std(self) -> float:
        """Sample std of C-index across folds (ddof=1); 0.0 for a single fold."""
        vals = [f.c_index for f in self.folds if not math.isnan(f.c_index)]
        # ddof=1 because we're estimating population std from a sample of folds.
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    @property
    def final_loss_mean(self) -> float:
        """Mean final validation loss across all folds."""
        return float(np.mean([f.final_val_loss for f in self.folds]))

    def per_fold_c_index(self) -> list[float]:
        """C-index for each fold in fold order."""
        return [f.c_index for f in self.folds]

    def per_fold_final_loss(self) -> list[float]:
        """Final validation loss for each fold in fold order."""
        return [f.final_val_loss for f in self.folds]


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
    idx = np.arange(n_train)
    # train_test_split handles the stratify-by-event case and keeps the
    # resulting indices aligned with the input order.
    inner_train, inner_val = train_test_split(
        idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=event_train,
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
    np.random.seed(config.seed + fold_id)  # torch-style global seed

    # Inner val split for early stopping.
    inner_train_idx, inner_val_idx = _inner_split(
        n_train=fold.x_train.shape[0],
        val_fraction=config.val_fraction,
        event_train=fold.event_train,
        seed=config.seed + fold_id,
    )

    def _to_t(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Cast a NumPy array to a typed tensor on the training device."""
        return torch.as_tensor(arr, dtype=dtype, device=device)

    x_tr = _to_t(fold.x_train[inner_train_idx])
    t_tr = _to_t(fold.time_train[inner_train_idx])
    e_tr = _to_t(fold.event_train[inner_train_idx])
    x_va = _to_t(fold.x_train[inner_val_idx])
    t_va = _to_t(fold.time_train[inner_val_idx])
    e_va = _to_t(fold.event_train[inner_val_idx])
    x_te = _to_t(fold.x_test)

    # NB: Cox PH is a rank-based loss; we want each mini-batch to contain
    # enough events that the risk set is meaningful. `batch_size=None`
    # means full-batch — the entire training fold is one "batch", which
    # is what the partial likelihood is actually defined over. For TCGA
    # cohorts (<~1200 patients) this fits on CPU trivially and removes
    # the approximation of treating each mini-batch as its own risk set.
    # An explicit int keeps the original mini-batch behavior.
    effective_bs = x_tr.shape[0] if config.batch_size is None else config.batch_size
    loader = DataLoader(
        TensorDataset(x_tr, t_tr, e_tr),
        batch_size=effective_bs,
        shuffle=config.batch_size is not None,  # no shuffle needed when batch == dataset
        drop_last=False,
    )

    model = MODEL_REGISTRY.get(config.model)(fold.x_train.shape[1], config, fold.selected_genes).to(
        device
    )
    optimizer = OPTIMIZER_REGISTRY.get(config.optimizer)(model.parameters(), config)
    loss_fn = LOSS_REGISTRY.get(config.loss)

    # ------------------------------------------------------------------ #
    # LR schedule: composable warmup + main schedule
    #
    # lr_warmup_epochs > 0: prepend a LinearLR warmup that ramps the LR
    #   from lr×0.01 → lr over `warmup_epochs` steps.  After warmup the
    #   main schedule takes over; for cosine its T_max is reduced by the
    #   warmup epochs so the total budget stays at config.epochs.
    #
    # lr_schedule == "cosine": CosineAnnealingLR decaying to lr×0.01.
    #   T_max covers the post-warmup epochs so the full decay happens in
    #   the remaining training budget.
    #
    # When both are set they are chained with SequentialLR.
    # ------------------------------------------------------------------ #
    warmup_epochs = config.lr_warmup_epochs
    post_warmup_epochs = max(config.epochs - warmup_epochs, 1)

    _schedulers: list = []
    _milestones: list[int] = []

    if warmup_epochs > 0:
        _schedulers.append(
            LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        )
        _milestones.append(warmup_epochs)

    if config.lr_schedule == "cosine":
        _schedulers.append(
            CosineAnnealingLR(optimizer, T_max=post_warmup_epochs, eta_min=config.lr * 0.01)
        )

    if len(_schedulers) > 1:
        scheduler = SequentialLR(optimizer, schedulers=_schedulers, milestones=_milestones)
    elif len(_schedulers) == 1:
        scheduler = _schedulers[0]
    else:
        scheduler = None

    # ------------------------------------------------------------------ #
    # SWA: Stochastic Weight Averaging
    #
    # When swa_start_fraction > 0, an AveragedModel starts accumulating a
    # uniform average of model weights from swa_start_epoch onward.
    # - Early stopping still operates during the pre-SWA phase; if it fires,
    #   best_state is restored and the SWA phase continues from there.
    # - During the SWA phase the LR is held at a constant low value
    #   (lr×0.05) instead of following the main schedule — this is the
    #   canonical SWA recipe (Izmailov et al., 2018).
    # - After all epochs, BatchNorm statistics are recalibrated on the
    #   training data using update_bn before final evaluation.
    # - If SWA never collects any weights (early stopping fired before
    #   swa_start_epoch), evaluation falls back to the best-checkpoint
    #   model as normal.
    # ------------------------------------------------------------------ #
    swa_start_epoch: int | None = (
        max(1, int(config.epochs * config.swa_start_fraction))
        if config.swa_start_fraction > 0.0
        else None
    )
    swa_model: AveragedModel | None = AveragedModel(model) if swa_start_epoch is not None else None
    swa_active = False  # True once swa_model.update_parameters() is called at least once
    swa_lr = config.lr * 0.05

    # Each model declares its own L1 target via RegularizableMixin.regularized_weight.
    # None means "skip L1 for this architecture" (e.g. GeneAttentionNet).
    reg_weight = model.regularized_weight if config.l1_weight > 0.0 else None  # type: ignore[union-attr]

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    final_train_loss = float("nan")
    epochs_trained = 0
    all_train_losses: list[float] = []
    all_val_losses: list[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, tb, eb in loader:
            optimizer.zero_grad(set_to_none=True)
            risk = model(xb)
            loss = loss_fn(risk, tb, eb)
            # Optional L1 penalty — target declared by model.regularized_weight.
            if reg_weight is not None and torch.isfinite(loss):
                loss = loss + config.l1_weight * reg_weight.abs().sum()
            if torch.isfinite(loss):
                loss.backward()  # type: ignore[no-untyped-call]
                # Gradient clipping: Cox PH can produce large gradients
                # when a single patient dominates the risk set. Clipping
                # to a global norm of `max_grad_norm` prevents NaN
                # explosions without materially hurting convergence.
                if config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                epoch_losses.append(loss.item())

        # SWA weight collection and LR override — must happen before val
        # so the SWA model's step count is in sync with the epoch count.
        if swa_start_epoch is not None and epoch >= swa_start_epoch:
            # Override LR to the constant SWA rate (bypasses the main schedule).
            for pg in optimizer.param_groups:
                pg["lr"] = swa_lr
            assert swa_model is not None
            swa_model.update_parameters(model)
            swa_active = True
        elif scheduler is not None:
            scheduler.step()

        final_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        all_train_losses.append(final_train_loss)

        # Validation (always on the base model, not the SWA average — SWA
        # BN stats are only recalibrated at the very end, so mid-training
        # SWA val loss would be misleading).
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_va), t_va, e_va).item()
        all_val_losses.append(val_loss)

        epochs_trained = epoch

        # Early stopping only operates before the SWA phase begins.
        # Once SWA kicks in we want to collect as many diverse checkpoints
        # as possible (stopping early would give only 1–2 averaged points).
        in_swa_phase = swa_start_epoch is not None and epoch >= swa_start_epoch
        if not in_swa_phase:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if config.early_stopping_patience > 0 and no_improve >= config.early_stopping_patience:
                    # Restore best weights, then continue into SWA phase
                    # (if enabled) from the best-checkpoint starting point.
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    if swa_start_epoch is None:
                        break  # no SWA — stop here
                    # SWA phase not yet started: fast-forward into it.
                    # This re-enters the loop from swa_start_epoch.
                    log.info(
                        "fold %d: early stopping at epoch %d; entering SWA phase from epoch %d",
                        fold_id,
                        epoch,
                        swa_start_epoch,
                    )
                    # Reset no_improve so the SWA phase always runs to max_epochs.
                    no_improve = 0
                    # Jump to the SWA start by continuing the outer loop — the
                    # swa_start_epoch guard above will catch epochs from here on.
                    continue

    # ------------------------------------------------------------------ #
    # Final evaluation
    # ------------------------------------------------------------------ #
    if swa_active:
        # Recalibrate BatchNorm running statistics on the training data.
        # The SWA model's BN buffers are a weighted average of the
        # constituent models' buffers, which is not statistically valid;
        # update_bn runs a forward pass over the training set to fix them.
        assert swa_model is not None
        _bn_loader = DataLoader(
            TensorDataset(x_tr), batch_size=x_tr.shape[0], shuffle=False
        )
        update_bn(_bn_loader, swa_model, device=device)
        swa_model.eval()
        with torch.no_grad():
            risk_te = swa_model(x_te).cpu().numpy()
        log.info("fold %d: using SWA model for evaluation", fold_id)
    else:
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
        train_losses=tuple(all_train_losses),
        val_losses=tuple(all_val_losses),
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
            cohort,
            train_idx,
            test_idx,
            top_k=config.top_k_genes,
            gene_selection=config.gene_selection,
        )
        result = train_one_fold(tensors, config, fold_id=fold_id, device=device)
        if verbose:
            log.info(
                "fold %d/%d  C-index=%.4f  val_loss=%.4f  " "epochs=%d (best@%d)  %.1fs",
                fold_id + 1,
                config.n_folds,
                result.c_index,
                result.final_val_loss,
                result.epochs_trained,
                result.best_epoch,
                result.wall_clock_sec,
            )
        folds.append(result)
    return CVResult(folds=folds)

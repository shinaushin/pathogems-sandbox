"""Model architectures for Stage 3.

Currently one model — `OmicsMLP`, the baseline from ADR 0001. Subsequent
experiments will add models alongside this one (never replace), so we can
compare them on the same harness.

The baseline is intentionally small: 2 hidden layers, 128 + 32 units,
ReLU, dropout, BatchNorm, and a single scalar output (the Cox risk score
— see ADR 0002). This leaves room for controlled experiments that change
exactly one axis: depth, width, activation, regularization, etc.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from .registry import Registry

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # avoid a runtime import cycle (config has no torch deps anyway)
    from .config import ExperimentConfig

# Registry maps a model name (as used in `ExperimentConfig.model`) to a
# *factory* callable `factory(in_features: int, config: ExperimentConfig) -> nn.Module`.
# The factory takes the runtime `in_features` (which the config doesn't know
# — it depends on `top_k_genes` and the preprocessor output) plus the full
# config so each model can pick out the knobs it cares about. Keeping the
# factory signature uniform lets `train.py` call `MODEL_REGISTRY.get(name)`
# and forget about which specific model it's constructing.
ModelFactory = Callable[[int, "ExperimentConfig"], nn.Module]
MODEL_REGISTRY: Registry[ModelFactory] = Registry("model")


@dataclass(frozen=True, slots=True)
class OmicsMLPConfig:
    """Configuration for `OmicsMLP`.

    Attributes:
        in_features: Number of input genes (== `top_k` from the
            preprocessor). Must match the data at runtime; a mismatch
            raises at construction time rather than silently producing
            garbage risk scores.
        hidden_dims: Sizes of the hidden layers, in order. Default
            (128, 32) follows the baseline spec from ADR 0001.
        dropout: Dropout probability applied after each hidden layer.
            0.3 is the DeepSurv default and works reasonably on TCGA-BRCA
            in pilot runs.
        use_batchnorm: BatchNorm1d after each linear layer. Useful because
            z-scored inputs + ReLU can still have per-layer distribution
            shift as training progresses.
    """

    in_features: int
    hidden_dims: tuple[int, ...] = (128, 32)
    dropout: float = 0.3
    use_batchnorm: bool = True

    def __post_init__(self) -> None:
        if self.in_features <= 0:
            raise ValueError("in_features must be positive.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")


class OmicsMLP(nn.Module):
    """Feed-forward MLP that maps an omics feature vector to a scalar risk.

    The network outputs a single scalar per patient — the Cox linear
    predictor `r = f(x)`. *We deliberately do not apply a final sigmoid
    or softplus*: the Cox partial likelihood is invariant to monotone
    transforms of `r`, so bounding it adds no information and makes
    gradients vanish for large |r|. Downstream code assumes unbounded
    scalar outputs.

    Shape contract:
        input:  (batch, in_features), float32
        output: (batch,),            float32  (risk scores)
    """

    def __init__(self, config: OmicsMLPConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        prev = config.in_features
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev, h))
            if config.use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Weight init: Kaiming for ReLU stacks is the standard choice and
        # avoids the "all zeros" trap that makes Cox training stall.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return risk scores of shape `(batch,)`."""
        if x.dim() != 2 or x.shape[1] != self.config.in_features:
            raise ValueError(
                f"Expected input of shape (batch, {self.config.in_features}), got {tuple(x.shape)}."
            )
        # BatchNorm1d requires batch size > 1 in train mode. The trainer
        # passes batches of reasonable size, but a size-1 edge case is easy
        # to hit in tests; squeeze(-1) tolerates both (N,1) and (1,1).
        return self.net(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("omics_mlp")
def _build_omics_mlp(in_features: int, config: "ExperimentConfig") -> nn.Module:
    """Factory: translate the generic `ExperimentConfig` into `OmicsMLPConfig`.

    Living next to `OmicsMLP` (instead of in `train.py`) means the
    responsibility for mapping config fields to constructor args stays with
    the owner of those fields. A second model in this file only has to add
    its own factory, with no change to `train.py`.
    """
    return OmicsMLP(
        OmicsMLPConfig(
            in_features=in_features,
            hidden_dims=tuple(config.hidden_dims),
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
        )
    )


# --------------------------------------------------------------------------- #
# LinearCox — "baseline of the baseline"
# --------------------------------------------------------------------------- #
class LinearCox(nn.Module):
    """Linear Cox predictor: risk = w^T x + b, no non-linearity.

    This is the *simplest possible* neural Cox model and serves as a
    sanity-check reference point: if the MLP can't beat a regularized
    linear Cox fit, either the MLP is mis-tuned or there is no non-linear
    signal to extract in the first place. Classic survival-ML papers
    (DeepSurv and successors) treat linear Cox as the baseline to beat,
    so we want it available on the exact same harness to make the
    comparison meaningful.

    Notes on shape: we implement this as a single `nn.Linear(in, 1)`
    rather than calling it a "logistic regression" — Cox PH's partial
    likelihood is already what this model optimizes end-to-end via
    `loss.cox_ph_loss`, so the terminology stays precise.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        self.linear = nn.Linear(in_features, 1)
        # Small init so early batches don't produce extreme risks that
        # destabilize the logcumsumexp in the loss.
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input of shape (batch, {self.in_features}), got {tuple(x.shape)}."
            )
        return self.linear(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("linear_cox")
def _build_linear_cox(in_features: int, config: "ExperimentConfig") -> nn.Module:
    """LinearCox has no hyperparameters beyond `in_features`.

    ``hidden_dims``, ``dropout``, and ``use_batchnorm`` are ignored. We
    log a warning when they differ from their defaults so a user who
    accidentally left ``dropout=0.5`` in a linear_cox config knows they're
    not doing what they think they're doing — while still allowing the
    common case of "copy the MLP config, change only `model`" to work
    without requiring field cleanup.
    """
    # Defaults mirror ExperimentConfig's field defaults. We hard-code them
    # here rather than reading class attributes because ExperimentConfig
    # uses `slots=True`, which removes class-level defaults and causes a
    # mypy error if accessed as `ExperimentConfig.hidden_dims`.
    _DEFAULT_HIDDEN_DIMS = (128, 32)
    _DEFAULT_DROPOUT = 0.3
    _DEFAULT_USE_BATCHNORM = True

    if config.hidden_dims != _DEFAULT_HIDDEN_DIMS:
        log.warning(
            "linear_cox ignores hidden_dims=%s (config has non-default value).",
            config.hidden_dims,
        )
    if config.dropout != _DEFAULT_DROPOUT:
        log.warning(
            "linear_cox ignores dropout=%.2f (config has non-default value).",
            config.dropout,
        )
    if config.use_batchnorm != _DEFAULT_USE_BATCHNORM:
        log.warning(
            "linear_cox ignores use_batchnorm=%s (config has non-default value).",
            config.use_batchnorm,
        )
    return LinearCox(in_features=in_features)

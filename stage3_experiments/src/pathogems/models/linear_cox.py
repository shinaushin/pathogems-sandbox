"""LinearCox — penalized linear Cox proportional-hazards model.

The simplest possible neural Cox model: a single linear layer with no
non-linearity.  Serves as the interpretable baseline and the target for
Cox-LASSO experiments (linear_cox + l1_weight).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from ._mixin import RegularizableMixin
from ._registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from ..config import ExperimentConfig

log = logging.getLogger(__name__)

# Defaults mirror ExperimentConfig field defaults.  Hard-coded here rather
# than read from the class because ExperimentConfig uses slots=True, which
# removes class-level defaults and causes a mypy error if accessed as
# ExperimentConfig.hidden_dims.
_DEFAULT_HIDDEN_DIMS = (128, 32)
_DEFAULT_DROPOUT = 0.3
_DEFAULT_USE_BATCHNORM = True


class LinearCox(RegularizableMixin, nn.Module):
    """Linear Cox predictor: risk = w^T x + b, no non-linearity.

    This is the *simplest possible* neural Cox model and serves as a
    sanity-check reference: if the MLP can't beat a regularized linear Cox
    fit, either the MLP is mis-tuned or there is no non-linear signal to
    extract.  Classic survival-ML papers (DeepSurv and successors) treat
    linear Cox as the baseline to beat, so we want it on the exact same
    harness for a meaningful comparison.

    Shape contract:
        input:  (batch, in_features), float32
        output: (batch,),             float32  (risk scores)
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
        """Return risk scores of shape ``(batch,)``."""
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input of shape (batch, {self.in_features}), "
                f"got {tuple(x.shape)}."
            )
        return self.linear(x).squeeze(-1)

    @property
    def regularized_weight(self) -> torch.Tensor | None:
        """Return the single linear layer's weight for L1 (Cox-LASSO)."""
        return self.linear.weight

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("linear_cox")
def _build_linear_cox(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],  # not used by linear model
) -> nn.Module:
    """LinearCox has no hyperparameters beyond ``in_features``.

    ``hidden_dims``, ``dropout``, and ``use_batchnorm`` are ignored.  We
    log a warning when they differ from defaults so a user who accidentally
    left ``dropout=0.5`` in a linear_cox config knows they're not doing
    what they think — while still allowing "copy the MLP config, change
    only ``model``" to work without requiring field cleanup.
    """
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

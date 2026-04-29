"""OmicsMLP — baseline feed-forward MLP for omics survival prediction.

Baseline model from ADR 0001.  Maps a z-scored gene expression vector to
a scalar Cox risk score through a sequence of Linear → BatchNorm → ReLU →
Dropout blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from ._mixin import RegularizableMixin
from ._registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from ..config import ExperimentConfig

log = logging.getLogger(__name__)


_SUPPORTED_ACTIVATIONS = frozenset({"relu", "gelu", "silu"})


def _make_activation(name: str) -> nn.Module:
    """Return an activation module for the given name.

    ReLU is inplace-eligible (saves one buffer allocation per layer); GELU
    and SiLU are not inplace because PyTorch's implementations don't support
    it and the memory saving is negligible at our model sizes.
    """
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(_SUPPORTED_ACTIVATIONS)}.")


@dataclass(frozen=True, slots=True)
class OmicsMLPConfig:
    """Construction-time configuration for ``OmicsMLP``.

    Attributes:
        in_features: Number of input genes (== ``top_k`` from the
            preprocessor).  Must match the data at runtime; a mismatch
            raises at construction time rather than silently producing
            garbage risk scores.
        hidden_dims: Sizes of the hidden layers, in order.  Default
            (128, 32) follows the baseline spec from ADR 0001.
        dropout: Dropout probability applied after each hidden layer.
            0.3 is the DeepSurv default and works reasonably on TCGA-BRCA
            in pilot runs.
        use_batchnorm: BatchNorm1d after each linear layer.  Useful because
            z-scored inputs + ReLU can still have per-layer distribution
            shift as training progresses.
        activation: Hidden-layer activation name — "relu" (default), "gelu",
            or "silu".  Affects both the activation module and weight
            initialisation: ReLU uses Kaiming (He) init; GELU and SiLU use
            Xavier (Glorot) uniform init which does not assume a particular
            nonlinearity shape.
    """

    in_features: int
    hidden_dims: tuple[int, ...] = (128, 32)
    dropout: float = 0.3
    use_batchnorm: bool = True
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.in_features <= 0:
            raise ValueError("in_features must be positive.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Choose from {sorted(_SUPPORTED_ACTIVATIONS)}."
            )


class OmicsMLP(RegularizableMixin, nn.Module):
    """Feed-forward MLP that maps an omics feature vector to a scalar risk.

    The network outputs a single scalar per patient — the Cox linear
    predictor ``r = f(x)``.  We deliberately do not apply a final sigmoid
    or softplus: the Cox partial likelihood is invariant to monotone
    transforms of ``r``, so bounding it adds no information and makes
    gradients vanish for large |r|.  Downstream code assumes unbounded
    scalar outputs.

    Shape contract:
        input:  (batch, in_features), float32
        output: (batch,),             float32  (risk scores)
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
            layers.append(_make_activation(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Weight initialisation depends on the activation function.
        # - ReLU: Kaiming (He) normal with nonlinearity="relu" — scales by
        #   sqrt(2) to compensate for the zero-half of ReLU's range.
        # - GELU / SiLU: Xavier uniform — does not assume a specific
        #   nonlinearity shape; empirically standard in transformer FFNs.
        # The final output layer always uses Xavier (it maps to a scalar with
        # no activation, so the ReLU-specific Kaiming scaling is wrong there).
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if config.activation == "relu":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Output layer always Xavier regardless of hidden activation.
        output_linear = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.xavier_uniform_(output_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return risk scores of shape ``(batch,)``."""
        if x.dim() != 2 or x.shape[1] != self.config.in_features:
            raise ValueError(
                f"Expected input of shape (batch, {self.config.in_features}), "
                f"got {tuple(x.shape)}."
            )
        # BatchNorm1d requires batch size > 1 in train mode. The trainer
        # passes batches of reasonable size, but a size-1 edge case is easy
        # to hit in tests; squeeze(-1) tolerates both (N,1) and (1,1).
        return self.net(x).squeeze(-1)

    @property
    def regularized_weight(self) -> torch.Tensor | None:
        """Return the first linear layer's weight (gene → hidden projection).

        This is the meaningful target for L1 regularization in a flat MLP:
        sparsifying the input projection drives the model to ignore genes
        with no survival signal, reducing overfitting on high-dimensional
        TCGA data.
        """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                return m.weight
        return None  # pragma: no cover

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("omics_mlp")
def _build_omics_mlp(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],  # not used by flat MLP
) -> nn.Module:
    """Factory: translate the generic ``ExperimentConfig`` into ``OmicsMLPConfig``.

    Living next to ``OmicsMLP`` (instead of in ``train.py``) means the
    responsibility for mapping config fields to constructor args stays with
    the owner of those fields.  A second model only adds its own factory —
    no change to ``train.py``.
    """
    return OmicsMLP(
        OmicsMLPConfig(
            in_features=in_features,
            hidden_dims=tuple(config.hidden_dims),
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
            activation=config.activation,
        )
    )

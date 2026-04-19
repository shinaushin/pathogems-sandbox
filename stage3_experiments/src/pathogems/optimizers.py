"""Optimizer registry.

We keep the optimizer factories in their own module (rather than inlining
them in `train.py`) for two reasons:

1. **Isolation.** `train.py` should orchestrate the training loop, not
   enumerate every optimizer we might ever try. When a future experiment
   adds a one-line AdamW wrapper, the diff touches this file only.
2. **Consistency with model/loss registration.** All "pluggable by name"
   components follow the same pattern: a `*_REGISTRY` at module top, one
   factory per name, with the factory living next to the thing it
   constructs. Uniformity makes the codebase predictable.

The factory signature takes the full `ExperimentConfig` so individual
optimizers can pick up their own knobs (e.g. SGD's momentum) without
having to extend the call sites in `train.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from torch import nn, optim

from .registry import Registry

if TYPE_CHECKING:
    from .config import ExperimentConfig


# `Iterable[nn.Parameter]` matches what `model.parameters()` returns. We
# materialize to a list inside the factory because some torch optimizers
# iterate the param group twice on construction.
OptimizerFactory = Callable[[Iterable[nn.Parameter], "ExperimentConfig"], optim.Optimizer]
OPTIMIZER_REGISTRY: Registry[OptimizerFactory] = Registry("optimizer")


@OPTIMIZER_REGISTRY.register("adam")
def _build_adam(params: Iterable[nn.Parameter], config: ExperimentConfig) -> optim.Optimizer:
    return optim.Adam(list(params), lr=config.lr, weight_decay=config.weight_decay)


@OPTIMIZER_REGISTRY.register("sgd")
def _build_sgd(params: Iterable[nn.Parameter], config: ExperimentConfig) -> optim.Optimizer:
    # Momentum = 0.9 is the textbook default for SGD on non-convex NN losses;
    # it's fixed here (rather than a config field) so switching adam→sgd is
    # a true one-field change in the experiment config.
    return optim.SGD(list(params), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)

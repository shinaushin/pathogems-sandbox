"""RegularizableMixin — explicit L1 regularization contract for all models.

Instead of guessing the "first nn.Linear" by BFS (which silently picks the
wrong layer in complex architectures like PathwayMLP), each model declares
which weight tensor the training loop should regularize when l1_weight > 0.
"""

from __future__ import annotations

import torch


class RegularizableMixin:
    """Mixin that gives each model an explicit L1 regularization target.

    Override ``regularized_weight`` in subclasses to return the specific
    parameter tensor that L1 regularization should target.  The default
    returns ``None``, meaning "apply no L1 penalty for this model type".

    The training loop calls ``model.regularized_weight`` and adds
    ``l1_weight * weight.abs().sum()`` to the Cox loss when it is non-None.
    This makes the choice of which weights to penalize an auditable decision
    of each architecture, not an opaque structural search.

    Design note: a Protocol could enforce this at type-check time, but a
    mixin is simpler and achieves the same runtime contract with less
    boilerplate for the common case (most new architectures will want to
    inherit the default None rather than override it).
    """

    @property
    def regularized_weight(self) -> torch.Tensor | None:
        """Return the weight tensor to L1-regularize, or ``None`` to skip."""
        return None

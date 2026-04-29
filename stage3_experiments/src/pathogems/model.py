"""Backward-compatibility shim — import from ``.models`` instead.

This file re-exports everything from the ``models/`` package so that
existing code using ``from .model import MODEL_REGISTRY`` continues to work
without modification.  New code should import from ``.models`` directly.
"""

from .models import (  # noqa: F401
    MODEL_REGISTRY,
    ModelFactory,
    RegularizableMixin,
    OmicsMLPConfig,
    OmicsMLP,
    LinearCox,
    MaskedLinear,
    PathwayMLP,
    GeneAttentionNet,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelFactory",
    "RegularizableMixin",
    "OmicsMLPConfig",
    "OmicsMLP",
    "LinearCox",
    "MaskedLinear",
    "PathwayMLP",
    "GeneAttentionNet",
]

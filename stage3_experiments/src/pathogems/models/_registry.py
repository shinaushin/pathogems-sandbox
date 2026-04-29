"""MODEL_REGISTRY — shared registry for all model factories.

Isolated in its own sub-module so every per-architecture file can import it
without creating circular imports through the parent package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..registry import Registry

if TYPE_CHECKING:
    import torch.nn as nn

    from ..config import ExperimentConfig

# Every factory has this signature:
#   factory(in_features, config, selected_genes) -> nn.Module
#
# in_features    — runtime gene count (top_k_genes after preprocessor filtering)
# config         — the full ExperimentConfig; factories pick the fields they need
# selected_genes — ordered HGNC symbols; pathway-aware factories use this to
#                  build the gene-to-pathway connectivity mask.
ModelFactory = Callable[[int, "ExperimentConfig", "list[str]"], "nn.Module"]
MODEL_REGISTRY: Registry[ModelFactory] = Registry("model")

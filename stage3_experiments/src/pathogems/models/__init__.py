"""Model architectures package for Stage 3.

Each submodule owns one architecture and registers its factory with
``MODEL_REGISTRY`` on import.  This ``__init__`` imports all submodules so
that a single ``from pathogems.models import MODEL_REGISTRY`` guarantees all
factories are registered.

Adding a new architecture:
    1. Create ``models/my_model.py`` with the ``nn.Module`` class and a
       factory decorated with ``@MODEL_REGISTRY.register("my_model")``.
    2. Add ``from . import my_model as my_model`` below.
    3. Re-export the class if callers will import it by name.

No changes to ``train.py`` are needed — it calls
``MODEL_REGISTRY.get(config.model)(...)`` and is oblivious to which file
the factory came from.

The ModelFactory callable signature is:
    factory(in_features: int, config: ExperimentConfig, selected_genes: list[str]) -> nn.Module
"""

# Import submodules to trigger @MODEL_REGISTRY.register(...) decorators.
from . import gene_attention as gene_attention
from . import linear_cox as linear_cox
from . import omics_mlp as omics_mlp
from . import pathway_mlp as pathway_mlp
from ._mixin import RegularizableMixin
from ._registry import MODEL_REGISTRY, ModelFactory

# Re-export model classes for callers that need them by name.
from .gene_attention import GeneAttentionNet
from .linear_cox import LinearCox
from .omics_mlp import OmicsMLP, OmicsMLPConfig
from .pathway_mlp import MaskedLinear, PathwayMLP

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

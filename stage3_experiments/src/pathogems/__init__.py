"""Stage 3 experiment harness for the PathoGems auto-research agent.

Public submodules:
    data       - Loading cBioPortal omics + clinical, train/test splits, 5-fold CV.
    model      - Model architectures (currently just an omics-only MLP).
    loss       - Loss functions (currently Cox partial likelihood).
    optimizers - Optimizer factories (adam, sgd).
    metrics    - Evaluation metrics (currently Harrell's C-index).
    train      - Single-fold and cross-validated training loops.
    registry   - Small generic name->factory registry used by model/loss/optimizers.
    run_log    - Structured JSON run-log writer (Stage 3 -> Stage 4 contract).
    cli        - Command-line entry point: `pathogems-train --config <path>`.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import submodules that own side-effectful @register calls so their
# entries appear in the relevant registries as soon as anyone does
# `import pathogems`. Without this, a consumer that reaches only for
# `pathogems.train` would still work (train already imports these), but
# a consumer that wants to list registered models via
# `pathogems.model.MODEL_REGISTRY.names()` would get an empty registry
# unless they first triggered the registration themselves.
from . import loss as loss  # noqa: F401
from . import model as model  # noqa: F401
from . import optimizers as optimizers  # noqa: F401

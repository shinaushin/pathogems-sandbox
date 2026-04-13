"""Stage 3 experiment harness for the PathoGems auto-research agent.

Public submodules:
    data     - Loading cBioPortal omics + clinical, train/test splits, 5-fold CV.
    model    - Model architectures (currently just an omics-only MLP).
    loss     - Loss functions (currently Cox partial likelihood).
    metrics  - Evaluation metrics (currently Harrell's C-index).
    train    - Single-fold and cross-validated training loops.
    run_log  - Structured JSON run-log writer (Stage 3 -> Stage 4 contract).
    cli      - Command-line entry point: `pathogems-train --config <path>`.
"""

__version__ = "0.1.0"

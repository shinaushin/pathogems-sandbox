"""Experiment configuration.

Every run is fully described by an `ExperimentConfig` dataclass that can
be serialized to and from JSON. Two runs that load from byte-identical
JSON files must produce byte-identical run logs (modulo wall-clock time),
which is what makes "change one thing at a time" actually mean something.

This module is deliberately conservative about adding fields: every knob
is a place where two experiments can accidentally diverge. New fields
arrive only when we have a concrete experiment that requires them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Everything needed to reproduce one Stage 3 experiment.

    Fields are grouped by concern: identity, data, model, training,
    evaluation. Adding a field requires bumping no schema version because
    the serializer round-trips whatever dict it is given, but run-log
    consumers (Stage 4) filter to known fields.

    Attributes:
        name: Unique identifier, becomes the log file basename.
        cohort: TCGA cohort id, currently only "TCGA-BRCA".
        seed: Master seed. Controls CV fold assignment AND
            torch / numpy RNG inside training, so the same config
            produces the same numbers on a given machine.

        study_data_dir: Directory produced by Stage-2-lite
            (contains `data_mrna_seq_v2_rsem.txt` etc.).

        modalities: For forward-compat with multimodal experiments.
            Only ("RNA_seq",) is supported in the baseline.
        top_k_genes: How many highly-variable genes to keep per fold.

        model: Model architecture name. Only "omics_mlp" for now.
        hidden_dims: Layer sizes for omics_mlp.
        dropout: Dropout probability for omics_mlp.
        use_batchnorm: Whether omics_mlp uses BatchNorm.

        loss: Loss function name. Only "cox_ph" for now.
        optimizer: "adam" or "sgd".
        lr: Learning rate.
        weight_decay: L2 coefficient.
        batch_size: Batch size. `None` means *full-batch* training —
            the entire training fold becomes the risk set at every
            gradient step. Full-batch is the theoretically correct
            setting for Cox PH partial likelihood (the risk set in the
            population loss is every at-risk patient), and TCGA cohorts
            are small enough (<~1200 patients) that it fits on CPU. For
            larger cohorts or when mini-batching is desired, set an
            explicit int. 128 was the old default and is known to work.
        max_grad_norm: Global gradient norm clipping threshold. Cox PH
            loss can produce large gradients when one patient dominates
            the risk set (common in small cohorts with extreme outliers).
            Clipping to norm 1.0 prevents NaN explosions without hurting
            convergence. ``None`` disables clipping.
        epochs: Max epochs. Early stopping may cut short.
        early_stopping_patience: Epochs of no validation improvement
            before stopping. 0 disables.
        val_fraction: Fraction of the training fold to hold out for early
            stopping. 0.1 balances signal and training data.

        n_folds: CV folds. Defaults to 5 (ADR 0004).
    """

    # --- identity ---
    name: str
    cohort: str = "TCGA-BRCA"
    seed: int = 20260413

    # --- data ---
    study_data_dir: str = "stage2_data/raw/brca_tcga_pan_can_atlas_2018"
    modalities: tuple[str, ...] = ("RNA_seq",)
    top_k_genes: int = 500

    # --- model ---
    model: str = "omics_mlp"
    hidden_dims: tuple[int, ...] = (128, 32)
    dropout: float = 0.3
    use_batchnorm: bool = True

    # --- training ---
    loss: str = "cox_ph"
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int | None = None
    max_grad_norm: float | None = 1.0  # gradient clipping; None disables
    epochs: int = 50
    early_stopping_patience: int = 10
    val_fraction: float = 0.1

    # --- evaluation ---
    n_folds: int = 5

    # --- pathway-informed model options ---
    # Used by the `pathway_mlp` model. Ignored by flat MLP and linear Cox.
    # `pathway_db` selects the MSigDB gene-set collection ("hallmark" or
    # "c2_kegg"). `pathway_cache_dir` is the local cache for GMT files;
    # None defaults to ~/.pathogems/gene_sets/.
    pathway_db: str = "hallmark"
    pathway_cache_dir: str | None = None

    # --- attention model options ---
    # Used by the `gene_attention` model. Ignored by other architectures.
    # `attn_d_model` must be divisible by `attn_n_heads`.
    attn_d_model: int = 64
    attn_n_heads: int = 4
    attn_n_layers: int = 2

    # --- experiment tracking (ADR 0008) ---
    # MLflow is optional: leaving `enable_mlflow=False` keeps the harness
    # fully self-contained (JSON run log is still the source of truth for
    # Stage 4). When enabled, every run writes params, per-fold metrics,
    # summary metrics, and the run-log JSON as an artifact to the
    # configured tracking URI.
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None  # None -> MLflow default (./mlruns)
    mlflow_experiment_name: str = "pathogems"

    # --- optional free-form notes for the run log ---
    notes: str | None = None

    # A version field is cheap and protects against silent drift if we
    # ever rename a field. Bump when we make a backwards-incompatible
    # change and update the loader accordingly.
    config_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Tuples become lists; dataclasses flatten to dicts."""
        d = asdict(self)
        # JSON has no tuple; convert so round-trip is stable.
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        d = dict(d)
        # Coerce JSON lists back to tuples for fields that expect them.
        if "modalities" in d and isinstance(d["modalities"], list):
            d["modalities"] = tuple(d["modalities"])
        if "hidden_dims" in d and isinstance(d["hidden_dims"], list):
            d["hidden_dims"] = tuple(d["hidden_dims"])
        if d.get("config_version", 1) != 1:
            raise ValueError(
                f"Unknown config_version={d.get('config_version')}; "
                "this code only handles version 1. Upgrade explicitly."
            )
        # Drop unknown keys loudly — silent ignore is a classic "why didn't
        # my hyperparam change have any effect?" source of pain. `cls` is
        # always a dataclass here (enforced by the `@dataclass` decorator),
        # so `__dataclass_fields__` is guaranteed to exist.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(path.read_text()))

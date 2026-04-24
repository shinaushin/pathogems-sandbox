"""Model architectures for Stage 3.

Currently three models:
  - `OmicsMLP`        — baseline feed-forward MLP (ADR 0001)
  - `PathwayMLP`      — pathway-informed sparse first layer (P-NET style)
  - `GeneAttentionNet`— Transformer encoder treating each gene as a token

Subsequent experiments add models alongside existing ones (never replace),
so every architecture remains comparable on the same harness.

The ModelFactory callable signature is:
    factory(in_features: int, config: ExperimentConfig, selected_genes: list[str]) -> nn.Module

`selected_genes` is the ordered list of gene symbols chosen by the
preprocessor. Most factories ignore it; `PathwayMLP` uses it to build the
gene-to-pathway connectivity mask via `pathways.build_connectivity`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .registry import Registry

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # avoid a runtime import cycle (config has no torch deps anyway)
    from .config import ExperimentConfig

# Registry maps a model name (as used in `ExperimentConfig.model`) to a
# *factory* callable:
#     factory(in_features: int, config: ExperimentConfig, selected_genes: list[str]) -> nn.Module
#
# `in_features` is the runtime gene count (depends on top_k_genes and the
# preprocessor).  `selected_genes` is the ordered list of HGNC gene symbols
# — needed by pathway-aware models to build the connectivity mask; flat MLP
# factories accept and ignore it.  Keeping the signature uniform lets
# `train.py` call MODEL_REGISTRY.get(name) and forget which model it builds.
ModelFactory = Callable[[int, "ExperimentConfig", list[str]], nn.Module]
MODEL_REGISTRY: Registry[ModelFactory] = Registry("model")


@dataclass(frozen=True, slots=True)
class OmicsMLPConfig:
    """Configuration for `OmicsMLP`.

    Attributes:
        in_features: Number of input genes (== `top_k` from the
            preprocessor). Must match the data at runtime; a mismatch
            raises at construction time rather than silently producing
            garbage risk scores.
        hidden_dims: Sizes of the hidden layers, in order. Default
            (128, 32) follows the baseline spec from ADR 0001.
        dropout: Dropout probability applied after each hidden layer.
            0.3 is the DeepSurv default and works reasonably on TCGA-BRCA
            in pilot runs.
        use_batchnorm: BatchNorm1d after each linear layer. Useful because
            z-scored inputs + ReLU can still have per-layer distribution
            shift as training progresses.
    """

    in_features: int
    hidden_dims: tuple[int, ...] = (128, 32)
    dropout: float = 0.3
    use_batchnorm: bool = True

    def __post_init__(self) -> None:
        if self.in_features <= 0:
            raise ValueError("in_features must be positive.")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be non-empty.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")


class OmicsMLP(nn.Module):
    """Feed-forward MLP that maps an omics feature vector to a scalar risk.

    The network outputs a single scalar per patient — the Cox linear
    predictor `r = f(x)`. *We deliberately do not apply a final sigmoid
    or softplus*: the Cox partial likelihood is invariant to monotone
    transforms of `r`, so bounding it adds no information and makes
    gradients vanish for large |r|. Downstream code assumes unbounded
    scalar outputs.

    Shape contract:
        input:  (batch, in_features), float32
        output: (batch,),            float32  (risk scores)
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
            layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        # Weight init: Kaiming for ReLU stacks is the standard choice and
        # avoids the "all zeros" trap that makes Cox training stall.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return risk scores of shape `(batch,)`."""
        if x.dim() != 2 or x.shape[1] != self.config.in_features:
            raise ValueError(
                f"Expected input of shape (batch, {self.config.in_features}), got {tuple(x.shape)}."
            )
        # BatchNorm1d requires batch size > 1 in train mode. The trainer
        # passes batches of reasonable size, but a size-1 edge case is easy
        # to hit in tests; squeeze(-1) tolerates both (N,1) and (1,1).
        return self.net(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("omics_mlp")
def _build_omics_mlp(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],  # not used by flat MLP
) -> nn.Module:
    """Factory: translate the generic `ExperimentConfig` into `OmicsMLPConfig`.

    Living next to `OmicsMLP` (instead of in `train.py`) means the
    responsibility for mapping config fields to constructor args stays with
    the owner of those fields. A second model in this file only has to add
    its own factory, with no change to `train.py`.
    """
    return OmicsMLP(
        OmicsMLPConfig(
            in_features=in_features,
            hidden_dims=tuple(config.hidden_dims),
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
        )
    )


# --------------------------------------------------------------------------- #
# LinearCox — "baseline of the baseline"
# --------------------------------------------------------------------------- #
class LinearCox(nn.Module):
    """Linear Cox predictor: risk = w^T x + b, no non-linearity.

    This is the *simplest possible* neural Cox model and serves as a
    sanity-check reference point: if the MLP can't beat a regularized
    linear Cox fit, either the MLP is mis-tuned or there is no non-linear
    signal to extract in the first place. Classic survival-ML papers
    (DeepSurv and successors) treat linear Cox as the baseline to beat,
    so we want it available on the exact same harness to make the
    comparison meaningful.

    Notes on shape: we implement this as a single `nn.Linear(in, 1)`
    rather than calling it a "logistic regression" — Cox PH's partial
    likelihood is already what this model optimizes end-to-end via
    `loss.cox_ph_loss`, so the terminology stays precise.
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
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input of shape (batch, {self.in_features}), got {tuple(x.shape)}."
            )
        return self.linear(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("linear_cox")
def _build_linear_cox(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],  # not used by linear model
) -> nn.Module:
    """LinearCox has no hyperparameters beyond `in_features`.

    ``hidden_dims``, ``dropout``, and ``use_batchnorm`` are ignored. We
    log a warning when they differ from their defaults so a user who
    accidentally left ``dropout=0.5`` in a linear_cox config knows they're
    not doing what they think they're doing — while still allowing the
    common case of "copy the MLP config, change only `model`" to work
    without requiring field cleanup.
    """
    # Defaults mirror ExperimentConfig's field defaults. We hard-code them
    # here rather than reading class attributes because ExperimentConfig
    # uses `slots=True`, which removes class-level defaults and causes a
    # mypy error if accessed as `ExperimentConfig.hidden_dims`.
    _DEFAULT_HIDDEN_DIMS = (128, 32)
    _DEFAULT_DROPOUT = 0.3
    _DEFAULT_USE_BATCHNORM = True

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


# --------------------------------------------------------------------------- #
# MaskedLinear — sparse linear layer with a fixed binary connectivity mask
# --------------------------------------------------------------------------- #
class MaskedLinear(nn.Module):
    """Linear layer where only pre-specified connections may be non-zero.

    The mask is a fixed binary tensor of shape (out_features, in_features).
    Weights are multiplied by the mask on every forward pass, which:
      - zeroes out disallowed connections in the output, and
      - zeroes the gradient for those weights (∂(w*mask)/∂w = mask), so
        Adam/SGD never move them away from zero regardless of momentum.

    No backward hook needed — the masking in the forward pass propagates
    through autograd correctly.

    Args:
        mask: Float32 binary tensor of shape (out_features, in_features).
              Typically the gene-to-pathway connectivity matrix produced by
              `pathways.build_connectivity`.
    """

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        out_features, in_features = mask.shape
        # Register mask as a buffer so it moves to the right device with .to()
        # and is saved/loaded with state_dict, but is NOT a trainable parameter.
        self.register_buffer("mask", mask.float())
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        # Zero out any non-zero initial weights outside the mask.
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        out_f, in_f = self.mask.shape
        n_active = int(self.mask.sum().item())
        return f"in={in_f}, out={out_f}, active_connections={n_active}/{in_f * out_f}"


# --------------------------------------------------------------------------- #
# PathwayMLP — P-NET style pathway-informed network
# --------------------------------------------------------------------------- #
class PathwayMLP(nn.Module):
    """Pathway-informed MLP with a sparse first layer.

    The first layer is a `MaskedLinear` whose connectivity is defined by
    membership in curated gene-set pathways (e.g. MSigDB Hallmark).  Each
    output unit in this layer represents one pathway and can only receive
    signal from its member genes.  Dense hidden layers follow, matching the
    OmicsMLP architecture.

    Biological motivation: by constraining the first layer to known biology
    we reduce the effective parameter count in the noisiest part of the
    network (the gene → latent mapping), which can improve generalization
    on small cohorts like TCGA-BRCA (~900 patients).  The hidden layers
    after the pathway layer are unconstrained so the model can learn
    non-trivial combinations of pathway activations.

    Shape contract:
        input:  (batch, n_selected_genes), float32
        output: (batch,),                  float32  (Cox risk scores)
    """

    def __init__(self, mask: torch.Tensor, config: OmicsMLPConfig) -> None:
        super().__init__()
        n_pathways = mask.shape[0]

        # Sparse pathway projection: genes → pathways
        self.pathway_layer = MaskedLinear(mask)

        # Dense layers after the pathway projection (same pattern as OmicsMLP)
        layers: list[nn.Module] = []
        if config.use_batchnorm:
            layers.append(nn.BatchNorm1d(n_pathways))
        layers.append(nn.ReLU(inplace=True))
        if config.dropout > 0:
            layers.append(nn.Dropout(p=config.dropout))
        prev = n_pathways
        for h in config.hidden_dims:
            layers.append(nn.Linear(prev, h))
            if config.use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        self.n_genes = mask.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.n_genes:
            raise ValueError(
                f"Expected input of shape (batch, {self.n_genes}), got {tuple(x.shape)}."
            )
        pathway_acts = self.pathway_layer(x)  # (batch, n_pathways)
        return self.head(pathway_acts).squeeze(-1)  # (batch,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("pathway_mlp")
def _build_pathway_mlp(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],
) -> nn.Module:
    """Build a PathwayMLP by downloading/caching MSigDB gene sets and
    constructing the gene-to-pathway connectivity mask.

    The pathway database and cache directory are read from
    `config.pathway_db` and `config.pathway_cache_dir`. These fields
    are optional in ExperimentConfig (defaults to "hallmark" and None,
    which caches in ~/.pathogems/gene_sets/).
    """
    from .pathways import build_connectivity, load_gene_sets

    cache_dir = Path(config.pathway_cache_dir) if config.pathway_cache_dir else None
    gene_sets = load_gene_sets(db=config.pathway_db, cache_dir=cache_dir)
    mask, pathway_names, assigned = build_connectivity(selected_genes, gene_sets)

    log.info(
        "[pathway_mlp] %d pathways x %d genes (%d/%d genes assigned)",
        len(pathway_names),
        in_features,
        len(assigned),
        in_features,
    )

    return PathwayMLP(
        mask=mask,
        config=OmicsMLPConfig(
            in_features=in_features,
            hidden_dims=tuple(config.hidden_dims),
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
        ),
    )


# --------------------------------------------------------------------------- #
# GeneAttentionNet — Transformer encoder over gene tokens
# --------------------------------------------------------------------------- #
class GeneAttentionNet(nn.Module):
    """Transformer-based survival model treating each gene as a sequence token.

    Each gene i is represented as a d_model-dimensional token:
        token_i = value_proj(x_i) + gene_emb_i

    where x_i is the scalar (z-scored log2 FPKM) expression value,
    `value_proj` is a shared linear (1 → d_model) that encodes the
    magnitude of expression, and `gene_emb_i` is a learnable per-gene
    identity embedding that tells the model *which* gene is in position i
    (analogous to positional encoding in NLP but gene-specific).

    The token sequence is passed through a multi-layer TransformerEncoder
    (multi-head self-attention + FFN), and the resulting per-gene
    representations are mean-pooled to a single (batch, d_model) vector.
    A linear head maps this to a scalar Cox risk score.

    Memory note: full self-attention is O(n_genes²) per sample. With
    n_genes=500 and batch_size=32, the attention tensors are ~128 MB —
    feasible on CPU. For n_genes > 1000, use mini-batching
    (set `batch_size` in the config, e.g. 32).

    Shape contract:
        input:  (batch, n_genes), float32
        output: (batch,),         float32  (Cox risk scores)
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model

        # Project each gene's scalar expression value into embedding space.
        self.value_proj = nn.Linear(1, d_model)
        nn.init.kaiming_normal_(self.value_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.value_proj.bias)

        # Learnable gene-identity embeddings (one per gene, not positional).
        # Small init to avoid dominating value_proj early in training.
        self.gene_emb = nn.Parameter(torch.empty(n_genes, d_model))
        nn.init.normal_(self.gene_emb, std=0.02)

        # Multi-layer self-attention encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # (batch, seq, d_model) convention
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scalar Cox risk head.
        self.head = nn.Linear(d_model, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.n_genes:
            raise ValueError(
                f"Expected input of shape (batch, {self.n_genes}), got {tuple(x.shape)}."
            )
        # Build per-gene tokens: (batch, n_genes, d_model)
        tokens = x.unsqueeze(-1)  # (batch, n_genes, 1)
        tokens = self.value_proj(tokens)  # (batch, n_genes, d_model)
        tokens = tokens + self.gene_emb.unsqueeze(0)  # broadcast gene identities

        # Self-attention encoder
        out = self.transformer(tokens)  # (batch, n_genes, d_model)

        # Aggregate: mean-pool across genes → (batch, d_model)
        pooled = out.mean(dim=1)

        return self.head(pooled).squeeze(-1)  # (batch,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("gene_attention")
def _build_gene_attention(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],  # used implicitly via in_features
) -> nn.Module:
    """Build a GeneAttentionNet from `ExperimentConfig`.

    Relevant config fields (all optional, with sensible defaults):
        attn_d_model  (int, default 64)  — token embedding dimension
        attn_n_heads  (int, default 4)   — attention heads (must divide d_model)
        attn_n_layers (int, default 2)   — TransformerEncoder layers
        dropout       (float)            — shared with other architectures

    For large gene sets (> ~500 genes) set `batch_size` in the config to
    avoid OOM on CPU (32 is a good starting point).
    """
    return GeneAttentionNet(
        n_genes=in_features,
        d_model=config.attn_d_model,
        n_heads=config.attn_n_heads,
        n_layers=config.attn_n_layers,
        dropout=config.dropout,
    )

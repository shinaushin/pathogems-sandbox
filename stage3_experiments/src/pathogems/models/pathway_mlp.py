"""PathwayMLP — pathway-informed MLP with a sparse first layer.

The first layer is a MaskedLinear whose connectivity is defined by
membership in curated gene-set pathways (MSigDB Hallmark or C2 KEGG).
Each output unit represents one pathway and can only receive signal from
its member genes.  Dense hidden layers follow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from ._mixin import RegularizableMixin
from ._registry import MODEL_REGISTRY
from .omics_mlp import OmicsMLPConfig

if TYPE_CHECKING:
    from ..config import ExperimentConfig

log = logging.getLogger(__name__)


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
              ``pathways.build_connectivity``.
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
        """Apply the masked linear transformation to ``x``."""
        return F.linear(x, self.weight * self.mask, self.bias)

    def extra_repr(self) -> str:
        """Summary shown by ``print(model)`` — reports active connections."""
        out_f, in_f = self.mask.shape
        n_active = int(self.mask.sum().item())
        return f"in={in_f}, out={out_f}, active_connections={n_active}/{in_f * out_f}"


# --------------------------------------------------------------------------- #
# PathwayMLP — P-NET style pathway-informed network
# --------------------------------------------------------------------------- #
class PathwayMLP(RegularizableMixin, nn.Module):
    """Pathway-informed MLP with a sparse first layer and optional ablation flags.

    The first layer is a ``MaskedLinear`` whose connectivity is defined by
    membership in curated gene-set pathways (e.g. MSigDB Hallmark or C2 KEGG).
    Each output unit represents one pathway and can only receive signal from
    its member genes.  Dense hidden layers follow.

    Four ablation flags extend the base architecture (all default to off):

    ``gene_filter_idx`` (*pathway_only*)
        A 1-D LongTensor of indices into the full ``n_input_genes`` input.
        When set, the forward pass first selects ``x[:, gene_filter_idx]``
        so only pathway-assigned genes reach the sparse layer.

    ``scaled_init`` (*pathway_scaled_init*)
        After Kaiming init, each pathway node's weights are divided by
        ``sqrt(n_member_genes_in_that_pathway)``.  Equalises expected
        pre-activation magnitude across large and small pathways.

    ``residual`` (*pathway_residual*)
        Adds a dense linear projection from the (filtered) gene input
        directly to the pathway-activation space (``y += W_skip @ x``).

    ``norm_type`` (*pathway_norm*)
        Controls the normalisation layer after the sparse projection.
        ``"batch"`` → ``BatchNorm1d``; ``"layer"`` → ``LayerNorm``;
        ``"none"`` → no normalisation.

    Shape contract:
        input:  (batch, n_input_genes), float32
        output: (batch,),               float32  (Cox risk scores)
    """

    # register_buffer sets this to a Tensor in pathway_only mode; mypy needs
    # the class-level annotation to know the attribute is not always None.
    gene_filter_idx: torch.Tensor | None

    def __init__(
        self,
        mask: torch.Tensor,
        config: OmicsMLPConfig,
        *,
        gene_filter_idx: torch.Tensor | None = None,
        n_input_genes: int | None = None,
        scaled_init: bool = False,
        residual: bool = False,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        n_pathways, n_pathway_genes = mask.shape

        # --- Gene filtering (pathway_only mode) ---
        if gene_filter_idx is not None:
            self.register_buffer("gene_filter_idx", gene_filter_idx)
            self.n_genes = n_input_genes if n_input_genes is not None else n_pathway_genes
        else:
            self.gene_filter_idx = None
            self.n_genes = n_pathway_genes

        # --- Sparse pathway projection: genes → pathways ---
        self.pathway_layer = MaskedLinear(mask)

        # --- Pathway-size-proportional weight rescaling ---
        if scaled_init:
            with torch.no_grad():
                n_members = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                scale = 1.0 / n_members.sqrt()
                self.pathway_layer.weight.data.mul_(scale)

        # --- Residual skip: gene input → pathway activation space ---
        self.residual_proj: nn.Linear | None = None
        if residual:
            self.residual_proj = nn.Linear(n_pathway_genes, n_pathways, bias=False)
            nn.init.kaiming_normal_(self.residual_proj.weight, nonlinearity="relu")

        # --- Dense head after pathway activations ---
        layers: list[nn.Module] = []
        if norm_type == "batch":
            layers.append(nn.BatchNorm1d(n_pathways))
        elif norm_type == "layer":
            layers.append(nn.LayerNorm(n_pathways))
        # "none" → no normalisation after pathway layer
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Cox risk scores of shape ``(batch,)``."""
        if x.dim() != 2 or x.shape[1] != self.n_genes:
            raise ValueError(
                f"Expected input of shape (batch, {self.n_genes}), got {tuple(x.shape)}."
            )
        if self.gene_filter_idx is not None:
            x = x[:, self.gene_filter_idx]
        pathway_acts = self.pathway_layer(x)
        if self.residual_proj is not None:
            pathway_acts = pathway_acts + self.residual_proj(x)
        return self.head(pathway_acts).squeeze(-1)

    @property
    def regularized_weight(self) -> torch.Tensor | None:
        """Return the MaskedLinear pathway projection weight for L1 penalty.

        L1 on the sparse pathway layer encourages the model to zero out
        entire gene→pathway connections, producing interpretable sparsity
        within the biologically-constrained first layer rather than in an
        arbitrary dense hidden layer.
        """
        return self.pathway_layer.weight

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("pathway_mlp")
def _build_pathway_mlp(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],
) -> nn.Module:
    """Build a ``PathwayMLP`` from config, applying any ablation flags set.

    Steps:
      1. Load the requested MSigDB gene-set collection.
      2. Build the initial gene-to-pathway connectivity mask.
      3. If ``pathway_only=True``, filter genes to pathway-assigned only,
         store filter indices, and rebuild the mask without UNASSIGNED row.
      4. Resolve the normalisation type from ``pathway_norm``.
      5. Construct and return the ``PathwayMLP``.
    """
    from ..pathways import build_connectivity, load_gene_sets

    pc = config.pathway
    cache_dir = Path(pc.cache_dir) if pc.cache_dir else None
    gene_sets = load_gene_sets(db=pc.db, cache_dir=cache_dir)
    mask, pathway_names, assigned = build_connectivity(selected_genes, gene_sets)

    # --- pathway_only: filter to pathway-assigned genes only ---
    gene_filter_idx: torch.Tensor | None = None
    if pc.only:
        assigned_set = set(assigned)
        filter_idx = [i for i, g in enumerate(selected_genes) if g in assigned_set]
        gene_filter_idx = torch.tensor(filter_idx, dtype=torch.long)
        filtered_genes = [selected_genes[i] for i in filter_idx]
        mask, pathway_names, assigned = build_connectivity(filtered_genes, gene_sets)
        log.info(
            "[pathway_mlp] pathway_only: %d → %d genes (%d pathways, 0 unassigned)",
            in_features,
            len(filtered_genes),
            len(pathway_names),
        )

    log.info(
        "[pathway_mlp] %d pathways x %d pathway-genes (%d/%d input genes assigned)",
        len(pathway_names),
        mask.shape[1],
        len(assigned),
        in_features,
    )

    # --- Resolve normalisation type ---
    raw_norm = pc.norm
    if raw_norm not in ("batch", "layer", "none"):
        raise ValueError(f"pathway_norm must be 'batch', 'layer', or 'none'; got {raw_norm!r}")
    norm_type = raw_norm if raw_norm != "batch" else ("batch" if config.use_batchnorm else "none")

    return PathwayMLP(
        mask=mask,
        config=OmicsMLPConfig(
            in_features=in_features,
            hidden_dims=tuple(config.hidden_dims),
            dropout=config.dropout,
            use_batchnorm=config.use_batchnorm,
        ),
        gene_filter_idx=gene_filter_idx,
        n_input_genes=in_features,
        scaled_init=pc.scaled_init,
        residual=pc.residual,
        norm_type=norm_type,
    )

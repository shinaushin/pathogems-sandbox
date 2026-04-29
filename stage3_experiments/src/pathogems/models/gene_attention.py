"""GeneAttentionNet — Transformer encoder treating each gene as a token.

Each gene's scalar expression value is projected into a d_model-dimensional
embedding space, augmented with a learnable gene-identity embedding, and
processed by multi-layer self-attention.  The resulting per-gene
representations are mean-pooled to produce a single Cox risk score.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from ._mixin import RegularizableMixin
from ._registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from ..config import ExperimentConfig

log = logging.getLogger(__name__)


class GeneAttentionNet(RegularizableMixin, nn.Module):
    """Transformer-based survival model treating each gene as a sequence token.

    Each gene i is represented as a d_model-dimensional token:
        token_i = value_proj(x_i) + gene_emb_i

    where x_i is the scalar (z-scored log₂ RSEM) expression value,
    ``value_proj`` is a shared linear (1 → d_model) encoding expression
    magnitude, and ``gene_emb_i`` is a learnable per-gene identity embedding
    (analogous to positional encoding in NLP but gene-specific).

    The token sequence is passed through a multi-layer TransformerEncoder
    (multi-head self-attention + FFN), then mean-pooled to a (batch, d_model)
    vector.  A linear head maps this to a scalar Cox risk score.

    Memory note: full self-attention is O(n_genes²) per sample.  With
    n_genes=500 and batch_size=32, attention tensors are ~128 MB — feasible
    on CPU.  For n_genes > 1000 set ``batch_size`` in the config (e.g. 32).

    Shape contract:
        input:  (batch, n_genes), float32
        output: (batch,),         float32  (Cox risk scores)

    ``regularized_weight`` returns ``None`` (inherited from
    ``RegularizableMixin``): L1 regularization on attention weight matrices
    is not a standard technique and is omitted.
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
        """Return Cox risk scores of shape ``(batch,)``."""
        if x.dim() != 2 or x.shape[1] != self.n_genes:
            raise ValueError(
                f"Expected input of shape (batch, {self.n_genes}), got {tuple(x.shape)}."
            )
        # Build per-gene tokens: (batch, n_genes, d_model)
        tokens = x.unsqueeze(-1)  # (batch, n_genes, 1)
        tokens = self.value_proj(tokens)  # (batch, n_genes, d_model)
        tokens = tokens + self.gene_emb.unsqueeze(0)  # broadcast gene identities

        out = self.transformer(tokens)  # (batch, n_genes, d_model)
        pooled = out.mean(dim=1)  # (batch, d_model)
        return self.head(pooled).squeeze(-1)  # (batch,)

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@MODEL_REGISTRY.register("gene_attention")
def _build_gene_attention(
    in_features: int,
    config: ExperimentConfig,
    selected_genes: list[str],  # used implicitly via in_features
) -> nn.Module:
    """Build a ``GeneAttentionNet`` from ``ExperimentConfig``.

    Relevant config fields (all optional, with sensible defaults):
        attn_d_model  (int, default 64)  — token embedding dimension
        attn_n_heads  (int, default 4)   — attention heads (must divide d_model)
        attn_n_layers (int, default 2)   — TransformerEncoder layers
        dropout       (float)            — shared with other architectures

    For large gene sets (> ~500 genes) set ``batch_size`` in the config to
    avoid OOM on CPU (32 is a good starting point).
    """
    ac = config.attn
    return GeneAttentionNet(
        n_genes=in_features,
        d_model=ac.d_model,
        n_heads=ac.n_heads,
        n_layers=ac.n_layers,
        dropout=config.dropout,
    )

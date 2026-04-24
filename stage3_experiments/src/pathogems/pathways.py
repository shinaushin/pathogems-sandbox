"""Gene-set loading and gene-to-pathway connectivity for pathway-informed models.

Provides two public functions:

    load_gene_sets(db, cache_dir) -> dict[str, list[str]]
        Downloads (or reads from cache) an MSigDB gene-set collection and
        returns a mapping {pathway_name: [HGNC_gene_symbol, ...]}.

    build_connectivity(selected_genes, gene_sets)
        -> (mask_tensor, pathway_names, assigned_genes)
        Given the list of genes selected by the preprocessor, returns a
        binary connectivity matrix of shape (n_pathways, n_selected_genes)
        where entry [j, i] = 1 means gene i belongs to pathway j.
        Genes that belong to no pathway in the collection are grouped into
        a synthetic "UNASSIGNED" pathway so no information is discarded.

Supported collections (db argument):
    "hallmark"  — MSigDB H: 50 curated hallmark gene sets, ~4 000 unique
                  human genes.  Well-maintained, minimal redundancy, ideal
                  for a first pathway experiment.
    "c2_kegg"   — MSigDB C2 KEGG: ~186 canonical pathway gene sets.
                  More granular than Hallmarks but noisier.

Gene-set files are cached as plain text (GMT format) in *cache_dir*
(default: ~/.pathogems/gene_sets/).  Re-running is instant once cached.
The download step requires internet access; if all URLs fail, a clear
error with manual-download instructions is raised.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GMT download URLs — try in order, fall back on HTTP errors.
# Source: Broad Institute public MSigDB release server.
# ---------------------------------------------------------------------------
_GMT_URLS: dict[str, list[str]] = {
    "hallmark": [
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/h.all.v7.5.1.symbols.gmt",
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.1.Hs/h.all.v2023.1.Hs.symbols.gmt",
    ],
    "c2_kegg": [
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/7.5.1/c2.cp.kegg.v7.5.1.symbols.gmt",
        "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.1.Hs/c2.cp.kegg_legacy.v2023.1.Hs.symbols.gmt",
    ],
}

_DEFAULT_CACHE = Path.home() / ".pathogems" / "gene_sets"
_TIMEOUT = 60  # seconds


def load_gene_sets(
    db: str = "hallmark",
    cache_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Return {pathway_name: [gene_symbol, ...]} for the requested collection.

    Downloads the GMT file on first call and caches it locally.  Subsequent
    calls read from cache (no network required).

    Args:
        db:        Gene-set collection name.  Currently "hallmark" or "c2_kegg".
        cache_dir: Directory for cached GMT files.  Defaults to
                   ~/.pathogems/gene_sets/.

    Returns:
        Dict mapping pathway name → list of HGNC gene symbols.

    Raises:
        ValueError:  Unknown *db* name.
        RuntimeError: All download URLs failed and no cache exists.
    """
    if db not in _GMT_URLS:
        raise ValueError(f"Unknown gene-set collection '{db}'. " f"Supported: {sorted(_GMT_URLS)}.")

    cache_root = cache_dir or _DEFAULT_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{db}.gmt"

    if not cache_path.exists():
        _download_gmt(db, cache_path)

    return _parse_gmt(cache_path)


def _download_gmt(db: str, dest: Path) -> None:
    """Try each URL in order; write raw GMT text to *dest*."""
    urls = _GMT_URLS[db]
    last_exc: Exception | None = None
    for url in urls:
        try:
            log.info("[pathways] Downloading %s gene sets from %s …", db, url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                dest.write_bytes(resp.read())
            log.info("[pathways] Saved to %s", dest)
            return
        except Exception as exc:
            log.warning("[pathways] %s failed: %s", url, exc)
            last_exc = exc

    raise RuntimeError(
        f"All download URLs for '{db}' gene sets failed. "
        f"Download the GMT file manually and place it at {dest}.\n"
        f"Hallmark: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp\n"
        f"Last error: {last_exc}"
    ) from last_exc


def _parse_gmt(path: Path) -> dict[str, list[str]]:
    """Parse a GMT file into {pathway: [genes]}.

    GMT format (tab-separated):
        PATHWAY_NAME<TAB>description_or_URL<TAB>GENE1<TAB>GENE2<TAB>...
    """
    gene_sets: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        name = parts[0]
        genes = [g.strip() for g in parts[2:] if g.strip()]
        if genes:
            gene_sets[name] = genes
    log.info(
        "[pathways] Loaded %d gene sets covering %d unique genes",
        len(gene_sets),
        len({g for gs in gene_sets.values() for g in gs}),
    )
    return gene_sets


def build_connectivity(
    selected_genes: list[str],
    gene_sets: dict[str, list[str]],
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Build a binary gene-to-pathway connectivity mask.

    Args:
        selected_genes: Ordered list of gene symbols chosen by the
            preprocessor (length == n_selected_genes).
        gene_sets: Mapping {pathway_name: [gene_symbol, ...]}.

    Returns:
        mask:          Float tensor of shape (n_pathways, n_genes).
                       mask[j, i] = 1.0  iff  selected_genes[i] ∈ pathway j.
        pathway_names: List of pathway names in row order.  The last entry
                       is "UNASSIGNED" if any genes fall outside all pathways.
        assigned:      The subset of selected_genes that appear in at least
                       one pathway (for diagnostic logging).
    """
    gene_index = {g: i for i, g in enumerate(selected_genes)}
    gene_set_index = sorted(gene_sets)  # deterministic ordering

    # Binary membership: shape (n_pathways, n_selected_genes)
    n_pathways = len(gene_set_index)
    n_genes = len(selected_genes)
    mask = torch.zeros(n_pathways, n_genes, dtype=torch.float32)

    assigned: set[str] = set()
    for j, pathway in enumerate(gene_set_index):
        for gene in gene_sets[pathway]:
            if gene in gene_index:
                mask[j, gene_index[gene]] = 1.0
                assigned.add(gene)

    pathway_names = list(gene_set_index)

    # Genes in no pathway → synthetic UNASSIGNED catch-all so no input is lost.
    unassigned_idx = [i for i, g in enumerate(selected_genes) if g not in assigned]
    if unassigned_idx:
        unassigned_row = torch.zeros(1, n_genes, dtype=torch.float32)
        for i in unassigned_idx:
            unassigned_row[0, i] = 1.0
        mask = torch.cat([mask, unassigned_row], dim=0)
        pathway_names.append("UNASSIGNED")

    log.info(
        "[pathways] Connectivity: %d selected genes × %d pathways "
        "(%d assigned, %d unassigned → UNASSIGNED pathway)",
        n_genes,
        len(pathway_names),
        len(assigned),
        len(unassigned_idx),
    )
    return mask, pathway_names, sorted(assigned)

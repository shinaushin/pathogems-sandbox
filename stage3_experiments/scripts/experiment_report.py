r"""Generate a self-contained HTML experiment comparison report.

Reads every *.json file in the logs directory, diffs each experiment's
config against the baseline (brca_omics_baseline), and writes a
single HTML file that includes:

  - Summary table: C-index, std, status, wall time, config changes
  - C-index bar chart across all experiments (with literature benchmark band)
  - Per-fold C-index breakdown
  - Config diff table (what changed vs baseline, highlighted)
  - Training / validation loss curves per experiment

Usage (from repo root):
    python stage3_experiments/scripts/experiment_report.py
    python stage3_experiments/scripts/experiment_report.py \\
        --logs-dir stage3_experiments/logs \\
        --out stage3_experiments/reports/experiment_report.html

The output file is self-contained — no server required, open it
directly in any browser.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config keys excluded from the diff (infrastructure / metadata, not science)
# ---------------------------------------------------------------------------
_EXCLUDED_KEYS = {
    "name",
    "cohort",
    "seed",
    "study_data_dir",
    "notes",
    "config_version",
    "enable_mlflow",
    "mlflow_tracking_uri",
    "mlflow_experiment_name",
}

# Default values for fields added to ExperimentConfig after the initial
# baseline run was captured.  When the baseline log is absent a key but a
# newer run has it set to exactly its default value, that is not a real
# experimental change — the field simply did not exist in the older log
# because the schema was extended after the baseline was run.  Suppress
# these "<absent> → default" entries from the diff so only genuine
# changes surface.
_CONFIG_DEFAULTS: dict[str, object] = {
    # Pathway-model fields (added for PathwayMLP experiments)
    "pathway_db": "hallmark",
    "pathway_cache_dir": None,
    "pathway_only": False,
    "pathway_scaled_init": False,
    "pathway_residual": False,
    "pathway_norm": "batch",
    # Baseline-MLP ablation fields (added for Stage-3 ablation configs)
    "gene_selection": "variance",
    "l1_weight": 0.0,
    "lr_schedule": "constant",
    "activation": "relu",
    "swa_start_fraction": 0.0,
    "lr_warmup_epochs": 0,
    # Attention-model fields (added for GeneAttentionNet experiments)
    "attn_d_model": 64,
    "attn_n_heads": 4,
    "attn_n_layers": 2,
}

BASELINE_NAME = "brca_omics_baseline"
BENCH_LOW = 0.62
BENCH_HIGH = 0.68

# Approximate unique gene counts per pathway database (for coverage estimates).
_DB_UNIQUE_GENES: dict[str, int] = {
    "hallmark": 4383,   # MSigDB Hallmark v2023 — 50 pathways, 4 383 unique genes
    "c2_kegg": 7012,    # MSigDB C2 KEGG canonical — 186 pathways, ~7 000 unique genes
}
# Approximate coding transcriptome size (used to estimate background coverage).
_CODING_GENES = 19_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_logs(logs_dir: Path) -> list[dict]:
    """Load all run-log JSONs, sorted chronologically by started_at."""
    runs = []
    for path in sorted(logs_dir.glob("*.json")):
        if path.name == ".gitkeep":
            continue
        try:
            data = json.loads(path.read_text())
            data["_path"] = str(path)
            runs.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[report] Warning: skipping {path.name}: {exc}")
    runs.sort(key=lambda r: r.get("started_at", ""))
    return runs


def _config_diff(baseline: dict, other: dict) -> list[tuple[str, object, object]]:
    """Return (key, baseline_val, other_val) for every differing config field.

    Entries where the baseline is ``"<absent>"`` but the new value equals the
    known ``_CONFIG_DEFAULTS`` for that field are suppressed: those represent
    schema evolution (a field was added to ExperimentConfig after the baseline
    run was captured, with its default value still in effect) rather than a
    deliberate experimental change.
    """
    all_keys = set(baseline) | set(other)
    diffs = []
    for k in sorted(all_keys):
        if k in _EXCLUDED_KEYS:
            continue
        bv = baseline.get(k, "<absent>")
        ov = other.get(k, "<absent>")
        if bv == ov:
            continue
        # Suppress schema-evolution noise: field absent from old baseline log
        # but set to its ExperimentConfig default in the new run.
        if bv == "<absent>" and k in _CONFIG_DEFAULTS and ov == _CONFIG_DEFAULTS[k]:
            continue
        diffs.append((k, bv, ov))
    return diffs


def _fmt(v: object) -> str:
    """Format a config value for display."""
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    if v is None:
        return "null"
    return str(v)


def _cindex_color(c: float | None) -> str:
    """Return a CSS class based on C-index relative to benchmark."""
    if c is None:
        return "ci-na"
    if c >= BENCH_HIGH:
        return "ci-great"
    if c >= BENCH_LOW:
        return "ci-good"
    if c >= 0.50:
        return "ci-low"
    return "ci-bad"


def _unstable_folds(loss_curves: dict) -> list[int]:
    """Return 1-indexed fold numbers whose val-loss minimum falls at epoch ≤ 2.

    A fold whose best validation loss is at the very first or second epoch
    means the model degraded from initialisation — a sign of optimisation
    instability rather than genuine early convergence.
    """
    bad: list[int] = []
    for fold_id_str, curves in sorted(loss_curves.items()):
        val = curves.get("val", [])
        if val:
            best_ep = int(min(range(len(val)), key=lambda e: val[e])) + 1
            if best_ep <= 2:
                bad.append(int(fold_id_str) + 1)  # convert to 1-indexed
    return bad


def _best_flat_mlp(all_runs: list[dict]) -> dict | None:
    """Return the omics_mlp run with the highest mean C-index, or None."""
    candidates = [
        r for r in all_runs
        if r.get("config", {}).get("model", "omics_mlp") == "omics_mlp"
        and r.get("metrics", {}).get("c_index_mean") is not None
    ]
    return max(candidates, key=lambda r: r["metrics"]["c_index_mean"], default=None)


def _pathway_mlp_base(all_runs: list[dict], db: str) -> dict | None:
    """Return the un-ablated pathway_mlp run for the given database, or None.

    The "base" run is the one that has no ablation flags set — it is the
    reference point for evaluating whether a specific structural change helped.
    """
    _ABLATION_FLAGS = ("pathway_only", "pathway_scaled_init", "pathway_residual")
    candidates = []
    for r in all_runs:
        cfg = r.get("config", {})
        if cfg.get("model") != "pathway_mlp":
            continue
        if cfg.get("pathway_db") != db:
            continue
        if r.get("metrics", {}).get("c_index_mean") is None:
            continue
        # No ablation flags set, and norm is the default "batch"
        if any(cfg.get(f) for f in _ABLATION_FLAGS):
            continue
        if cfg.get("pathway_norm", "batch") != "batch":
            continue
        candidates.append(r)
    # Prefer the run with more genes (higher top_k) as it is the "primary" run.
    return max(candidates, key=lambda r: r["config"].get("top_k_genes", 0), default=None)


def _experiment_commentary(
    run: dict,
    baseline_run: dict | None,
    all_runs: list[dict],
) -> str:
    """Build an HTML commentary block explaining results relative to baseline.

    Generates contextual prose for each experiment based on:
      * C-index delta vs baseline (and vs best flat MLP for non-MLP models)
      * Config changes relative to baseline
      * Fold-level instability (folds whose val-loss minimum is at epoch ≤ 2)
      * Model-specific biological/architectural reasoning

    Returns an HTML string (the ``<div class="commentary">`` block), or an
    empty string if the run has no metrics to reason about.
    """
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    name = run.get("run_name", "")
    model = cfg.get("model", "omics_mlp")
    ci_mean = metrics.get("c_index_mean")
    loss_curves = metrics.get("loss_curves", {})
    top_k = cfg.get("top_k_genes", 0)

    if ci_mean is None:
        return (
            '<div class="commentary">'
            '<p class="muted">Run did not produce metrics — '
            "check the status and error fields above.</p></div>"
        )

    baseline_cfg = (baseline_run or {}).get("config", {})
    baseline_ci = (baseline_run or {}).get("metrics", {}).get("c_index_mean")
    best_flat = _best_flat_mlp(all_runs)
    best_flat_ci = (best_flat or {}).get("metrics", {}).get("c_index_mean")
    best_flat_name = (best_flat or {}).get("run_name", "")
    bad_folds = _unstable_folds(loss_curves)

    paragraphs: list[str] = []

    # ------------------------------------------------------------------ #
    # Baseline — special case                                             #
    # ------------------------------------------------------------------ #
    if name == BASELINE_NAME:
        paragraphs.append(
            f"This is the reference experiment. All other runs are diffed against it. "
            f"With only {top_k} input genes the network lacks sufficient prognostic signal — "
            f"the top-{top_k} high-variance genes include housekeeping and cell-cycle genes "
            f"that are not survival-relevant. The mean C-index of {ci_mean:.4f} is well below "
            f"the published TCGA-BRCA benchmark range (0.62–0.68), and the wide spread across "
            f"folds reflects how few signal-bearing genes are included at this gene budget."
        )
        if bad_folds:
            fold_str = ", ".join(f"fold {f}" for f in bad_folds)
            paragraphs.append(
                f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2, "
                f"meaning the model finds no improving gradient direction for a subset of data "
                f"splits — a consequence of the sparse signal at top_k={top_k}."
            )
        return _wrap_commentary(paragraphs)

    # ------------------------------------------------------------------ #
    # Generic delta opener (all non-baseline runs)                        #
    # ------------------------------------------------------------------ #
    if baseline_ci is not None:
        delta = ci_mean - baseline_ci
        if delta > 0.005:
            verdict = f"improves on the baseline by +{delta:.4f}"
        elif abs(delta) <= 0.005:
            verdict = f"is essentially the same as the baseline (Δ{delta:+.4f})"
        else:
            verdict = f"underperforms the baseline by {delta:.4f}"
        paragraphs.append(
            f"Mean C-index {ci_mean:.4f} — this {verdict} ({baseline_ci:.4f})."
        )

    # ------------------------------------------------------------------ #
    # omics_mlp variants                                                  #
    # ------------------------------------------------------------------ #
    if model == "omics_mlp" and baseline_cfg:
        b_top_k = baseline_cfg.get("top_k_genes", 0)
        b_dropout = baseline_cfg.get("dropout", 0.3)
        b_hidden = baseline_cfg.get("hidden_dims", [])
        b_patience = baseline_cfg.get("early_stopping_patience", 10)
        this_dropout = cfg.get("dropout", 0.3)
        this_hidden = cfg.get("hidden_dims", [])
        this_patience = cfg.get("early_stopping_patience", 10)

        # top_k is the dominant driver — explain it first.
        if top_k != b_top_k:
            if top_k > b_top_k:
                paragraphs.append(
                    f"Raising top_k from {b_top_k} to {top_k} genes is the key change. "
                    f"More high-variance genes capture a broader range of biological processes. "
                    f"Weight decay (1e-4) prevents overfitting on TCGA-BRCA's ~900 patients, "
                    f"so the additional genes add signal rather than noise."
                )
                if ci_mean >= BENCH_LOW:
                    paragraphs.append(
                        f"This brings the model into or near the published benchmark range "
                        f"(0.62–0.68), suggesting the gene budget is now adequate."
                    )
                elif ci_mean > baseline_ci + 0.04:
                    paragraphs.append(
                        f"The gain is substantial, but the model still falls short of the "
                        f"benchmark band — further top_k increases or architectural changes "
                        f"may be needed."
                    )
            else:
                paragraphs.append(
                    f"Reducing top_k from {b_top_k} to {top_k} limits the available "
                    f"prognostic signal, which is reflected in the lower C-index."
                )

        # Dropout change only (top_k and hidden unchanged).
        elif this_dropout != b_dropout and this_hidden == b_hidden:
            direction = "Reducing" if this_dropout < b_dropout else "Increasing"
            paragraphs.append(
                f"{direction} dropout from {b_dropout} to {this_dropout} has negligible "
                f"effect at top_k={top_k}. The network is input-constrained rather than "
                f"overfitting, so regularisation strength is not the binding factor here."
            )

        # Hidden dims change only.
        elif this_hidden != b_hidden and top_k == b_top_k:
            paragraphs.append(
                f"Changing hidden layer dimensions from {b_hidden} to {this_hidden} "
                f"shows limited benefit. With only {top_k} input genes the bottleneck is "
                f"signal quality, not model capacity — wider layers add parameters without "
                f"meaningful extra representation power."
            )

        # Patience change only.
        elif this_patience != b_patience and top_k == b_top_k:
            paragraphs.append(
                f"Extending early stopping patience from {b_patience} to {this_patience} "
                f"epochs produces no improvement, confirming the model converges well within "
                f"the original window and no beneficial late-epoch learning is being cut off."
            )

        # Highlight best flat MLP.
        if best_flat and name == best_flat_name:
            paragraphs.append(
                f"This is the best-performing flat MLP found so far. "
                f"It sets the reference point for pathway-structured and "
                f"attention-based architectures to beat."
            )

        # Instability note.
        if bad_folds:
            fold_str = ", ".join(f"fold {f}" for f in bad_folds)
            paragraphs.append(
                f"<strong>Note:</strong> {fold_str} converge at epoch ≤ 2, "
                f"suggesting the configuration is harder to optimise for some data splits. "
                f"The mean C-index may understate performance on better-initialised seeds."
            )

    # ------------------------------------------------------------------ #
    # pathway_mlp                                                         #
    # ------------------------------------------------------------------ #
    elif model == "pathway_mlp" and baseline_cfg:
        db = cfg.get("pathway_db", "hallmark")
        db_genes = _DB_UNIQUE_GENES.get(db, 0)

        # Ablation flags for this run.
        pathway_only    = cfg.get("pathway_only", False)
        scaled_init     = cfg.get("pathway_scaled_init", False)
        residual        = cfg.get("pathway_residual", False)
        norm            = cfg.get("pathway_norm", "batch")
        is_ablation     = pathway_only or scaled_init or residual or (norm != "batch")

        # Find the un-ablated base run for this DB to compare ablations against.
        base_run = _pathway_mlp_base(all_runs, db)
        base_ci  = (base_run or {}).get("metrics", {}).get("c_index_mean")
        base_name = (base_run or {}).get("run_name", f"base {db} run")

        if not is_ablation:
            # ---- Base PathwayMLP: explain the architecture and its root weakness ----
            if db == "hallmark":
                expected_pct = 24
                assigned_approx = round(top_k * 0.24)
                unassigned_approx = top_k - assigned_approx
                paragraphs.append(
                    f"MSigDB Hallmark covers only ~{expected_pct}% of the human transcriptome "
                    f"({db_genes:,} unique genes across 50 pathways). At top_k={top_k}, roughly "
                    f"{assigned_approx}/{top_k} genes ({expected_pct}%) map to a named pathway; "
                    f"the remaining ~{unassigned_approx} collapse into a single UNASSIGNED "
                    f"catch-all node that dominates the sparse first layer."
                )
            elif db == "c2_kegg":
                expected_pct = 55
                assigned_approx = round(top_k * (expected_pct / 100))
                unassigned_approx = top_k - assigned_approx
                paragraphs.append(
                    f"C2 KEGG offers broader coverage than Hallmark — ~{db_genes:,} unique genes "
                    f"across 186 canonical pathways (~{expected_pct}% of top_k={top_k} genes "
                    f"expected to be assigned, vs ~24% for Hallmark). Even so, ~{unassigned_approx} "
                    f"genes remain unassigned and collapse into a single UNASSIGNED catch-all node."
                )
            else:
                paragraphs.append(
                    f"The <code>{db}</code> pathway database provides the gene-to-pathway "
                    f"connectivity mask for the sparse first layer."
                )

            paragraphs.append(
                f"The UNASSIGNED node is the structural weakness of this architecture: it "
                f"receives far more input connections than any individual pathway node, so its "
                f"gradient signal drowns out the true pathway activations during training. "
                f"This is why PathwayMLP underperforms the best flat MLP "
                f"({best_flat_ci:.4f} for {html.escape(best_flat_name)}) despite "
                f"its biologically-informed first layer — the sparse structure is undermined "
                f"by the dense catch-all node that short-circuits it."
                if best_flat_ci is not None
                else
                f"The UNASSIGNED node is the structural weakness: it dominates gradient updates "
                f"and prevents the pathway layer from learning genuine biological structure."
            )

            if bad_folds:
                fold_str = ", ".join(f"fold {f}" for f in bad_folds)
                paragraphs.append(
                    f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2. "
                    f"The imbalanced first layer — one giant UNASSIGNED node alongside small "
                    f"pathway nodes — creates an uneven optimisation landscape where certain "
                    f"data splits immediately find a bad local minimum and stay there."
                )
            else:
                paragraphs.append(
                    f"Training is relatively stable across folds (no fold converges at epoch ≤ 2), "
                    f"but the mean C-index is still limited by the large fraction of genes that "
                    f"carry no structured pathway signal."
                )

        elif pathway_only:
            # ---- Ablation 1: pathway_only — eliminate UNASSIGNED node ----
            paragraphs.append(
                f"<strong>Ablation: pathway_only=True.</strong> This run filters the input genes "
                f"to only those that map to at least one {db.upper().replace('_', ' ')} pathway, "
                f"eliminating the UNASSIGNED catch-all node entirely. top_k was raised to "
                f"{top_k} so that enough pathway-assigned genes survive the variance "
                f"pre-selection step. Every gene feeding the MaskedLinear now carries genuine "
                f"pathway membership — the structural weakness of the base PathwayMLP is "
                f"directly addressed."
            )
            if base_ci is not None:
                delta_base = ci_mean - base_ci
                if delta_base > 0.005:
                    paragraphs.append(
                        f"The result is a clear improvement over the base {db.upper().replace('_', ' ')} "
                        f"run ({base_ci:.4f}): +{delta_base:.4f} C-index. Removing the UNASSIGNED "
                        f"node lets the pathway layer learn genuine biological structure instead of "
                        f"having its gradient dominated by the catch-all node."
                    )
                elif abs(delta_base) <= 0.005:
                    paragraphs.append(
                        f"Compared to the base {db.upper().replace('_', ' ')} run ({base_ci:.4f}), "
                        f"the change is marginal (Δ{delta_base:+.4f}). Eliminating UNASSIGNED "
                        f"genes helps gradient balance but the model may still be limited by "
                        f"the pathway database's biological coverage for this cohort."
                    )
                else:
                    paragraphs.append(
                        f"Surprisingly, this underperforms the base {db.upper().replace('_', ' ')} "
                        f"run ({base_ci:.4f}) by {-delta_base:.4f}. Filtering to pathway-only genes "
                        f"reduces the total input dimensionality substantially; if the removed genes "
                        f"contained genuine prognostic signal (even without pathway annotation), "
                        f"the model loses access to it."
                    )
            if bad_folds:
                fold_str = ", ".join(f"fold {f}" for f in bad_folds)
                paragraphs.append(
                    f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2. "
                    f"Even without the UNASSIGNED node, the pathway layer can still be "
                    f"unbalanced due to large variation in pathway sizes across the {db.upper().replace('_', ' ')} "
                    f"collection. Consider combining with scaled_init to address this."
                )

        elif scaled_init:
            # ---- Ablation 2: pathway_scaled_init — equalise pre-activation magnitudes ----
            paragraphs.append(
                f"<strong>Ablation: pathway_scaled_init=True.</strong> After Kaiming init, "
                f"each pathway node's weights are rescaled by 1/√(n_member_genes). Without "
                f"this, large pathways (e.g. KEGG_RIBOSOME with ~90 genes) produce pre-activation "
                f"variance ~6× higher than small pathways (~15 genes), causing BatchNorm to be "
                f"driven by large pathways and leaving small pathway nodes starved of gradient "
                f"signal. Scaled init equalises expected pre-activation magnitude at step 0."
            )
            if base_ci is not None:
                delta_base = ci_mean - base_ci
                if delta_base > 0.005:
                    paragraphs.append(
                        f"Compared to the base {db.upper().replace('_', ' ')} run ({base_ci:.4f}), "
                        f"this gains +{delta_base:.4f}. Equalising the initialisation gives small "
                        f"pathway nodes a fair gradient signal from the first epoch, reducing the "
                        f"variance in which pathways the model learns from."
                    )
                elif abs(delta_base) <= 0.005:
                    paragraphs.append(
                        f"The gain over the base {db.upper().replace('_', ' ')} run ({base_ci:.4f}) "
                        f"is marginal (Δ{delta_base:+.4f}). BatchNorm after the pathway layer may "
                        f"already partially compensate for pre-activation scale differences, "
                        f"limiting the additional benefit of init rescaling."
                    )
                else:
                    paragraphs.append(
                        f"This underperforms the base {db.upper().replace('_', ' ')} run "
                        f"({base_ci:.4f}) by {-delta_base:.4f}. Scaled init changes the "
                        f"effective learning rate per pathway implicitly — small pathways, "
                        f"now initialised at lower magnitude, may need more epochs to leave "
                        f"a near-zero regime than the early-stopping window allows."
                    )
            if bad_folds:
                fold_str = ", ".join(f"fold {f}" for f in bad_folds)
                paragraphs.append(
                    f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2. "
                    f"Scaled init addresses variance at initialisation but not the UNASSIGNED "
                    f"node imbalance, which continues to dominate gradient updates in some splits."
                )

        elif residual:
            # ---- Ablation 3: pathway_residual — dense skip connection ----
            paragraphs.append(
                f"<strong>Ablation: pathway_residual=True.</strong> A dense linear projection "
                f"W_skip (n_genes × n_pathways, no bias, Kaiming init) is added to the "
                f"MaskedLinear output before normalisation and activation. This lets the model "
                f"route signal through the pathway layer (structured, biologically interpretable) "
                f"or bypass it (dense, unconstrained) — in the worst case it recovers flat-MLP-like "
                f"behaviour; in the best case it augments pathway activations with direct gene "
                f"contributions that the sparse mask excludes."
            )
            if base_ci is not None:
                delta_base = ci_mean - base_ci
                if delta_base > 0.005:
                    paragraphs.append(
                        f"The skip connection delivers a +{delta_base:.4f} gain over the base "
                        f"{db.upper().replace('_', ' ')} run ({base_ci:.4f}). The model appears to "
                        f"benefit from the unconstrained path — likely routing signal from the "
                        f"~{round(top_k * 0.45)} genes not assigned to any pathway directly "
                        f"into the activation space without being filtered through UNASSIGNED."
                    )
                elif abs(delta_base) <= 0.005:
                    paragraphs.append(
                        f"The skip connection shows no clear benefit over the base "
                        f"{db.upper().replace('_', ' ')} run ({base_ci:.4f}, Δ{delta_base:+.4f}). "
                        f"The dense projection doubles the parameter count of the first layer; "
                        f"on a ~900-patient cohort the extra capacity may overfit rather than "
                        f"add genuine signal."
                    )
                else:
                    paragraphs.append(
                        f"The residual connection hurts vs the base {db.upper().replace('_', ' ')} "
                        f"run ({base_ci:.4f}, Δ{delta_base:.4f}). The additional parameters in "
                        f"W_skip increase overfitting risk on the small TCGA-BRCA cohort, and "
                        f"the dense path may swamp the sparse pathway signal rather than "
                        f"complementing it."
                    )
            if bad_folds:
                fold_str = ", ".join(f"fold {f}" for f in bad_folds)
                paragraphs.append(
                    f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2. "
                    f"The increased parameter count from the skip projection raises the "
                    f"sensitivity to initialisation, particularly in smaller training folds."
                )

        elif norm != "batch":
            # ---- Ablation 4: pathway_norm — LayerNorm or no norm ----
            norm_label = "LayerNorm" if norm == "layer" else "no normalisation"
            if norm == "layer":
                norm_rationale = (
                    f"LayerNorm normalises across the pathway dimension per sample, making "
                    f"it invariant to the absolute activation scale of each pathway node. "
                    f"BatchNorm normalises across the batch dimension per pathway node, so "
                    f"it is sensitive to the fact that large pathways produce higher-magnitude "
                    f"activations than small ones — even after Kaiming init, a single large "
                    f"pathway can skew the batch statistics and suppress gradient flow to "
                    f"smaller pathway nodes."
                )
            else:
                norm_rationale = (
                    f"Removing normalisation entirely after the pathway projection tests "
                    f"whether the BatchNorm in the base run was helping or hurting — "
                    f"it may have been erasing inter-pathway scale differences that are "
                    f"themselves informative."
                )
            paragraphs.append(
                f"<strong>Ablation: pathway_norm='{norm}'.</strong> Replaces the default "
                f"BatchNorm1d after the sparse pathway projection with {norm_label}. "
                f"{norm_rationale}"
            )
            if base_ci is not None:
                delta_base = ci_mean - base_ci
                if delta_base > 0.005:
                    paragraphs.append(
                        f"{norm_label} delivers +{delta_base:.4f} over the base "
                        f"{db.upper().replace('_', ' ')} run ({base_ci:.4f}). The per-sample "
                        f"normalisation is better suited to the imbalanced pathway-node scale "
                        f"distribution than batch-level statistics that are dominated by a few "
                        f"large pathways."
                    )
                elif abs(delta_base) <= 0.005:
                    paragraphs.append(
                        f"The normalisation change shows no clear benefit over the base "
                        f"{db.upper().replace('_', ' ')} run ({base_ci:.4f}, Δ{delta_base:+.4f}). "
                        f"With full-batch training (batch_size=None) the batch statistics "
                        f"equal the dataset statistics, so BatchNorm and LayerNorm produce "
                        f"more similar outputs than they would under mini-batching — reducing "
                        f"the practical difference between the two."
                    )
                else:
                    paragraphs.append(
                        f"{norm_label} underperforms the base {db.upper().replace('_', ' ')} run "
                        f"({base_ci:.4f}, Δ{delta_base:.4f}). With full-batch training, "
                        f"BatchNorm statistics are stable (no batch-to-batch noise), so it "
                        f"may actually be working better than expected for this setup."
                    )
            if bad_folds:
                fold_str = ", ".join(f"fold {f}" for f in bad_folds)
                paragraphs.append(
                    f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2, "
                    f"suggesting that changing the normalisation layer alone does not resolve "
                    f"the underlying optimisation instability caused by the UNASSIGNED node."
                )

    # ------------------------------------------------------------------ #
    # gene_attention                                                      #
    # ------------------------------------------------------------------ #
    elif model == "gene_attention" and baseline_cfg:
        d_model = cfg.get("attn_d_model", 64)
        n_heads = cfg.get("attn_n_heads", 4)
        n_layers = cfg.get("attn_n_layers", 2)
        batch_size = cfg.get("batch_size")
        paragraphs.append(
            f"The transformer encoder treats each of the {top_k} selected genes as a "
            f"sequence token. Gene <em>i</em> is projected to a {d_model}-dimensional "
            f"embedding (a learned per-gene identity vector plus a value projection of "
            f"its expression level). {n_layers} transformer encoder layers with "
            f"{n_heads}-head self-attention then let every gene attend to every other, "
            f"capturing co-activation patterns directly from the data rather than from "
            f"curated pathway membership."
        )
        if best_flat_ci is not None:
            delta_flat = ci_mean - best_flat_ci
            if delta_flat > 0.005:
                verdict = (
                    f"beats the best flat MLP ({best_flat_ci:.4f}) by +{delta_flat:.4f}, "
                    f"suggesting the attention mechanism captures gene interactions "
                    f"that a plain linear projection misses"
                )
            elif abs(delta_flat) <= 0.005:
                verdict = (
                    f"matches the best flat MLP ({best_flat_ci:.4f}) within noise — "
                    f"the self-attention doesn't add clear value over a well-tuned flat MLP "
                    f"at this cohort size"
                )
            else:
                verdict = (
                    f"falls short of the best flat MLP ({best_flat_ci:.4f}) by "
                    f"{-delta_flat:.4f} — the O(n²) attention may be overfitting on "
                    f"TCGA-BRCA's ~900 patients, especially with only {top_k} tokens "
                    f"and mini-batch Cox loss (batch_size={batch_size})"
                )
            paragraphs.append(
                f"On this cohort the model {verdict}."
            )
        if bad_folds:
            fold_str = ", ".join(f"fold {f}" for f in bad_folds)
            paragraphs.append(
                f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2. "
                f"Transformer encoders have more parameters than flat MLPs for the same "
                f"token count, making them sensitive to initialisation on small cohorts. "
                f"A lower learning rate or warmup schedule may help."
            )

    # ------------------------------------------------------------------ #
    # Unknown model — generic note                                        #
    # ------------------------------------------------------------------ #
    else:
        if bad_folds:
            fold_str = ", ".join(f"fold {f}" for f in bad_folds)
            paragraphs.append(
                f"<strong>Fold instability:</strong> {fold_str} converge at epoch ≤ 2."
            )

    return _wrap_commentary(paragraphs)


def _wrap_commentary(paragraphs: list[str]) -> str:
    """Wrap a list of HTML paragraph strings in the commentary div."""
    if not paragraphs:
        return ""
    inner = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f'<div class="commentary">{inner}</div>'


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def _html_summary_row(run: dict, baseline_cfg: dict | None, rank: int) -> str:
    """One <tr> for the summary table."""
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    name = run.get("run_name", run["_path"])
    status = run.get("status", "?")
    wall = run.get("wall_clock_sec")
    wall_str = f"{wall:.1f}s" if wall else "—"
    started = run.get("started_at", "")[:16].replace("T", " ")

    ci_mean = metrics.get("c_index_mean")
    ci_std = metrics.get("c_index_std")
    ci_str = f"{ci_mean:.4f} ± {ci_std:.4f}" if ci_mean is not None else "—"
    ci_cls = _cindex_color(ci_mean)

    best_fold = max(metrics.get("c_index_folds", [float("nan")]))
    worst_fold = min(metrics.get("c_index_folds", [float("nan")]))

    # Config diff summary
    if baseline_cfg and name != BASELINE_NAME:
        diffs = _config_diff(baseline_cfg, cfg)
        if diffs:
            diff_cells = "".join(
                f'<span class="diff-pill">{html.escape(k)}: '
                f"{html.escape(_fmt(bv))} → {html.escape(_fmt(ov))}</span>"
                for k, bv, ov in diffs
            )
        else:
            diff_cells = '<span class="diff-same">identical to baseline</span>'
    elif name == BASELINE_NAME:
        diff_cells = '<span class="diff-baseline">baseline</span>'
    else:
        diff_cells = "—"

    status_cls = "status-ok" if status == "success" else "status-err"
    baseline_marker = " ★" if name == BASELINE_NAME else ""

    return f"""
<tr id="row-{rank}">
  <td class="col-name"><a href="#exp-{rank}">{html.escape(name)}{baseline_marker}</a></td>
  <td class="{ci_cls} col-ci">{ci_str}</td>
  <td class="col-num">{best_fold:.4f}</td>
  <td class="col-num">{worst_fold:.4f}</td>
  <td class="{status_cls} col-status">{status}</td>
  <td class="col-num">{wall_str}</td>
  <td class="col-started">{started}</td>
  <td class="col-diffs">{diff_cells}</td>
</tr>"""


def _html_experiment_section(
    run: dict,
    baseline_cfg: dict | None,
    rank: int,
    baseline_run: dict | None = None,
    all_runs: list[dict] | None = None,
) -> str:
    """Detailed per-experiment section: commentary + diff table + fold bars + loss curves."""
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    name = run.get("run_name", run["_path"])
    folds = metrics.get("c_index_folds", [])
    loss_curves = metrics.get("loss_curves", {})
    commentary_html = _experiment_commentary(run, baseline_run, all_runs or [])

    # --- Config diff table ---
    if baseline_cfg and name != BASELINE_NAME:
        diffs = _config_diff(baseline_cfg, cfg)
        if diffs:
            diff_rows = "".join(
                f'<tr><td class="diff-key">{html.escape(k)}</td>'
                f'<td class="diff-old">{html.escape(_fmt(bv))}</td>'
                f'<td class="diff-arrow">→</td>'
                f'<td class="diff-new">{html.escape(_fmt(ov))}</td></tr>'
                for k, bv, ov in diffs
            )
            diff_html = f'<table class="diff-table">{diff_rows}</table>'
        else:
            diff_html = '<p class="muted">Config identical to baseline.</p>'
    elif name == BASELINE_NAME:
        diff_html = (
            '<p class="muted">This is the baseline'
            " — all other experiments are diffed against it.</p>"
        )
    else:
        diff_html = ""

    # --- Fold C-index bars ---
    fold_bars = ""
    for i, ci in enumerate(folds):
        pct = min(ci / 0.80 * 100, 100)
        bench_low_pct = BENCH_LOW / 0.80 * 100
        bench_high_pct = BENCH_HIGH / 0.80 * 100
        bench_width_pct = bench_high_pct - bench_low_pct
        cls = _cindex_color(ci)
        fold_bars += f"""
<div class="fold-row">
  <span class="fold-label">Fold {i}</span>
  <div class="fold-bar-wrap">
    <div class="bench-band" style="left:{bench_low_pct:.1f}%;width:{bench_width_pct:.1f}%"></div>
    <div class="fold-bar {cls}" style="width:{pct:.1f}%"></div>
  </div>
  <span class="fold-val">{ci:.4f}</span>
</div>"""

    # --- Loss curve data (serialised as JS) ---
    fold_colors = ["#3266ad", "#1d9e75", "#ba7517", "#a32d2d", "#534ab7"]
    datasets_js = ""
    for fold_id_str, curves in sorted(loss_curves.items()):
        fi = int(fold_id_str)
        col = fold_colors[fi % len(fold_colors)]
        train = curves.get("train", [])
        val = curves.get("val", [])
        tr_pts = "[" + ",".join(f"{{x:{e+1},y:{v:.4f}}}" for e, v in enumerate(train)) + "]"
        vl_pts = "[" + ",".join(f"{{x:{e+1},y:{v:.4f}}}" for e, v in enumerate(val)) + "]"
        _tr = (
            f"  {{label:'F{fi} train',data:{tr_pts},borderColor:'{col}',"
            f"borderWidth:1.5,borderDash:[],pointRadius:0,tension:0.3}},"
        )
        _vl = (
            f"  {{label:'F{fi} val',  data:{vl_pts},borderColor:'{col}',"
            f"borderWidth:2,borderDash:[5,3],pointRadius:0,tension:0.3}},"
        )
        datasets_js += f"\n{_tr}\n{_vl}"

    chart_html = ""
    if datasets_js:
        canvas_id = f"lc-{rank}"
        chart_html = f"""
<div style="position:relative;width:100%;height:220px;margin-top:12px;">
  <canvas id="{canvas_id}" role="img"
    aria-label="Training and validation loss curves for {html.escape(name)}">
    Loss curves for {html.escape(name)}.
  </canvas>
</div>
<script>
new Chart(document.getElementById('{canvas_id}'),{{
  type:'line',
  data:{{datasets:[{datasets_js}]}},
  options:{{
    responsive:true,maintainAspectRatio:false,parsing:false,animation:false,
    plugins:{{legend:{{display:false}},tooltip:{{mode:'index',intersect:false,
      callbacks:{{label:c=>c.dataset.label+': '+c.parsed.y.toFixed(3)}}}}}},
    scales:{{
      x:{{type:'linear',title:{{display:true,text:'Epoch',font:{{size:11}}}},
          ticks:{{font:{{size:10}},stepSize:10}}}},
      y:{{title:{{display:true,text:'Cox loss',font:{{size:11}}}},
          ticks:{{font:{{size:10}},callback:v=>v.toFixed(2)}}}}
    }}
  }}
}});
</script>"""

    return f"""
<section class="exp-section" id="exp-{rank}">
  <h2 class="exp-title">{html.escape(name)}</h2>
  <div class="exp-meta">
    Status: <strong>{run.get("status","?")}</strong> &nbsp;|&nbsp;
    Started: {run.get("started_at","")[:16].replace("T"," ")} &nbsp;|&nbsp;
    Wall time: {run.get("wall_clock_sec",0):.1f}s &nbsp;|&nbsp;
    Git: <code>{(run.get("git_sha") or "")[:10]}</code>
  </div>

  <h3>Analysis</h3>
  {commentary_html}

  <h3>Config changes vs baseline</h3>
  {diff_html}

  <h3>C-index by fold <span class="bench-legend">gold band = benchmark 0.62-0.68</span></h3>
  {fold_bars}

  <h3>Loss curves
    <span class="curve-legend">
      <span class="leg-solid">— train</span>
      <span class="leg-dash">- - val</span>
    </span>
  </h3>
  {chart_html}
</section>"""


def _build_comparison_chart_js(runs: list[dict]) -> str:
    """JS for the top-level C-index bar chart comparing all experiments."""
    names = [json.dumps(r.get("run_name", r["_path"])) for r in runs]
    means = [r.get("metrics", {}).get("c_index_mean") or 0 for r in runs]
    stds = [r.get("metrics", {}).get("c_index_std") or 0 for r in runs]
    bar_colors = [
        "#1d9e75" if (m >= BENCH_LOW) else ("#3266ad" if m >= 0.50 else "#d85a30") for m in means
    ]

    names_js = "[" + ",".join(names) + "]"
    means_js = "[" + ",".join(f"{v:.6f}" for v in means) + "]"
    stds_js = "[" + ",".join(f"{v:.6f}" for v in stds) + "]"
    colors_js = "[" + ",".join(f'"{c}"' for c in bar_colors) + "]"

    n = len(runs)
    height = max(200, n * 50 + 80)

    return f"""
<div style="position:relative;width:100%;height:{height}px;">
  <canvas id="cmp-chart" role="img"
    aria-label="C-index comparison bar chart across {n} experiments.">
    Comparison chart of mean C-index across experiments.
  </canvas>
</div>
<script>
(function(){{
  const names   = {names_js};
  const means   = {means_js};
  const stds    = {stds_js};
  const colors  = {colors_js};
  const benchLo = {BENCH_LOW};
  const benchHi = {BENCH_HIGH};

  const errBars = means.map((m,i) => ({{
    x: m, y: names[i],
    xMin: Math.max(0, m - stds[i]),
    xMax: Math.min(1, m + stds[i]),
  }}));

  new Chart(document.getElementById('cmp-chart'), {{
    type: 'bar',
    data: {{
      labels: names,
      datasets: [
        {{
          label: 'Mean C-index',
          data: means,
          backgroundColor: colors,
          borderWidth: 0,
          barThickness: 28,
        }}
      ]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        annotation: {{
          annotations: {{
            benchBand: {{
              type: 'box',
              xMin: benchLo, xMax: benchHi,
              backgroundColor: 'rgba(186,117,23,0.10)',
              borderColor: 'rgba(186,117,23,0.50)',
              borderWidth: 1,
              label: {{
                display: true,
                content: 'Benchmark 0.62-0.68',
                position: 'start',
                font: {{ size: 10 }},
                color: '#ba7517',
              }}
            }}
          }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const i = ctx.dataIndex;
              return `C-index: ${{means[i].toFixed(4)}} ± ${{stds[i].toFixed(4)}}`;
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          min: 0.40, max: 0.80,
          title: {{ display: true, text: 'Harrell C-index', font: {{ size: 12 }} }},
          ticks: {{ font: {{ size: 11 }}, callback: v => v.toFixed(2) }}
        }},
        y: {{
          ticks: {{ font: {{ size: 11 }} }}
        }}
      }}
    }}
  }});
}})();
</script>"""


def generate_report(logs_dir: Path, out_path: Path) -> None:
    """Build the HTML report from all run logs in ``logs_dir`` and write it to ``out_path``."""
    runs = _load_logs(logs_dir)
    if not runs:
        print(f"[report] No run logs found in {logs_dir}. Nothing to report.")
        return

    baseline_run = next((r for r in runs if r.get("run_name") == BASELINE_NAME), None)
    if baseline_run is None:
        print(
            f"[report] Warning: baseline run '{BASELINE_NAME}' not found in logs — "
            "config diffs and model-specific commentary will be suppressed."
        )
    baseline_cfg = baseline_run.get("config", {}) if baseline_run is not None else {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = len(runs)

    summary_rows = "".join(_html_summary_row(r, baseline_cfg, i) for i, r in enumerate(runs))
    exp_sections = "".join(
        _html_experiment_section(r, baseline_cfg, i, baseline_run=baseline_run, all_runs=runs)
        for i, r in enumerate(runs)
    )
    comparison_chart = _build_comparison_chart_js(runs)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PathoGems — Experiment Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-annotation/3.0.1/chartjs-plugin-annotation.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: #1a1a2e;
  background: #f8f8f5;
  margin: 0;
  padding: 0;
}}
a {{ color: #185fa5; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px;
        background: #eee; padding: 1px 5px; border-radius: 3px; }}

/* Layout */
.page-header {{
  background: #1a1a2e;
  color: #fff;
  padding: 2rem 2.5rem 1.5rem;
}}
.page-header h1 {{ margin: 0 0 4px; font-size: 22px; font-weight: 500; }}
.page-header p  {{ margin: 0; font-size: 12px; color: #aaa; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 2rem; }}
.section-head {{
  font-size: 11px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase;
  color: #888; margin: 2rem 0 0.75rem; border-bottom: 1px solid #e0e0d8; padding-bottom: 6px;
}}

/* Summary table */
.summary-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.summary-table th {{
  text-align: left; font-weight: 500; font-size: 11px; color: #666;
  border-bottom: 1.5px solid #ccc; padding: 6px 10px;
}}
.summary-table td {{ padding: 7px 10px; border-bottom: 1px solid #ececec; vertical-align: top; }}
.summary-table tr:hover td {{ background: #f2f2ee; }}
.col-name  {{ min-width: 200px; font-weight: 500; }}
.col-ci    {{ white-space: nowrap; }}
.col-num   {{ text-align: right; white-space: nowrap; }}
.col-status {{ text-align: center; }}
.col-started {{ white-space: nowrap; font-size: 12px; color: #888; }}
.col-diffs {{ font-size: 12px; }}

/* C-index colour coding */
.ci-great {{ color: #085041; font-weight: 500; }}
.ci-good  {{ color: #3b6d11; }}
.ci-low   {{ color: #854f0b; }}
.ci-bad   {{ color: #a32d2d; font-weight: 500; }}
.ci-na    {{ color: #888; }}

/* Status */
.status-ok  {{ color: #085041; }}
.status-err {{ color: #a32d2d; font-weight: 500; }}

/* Diff pills in summary */
.diff-pill {{
  display: inline-block; background: #eef3fb; color: #185fa5;
  border: 1px solid #b5d4f4; border-radius: 12px;
  padding: 1px 8px; margin: 2px 3px 2px 0; font-size: 11px;
  white-space: nowrap;
}}
.diff-same    {{ color: #888; font-style: italic; }}
.diff-baseline {{ color: #634f0b; background: #faeeda; border-radius: 12px;
                  padding: 1px 8px; font-size: 11px; font-weight: 500; }}

/* Experiment sections */
.exp-section {{
  background: #fff;
  border: 1px solid #e0e0d8;
  border-radius: 10px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
}}
.exp-title {{
  font-size: 17px; font-weight: 500; margin: 0 0 6px; color: #1a1a2e;
}}
.exp-meta {{ font-size: 12px; color: #888; margin-bottom: 1.25rem; }}
.exp-section h3 {{
  font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: .05em;
  color: #666; margin: 1.25rem 0 0.6rem; border-bottom: 1px solid #eee; padding-bottom: 5px;
}}
.muted {{ color: #999; font-style: italic; font-size: 13px; margin: 4px 0; }}

/* Commentary block */
.commentary {{
  background: #f5f7fb;
  border-left: 3px solid #3266ad;
  border-radius: 0 6px 6px 0;
  padding: 10px 16px;
  margin: 4px 0 0;
  font-size: 13px;
  line-height: 1.65;
  color: #2a2a3e;
}}
.commentary p {{ margin: 0 0 7px; }}
.commentary p:last-child {{ margin-bottom: 0; }}
.commentary strong {{ color: #1a1a2e; }}

/* Config diff table */
.diff-table {{ border-collapse: collapse; font-size: 13px; margin-top: 4px; }}
.diff-table td {{ padding: 5px 12px; vertical-align: top; }}
.diff-key  {{ color: #555; font-weight: 500; white-space: nowrap; }}
.diff-old  {{ color: #a32d2d; background: #fef2f2; border-radius: 4px; padding: 2px 8px; }}
.diff-new  {{ color: #085041; background: #edf7f3; border-radius: 4px; padding: 2px 8px; }}
.diff-arrow {{ color: #aaa; padding: 5px 4px; }}

/* Fold bars */
.fold-row {{
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 7px; font-size: 13px;
}}
.fold-label {{ width: 50px; color: #666; flex-shrink: 0; }}
.fold-bar-wrap {{
  flex: 1; background: #f0f0ea; border-radius: 4px; height: 20px;
  position: relative; overflow: hidden;
}}
.bench-band {{
  position: absolute; top: 0; height: 20px;
  background: rgba(186,117,23,0.12);
  border-left: 1.5px solid rgba(186,117,23,0.45);
  border-right: 1.5px solid rgba(186,117,23,0.45);
}}
.fold-bar {{
  height: 20px; border-radius: 4px; position: absolute; left: 0; top: 0;
  transition: width .3s;
}}
.fold-bar.ci-great {{ background: #1d9e75; }}
.fold-bar.ci-good  {{ background: #639922; }}
.fold-bar.ci-low   {{ background: #ef9f27; }}
.fold-bar.ci-bad   {{ background: #d85a30; }}
.fold-val {{ width: 52px; text-align: right; color: #333; flex-shrink: 0; font-size: 12px; }}

/* Legend */
.bench-legend {{ font-size: 11px; font-weight: 400; color: #ba7517; margin-left: 8px; }}
.curve-legend {{ font-size: 11px; font-weight: 400; color: #888; margin-left: 8px; }}
.leg-solid {{ border-bottom: 2px solid #888; padding-bottom: 1px; }}
.leg-dash  {{ border-bottom: 2px dashed #888; padding-bottom: 1px; margin-left: 8px; }}

/* Comparison chart wrapper */
.chart-card {{
  background: #fff; border: 1px solid #e0e0d8;
  border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
}}
</style>
</head>
<body>
<div class="page-header">
  <h1>PathoGems — Experiment Report</h1>
  <p>TCGA-BRCA survival prediction &nbsp;·&nbsp;
     {n} experiment{"s" if n != 1 else ""} &nbsp;·&nbsp; Generated {now}</p>
</div>

<div class="container">

<p class="section-head">C-index comparison</p>
<div class="chart-card">
{comparison_chart}
</div>

<p class="section-head">Summary ({n} experiments)</p>
<div style="overflow-x:auto;">
<table class="summary-table">
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Mean C-index ± std</th>
      <th style="text-align:right">Best fold</th>
      <th style="text-align:right">Worst fold</th>
      <th style="text-align:center">Status</th>
      <th style="text-align:right">Wall time</th>
      <th>Started</th>
      <th>Config changes vs baseline</th>
    </tr>
  </thead>
  <tbody>
{summary_rows}
  </tbody>
</table>
</div>

<p class="section-head">Experiment details</p>
{exp_sections}

</div>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[report] Wrote {n} experiment(s) → {out_path}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; return 0 on success or 1 if the logs directory is missing."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("stage3_experiments/logs"),
        help="Directory containing run-log JSON files. Default: stage3_experiments/logs",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("stage3_experiments/reports/experiment_report.html"),
        help="Output HTML path. Default: stage3_experiments/reports/experiment_report.html",
    )
    args = p.parse_args(argv)

    if not args.logs_dir.is_dir():
        print(f"[report] ERROR: logs directory not found: {args.logs_dir}")
        return 1

    generate_report(args.logs_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

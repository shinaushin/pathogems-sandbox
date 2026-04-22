"""Exploratory data analysis of TCGA-BRCA PanCancer Atlas 2018.

Produces a multi-page PDF report at stage2_data/brca_eda.pdf covering:
  - Dataset inventory (files, dimensions)
  - Patient demographics (age, sex, ancestry, race)
  - Cancer characteristics (subtype, AJCC stage, grade)
  - Survival overview (OS, DFS, PFS status + KM curves by subtype)
  - Sample-level genomic instability (TMB, aneuploidy, MSI)
  - RNA-seq expression overview (per-sample totals, top variable genes)
  - Somatic mutation landscape (top mutated genes, variant classes)

Usage (from repo root, with the pathogems conda env active):
    python stage2_data/explore_brca.py [--data-dir PATH] [--out PATH]

Dependencies: pandas, numpy, matplotlib, lifelines
All are in the pathogems conda environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

try:
    from lifelines import KaplanMeierFitter

    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("[explore] lifelines not found — KM curves will be skipped.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Colour palette: one colour per BRCA subtype, consistent across all plots.
# ---------------------------------------------------------------------------
SUBTYPE_COLOURS = {
    "BRCA_LumA": "#4C72B0",
    "BRCA_LumB": "#DD8452",
    "BRCA_Her2": "#55A868",
    "BRCA_Basal": "#C44E52",
    "BRCA_Normal": "#8172B2",
}
DEFAULT_COLOUR = "#909090"

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _load_clinical(path: Path) -> pd.DataFrame:
    """Read a cBioPortal clinical TSV (4 comment lines then real header)."""
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def _parse_os_status(series: pd.Series) -> pd.Series:
    """Convert '0:LIVING' / '1:DECEASED' → integer 0 / 1."""
    return series.str.split(":").str[0].astype(float)


def _load_expression(path: Path) -> pd.DataFrame:
    """Load RSEM matrix; drop Entrez column, use Hugo_Symbol as index.

    Genes with a blank Hugo_Symbol are kept under their Entrez ID as a
    string to avoid silent row loss.
    """
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    # Drop the Entrez_Gene_Id column — we work with gene symbols.
    df = df.drop(columns=["Entrez_Gene_Id"], errors="ignore")
    # Fill blank gene symbols with their positional index so we don't lose rows.
    df.index = [str(g).strip() if str(g).strip() else f"GENE_{i}" for i, g in enumerate(df.index)]
    return df.astype(float)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------
def _bar(ax: plt.Axes, counts: pd.Series, title: str, colour: str | list = "#4C72B0",
         horizontal: bool = False, xlabel: str = "", ylabel: str = "Count") -> None:
    """Generic bar chart helper."""
    if horizontal:
        ax.barh(counts.index.astype(str), counts.values, color=colour)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.invert_yaxis()
    else:
        ax.bar(counts.index.astype(str), counts.values, color=colour)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
    ax.set_title(title)


def _hist(ax: plt.Axes, values: pd.Series, title: str, xlabel: str,
          colour: str = "#4C72B0", bins: int = 40) -> None:
    """Generic histogram helper."""
    vals = values.dropna()
    ax.hist(vals, bins=bins, color=colour, edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Patients")
    ax.axvline(vals.median(), color="crimson", linestyle="--", linewidth=1.2,
               label=f"Median {vals.median():.1f}")
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------
def page_inventory(pdf: PdfPages, data_dir: Path) -> None:
    """Text page: dataset file inventory with row counts."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    lines = ["TCGA-BRCA PanCancer Atlas 2018 — Dataset Inventory", ""]
    lines.append(f"Directory: {data_dir}\n")

    data_files = sorted(data_dir.glob("data_*.txt"))
    rows = [("File", "Size (KB)", "Lines")]
    for f in data_files:
        size_kb = f.stat().st_size / 1024
        with open(f) as fh:
            n = sum(1 for _ in fh)
        rows.append((f.name, f"{size_kb:,.0f}", f"{n:,}"))

    col_w = [0.55, 0.2, 0.15]
    y = 0.92
    ax.text(0.02, y + 0.04, lines[0], transform=ax.transAxes,
            fontsize=14, fontweight="bold")
    ax.text(0.02, y, lines[2], transform=ax.transAxes, fontsize=9, color="#555")

    y -= 0.04
    for i, (fname, size, nlines) in enumerate(rows):
        weight = "bold" if i == 0 else "normal"
        colour = "#eef" if i % 2 == 1 else "white"
        ax.add_patch(plt.Rectangle((0.01, y - 0.025), 0.97, 0.028,
                                   transform=ax.transAxes, color=colour, zorder=0))
        ax.text(0.02, y, fname, transform=ax.transAxes, fontsize=8, fontweight=weight)
        ax.text(0.58, y, size, transform=ax.transAxes, fontsize=8,
                fontweight=weight, ha="right")
        ax.text(0.80, y, nlines, transform=ax.transAxes, fontsize=8,
                fontweight=weight, ha="right")
        y -= 0.028
        if y < 0.02:
            break

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 1: dataset inventory")


def page_demographics(pdf: PdfPages, clin: pd.DataFrame) -> None:
    """Patient demographics: subtype, age, sex, race, ancestry."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Patient Demographics  (n = {:,})".format(len(clin)), fontsize=13, y=1.01)

    # BRCA subtype
    ax = axes[0, 0]
    sub_counts = clin["SUBTYPE"].value_counts()
    colours = [SUBTYPE_COLOURS.get(s, DEFAULT_COLOUR) for s in sub_counts.index]
    _bar(ax, sub_counts, "BRCA Molecular Subtype", colour=colours, xlabel="Subtype")

    # Age at diagnosis
    _hist(axes[0, 1], clin["AGE"], "Age at Diagnosis", "Age (years)", colour="#4C72B0")

    # Sex
    ax = axes[0, 2]
    sex_counts = clin["SEX"].value_counts()
    ax.pie(sex_counts, labels=sex_counts.index, autopct="%1.1f%%",
           colors=["#4C72B0", "#DD8452"], startangle=90)
    ax.set_title("Sex")

    # Genetic ancestry
    ax = axes[1, 0]
    anc_counts = clin["GENETIC_ANCESTRY_LABEL"].value_counts()
    _bar(ax, anc_counts, "Genetic Ancestry", colour="#55A868", xlabel="Ancestry")

    # Race
    ax = axes[1, 1]
    race_counts = clin["RACE"].value_counts().head(6)
    _bar(ax, race_counts, "Self-Reported Race", colour="#8172B2", xlabel="Race")

    # Ethnicity
    ax = axes[1, 2]
    eth_counts = clin["ETHNICITY"].value_counts()
    _bar(ax, eth_counts, "Ethnicity", colour="#CCB974", xlabel="Ethnicity")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 2: demographics")


def page_cancer_characteristics(pdf: PdfPages, clin: pd.DataFrame,
                                samp: pd.DataFrame) -> None:
    """AJCC stage, T/N/M codes, tumour grade, and cancer status."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Cancer Characteristics", fontsize=13, y=1.01)

    # AJCC pathologic stage (simplified)
    ax = axes[0, 0]
    stage_counts = clin["AJCC_PATHOLOGIC_TUMOR_STAGE"].value_counts()
    _bar(ax, stage_counts, "AJCC Pathologic Stage", colour="#C44E52", xlabel="Stage")

    # Pathologic T stage
    t_counts = clin["PATH_T_STAGE"].value_counts().head(12)
    _bar(axes[0, 1], t_counts, "Pathologic T Stage", colour="#4C72B0", xlabel="T")

    # Pathologic N stage
    n_counts = clin["PATH_N_STAGE"].value_counts().head(10)
    _bar(axes[0, 2], n_counts, "Pathologic N Stage", colour="#DD8452", xlabel="N")

    # Tumour grade (from sample file)
    ax = axes[1, 0]
    if "GRADE" in samp.columns:
        grade_counts = samp["GRADE"].value_counts()
        _bar(ax, grade_counts, "Histologic Grade (Sample)", colour="#55A868", xlabel="Grade")
    else:
        ax.text(0.5, 0.5, "GRADE not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Histologic Grade")

    # Person neoplasm cancer status
    status_counts = clin["PERSON_NEOPLASM_CANCER_STATUS"].value_counts()
    _bar(axes[1, 1], status_counts, "Neoplasm Cancer Status",
         colour="#8172B2", xlabel="Status")

    # New tumour event after initial treatment
    new_event = clin["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].value_counts()
    _bar(axes[1, 2], new_event, "New Tumour Event After Tx",
         colour="#CCB974", xlabel="")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 3: cancer characteristics")


def page_survival(pdf: PdfPages, clin: pd.DataFrame) -> None:
    """Survival endpoints: OS, DFS, PFS status + duration histograms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Survival Endpoints", fontsize=13, y=1.01)

    def _survival_pair(ax_pie: plt.Axes, ax_hist: plt.Axes,
                       status_col: str, months_col: str, label: str) -> None:
        if status_col not in clin.columns or months_col not in clin.columns:
            ax_pie.text(0.5, 0.5, "Not available", ha="center", va="center",
                        transform=ax_pie.transAxes)
            return
        status_raw = clin[status_col].dropna()
        event = _parse_os_status(status_raw)
        n_event = int(event.sum())
        n_total = len(event)
        ax_pie.pie([n_event, n_total - n_event],
                   labels=[f"Event ({n_event})", f"Censored ({n_total - n_event})"],
                   autopct="%1.1f%%", colors=["#C44E52", "#4C72B0"], startangle=90)
        ax_pie.set_title(f"{label} Status")
        months = clin[months_col].dropna()
        ax_hist.hist(months, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax_hist.axvline(months.median(), color="crimson", linestyle="--",
                        linewidth=1.2, label=f"Median {months.median():.1f} mo")
        ax_hist.set_title(f"{label} Duration")
        ax_hist.set_xlabel("Months")
        ax_hist.set_ylabel("Patients")
        ax_hist.legend(fontsize=7)

    _survival_pair(axes[0, 0], axes[0, 1], "OS_STATUS", "OS_MONTHS", "Overall Survival")
    _survival_pair(axes[0, 2], axes[1, 0], "DFS_STATUS", "DFS_MONTHS",
                   "Disease-Free Survival")
    _survival_pair(axes[1, 1], axes[1, 2], "PFS_STATUS", "PFS_MONTHS",
                   "Progression-Free Survival")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 4: survival endpoints")


def page_km_curves(pdf: PdfPages, clin: pd.DataFrame) -> None:
    """Kaplan-Meier overall-survival curves stratified by BRCA subtype."""
    if not HAS_LIFELINES:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Overall Survival by BRCA Molecular Subtype", fontsize=12)

    df = clin[["SUBTYPE", "OS_MONTHS", "OS_STATUS"]].dropna()
    df = df.copy()
    df["event"] = _parse_os_status(df["OS_STATUS"])
    df["time"] = df["OS_MONTHS"]

    for subtype in sorted(df["SUBTYPE"].unique()):
        sub = df[df["SUBTYPE"] == subtype]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub["time"], event_observed=sub["event"], label=f"{subtype} (n={len(sub)})")
        colour = SUBTYPE_COLOURS.get(subtype, DEFAULT_COLOUR)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colour, linewidth=1.8,
                                   ci_alpha=0.12)

    ax.set_xlabel("Months from Diagnosis")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 5: Kaplan-Meier curves")


def page_genomic_instability(pdf: PdfPages, samp: pd.DataFrame) -> None:
    """Sample-level genomic instability metrics: TMB, aneuploidy, MSI."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Genomic Instability Metrics (Sample-Level)", fontsize=12, y=1.02)

    if "TMB_NONSYNONYMOUS" in samp.columns:
        vals = samp["TMB_NONSYNONYMOUS"].dropna()
        axes[0].hist(vals, bins=60, color="#C44E52", edgecolor="white", linewidth=0.4)
        axes[0].set_title("Tumour Mutational Burden (TMB)")
        axes[0].set_xlabel("Mutations / Mb (nonsynonymous)")
        axes[0].set_ylabel("Samples")
        axes[0].axvline(vals.median(), color="navy", linestyle="--",
                        linewidth=1.2, label=f"Median {vals.median():.2f}")
        axes[0].legend(fontsize=7)

    if "ANEUPLOIDY_SCORE" in samp.columns:
        vals = samp["ANEUPLOIDY_SCORE"].dropna()
        axes[1].hist(vals, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
        axes[1].set_title("Aneuploidy Score")
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Samples")
        axes[1].axvline(vals.median(), color="crimson", linestyle="--",
                        linewidth=1.2, label=f"Median {vals.median():.1f}")
        axes[1].legend(fontsize=7)

    if "MSI_SENSOR_SCORE" in samp.columns:
        vals = samp["MSI_SENSOR_SCORE"].dropna()
        axes[2].hist(vals, bins=40, color="#55A868", edgecolor="white", linewidth=0.4)
        axes[2].set_title("MSI Sensor Score")
        axes[2].set_xlabel("Score (>10 = MSI-H)")
        axes[2].set_ylabel("Samples")
        axes[2].axvline(10, color="crimson", linestyle="--",
                        linewidth=1.2, label="MSI-H threshold (10)")
        axes[2].legend(fontsize=7)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 6: genomic instability")


def page_expression(pdf: PdfPages, data_dir: Path) -> None:
    """RNA-seq overview: per-sample totals and top variable genes."""
    print("[explore]   Loading expression matrix (~20k genes × 1k samples) …")
    expr = _load_expression(data_dir / "data_mrna_seq_v2_rsem.txt")
    n_genes, n_samples = expr.shape
    print(f"[explore]   Expression matrix: {n_genes:,} genes × {n_samples:,} samples")

    log_expr = np.log2(expr + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"RNA-seq Expression Overview  ({n_genes:,} genes × {n_samples:,} samples)",
        fontsize=12, y=1.02,
    )

    # Per-sample total log2 expression (box over a random subset of 80 samples)
    ax = axes[0]
    sample_subset = log_expr.sample(n=min(80, n_samples), axis=1, random_state=0)
    ax.boxplot(sample_subset.values, notch=False, patch_artist=True,
               boxprops=dict(facecolor="#4C72B080"),
               medianprops=dict(color="crimson"),
               whiskerprops=dict(linewidth=0.5),
               flierprops=dict(marker=".", markersize=1, alpha=0.3))
    ax.set_xlabel("Sample (random subset of 80)")
    ax.set_ylabel("log₂(RSEM + 1)")
    ax.set_title("Per-Sample Expression Distribution")
    ax.set_xticks([])

    # Top 20 most variable genes
    ax = axes[1]
    gene_var = log_expr.var(axis=1).sort_values(ascending=False)
    top20 = gene_var.head(20)
    ax.barh(top20.index[::-1], top20.values[::-1], color="#DD8452")
    ax.set_xlabel("Variance of log₂(RSEM + 1)")
    ax.set_title("Top 20 Most Variable Genes")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 7: expression overview")


def page_mutations(pdf: PdfPages, data_dir: Path) -> None:
    """Somatic mutation landscape: top genes and variant class breakdown."""
    print("[explore]   Loading mutations (~130k rows) …")
    mut = pd.read_csv(
        data_dir / "data_mutations.txt",
        sep="\t",
        usecols=["Hugo_Symbol", "Variant_Classification", "Variant_Type",
                 "Tumor_Sample_Barcode"],
        low_memory=False,
    )
    n_patients = mut["Tumor_Sample_Barcode"].nunique()
    print(f"[explore]   Mutations: {len(mut):,} rows, {n_patients:,} tumour samples")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Somatic Mutation Landscape  ({len(mut):,} mutations, {n_patients:,} tumours)",
        fontsize=12, y=1.02,
    )

    # Top 25 most frequently mutated genes (by number of unique samples)
    ax = axes[0]
    gene_freq = (
        mut.groupby("Hugo_Symbol")["Tumor_Sample_Barcode"]
        .nunique()
        .sort_values(ascending=False)
        .head(25)
    )
    pct = (gene_freq / n_patients * 100).round(1)
    colours = ["#C44E52" if v >= 5 else "#4C72B0" for v in pct]
    ax.barh(gene_freq.index[::-1], pct.values[::-1], color=colours[::-1])
    ax.set_xlabel("% Tumours Mutated")
    ax.set_title("Top 25 Mutated Genes\n(red = ≥5% of tumours)")
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(5, color="crimson", linestyle="--", linewidth=0.8, alpha=0.6)

    # Variant classification breakdown
    ax = axes[1]
    vc_counts = mut["Variant_Classification"].value_counts()
    cmap = plt.get_cmap("Set2")
    colours_vc = [cmap(i / max(len(vc_counts) - 1, 1)) for i in range(len(vc_counts))]
    ax.barh(vc_counts.index[::-1], vc_counts.values[::-1], color=colours_vc[::-1])
    ax.set_xlabel("Number of Mutations")
    ax.set_title("Variant Classification Breakdown")
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 8: mutation landscape")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("stage2_data/raw/brca_tcga_pan_can_atlas_2018"),
        help="Directory containing the cBioPortal data files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("stage2_data/brca_eda.pdf"),
        help="Output PDF path. Default: stage2_data/brca_eda.pdf",
    )
    args = p.parse_args(argv)

    data_dir: Path = args.data_dir
    out_pdf: Path = args.out

    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    print(f"[explore] Loading clinical data from {data_dir} …")
    clin = _load_clinical(data_dir / "data_clinical_patient.txt")
    samp = _load_clinical(data_dir / "data_clinical_sample.txt")
    print(f"[explore] Patients: {len(clin):,}  |  Samples: {len(samp):,}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    print(f"[explore] Writing report to {out_pdf} …\n")

    with PdfPages(out_pdf) as pdf:
        page_inventory(pdf, data_dir)
        page_demographics(pdf, clin)
        page_cancer_characteristics(pdf, clin, samp)
        page_survival(pdf, clin)
        page_km_curves(pdf, clin)
        page_genomic_instability(pdf, samp)
        page_expression(pdf, data_dir)
        page_mutations(pdf, data_dir)

        # Attach metadata to the PDF
        info = pdf.infodict()
        info["Title"] = "TCGA-BRCA PanCancer Atlas 2018 — EDA Report"
        info["Author"] = "PathoGems Stage 2 explore_brca.py"
        info["Subject"] = "Exploratory data analysis"

    print(f"\n[explore] Done. Report written to {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

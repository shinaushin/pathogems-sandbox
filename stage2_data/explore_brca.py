"""Exploratory data analysis of TCGA-BRCA PanCancer Atlas 2018.

Produces a multi-page PDF report at stage2_data/brca_eda.pdf covering:
  - Dataset inventory (files, dimensions)
  - Patient demographics (age, sex, ancestry, race)
  - Cancer characteristics (subtype, AJCC stage, grade)
  - Survival overview (OS, DFS, PFS status + KM curves by subtype)
  - Sample-level genomic instability (TMB, aneuploidy, MSI)
  - RNA-seq expression overview (per-sample totals, top variable genes)
  - Somatic mutation landscape (top mutated genes, variant classes)

Each page includes a plain-language explanation panel so that readers
without a genomics background can follow along.

Usage (from repo root, with the pathogems conda env active):
    python stage2_data/explore_brca.py [--data-dir PATH] [--out PATH]

Dependencies: pandas, numpy, matplotlib, lifelines
All are in the pathogems conda environment.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
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
# Explanation panel
# ---------------------------------------------------------------------------
_PANEL_HEIGHT_FRAC = 0.30   # fraction of figure height reserved for text
_PANEL_BG = "#F7F7F2"
_PANEL_EDGE = "#CCCCBB"
_BULLET = "\u2022"          # •


def _add_explanation(fig: plt.Figure, paragraphs: list[str], wrap_width: int = 115) -> None:
    """Draw a lightly shaded explanation panel across the bottom of *fig*.

    Call this after all axes have been added but before savefig. The
    function adjusts the subplot layout to leave room, then renders each
    string in *paragraphs* as a separate indented bullet point.

    Args:
        fig: The figure to annotate.
        paragraphs: List of plain-text strings, one per bullet.
        wrap_width: Target character width for text wrapping.
    """
    # Shrink the chart area to leave the bottom strip free.
    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])

    # Background rectangle.
    rect = mpatches.FancyBboxPatch(
        (0.01, 0.01), 0.98, _PANEL_HEIGHT_FRAC - 0.02,
        transform=fig.transFigure,
        boxstyle="round,pad=0.005",
        facecolor=_PANEL_BG, edgecolor=_PANEL_EDGE, linewidth=0.8,
        zorder=0, clip_on=False,
    )
    fig.add_artist(rect)

    # Header label.
    fig.text(
        0.025, _PANEL_HEIGHT_FRAC - 0.025,
        "How to read this page",
        transform=fig.transFigure,
        fontsize=8, fontweight="bold", color="#444",
        va="top", ha="left",
    )

    # Bullet points.
    y_start = _PANEL_HEIGHT_FRAC - 0.055
    line_height = 0.013
    x_bullet = 0.025
    x_text = 0.040
    y = y_start

    for para in paragraphs:
        wrapped = textwrap.wrap(para, width=wrap_width)
        if not wrapped:
            y -= line_height * 0.5
            continue
        # First line gets the bullet.
        fig.text(x_bullet, y, _BULLET, transform=fig.transFigure,
                 fontsize=7.5, color="#666", va="top")
        fig.text(x_text, y, wrapped[0], transform=fig.transFigure,
                 fontsize=7.5, color="#333", va="top")
        y -= line_height
        for continuation in wrapped[1:]:
            fig.text(x_text, y, continuation, transform=fig.transFigure,
                     fontsize=7.5, color="#333", va="top")
            y -= line_height
        y -= line_height * 0.4   # small gap between bullets
        if y < 0.015:
            break


# ---------------------------------------------------------------------------
# Structured text-page builder
# ---------------------------------------------------------------------------

def _text_page(
    pdf: PdfPages,
    sections: list[tuple[str, list[str]]],
    title: str,
    subtitle: str = "",
) -> None:
    """Render a formatted text-only page with titled sections and body paragraphs.

    Args:
        pdf: Open PdfPages handle.
        sections: List of (section_heading, [paragraph, ...]) tuples.
        title: Large page title at the top.
        subtitle: Optional smaller line under the title.
    """
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    TW = 105          # wrap width (characters)
    x_margin = 0.04
    y = 0.96          # start position (figure fraction, top of page)

    # ---- Page title ----
    ax.text(x_margin, y, title,
            transform=ax.transAxes,
            fontsize=15, fontweight="bold", color="#1a1a2e", va="top")
    y -= 0.055
    if subtitle:
        ax.text(x_margin, y, subtitle,
                transform=ax.transAxes,
                fontsize=9, color="#555", va="top", style="italic")
        y -= 0.035

    # Thin horizontal rule under the title block.
    ax.plot([x_margin, 0.96], [y, y], color="#AAAAAA", linewidth=0.6,
            transform=ax.transAxes, clip_on=False)
    y -= 0.025

    SECTION_FONT = 9.5
    BODY_FONT = 8.5
    SECTION_GAP = 0.022
    PARA_GAP = 0.012
    LINE_H = 0.018       # height of one body-text line

    for heading, paragraphs in sections:
        if y < 0.04:
            break

        # Section heading.
        ax.text(x_margin, y, heading,
                transform=ax.transAxes,
                fontsize=SECTION_FONT, fontweight="bold", color="#16213e", va="top")
        y -= SECTION_GAP

        for para in paragraphs:
            wrapped_lines = textwrap.wrap(para, width=TW)
            for line in wrapped_lines:
                if y < 0.04:
                    break
                ax.text(x_margin + 0.015, y, line,
                        transform=ax.transAxes,
                        fontsize=BODY_FONT, color="#333333", va="top")
                y -= LINE_H
            y -= PARA_GAP

        y -= 0.010   # extra gap between sections

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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
    """Load RSEM matrix; drop Entrez column, use Hugo_Symbol as index."""
    df = pd.read_csv(path, sep="\t", index_col=0, low_memory=False)
    df = df.drop(columns=["Entrez_Gene_Id"], errors="ignore")
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

def page_project_overview(pdf: PdfPages) -> None:
    """Text page 1: problem statement, ranking rationale, WSI roadmap."""
    _text_page(
        pdf,
        title="PathoGems — Project Overview",
        subtitle="Predicting breast cancer patient survival from molecular tumour data",
        sections=[
            (
                "The Problem",
                [
                    "Breast cancer is the most commonly diagnosed cancer in women worldwide. After a patient "
                    "is diagnosed and treated, clinicians face a critical question: how aggressive is this "
                    "particular tumour, and how much time does this patient have? The answer determines "
                    "whether to recommend intensive chemotherapy (with serious side effects) or lighter "
                    "hormone-blocking therapy — getting it wrong means either over-treating patients who "
                    "didn't need it, or under-treating patients whose cancer will return.",

                    "Current clinical tools (AJCC staging, Oncotype DX gene panels) give partial answers "
                    "but were derived from relatively small datasets and capture only a fraction of the "
                    "molecular complexity of a tumour. PathoGems asks: can a machine-learning model trained "
                    "on large-scale molecular data — starting with RNA-seq gene expression from ~1,000 "
                    "TCGA-BRCA patients — produce better, more individualised survival predictions?",
                ],
            ),
            (
                "Why Survival Ranking, Not Classification or Regression",
                [
                    "A natural first instinct might be to frame this as classification ('will this patient "
                    "die within 5 years: yes or no?') or regression ('predict exact survival time in "
                    "months'). Both are problematic for survival data.",

                    "Classification throws away information. Whether a patient died at month 6 or month 59 "
                    "of a 60-month study both count as the same 'yes' label — yet they are very different "
                    "clinical situations. A model trained this way cannot distinguish rapid progressors "
                    "from slow ones.",

                    "Regression cannot handle censoring. About 80% of TCGA-BRCA patients were still alive "
                    "when the study ended. We know they survived at least X months, but not how long they "
                    "will ultimately live. A standard regression model would either have to drop these "
                    "patients (losing 80% of the data) or treat their last follow-up time as their true "
                    "survival time (which would severely underestimate survival).",

                    "Ranking survives both problems. Instead of predicting an absolute number, we ask "
                    "the model to assign a relative risk score to each patient — higher score means higher "
                    "predicted hazard of death. We evaluate quality using Harrell's C-index: out of all "
                    "patient pairs where one definitely died before the other, how often does the model "
                    "rank the deceased patient as higher risk? A C-index of 0.5 is random; 1.0 is "
                    "perfect. Crucially, censored patients can still contribute to this comparison "
                    "whenever the other patient in the pair had an observed death.",
                ],
            ),
            (
                "Why We Are Not Using Pathology Images (WSIs) Yet",
                [
                    "Whole Slide Images (WSIs) are high-resolution scans of tumour tissue slides taken "
                    "at surgery — essentially a photograph of the cancer at 20× or 40× magnification, "
                    "capturing individual cell shapes, tissue architecture, and spatial patterns that "
                    "a pathologist normally reads by eye. TCGA-BRCA has WSIs for most patients, hosted "
                    "separately by the NCI's Genomic Data Commons (GDC).",

                    "We are deferring WSIs for two reasons. First, practical: WSI files are very large "
                    "(often 1–5 GB each), require GPU-accelerated deep-learning pipelines to process, "
                    "and need substantial infrastructure to download and store. Building that pipeline "
                    "before we have a working omics baseline would mean debugging two complex systems "
                    "at once, making it hard to isolate problems.",

                    "Second, scientific: establishing an omics-only baseline first gives us a clear "
                    "performance benchmark to beat. If our RNA-seq model achieves a C-index of 0.65, "
                    "we can ask a precise question when we add WSIs: 'Does adding image features push "
                    "this above 0.65?' Without the baseline, we wouldn't know whether any gain came "
                    "from the images or just from model improvements.",
                ],
            ),
            (
                "How Pathology Images Will Help When We Add Them",
                [
                    "Gene expression tells us which biological programmes are active inside the cells. "
                    "Pathology images tell us something different: what the tissue looks like — how "
                    "densely packed the tumour cells are, whether they are invading surrounding tissue, "
                    "how much immune cell infiltration exists. These two views are partially redundant "
                    "(highly proliferating tumours tend to look abnormal AND have high expression of "
                    "growth genes) but also partially complementary.",

                    "Specifically, images can capture spatial heterogeneity — some regions of a tumour "
                    "may be far more aggressive than others, and this regional variation is invisible "
                    "to bulk RNA-seq (which averages across the whole sample). Pathologists already "
                    "use slide morphology to assign tumour grade; a deep-learning model could learn "
                    "far finer-grained prognostic features from the same images.",

                    "The planned approach is to extract patch-level features from the WSIs using a "
                    "pre-trained pathology vision encoder, aggregate them per patient, and concatenate "
                    "the resulting embedding with the RNA-seq features before the final Cox risk head. "
                    "This multimodal fusion is the long-term goal of the PathoGems project.",
                ],
            ),
        ],
    )
    print("[explore] ✓ Page 1: project overview")


def page_design_choices(pdf: PdfPages) -> None:
    """Text page 2: loss function and MLP architecture justifications."""
    _text_page(
        pdf,
        title="Model Design Choices",
        subtitle="Why we made the architectural decisions we did for the baseline experiment",
        sections=[
            (
                "Loss Function Options",
                [
                    "The loss function is what the model is trained to minimise. In survival analysis "
                    "several options exist, each with different assumptions and trade-offs.",

                    "Cox Partial Likelihood (our choice): The Cox proportional-hazards model assumes "
                    "that every patient's risk of death at any moment is a fixed multiple of a shared "
                    "baseline hazard — a patient twice as risky is always twice as risky, regardless "
                    "of time. The loss derived from this assumption — the negative log partial "
                    "likelihood — only involves the ordering of risk scores among patients who are "
                    "'at risk' at each observed death time. This means it never requires predicting "
                    "absolute survival time, only relative ordering.",

                    "Ranking losses (e.g. pairwise log-rank): Directly optimise the C-index or an "
                    "approximation of it. These are conceptually very clean — optimise exactly what "
                    "you evaluate — but tend to have noisy gradients because each gradient step "
                    "depends on randomly sampled pairs of patients, not the full dataset.",

                    "DeepHit: A neural network approach that abandons the proportional-hazards "
                    "assumption and instead predicts the full probability distribution over discrete "
                    "time intervals. More flexible, but requires many more parameters, is harder to "
                    "train stably, and makes the outputs harder to interpret clinically.",

                    "Accelerated Failure Time (AFT) models: Predict log(survival time) directly as "
                    "a regression target, with a special likelihood that handles censoring. Intuitive "
                    "output, but the log-time regression target is sensitive to model mis-specification.",
                ],
            ),
            (
                "Why We Chose Cox Partial Likelihood to Start",
                [
                    "The Cox loss has four practical advantages for a baseline. First, it is the "
                    "dominant standard in clinical survival literature, so our C-index results are "
                    "directly comparable to decades of published benchmarks on TCGA-BRCA.",

                    "Second, it does not require us to predict absolute survival time — only to rank "
                    "patients correctly. This is a strictly easier task that allows the model to "
                    "focus its capacity on learning which molecular features are prognostically "
                    "discriminative, rather than also trying to learn the shape of the baseline "
                    "hazard function.",

                    "Third, the Breslow approximation (our implementation) reduces the O(N²) "
                    "all-pairs computation to O(N log N) using a sorted cumulative log-sum-exp "
                    "trick, making full-batch training on ~900 patients fast even on CPU.",

                    "Fourth, the Cox loss is well-behaved numerically and easy to unit-test: "
                    "for small examples, the closed-form partial likelihood can be computed by "
                    "hand and compared against our implementation — which is exactly what our "
                    "test suite does.",
                ],
            ),
            (
                "Why ReLU Activations",
                [
                    "After each linear layer, we apply a Rectified Linear Unit (ReLU): output = "
                    "max(0, input). Negative values are set to zero; positive values pass through "
                    "unchanged. ReLU has three advantages over the historically popular sigmoid and "
                    "tanh activations.",

                    "It does not saturate for large positive inputs. Sigmoid and tanh both squash "
                    "large values toward a ceiling (1.0), making their gradients near-zero and "
                    "causing the 'vanishing gradient' problem that makes deep networks hard to train. "
                    "ReLU gradients are either exactly 0 (for negative pre-activations) or exactly "
                    "1 (for positive ones), so gradients flow cleanly to early layers.",

                    "It is computationally trivial — a comparison and a threshold — which matters "
                    "when training hundreds of experiments. It also tends to produce sparse "
                    "activations (many neurons outputting exactly 0), which acts as an implicit "
                    "regulariser and can make the learned representations more interpretable.",
                ],
            ),
            (
                "Why Batch Normalisation",
                [
                    "Batch Normalisation (BatchNorm) is placed after each linear layer and before "
                    "the ReLU. It re-centres and re-scales the activations within each mini-batch "
                    "so they have approximately zero mean and unit variance, then applies learned "
                    "scale and shift parameters.",

                    "Even though we z-score the input genes before training, as the signal passes "
                    "through successive linear layers the distribution of activations shifts and "
                    "widens — a phenomenon called 'internal covariate shift'. Without BatchNorm, "
                    "deeper layers are constantly chasing a moving target, slowing convergence. "
                    "BatchNorm stabilises each layer's input distribution so each layer can learn "
                    "more independently.",

                    "On small cohorts like TCGA-BRCA (~900 training patients), BatchNorm also "
                    "provides mild regularisation: the per-batch statistics introduce small "
                    "stochastic perturbations that reduce overfitting, similar in spirit to dropout "
                    "but at the activation distribution level.",
                ],
            ),
            (
                "Why Dropout",
                [
                    "Dropout randomly sets a fraction of neuron outputs to zero during each "
                    "training step (we use a rate of 0.3, meaning 30% of activations are zeroed). "
                    "At inference time, all neurons are active and their outputs are scaled "
                    "accordingly.",

                    "The motivation is to prevent co-adaptation: if neurons can always rely on "
                    "their neighbours being present, they learn to correct each other's errors "
                    "rather than extract independent useful features. Dropout forces each neuron "
                    "to be useful on its own, producing a more robust representation.",

                    "With only ~900 training patients and 500 input genes, our model has enough "
                    "parameters to overfit — memorising the training fold rather than generalising "
                    "to new patients. Dropout is one of the most effective and simplest tools for "
                    "preventing this. The 0.3 rate comes from the DeepSurv paper (Katzman et al. "
                    "2018), which used the same architecture on several TCGA cohorts and is the "
                    "closest published baseline to our own setup.",
                ],
            ),
        ],
    )
    print("[explore] ✓ Page 2: design choices")


def page_inventory(pdf: PdfPages, data_dir: Path) -> None:
    """Text page: dataset file inventory with row counts."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Shrink chart area so explanation panel fits.
    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])

    lines = ["TCGA-BRCA PanCancer Atlas 2018 — Dataset Inventory", ""]
    lines.append(f"Directory: {data_dir}\n")

    data_files = sorted(data_dir.glob("data_*.txt"))
    rows = [("File", "Size (KB)", "Lines")]
    for f in data_files:
        size_kb = f.stat().st_size / 1024
        with open(f) as fh:
            n = sum(1 for _ in fh)
        rows.append((f.name, f"{size_kb:,.0f}", f"{n:,}"))

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

    _add_explanation(fig, [
        "This page lists every data file in the TCGA-BRCA dataset. TCGA stands for The Cancer Genome "
        "Atlas — a large US government-funded project that collected and shared molecular data from "
        "thousands of cancer patients across dozens of cancer types. BRCA is their breast cancer "
        "cohort (~1,100 patients).",

        "Each file captures a different 'view' of each tumour. For example: data_mrna_seq_v2_rsem.txt "
        "contains RNA-seq gene expression (which genes are active and by how much); "
        "data_mutations.txt lists every DNA mutation found in the tumour; "
        "data_clinical_patient.txt records patient outcomes like survival time.",

        "RNA-seq (RNA sequencing) works by reading the messenger RNA molecules inside a tumour cell. "
        "These molecules carry instructions from DNA to make proteins, so measuring them tells us "
        "which genes are 'switched on' or 'switched off' in the cancer. The result is a number for "
        "each of ~20,000 human genes per patient.",

        "Our survival-prediction model currently uses only the RNA-seq file and the clinical file. "
        "The other files (mutations, copy-number alterations, methylation, protein) are available "
        "for future experiments when we want to test whether adding more data modalities improves "
        "prediction accuracy.",
    ])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 1: dataset inventory")


def page_demographics(pdf: PdfPages, clin: pd.DataFrame) -> None:
    """Patient demographics: subtype, age, sex, race, ancestry."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Patient Demographics  (n = {:,})".format(len(clin)), fontsize=13, y=1.01)

    ax = axes[0, 0]
    sub_counts = clin["SUBTYPE"].value_counts()
    colours = [SUBTYPE_COLOURS.get(s, DEFAULT_COLOUR) for s in sub_counts.index]
    _bar(ax, sub_counts, "BRCA Molecular Subtype", colour=colours, xlabel="Subtype")

    _hist(axes[0, 1], clin["AGE"], "Age at Diagnosis", "Age (years)", colour="#4C72B0")

    ax = axes[0, 2]
    sex_counts = clin["SEX"].value_counts()
    ax.pie(sex_counts, labels=sex_counts.index, autopct="%1.1f%%",
           colors=["#4C72B0", "#DD8452"], startangle=90)
    ax.set_title("Sex")

    ax = axes[1, 0]
    anc_counts = clin["GENETIC_ANCESTRY_LABEL"].value_counts()
    _bar(ax, anc_counts, "Genetic Ancestry", colour="#55A868", xlabel="Ancestry")

    ax = axes[1, 1]
    race_counts = clin["RACE"].value_counts().head(6)
    _bar(ax, race_counts, "Self-Reported Race", colour="#8172B2", xlabel="Race")

    ax = axes[1, 2]
    eth_counts = clin["ETHNICITY"].value_counts()
    _bar(ax, eth_counts, "Ethnicity", colour="#CCB974", xlabel="Ethnicity")

    _add_explanation(fig, [
        "BRCA Molecular Subtype (top-left): Breast cancer is not one disease — it is several, "
        "defined by which genes are abnormally active. Luminal A (LumA) tumours grow slowly and "
        "respond well to hormone-blocking drugs. Luminal B (LumB) are similar but faster-growing. "
        "HER2-enriched tumours overproduce a growth-promoting protein called HER2 and are treated "
        "with targeted drugs like Herceptin. Basal-like tumours (often called 'triple-negative') "
        "lack the three common receptors and are the hardest to treat. Each subtype has a different "
        "expected survival trajectory, which is why subtype is one of the most important variables "
        "in our model.",

        "Age at Diagnosis (top-middle): Shows how old patients were when their cancer was found. "
        "The red dashed line is the median age. Breast cancer risk increases with age; the dataset "
        "skews toward post-menopausal patients, which is typical for TCGA cohorts.",

        "Genetic Ancestry (bottom-left): Derived computationally from the patient's DNA — not "
        "self-reported. Different ancestry groups can have different baseline mutation rates and "
        "different typical subtypes (e.g. Basal-like tumours are more common in patients of "
        "African ancestry). This matters for model fairness: if the dataset is skewed toward one "
        "group, predictions may be less accurate for underrepresented groups.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 2: demographics")


def page_cancer_characteristics(pdf: PdfPages, clin: pd.DataFrame,
                                samp: pd.DataFrame) -> None:
    """AJCC stage, T/N/M codes, tumour grade, and cancer status."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Cancer Characteristics", fontsize=13, y=1.01)

    ax = axes[0, 0]
    stage_counts = clin["AJCC_PATHOLOGIC_TUMOR_STAGE"].value_counts()
    _bar(ax, stage_counts, "AJCC Pathologic Stage", colour="#C44E52", xlabel="Stage")

    t_counts = clin["PATH_T_STAGE"].value_counts().head(12)
    _bar(axes[0, 1], t_counts, "Pathologic T Stage", colour="#4C72B0", xlabel="T")

    n_counts = clin["PATH_N_STAGE"].value_counts().head(10)
    _bar(axes[0, 2], n_counts, "Pathologic N Stage", colour="#DD8452", xlabel="N")

    ax = axes[1, 0]
    if "GRADE" in samp.columns:
        grade_counts = samp["GRADE"].value_counts()
        _bar(ax, grade_counts, "Histologic Grade (Sample)", colour="#55A868", xlabel="Grade")
    else:
        ax.text(0.5, 0.5, "GRADE not available", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Histologic Grade")

    status_counts = clin["PERSON_NEOPLASM_CANCER_STATUS"].value_counts()
    _bar(axes[1, 1], status_counts, "Neoplasm Cancer Status",
         colour="#8172B2", xlabel="Status")

    new_event = clin["NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"].value_counts()
    _bar(axes[1, 2], new_event, "New Tumour Event After Tx",
         colour="#CCB974", xlabel="")

    _add_explanation(fig, [
        "AJCC Pathologic Stage (top-left): The standard way oncologists describe how far a cancer "
        "has progressed, using Roman numerals I–IV. Stage I means a small tumour confined to the "
        "breast. Stage II means it has grown larger or reached nearby lymph nodes. Stage III "
        "means it has spread extensively to lymph nodes or nearby tissue. Stage IV means it has "
        "spread (metastasised) to distant organs. Higher stage generally means shorter survival, "
        "so stage is a strong predictor our model must account for.",

        "T / N Stages (top-middle and top-right): The full AJCC system breaks down into three "
        "sub-scores. T (Tumour) describes the size of the primary tumour — T1 is small, T4 is "
        "very large or has invaded surrounding skin or chest wall. N (Node) describes whether "
        "cancer cells have been found in nearby lymph nodes — N0 means none, N3 means many. "
        "A third score M (Metastasis, not shown separately) records distant spread. Together, "
        "T + N + M determine the overall Stage.",

        "Histologic Grade (bottom-left): A pathologist looks at the tumour cells under a "
        "microscope and scores how abnormal they look compared to normal breast cells. Grade 1 "
        "(low) means cells still look fairly normal and tend to grow slowly. Grade 3 (high) "
        "means cells look very abnormal, divide rapidly, and tend to be more aggressive. Grade "
        "is independent of how far the cancer has spread.",

        "New Tumour Event After Treatment (bottom-right): Records whether the cancer came back "
        "after initial treatment (surgery, chemo, radiation). A recurrence event is closely "
        "related to disease-free survival and is one of the harder outcomes for models to predict.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
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

    _add_explanation(fig, [
        "What 'Event' and 'Censored' mean: When we study survival, we track whether each patient "
        "experienced a specific outcome (the 'event') by the time the study ended. 'Event' (red) "
        "means the outcome was observed — for Overall Survival, that means the patient died during "
        "the study. 'Censored' (blue) means the study ended before we could observe the outcome: "
        "the patient was still alive at their last clinic visit, or they left the study early. "
        "Censored patients are not failures or missing data — their data still contributes "
        "valuable information ('this person survived at least X months').",

        "Overall Survival (OS, top row): The most fundamental endpoint. The pie chart shows what "
        "fraction of patients died during follow-up vs. were still alive. The histogram shows "
        "how long patients were followed — each bar is a count of patients whose follow-up lasted "
        "that many months. The red dashed line marks the median follow-up time. Because TCGA-BRCA "
        "patients were enrolled over many years, follow-up lengths vary widely.",

        "Disease-Free Survival (DFS, middle): Measures the time from treatment until the cancer "
        "returns OR the patient dies — whichever comes first. This is stricter than OS because "
        "a patient whose cancer returned but who is still alive counts as an 'event' here. It "
        "answers: 'How long does treatment actually keep the disease at bay?'",

        "Progression-Free Survival (PFS, bottom): Similar to DFS, but the 'event' is any sign "
        "that the tumour is growing again, even if the patient is not yet symptomatic. It is "
        "commonly used in clinical trials to evaluate whether a drug is working before waiting "
        "for patients to actually die.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 4: survival endpoints")


def page_km_curves(pdf: PdfPages, clin: pd.DataFrame) -> None:
    """Kaplan-Meier overall-survival curves stratified by BRCA subtype."""
    if not HAS_LIFELINES:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.set_ylabel("Probability of Still Being Alive")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_explanation(fig, [
        "What a Kaplan-Meier (KM) curve is: Each line shows the probability that a patient of "
        "a given subtype is still alive as time passes. At month 0 (diagnosis), everyone is "
        "alive so all curves start at 1.0 (100%). Every time a patient in that subtype group "
        "dies, the curve drops by a small step. The curve never rises — once a patient has died, "
        "that is not reversed. A curve that stays high for longer means patients in that subtype "
        "tend to survive longer.",

        "The staircase shape: The curve drops in discrete steps rather than a smooth line because "
        "deaths happen at specific moments in time. Each step down represents one or more patients "
        "dying at that time point. Between deaths, the curve stays flat because no new information "
        "has arrived.",

        "The shaded bands: Around each line is a shaded confidence interval. This represents "
        "statistical uncertainty — the true survival probability for all possible patients of "
        "that subtype lies within the shaded region with 95% confidence. Wider bands mean fewer "
        "patients in that group (less data = more uncertainty). Bands that overlap between "
        "subtypes mean the difference in survival is not statistically clear-cut.",

        "What to look for: Luminal A (blue) tends to have the best long-term survival; Basal-like "
        "(red) tends to drop fastest in the early years. These differences are precisely what our "
        "model is trying to learn from the RNA-seq data — ideally, gene expression alone should "
        "be enough to reproduce and refine this subtype-level ordering at the individual patient level.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
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
        axes[2].set_xlabel("Score (>10 = MSI-High)")
        axes[2].set_ylabel("Samples")
        axes[2].axvline(10, color="crimson", linestyle="--",
                        linewidth=1.2, label="MSI-High threshold (10)")
        axes[2].legend(fontsize=7)

    _add_explanation(fig, [
        "Tumour Mutational Burden — TMB (left): Counts how many DNA mutations exist per million "
        "base pairs of DNA in the tumour, counting only mutations that change a protein "
        "('nonsynonymous'). A higher TMB means the tumour's DNA repair machinery was more broken, "
        "allowing errors to accumulate. Very high TMB (>10) is clinically important because those "
        "tumours often respond well to immunotherapy drugs — the immune system can 'see' the "
        "many mutant proteins. Most breast cancers have low-to-moderate TMB; a long right tail "
        "in this histogram represents the rare hypermutated tumours.",

        "Aneuploidy Score (middle): Normal human cells have 46 chromosomes arranged in 23 pairs. "
        "Cancer cells often gain or lose entire chromosomes or large chromosomal arms — this is "
        "called aneuploidy. The score counts how many chromosomal arms have abnormal copy numbers. "
        "A score of 0 means the tumour's chromosomes are largely intact. High scores indicate "
        "chaotic genomes that have lost control of cell division machinery. Aneuploidy tends to "
        "correlate with higher grade and worse prognosis.",

        "MSI Sensor Score (right): Microsatellites are short repetitive DNA sequences scattered "
        "throughout the genome. When DNA mismatch repair is defective, these regions mutate "
        "especially rapidly — this is called Microsatellite Instability (MSI). The red dashed "
        "line at 10 is the clinical cut-off: samples above it are classified as MSI-High (MSI-H) "
        "and are eligible for certain immunotherapy treatments. Breast cancer is rarely MSI-H "
        "(unlike colorectal cancer), so most samples cluster near zero.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
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

    # Per-sample distribution (box over a random subset of 80 samples)
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

    _add_explanation(fig, [
        "What RNA-seq measures: Every cell in the body contains the same DNA, but different genes "
        "are turned on or off depending on the cell type and its state. RNA sequencing measures "
        "which genes are active (being 'read') and by how much. The result is a number per gene "
        "called RSEM — roughly, how many RNA molecules from that gene were found in the sample. "
        "A high RSEM value for a gene means that gene is highly active in that tumour.",

        "Why log₂ transform (left plot): Raw RSEM values span an enormous range — a highly "
        "expressed gene might have a value 10,000× higher than a low-expressed one. Plotting or "
        "training on raw values would make the few ultra-high genes dominate everything. Taking "
        "log₂(value + 1) compresses this range so that differences at all expression levels are "
        "treated more equally. The '+1' prevents log(0) errors for genes with zero reads.",

        "Per-sample distribution (left): Each vertical box shows the spread of expression values "
        "across all genes for one tumour sample. The red line inside each box is the median gene "
        "expression for that sample. The boxes look similar across samples, which is reassuring — "
        "it means there are no wildly outlier samples with systematically different expression "
        "that would bias the model.",

        "Top 20 most variable genes (right): Of the ~20,000 genes measured, most barely change "
        "between patients — they are 'housekeeping' genes that every cell needs at about the same "
        "level. The genes with the highest variance across patients are the most informative for "
        "distinguishing tumour subtypes and predicting outcomes. Our model uses the top 500 most "
        "variable genes as its input features. The genes shown here (e.g. MUCL1, FABP7) are "
        "known markers of breast cancer biology.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[explore] ✓ Page 7: expression overview")


def page_data_preprocessing(pdf: PdfPages) -> None:
    """Text page: full data pre-processing pipeline with plain-language explanations."""
    _text_page(
        pdf,
        title="Data Pre-Processing Pipeline",
        subtitle="How raw TCGA-BRCA data is cleaned and transformed before model training",
        sections=[
            (
                "Why Pre-Processing Matters",
                [
                    "Raw genomic data is never ready to hand directly to a machine-learning model. "
                    "It contains measurement artefacts, wildly different numerical scales across "
                    "genes, and quality issues that have nothing to do with biology. Pre-processing "
                    "is the set of steps that turns messy raw data into a clean, consistently scaled "
                    "feature matrix that a model can learn from reliably. Getting pre-processing "
                    "right is often more important than the choice of model architecture — a "
                    "beautifully designed neural network trained on poorly prepared data will "
                    "consistently underperform a simpler model trained on clean data.",

                    "The most critical rule in survival analysis pre-processing is the "
                    "'no-leakage' constraint: any statistics computed from the data (means, "
                    "variances, gene rankings) must be computed on the training fold only, then "
                    "applied to the test fold without revision. Violating this rule makes "
                    "evaluation results overly optimistic — the model appears to perform better "
                    "than it actually does on new patients.",
                ],
            ),
            (
                "Step 1 — Primary Tumour Sample Selection",
                [
                    "TCGA-BRCA contains expression profiles from both tumour tissue and, in some "
                    "cases, matched normal tissue from the same patient. Normal tissue samples act "
                    "as a baseline for understanding what 'healthy' breast cells look like. "
                    "Because we are modelling tumour-driven prognosis, we keep only primary tumour "
                    "samples (identified by the '-01' suffix in the TCGA barcode). Normal tissue "
                    "samples ('-11') are discarded.",

                    "A small number of patients also have multiple primary tumour samples — for "
                    "example, if a second biopsy was taken after treatment. For these patients, "
                    "we average the expression values across all their primary samples to produce "
                    "a single patient-level expression profile. This is the standard approach "
                    "in TCGA-BRCA survival literature.",
                ],
            ),
            (
                "Step 2 — Zero Survival-Time Removal (QC addition)",
                [
                    "A small number of TCGA-BRCA patients have a recorded Overall Survival time "
                    "of exactly zero months. These are almost certainly administrative or data-entry "
                    "artefacts — a patient cannot be enrolled in a study, have tissue collected, "
                    "have their RNA sequenced, and then die zero months later in any clinically "
                    "meaningful sense. Keeping these records would inject mathematically "
                    "degenerate rows into the Cox loss function: the first risk set (all patients "
                    "'at risk' at time zero) collapses to a single patient, making the log-"
                    "likelihood numerically extreme. These patients are removed before any "
                    "further analysis.",
                ],
            ),
            (
                "Step 3 — PCA-Based Expression Outlier Removal (QC addition)",
                [
                    "Even after standard TCGA quality checks, a small number of samples can "
                    "have systematically aberrant expression profiles — possibly due to RNA "
                    "degradation during sample storage, sequencing failures, or sample "
                    "contamination. These 'expression outliers' can distort the gene-selection "
                    "and normalisation statistics computed in later steps, affecting every "
                    "patient in the cohort.",

                    "To identify outliers, we first apply a log₂ transformation to the full "
                    "expression matrix, then run PCA (Principal Component Analysis) to "
                    "compress the ~20,000-gene space into the 10 largest axes of variation. "
                    "We then compute, for each axis, how far each patient's score lies from "
                    "the median of all patients, measured in MADs (median absolute deviations "
                    "— a robust version of 'standard deviations'). Patients whose score on "
                    "any axis exceeds 5 MADs from the median are flagged as outliers and "
                    "removed. The 5-MAD threshold is deliberately conservative — it catches "
                    "only extreme technical artefacts, not genuine biological extremes.",
                ],
            ),
            (
                "Step 4 — Survival Time Clipping at 10 Years (QC addition)",
                [
                    "A small proportion of TCGA-BRCA patients have follow-up times exceeding "
                    "10 years (120 months). These very long follow-up observations exert "
                    "disproportionate leverage during model training: the model can waste "
                    "capacity trying to rank patients who survived 12 years against those "
                    "who survived 13 years — a clinically irrelevant distinction supported "
                    "by very few data points. This is known as the 'long-tail leverage' problem.",

                    "We apply administrative censoring at 120 months: any patient with a "
                    "follow-up time beyond 10 years is treated as censored at 10 years, "
                    "regardless of their actual recorded outcome. A patient who died at "
                    "month 150 becomes 'censored at month 120' — we do not falsify their "
                    "outcome, we simply stop observing them at the cut-off, exactly as if "
                    "the study had ended then. This is the standard treatment in TCGA-BRCA "
                    "survival publications and focuses the model's attention on the "
                    "clinically actionable 0–10 year horizon.",
                ],
            ),
            (
                "Step 5 — Log₂ Transformation",
                [
                    "Raw RSEM values (the unit in which RNA-seq expression is delivered by "
                    "cBioPortal) span an enormous dynamic range — a highly expressed gene "
                    "might have a value of 50,000 while a lowly expressed gene has a value "
                    "of 0.5. If we handed raw values to a neural network, the few "
                    "ultra-high-expression genes would numerically dominate every computation, "
                    "preventing the model from learning from the thousands of moderately "
                    "expressed genes that collectively carry much of the prognostic signal.",

                    "Taking log₂(RSEM + 1) compresses this range dramatically — a value of "
                    "50,000 becomes ~15.6, while 0.5 becomes ~0.58 — so that differences at "
                    "all expression levels are treated more equally. The '+1' prevents "
                    "log(0) errors for genes with zero reads. All downstream steps (gene "
                    "selection, normalisation) operate on this log-transformed scale.",
                ],
            ),
            (
                "Step 6 — Minimum Expression Filter (QC addition)",
                [
                    "Of the ~20,000 genes in the RSEM matrix, many are expressed at near-zero "
                    "levels in almost every patient. These genes add noise to variance "
                    "calculations because their small non-zero values in a handful of samples "
                    "can produce an artificially high variance, causing them to be selected "
                    "as 'informative' features when they are actually measurement noise.",

                    "Before computing gene variance, we filter out genes that are not "
                    "'expressed' in at least 20% of training patients. A gene is considered "
                    "expressed in a sample if log₂(RSEM + 1) > 1, corresponding to an RSEM "
                    "value above 1. This filter is applied to training data only — the "
                    "selected gene list is then fixed and applied to test data without "
                    "re-evaluation.",
                ],
            ),
            (
                "Step 7 — Variance-Based Gene Selection",
                [
                    "After filtering the gene universe, we compute the variance of each "
                    "remaining gene's log-expression across training patients and select "
                    "the top 500 most variable genes. High-variance genes are those that "
                    "differ most between patients — these are the genes most likely to "
                    "carry patient-specific prognostic information. Low-variance 'housekeeping' "
                    "genes (like ribosomal proteins) are active at roughly the same level in "
                    "every patient and contribute nothing to distinguishing high-risk from "
                    "low-risk individuals.",

                    "The 500-gene count is a hyperparameter chosen to balance information "
                    "content against model complexity given the ~900 training patients "
                    "available per fold. Selecting too many genes risks overfitting; "
                    "selecting too few loses signal. The threshold k=500 is consistent with "
                    "published TCGA-BRCA survival baselines.",
                ],
            ),
            (
                "Step 8 — Robust Z-Score Normalisation (improved from mean/std)",
                [
                    "After gene selection, each gene's values are standardised so they are "
                    "on a comparable numerical scale for the neural network. Previously this "
                    "used standard z-scoring (subtract the mean, divide by the standard "
                    "deviation). We now use robust z-scoring: subtract the median, divide "
                    "by the MAD (median absolute deviation).",

                    "The improvement matters because both the mean and the standard deviation "
                    "are sensitive to outlier patients. If even one patient has an extreme "
                    "expression value for a gene, the mean is pulled toward that patient and "
                    "the standard deviation is inflated — causing every other patient's "
                    "normalised value to be slightly wrong. The median and MAD are "
                    "'breakdown-resistant': up to half the patients in a fold would need to "
                    "be outliers before the centre or scale estimates become meaningless. "
                    "This is especially valuable on small training folds (~720 patients) "
                    "where a single technical artefact could otherwise bias the entire fold's "
                    "normalisation.",

                    "All statistics (median, MAD) are computed on the training fold only and "
                    "applied to the test fold — preserving the no-leakage guarantee that "
                    "makes our C-index estimates trustworthy.",
                ],
            ),
        ],
    )
    print("[explore] ✓ Preprocessing page: data pre-processing pipeline")


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

    # Top 25 most frequently mutated genes
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
    ax.set_xlabel("% Tumours with a Mutation in This Gene")
    ax.set_title("Top 25 Mutated Genes\n(red = mutated in ≥5% of tumours)")
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(5, color="crimson", linestyle="--", linewidth=0.8, alpha=0.6)

    # Variant classification breakdown
    ax = axes[1]
    vc_counts = mut["Variant_Classification"].value_counts()
    cmap = plt.get_cmap("Set2")
    colours_vc = [cmap(i / max(len(vc_counts) - 1, 1)) for i in range(len(vc_counts))]
    ax.barh(vc_counts.index[::-1], vc_counts.values[::-1], color=colours_vc[::-1])
    ax.set_xlabel("Number of Mutations")
    ax.set_title("Mutation Type Breakdown")
    ax.tick_params(axis="y", labelsize=8)

    _add_explanation(fig, [
        "Somatic mutations are DNA changes that happened in the tumour cell during a person's "
        "lifetime — they are not inherited and not present in normal cells. Finding which genes "
        "are mutated, and in how many patients, helps identify the 'drivers' of cancer growth.",

        "Top mutated genes (left): Each bar shows what percentage of tumours have at least one "
        "mutation in that gene. Bars in red are mutated in 5% or more of tumours — these are "
        "likely driver genes. TP53 is the most commonly mutated gene in cancer overall; it "
        "normally acts as the cell's 'guardian of the genome', stopping damaged cells from "
        "dividing. PIK3CA encodes a key protein in a signalling pathway that controls cell "
        "growth; mutations here are especially common in Luminal A tumours and can be targeted "
        "with specific drugs.",

        "Mutation type breakdown (right): Not all mutations have the same effect. A Missense "
        "Mutation changes one amino acid in the resulting protein — the protein still forms but "
        "may behave differently. A Nonsense Mutation introduces a premature stop signal, "
        "producing a shortened, usually non-functional protein. A Frame Shift (insertion or "
        "deletion) shifts the entire reading frame downstream, almost always destroying the "
        "protein's function. Silent Mutations change the DNA but produce the same amino acid "
        "and usually have no functional effect. Splice Site Mutations affect how the gene is "
        "processed into RNA and can alter or destroy the protein.",
    ])

    fig.tight_layout(rect=[0, _PANEL_HEIGHT_FRAC, 1, 1])
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
        page_project_overview(pdf)
        page_design_choices(pdf)
        page_data_preprocessing(pdf)
        page_inventory(pdf, data_dir)
        page_demographics(pdf, clin)
        page_cancer_characteristics(pdf, clin, samp)
        page_survival(pdf, clin)
        page_km_curves(pdf, clin)
        page_genomic_instability(pdf, samp)
        page_expression(pdf, data_dir)
        page_mutations(pdf, data_dir)

        info = pdf.infodict()
        info["Title"] = "TCGA-BRCA PanCancer Atlas 2018 — EDA Report"
        info["Author"] = "PathoGems Stage 2 explore_brca.py"
        info["Subject"] = "Exploratory data analysis"

    print(f"\n[explore] Done. Report written to {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

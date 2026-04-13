# ADR 0001 — Omics-only MLP as the first baseline

- **Date:** 2026-04-13
- **Status:** Accepted

## Context

The reference architecture (PathoGems) is a multimodal model combining
whole-slide pathology images (WSI) with three genomics modalities (RNA-seq,
CNV, somatic mutation) via a histology-guided co-attention fusion. The brief's
example Stage 3 config reflects that full pipeline.

However, a realistic WSI pipeline requires:

1. Downloading hundreds of gigabytes of SVS slides from GDC (multi-day).
2. Tiling each slide into ~10k patches at 20x magnification.
3. Running a CNN (ResNet-50 or UNI) to extract per-patch features.
4. Storing those features on disk in a retrievable layout.

None of that is hard, but it is a multi-day engineering effort that delivers
no learning signal about *our own* model or training loop until it completes.
Meanwhile, the brief explicitly prioritizes a "not complex first draft"
baseline that we can iterate from by changing one variable at a time.

## Decision

The first baseline in Stage 3 will be an **omics-only multi-layer perceptron
(MLP) predicting overall survival on TCGA-BRCA**, trained on the top-N most
variable RNA-seq genes. No WSI, no CNV, no mutation data in the first
iteration.

Concretely:

- Cohort: TCGA-BRCA (the largest of the four cohorts in the brief; ~1000
  patients with RNA-seq + clinical available via cBioPortal).
- Input: Top-500 most-variable protein-coding genes (log2(TPM + 1),
  z-scored per gene on the training fold only, to prevent leakage).
- Model: 2-layer MLP (500 → 128 → 32 → 1) with ReLU, dropout 0.3, BatchNorm.
- Output: Scalar risk score (no sigmoid — the Cox loss is scale-invariant).

## Consequences

### Positive

- **Fast feedback loop.** Training completes in seconds on CPU, so we can
  exercise the whole harness (config → data → model → loss → metric → log →
  Stage 4 analysis) in a single session and catch integration bugs early.
- **Defensible baseline number.** Omics-only MLP survival models are a
  standard baseline in the literature (Chen et al. PORPOISE 2022; Wang et al.
  TMI 2021). Reporting an omics-only C-index gives every subsequent
  multimodal experiment an honest yardstick — if WSI + omics doesn't beat
  omics alone on our pipeline, something is wrong.
- **Decoupled from Stage 2 WSI work.** We can build the WSI feature extractor
  in parallel without blocking Stage 3 iteration.

### Negative

- The baseline will not match PathoGems' reported C-index, because PathoGems
  *is* multimodal. This is expected and explicitly called out in every run
  log so the Stage 4 analysis agent does not falsely flag it as a regression.

### Follow-ups

- ADR 0002: Loss choice (Cox PH).
- ADR 0003: Where the omics data comes from (cBioPortal).
- A later ADR will document the transition to multimodal once WSI features
  exist and the omics-only number is stable.

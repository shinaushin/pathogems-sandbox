# ADR 0003 — cBioPortal as the Stage-2-lite data source for omics + clinical

- **Date:** 2026-04-13
- **Status:** Accepted

## Context

The Stage 2 agent in the brief targets the GDC API and downloads WSIs, RNA-seq,
CNV, and mutations for the full multimodal pipeline. For the omics-only
baseline (ADR 0001) we only need RNA-seq expression + clinical survival
metadata for TCGA-BRCA, and we need it quickly and reproducibly.

The three realistic sources:

1. **GDC Data Portal API** — authoritative, but returns raw per-sample STAR
   counts files. Harmonizing into a gene-by-sample matrix requires extra
   processing (loading hundreds of TSVs, summing over transcripts, mapping
   Ensembl IDs to HGNC symbols).
2. **cBioPortal API** — curates TCGA PanCancer Atlas studies into ready-made
   gene-by-sample matrices (`data_mrna_seq_v2_rsem.txt`) and clinical
   tables with standardized `OS_STATUS` / `OS_MONTHS`. Public, no auth.
3. **UCSC Xena** — similar curation; excellent for exploration, but their
   matrix format is less stable across releases.

## Decision

Use **cBioPortal's bulk data files** (fetched via their documented S3 URLs,
one zip per study) for the omics-only baseline. Concretely we will target
the `brca_tcga_pan_can_atlas_2018` study.

## Rationale

- **Ready-made matrices.** Gene × sample RSEM matrix in a single TSV, no
  per-sample loop. One afternoon of code vs. one week.
- **Clinical harmonization.** `OS_STATUS` ("0:LIVING" / "1:DECEASED") and
  `OS_MONTHS` are already computed, so there is no ambiguity about which
  clinical column maps to time-to-event.
- **Reproducibility.** The PanCancer Atlas 2018 release is a frozen,
  versioned snapshot; a URL fetch today and a URL fetch in 2027 will return
  the same data. GDC files can shift under harmonization pipeline updates.
- **Leaves the brief's Stage 2 agent intact.** The full-pipeline Stage 2
  script (GDC + manifests + WSI download scripts) stays as planned; this is
  an *additive* "Stage 2 lite" script that only covers the omics baseline
  and can be deleted later without disturbing the rest.

## Consequences

### Positive

- Baseline unblocks Stage 3 development today.
- Single-file data product (one TSV + one clinical TSV) is trivial to cache
  and to include in manifests.

### Negative

- cBioPortal's RSEM matrix is already RSEM-normalized; we do not have raw
  counts to redo normalization. For the baseline this is fine — we use
  log2(x + 1) + z-score per gene. Future experiments that want to
  experiment with different normalizations (e.g., DESeq2 VST) will need to
  go to GDC raw data. That ADR is deferred until it matters.
- cBioPortal uses HGNC symbols only. If a later experiment requires Ensembl
  IDs (for pathway mapping, say) we will pay the cost to remap.

### Follow-ups

- ADR when we switch to raw GDC counts (if ever).
- The `stage2_data/fetch_cbioportal_brca.py` script writes both the TSVs and
  a small manifest JSON documenting the exact study ID, URL, and SHA256 of
  the downloaded bundle for reproducibility.

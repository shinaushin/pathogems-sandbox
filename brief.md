# PathoGems Auto-Research Agent — Project Brief

> This is the original brief as pasted into the Cowork session. It is kept
> verbatim for reference. Any deviations from the brief are documented in
> [`docs/decisions/`](docs/decisions) as Architectural Decision Records.

## What This Project Is

An end-to-end AI research agent for multimodal cancer survival prediction,
using the PathoGems framework as the reference architecture. The dual goal:
do real ML research AND learn how to build agentic AI workflows using
Claude Cowork.

Research domain: Predict patient survival from paired whole-slide pathology
images (WSI) and genomics data (RNA-seq, CNV, somatic mutations) using deep
learning, applied to TCGA cancer cohorts.

Reference paper: "Deep learning-based multimodal pathogenomics integration
for precision cancer prognosis" (PathoGems)
https://www.sciencedirect.com/science/article/abs/pii/S1532046425000656

## Key Decisions from the Brief

- **Framework:** PathoGems chosen over an attention-based survival framework
  because data is fully public (GDC, cBioPortal, TCIA) and the architecture
  is modular with many axes to experiment on.
- **Cohorts:** BRCA (949 patients), CRC (335), GBMLGG (522), ESCA (159).
  Start with BRCA or GBMLGG (smaller, faster to iterate).
- **Inclusion criterion:** Only patients with all four modalities — WSI +
  RNA-seq + CNV + mutation — plus survival data. *(Relaxed for the first
  baseline — see ADR 0001.)*
- **Stack:** All 4 stages in Python, run via Claude Cowork on a local
  machine.
- **Agent framework:** No heavy framework (LangChain, AutoGen) — use the
  Anthropic Python SDK directly so the logic is transparent and easy to
  debug.

## The Four Stages

1. **Stage 1 — Literature Review Agent.** Given a query, search the web and
   return structured output (key papers, methods, open problems, testable
   hypotheses, next steps). Saves JSON to `stage1_literature/outputs/`.
2. **Stage 2 — Data Acquisition Agent.** Query GDC and cBioPortal APIs for
   TCGA patients with all required modalities. Produce a validated
   manifest CSV and download scripts for WSIs.
3. **Stage 3 — Experiment Agent.** Accept an experiment config, train a
   model, log structured metrics. *This is the focus of the initial
   implementation.*
4. **Stage 4 — Results & Iteration Agent.** Read logs, summarize, generate
   next experiment config. Closes the loop back to Stage 3.

## Reference Links

| Resource | URL |
|----------|-----|
| PathoGems paper | https://www.sciencedirect.com/science/article/abs/pii/S1532046425000656 |
| GDC API docs | https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/ |
| GDC portal | https://portal.gdc.cancer.gov |
| cBioPortal | https://www.cbioportal.org |
| TCIA (WSIs) | https://www.cancerimagingarchive.net |
| TIL-WSI-TCGA feature maps | https://www.cancerimagingarchive.net/analysis-result/til-wsi-tcga/ |

## Author directives

- Progress pushed and committed in bite-sized pieces, good SW practice.
- Output at the proficiency of a senior ML engineer.
- Every step and decision well-documented and justified, code
  well-commented.
- Easily understandable to other engineers in the field.
- Stage 3 is the initial focus. Start with a non-complex first-draft
  baseline, then change one thing at a time.

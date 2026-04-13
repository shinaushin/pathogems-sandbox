# How to run Stage 3 locally

The sandbox used in this Cowork session can't install `torch` /
`scikit-learn` / `pytest`, so every commit here is syntax-checked but
not executed. Below is the exact sequence to run the harness on your
own machine and produce the first baseline run log.

## 1. Create the conda env

```bash
cd pathogems-agent
conda env create -f environment.yml
conda activate pathogems
```

## 2. Install the Stage 3 package in editable mode

```bash
cd stage3_experiments
pip install -e .
cd ..
```

## 3. Run the tests

```bash
pytest stage3_experiments/tests -q
```

Expected output: all tests pass. The end-to-end smoke test
(`test_smoke_end_to_end.py`) takes ~5–10 seconds on CPU and confirms the
full harness can recover a synthetic Cox signal end-to-end.

## 4. Fetch the BRCA omics + clinical data

```bash
python stage2_data/fetch_cbioportal_brca.py
```

This downloads ~100 MB from cBioPortal and writes to
`stage2_data/raw/brca_tcga_pan_can_atlas_2018/`. Idempotent — safe to
re-run.

## 5. Run the baseline experiment

```bash
pathogems-train --config stage3_experiments/configs/brca_omics_baseline.json
```

This runs 5-fold stratified CV on TCGA-BRCA with the ADR-0001 baseline
(omics-only MLP, Cox PH loss, top-500 variable genes). Expected to
complete in a couple of minutes on CPU. The run log is written to
`stage3_experiments/logs/brca_omics_baseline_run.json`.

Literature-reported omics-only C-index on TCGA-BRCA is typically in the
0.62–0.68 range; if the baseline lands there, the harness is working.

## 6. Commit the first real log as an artifact

```bash
git add stage3_experiments/logs/brca_omics_baseline_run.json
git commit -m "Baseline run log: BRCA omics-only MLP, Cox PH, 5-fold CV"
```

## 7. Subsequent experiments

Create a new JSON config next to the baseline, changing **one field** at
a time. Examples we expect to queue up:

- `brca_omics_mlp_wd0.json` — same as baseline with `weight_decay=0`
- `brca_omics_mlp_deeper.json` — `hidden_dims=[256,128,32]`
- `brca_omics_sgd.json` — `optimizer="sgd"`, same LR schedule
- `brca_omics_dropout05.json` — `dropout=0.5`

Each produces a run log. Stage 4 (not yet implemented) will consume the
logs directory and propose the next experiment.

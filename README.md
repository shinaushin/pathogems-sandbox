# PathoGems Auto-Research Agent

An end-to-end AI research agent for multimodal cancer survival prediction, using
the [PathoGems](https://www.sciencedirect.com/science/article/abs/pii/S1532046425000656)
framework as the reference architecture.

The project has two goals running in parallel:

1. **Do real ML research** on TCGA cancer cohorts — predict patient survival
   from paired whole-slide pathology images (WSI) and genomics data.
2. **Learn how to build agentic AI workflows** using Claude Cowork, structured
   across four stages: literature review, data acquisition, experiments, and
   results analysis.

See [`brief.md`](brief.md) for the full project brief and
[`docs/decisions/`](docs/decisions) for architectural decision records (ADRs)
explaining the "why" behind each choice.

## Status

Currently building the Stage 3 experiment harness. We are starting with the
simplest defensible baseline — an omics-only MLP predicting TCGA-BRCA overall
survival, trained with the Cox partial likelihood loss and evaluated by
Harrell's C-index under 5-fold cross-validation. From that baseline we will
change one variable at a time (architecture, loss, optimizer, modality, cohort)
and measure the effect on held-out C-index.

See [`docs/decisions/0001-omics-only-first-baseline.md`](docs/decisions/0001-omics-only-first-baseline.md)
for the reasoning behind starting here rather than jumping to full multimodal.

## Repository layout

```
pathogems-agent/
├── brief.md                     Project brief (pasted at session start)
├── README.md                    This file
├── environment.yml              Conda environment definition
├── pyproject.toml               Tool config (pytest, mypy, ruff)
├── docs/decisions/              Architectural Decision Records (ADRs)
├── stage1_literature/           Literature-review agent + JSON outputs
├── stage2_data/                 Data acquisition — manifests and raw data
├── stage3_experiments/          Training harness, models, configs, logs
│   ├── src/pathogems/           Installable Python package (data, model, train)
│   ├── configs/                 One JSON per experiment
│   ├── logs/                    Per-run JSON logs (consumed by Stage 4)
│   ├── checkpoints/             Model weights (best + last) — gitignored
│   └── tests/                   pytest suite
└── stage4_results/              Analysis reports + suggested next configs
```

## Quickstart

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate pathogems

# Install the stage3 package in editable mode so imports work
pip install -e stage3_experiments

# Run the test suite
pytest stage3_experiments/tests -q

# Run the baseline experiment (once data is available)
python -m pathogems.cli train --config stage3_experiments/configs/brca_omics_baseline.json
```

## Development conventions

- **One change per commit.** The whole point of the experiment loop is to
  isolate effects — if commits bundle unrelated changes, the loop breaks.
- **Tests before training runs.** Every new piece of logic in the harness
  (loss, metric, data split, etc.) ships with a unit test. Training code is
  cheap to run but expensive to trust.
- **ADRs for non-obvious choices.** Anything a future reader would reasonably
  question ("why this loss?", "why this split?") gets a short ADR under
  `docs/decisions/`.
- **Type hints everywhere, checked with mypy.** The harness is small enough
  that strict typing costs little and prevents a class of silent bugs (e.g.
  passing event indicators and times in the wrong order to the Cox loss).

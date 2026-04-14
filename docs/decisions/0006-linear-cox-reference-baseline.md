# 0006 — Keep a linear Cox reference baseline alongside the MLP baseline

Status: Accepted — 2026-04-13

## Context

ADR 0001 selected a small omics-only MLP as the first Stage 3 model.
In isolation, the MLP's headline C-index is hard to interpret: is 0.65
good? bad? a bug? The field's convention — followed by DeepSurv and
every subsequent paper we intend to compare to — is to report a
*linear* Cox regression on the same features as a floor. If the MLP
doesn't clear that floor, we either have a harness bug or no non-linear
signal to extract. Either way, we want to know.

## Decision

Register a `LinearCox` model (`nn.Linear(in_features, 1)`, no hidden
layers, no BN, no dropout) in `model.py` and ship a companion config
`brca_linear_cox_baseline.json`. It uses the same preprocessing, the
same 5-fold split, the same Cox PH loss, and the same CV harness as
the MLP baseline — only the model differs. Every future model config
is expected to be compared against both this linear floor and the MLP
baseline.

## Consequences

- Every Stage 4 experiment-ranking step has two reference points, not
  one. The "is it working?" signal gets substantially sharper.
- LinearCox ignores `hidden_dims`, `dropout`, and `use_batchnorm` from
  the config. We document this inside the factory and keep those
  fields in the config unchanged, because the harness treats config
  swaps as a one-field change and we don't want "switch model" to
  require editing unrelated fields.
- We use a slightly higher LR (1e-3) and weight decay (1e-3) for the
  linear model than for the MLP — the MLP's batchnorm + dropout act as
  implicit regularizers, so the linear model needs explicit L2 and a
  bigger step size to converge in the same number of epochs.

## Alternatives considered

- **Regularized linear Cox via scikit-survival (CoxnetSurvivalAnalysis).**
  More literally the "classical baseline", but it runs outside our
  torch training loop and wouldn't share the same fold tensors or run
  log. Harder to compare cleanly. We may add it as a separate
  non-NN baseline in a later ADR.
- **Skip the linear baseline.** Saves a config file but leaves the
  MLP number hard to interpret. Not worth it.

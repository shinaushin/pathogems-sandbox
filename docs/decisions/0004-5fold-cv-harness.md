# ADR 0004 — 5-fold stratified cross-validation as the evaluation harness

- **Date:** 2026-04-13
- **Status:** Accepted

## Context

TCGA-BRCA has ~1000 patients with RNA-seq + survival, of which typically
~15% experience the event (death) within follow-up. With that event rate,
a single train/validation split is noisy: a lucky or unlucky test fold can
move the reported C-index by ±0.05, which is larger than most real model
deltas we care about. That makes "change one thing at a time" unreliable.

## Decision

Every experiment is evaluated by **5-fold cross-validation, stratified by
event indicator**, and the reported number is the mean C-index across folds
together with its bootstrap 95% confidence interval from the per-fold
scores. The fold index is seeded deterministically from the experiment
config so two runs of the same config partition patients identically.

The train split inside each fold is further split 90/10 into train /
validation for early stopping and learning-rate scheduling; only the held-out
test fold contributes to the reported C-index.

## Rationale

- **Noise reduction.** Averaging across 5 folds cuts the standard error of
  the reported metric by √5 ≈ 2.2 — enough to resolve typical
  architecture / loss deltas (~0.01 C-index) with a handful of runs.
- **Stratification by event.** Random CV folds can accidentally land on
  folds with 5% or 25% event rate, which blows up the C-index variance.
  Stratifying on the event indicator keeps event rate roughly constant
  across folds.
- **Deterministic seeding.** Making the fold assignment a pure function of
  the config means reruns are byte-identical. This is essential for
  debugging: if Stage 4 flags an experiment as anomalous, we can reproduce
  the exact fold split.
- **Nested val split for early stopping.** Using the held-out test fold
  for early stopping would be a leak; the inner 90/10 split prevents that
  without reducing test-fold size enough to matter.

## Consequences

### Positive

- A single experiment is 5× more expensive than a single train/test run,
  but the baseline trains in seconds, so the total cost is still seconds.
- Every run writes per-fold metrics *and* the aggregate, so Stage 4 can
  compute significance across experiments (e.g., paired t-test on fold
  C-indices).

### Negative

- Slightly more code complexity for the trainer. We encapsulate it in a
  `CrossValidatedTrainer` that wraps the single-fold trainer — see
  `pathogems/train.py`.
- 5 folds is a convention, not a truth. For very small cohorts (e.g.,
  ESCA with 159 patients) we may want 10-fold to keep test folds usable.
  The harness accepts `n_folds` as a config field; the *default* is 5.

# 0007 — Full-batch Cox PH training as the default

Status: Accepted — 2026-04-13, supersedes the mini-batch default in 0001's companion config

## Context

The Cox partial likelihood is defined over the *population* risk set:
at each event time, the denominator is a sum over *every* patient
still at risk. Mini-batching breaks this: each batch pretends the risk
set is only the patients in that batch. For small cohorts the
approximation is poor — a batch of 128 out of ~1000 patients changes
every risk-set sum by a factor of roughly 8×, and the loss landscape
no longer matches what we ultimately evaluate with.

TCGA omics cohorts (~300-1200 patients × ~500 genes after filtering)
fit comfortably on CPU as a single batch. There's no compute reason
to mini-batch.

## Decision

1. `ExperimentConfig.batch_size` becomes `int | None` with `None` as
   the default. `None` means "use the entire training fold as one
   batch".
2. Both baseline configs (`brca_omics_baseline.json`,
   `brca_linear_cox_baseline.json`) are updated to `"batch_size": null`.
3. An explicit integer still works — this is a pure additive change
   to the config surface.

## Consequences

- All baseline numbers going forward are measured under the "correct"
  Cox risk set. Any prior mini-batch results we collected are no
  longer directly comparable; as of this commit none had been logged
  yet, so there is nothing to invalidate.
- With `batch_size=None` the DataLoader does no shuffling (a single
  batch has nothing to shuffle). We kept the DataLoader indirection
  to preserve one code path.
- BatchNorm under full-batch is still well-defined (batch > 1) and
  now sees more stable population-level statistics at every step.
- Future experiments that *want* to test mini-batch behavior (e.g.
  for noise-as-regularizer studies) simply set an explicit int — no
  harness change required.

## Alternatives considered

- **Keep mini-batch default.** Easier rollout, but leaves the subtle
  risk-set mismatch in place. Every paper that has serious Cox PH
  results uses full-batch or careful batching with explicit risk-set
  tracking; copying the DeepSurv README's `batch_size=128` was the
  wrong choice.
- **Batch-aware risk-set computation inside the loss.** Keeps
  mini-batch compatibility, but the implementation is much trickier
  and the failure mode (patients in a batch see only a subset of the
  true risk set) is still approximated. Not worth the complexity for
  cohort sizes we actually handle.

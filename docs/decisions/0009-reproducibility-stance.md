# 0009 — Reproducibility stance: deterministic seeding, no bit-exact guarantee

Status: Accepted — 2026-04-14

## Context

PyTorch offers `torch.use_deterministic_algorithms(True)` which forces
every operation to use a deterministic implementation, raising
`RuntimeError` if one is not available. This gives *bit-exact*
reproducibility across identical hardware + driver combos.

The catch: several ops we rely on (or may rely on) either do not have
deterministic implementations or have slower deterministic alternatives:

- `torch.nn.functional.batch_norm` with CUDA (fallback exists on CPU)
- `scatter_add_` (used by some loss formulations)
- `index_select` backward pass

Enabling deterministic mode breaks or slows training in non-obvious
ways, and still does *not* reproduce across different hardware, CUDA
versions, or cuBLAS/cuDNN builds.

## Decision

1. **Seed everything explicitly.** `ExperimentConfig.seed` feeds into:
   - `torch.manual_seed(seed + fold_id)`
   - `np.random.seed(seed + fold_id)`
   - `sklearn`'s `random_state` for splits
   So the same config on the same machine produces the same C-index
   within floating-point noise.

2. **Do not enable `torch.use_deterministic_algorithms(True)`.**
   The cost (breakage, debugging false-positive RuntimeErrors) exceeds
   the benefit for a research harness. We care about
   *experiment-level* reproducibility ("does re-running the config
   give the same C-index to 3 decimals?"), not bit-exact loss
   trajectories.

3. **Document the expectation.** Two runs of the same config on the
   same machine may differ at the 4th-5th decimal of C-index due to
   non-deterministic GPU atomics and cuDNN auto-tuner choices. This
   is within noise for survival-prediction benchmarks, where inter-fold
   variance is typically 0.02-0.05.

4. **Lock the environment.** `environment.yml` pins major.minor
   versions of torch, numpy, and scikit-learn. Patch-level variation
   may change floating-point results but not conclusions.

## Consequences

- Results are reproducible *enough* for controlled experiments ("change
  one thing at a time") without the maintenance burden of deterministic
  mode.
- If a future experiment requires strict bit-exactness (e.g. for a
  reproducibility audit), the contributor adds a one-line
  `torch.use_deterministic_algorithms(True)` and debugs any resulting
  errors on their specific hardware — this is a per-environment
  decision, not a repo-wide one.
- GPU runs (CUDA) will see more inter-run variance than CPU runs.
  The baseline is designed to run on CPU, so this does not affect the
  first experiments.

## Alternatives considered

- **Always enforce deterministic mode.** Appealing on paper but
  unreliable across torch versions and hardware. Would create a
  maintenance burden disproportionate to its value for this project's
  current scope.
- **Pin patch-level versions of everything.** Pins stale fast and
  block security patches. Major.minor pins are the right granularity.

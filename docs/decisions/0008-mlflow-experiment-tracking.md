# 0008 — MLflow as an optional observability layer for Stage 3

Status: Accepted — 2026-04-13

## Context

By the time Stage 4 starts proposing experiments, we expect ~10-50 runs
sitting in `stage3_experiments/logs/*.json`. Scanning that by hand —
"what did I try, which lr won?" — gets old fast, and a plot of C-index
vs. hyperparameter is exactly the kind of thing an experiment-tracking
tool is for.

MLflow is the default choice in industry survival-analysis tooling
(e.g. used in the Hugging Face Biomedical benchmarks) and is simple to
self-host. The main risk in adding it is *creep*: MLflow becoming a
second source of truth alongside the JSON run logs, with people
checking in only one or the other.

## Decision

1. MLflow is **optional and opt-in**:
   - `ExperimentConfig.enable_mlflow: bool = False` (default off).
   - `mlflow_tracking_uri: str | None = None` (None => MLflow default
     `./mlruns` directory).
   - `mlflow_experiment_name: str = "pathogems"`.
2. Tracking lives in its own module (`tracking.py`) with **one**
   place allowed to `import mlflow`. The rest of the codebase imports
   the `track_run` context manager, which yields either a
   `_MLflowTracker` or a `_NullTracker` depending on config.
3. If `enable_mlflow=True` but mlflow is not installed, we log a
   warning and fall back to the null tracker. A real run is never
   killed by a broken observability sidecar.
4. The JSON run log remains the **source of truth** for the Stage 3 ->
   Stage 4 contract (ADR 0005). MLflow is an observer: when enabled,
   every run's JSON log is also attached to its MLflow run as an
   artifact, so the tracker has a superset of what MLflow alone has.

## Consequences

- Stage 4 continues to read `logs/*.json` and does not depend on
  MLflow being set up. Contributors without MLflow credentials can
  still run experiments and produce usable output.
- The fake-mlflow tests in `test_tracking.py` make the tracking
  module's internals exercisable without the real package, which
  matches the "every commit is syntax-checked but not executed" rule
  in our sandbox.
- Adding `mlflow>=2.12,<3.0` to `environment.yml` bumps image size
  but not dev ergonomics, because mlflow is pip-only and we already
  have pip in the env.

## Alternatives considered

- **Weights & Biases.** Nicer UI but requires an external account and
  network egress, which is a real blocker for contributors behind
  air-gapped setups.
- **Just keep using JSON logs.** Fine for 10 runs, painful for 50.
  The threshold will be crossed during this project's lifetime.
- **Require MLflow everywhere.** Simpler code path, but couples
  correctness of training runs to correctness of observability setup.
  Always the wrong trade for a research harness.

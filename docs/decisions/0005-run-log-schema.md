# ADR 0005 — Structured JSON run logs as the Stage 3 → Stage 4 contract

- **Date:** 2026-04-13
- **Status:** Accepted

## Context

Stage 4 (the results agent) reads every Stage 3 run log, summarizes them,
and asks Claude to propose the next experiment. The brief's example
implementation grep'd stdout for a "c-index" substring. That is fragile:
one print-statement rename breaks Stage 4, and stdout does not carry the
information Stage 4 actually needs (per-fold metrics, config diff vs
baseline, wall-clock, loss curve).

## Decision

Every experiment writes a single JSON file,
`stage3_experiments/logs/<run_name>_run.json`, conforming to a versioned
schema (`schema_version: 1`). Stage 4 reads from this file only — never
from stdout — and fails loudly if `schema_version` is unknown.

### Schema v1 (fields)

```
{
  "schema_version": 1,
  "run_name":          str,         # unique, matches config.name
  "config":            { ... },     # verbatim copy of the input config
  "git_sha":           str|None,    # commit the run was produced on
  "started_at":        str,         # ISO 8601 UTC
  "finished_at":       str,         # ISO 8601 UTC
  "wall_clock_sec":    float,
  "status":            "success" | "failed",
  "error":             str|None,    # traceback if failed
  "metrics": {
    "c_index_mean":    float,
    "c_index_std":     float,
    "c_index_folds":   [float, ...],   # length == n_folds
    "final_loss_mean": float,
    "final_loss_folds":[float, ...]
  },
  "environment": {
    "python":          str,
    "torch":           str,
    "cuda":            str|None,
    "hostname":        str
  },
  "notes":             str|None     # optional free-form commentary
}
```

## Rationale

- **Decoupled stages.** Stage 3 can change its internal training loop
  freely as long as it produces a schema-compliant log. Stage 4 only needs
  to know the schema.
- **Diffability.** All the primitives are JSON-serializable, so `git diff`
  on run logs is readable. The full config is embedded so logs are
  self-describing — a log file alone is enough to understand what ran.
- **Versioned schema.** When we add fields (e.g., per-epoch loss curves),
  we bump `schema_version` and teach Stage 4 to handle both. Unknown
  versions fail loudly rather than silently producing wrong summaries.
- **Environment capture.** `torch.__version__`, CUDA version, and hostname
  are captured automatically; these turn "my C-index went from 0.68 to
  0.64 overnight" from a mystery into a diff.

## Consequences

### Positive

- Stage 4 agent becomes a pure function of the logs directory. That is
  straightforward to test and to re-run offline.
- Downstream tooling (a future results dashboard, a regression tracker)
  can use the same schema without negotiation.

### Negative

- One more place to keep up to date when adding fields. We mitigate this by
  placing the write logic in one module (`pathogems.run_log`) with a pytest
  that round-trips every schema field.

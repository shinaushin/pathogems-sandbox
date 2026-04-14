"""Command-line entry point: `pathogems-train --config <path>`.

The CLI is intentionally tiny: parse args, load config, load data, run CV,
write run log, print a one-line summary. Anything more belongs in the
library modules where it can be tested without spinning up a subprocess.

Failure handling: if anything below config-load throws, we still write a
`status="failed"` run log with the traceback so Stage 4 sees an entry
for the run instead of silence. Config load itself fails fast — we do
not have a run name yet at that point, so there is nothing to log.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import assemble_cohort
from .run_log import write_run_log
from .tracking import track_run
from .train import cross_validate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pathogems-train",
        description="Train one Stage 3 experiment end-to-end and write a run log.",
    )
    p.add_argument("--config", type=Path, required=True, help="Path to experiment config JSON.")
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("stage3_experiments/logs"),
        help="Directory to write the JSON run log. Default: stage3_experiments/logs",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help='Torch device string ("cpu" or "cuda"). The baseline runs in seconds on cpu.',
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-fold progress lines.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Fail fast on config errors — we have no run name to log against yet.
    config = ExperimentConfig.from_json(args.config)

    started_at = datetime.now(UTC)
    # `track_run` yields a no-op tracker when enable_mlflow=False, so the
    # happy and tracked paths are identical code.
    with track_run(config) as tracker:
        try:
            study_dir = Path(config.study_data_dir)
            cohort = assemble_cohort(
                expression_path=study_dir / "data_mrna_seq_v2_rsem.txt",
                clinical_patient_path=study_dir / "data_clinical_patient.txt",
                clinical_sample_path=study_dir / "data_clinical_sample.txt",
                study_id=config.cohort,
            )
            print(
                f"[cli] Cohort {config.cohort}: {cohort.n_patients} patients, "
                f"{cohort.n_genes} genes, event rate {cohort.event_rate:.2%}"
            )
            tracker.log_metric("cohort_n_patients", float(cohort.n_patients))
            tracker.log_metric("cohort_n_genes", float(cohort.n_genes))
            tracker.log_metric("cohort_event_rate", float(cohort.event_rate))

            device = torch.device(args.device)
            result = cross_validate(cohort, config, device=device, verbose=not args.quiet)
            tracker.log_cv_result(result)

            finished_at = datetime.now(UTC)
            log_path = write_run_log(
                config=config,
                result=result,
                logs_dir=args.logs_dir,
                status="success",
                error=None,
                started_at=started_at,
                finished_at=finished_at,
            )
            # Attach the JSON run log as an MLflow artifact so the tracker
            # has everything the run log has — single pane of glass when
            # enabled, without sacrificing the run log as source of truth.
            tracker.log_artifact(log_path)
            print(
                f"[cli] DONE  C-index = {result.c_index_mean:.4f} +/- {result.c_index_std:.4f}  "
                f"(n_folds={config.n_folds})  log={log_path}"
            )
            return 0

        except BaseException as exc:  # noqa: BLE001 - we want to catch everything, log, re-raise
            # Any error past this point must produce a run log the schema
            # recognizes (status="failed"), otherwise Stage 4 will treat
            # the run as if it never happened.
            finished_at = datetime.now(UTC)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            try:
                log_path = write_run_log(
                    config=config,
                    result=None,
                    logs_dir=args.logs_dir,
                    status="failed",
                    error=tb,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                # Even on failure, attach what we have to the MLflow run
                # so the tracker reflects reality.
                tracker.log_artifact(log_path)
                print(f"[cli] FAILED  {type(exc).__name__}: {exc}  log={log_path}", file=sys.stderr)
            except Exception as log_exc:  # pragma: no cover - extremely defensive
                # If even writing the failure log fails, surface both errors.
                print(
                    f"[cli] FAILED and could not write run log. "
                    f"Original error: {type(exc).__name__}: {exc}. "
                    f"Log write error: {type(log_exc).__name__}: {log_exc}",
                    file=sys.stderr,
                )
            # Re-raise so shell `$?` reflects the failure and any wrapping
            # script sees a non-zero exit. `SystemExit` and `KeyboardInterrupt`
            # propagate naturally.
            raise


if __name__ == "__main__":
    sys.exit(main())

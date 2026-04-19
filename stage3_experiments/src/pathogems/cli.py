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

import logging

import torch

from .config import ExperimentConfig
from .data import assemble_cohort
from .run_log import write_run_log
from .tracking import track_run
from .train import cross_validate

log = logging.getLogger(__name__)


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


def _validate_study_dir(study_dir: Path) -> None:
    """Fail fast with a helpful message if the study directory is incomplete.

    Checks that the directory exists and that the three expected cBioPortal
    files are present. Running `assemble_cohort` against a bad path would
    eventually raise a cryptic pandas error; catching it here saves time.
    """
    expected_files = [
        "data_mrna_seq_v2_rsem.txt",
        "data_clinical_patient.txt",
        "data_clinical_sample.txt",
    ]
    if not study_dir.is_dir():
        raise FileNotFoundError(
            f"Study data directory does not exist: {study_dir}\n"
            f"Run `python stage2_data/fetch_cbioportal_brca.py` to download the data."
        )
    missing = [f for f in expected_files if not (study_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Study data directory {study_dir} is missing expected files: {missing}\n"
            f"Run `python stage2_data/fetch_cbioportal_brca.py` to re-download."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Configure logging early so all modules that use `logging` are wired up.
    logging.basicConfig(
        level=logging.DEBUG if not args.quiet else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Fail fast on config errors — we have no run name to log against yet.
    config = ExperimentConfig.from_json(args.config)

    study_dir = Path(config.study_data_dir)
    started_at = datetime.now(UTC)
    # `track_run` yields a no-op tracker when enable_mlflow=False, so the
    # happy and tracked paths are identical code.
    with track_run(config) as tracker:
        try:
            # Validate study directory before entering the (slow) training loop.
            _validate_study_dir(study_dir)
            cohort = assemble_cohort(
                expression_path=study_dir / "data_mrna_seq_v2_rsem.txt",
                clinical_patient_path=study_dir / "data_clinical_patient.txt",
                clinical_sample_path=study_dir / "data_clinical_sample.txt",
                study_id=config.cohort,
            )
            log.info(
                "Cohort %s: %d patients, %d genes, event rate %.2f%%",
                config.cohort,
                cohort.n_patients,
                cohort.n_genes,
                cohort.event_rate * 100,
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
            log.info(
                "DONE  C-index = %.4f +/- %.4f  (n_folds=%d)  log=%s",
                result.c_index_mean,
                result.c_index_std,
                config.n_folds,
                log_path,
            )
            return 0

        except BaseException as exc:  # we want to catch everything, log, re-raise
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
                log.error("FAILED  %s: %s  log=%s", type(exc).__name__, exc, log_path)
            except Exception as log_exc:  # pragma: no cover - extremely defensive
                # If even writing the failure log fails, surface both errors.
                log.error(
                    "FAILED and could not write run log. "
                    "Original error: %s: %s. Log write error: %s: %s",
                    type(exc).__name__,
                    exc,
                    type(log_exc).__name__,
                    log_exc,
                )
            # Re-raise so shell `$?` reflects the failure and any wrapping
            # script sees a non-zero exit. `SystemExit` and `KeyboardInterrupt`
            # propagate naturally.
            raise


if __name__ == "__main__":
    sys.exit(main())

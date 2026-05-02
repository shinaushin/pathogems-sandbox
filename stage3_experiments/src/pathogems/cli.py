"""Command-line entry points for pathogems training.

Two entry points are registered:

``pathogems-train`` (legacy, backward-compatible)
    Reads a JSON experiment config file via ``--config``.  All existing
    configs and tests continue to work without modification.

    Usage::

        pathogems-train --config stage3_experiments/configs/brca_omics_baseline.json

``pathogems-train-hydra`` (recommended)
    Hydra-based entry point.  Base defaults come from
    ``stage3_experiments/configs/base.yaml``; per-experiment overrides from
    ``stage3_experiments/configs/experiment/<name>.yaml``; any field can be
    further overridden on the command line.

    Usage::

        # Run a pre-defined experiment:
        pathogems-train-hydra +experiment=brca_omics_baseline

        # Override individual fields:
        pathogems-train-hydra +experiment=brca_omics_baseline lr=1e-3 epochs=50

        # Sweep with multirun (-m):
        pathogems-train-hydra -m +experiment=brca_omics_baseline,brca_omics_topk1000 \\
            lr=1e-4,1e-3

Failure handling: if anything past config-load throws, we still write a
``status="failed"`` run log so the results layer always has a record of the
attempt. Config load itself fails fast (no run name yet, nothing to log).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .data import (
    assemble_cohort,
    clip_survival_time,
    filter_zero_time_patients,
    remove_outlier_samples,
)
from .run_log import write_run_log
from .tracking import track_run
from .train import cross_validate

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


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


def _refresh_report(logs_dir: Path) -> None:
    """Regenerate the HTML experiment report after a successful run.

    The report script and output directory are located relative to logs_dir:
        <stage3_root>/scripts/experiment_report.py
        <stage3_root>/reports/experiment_report.html

    Failure is non-fatal: a warning is logged but the CLI still exits 0,
    because a missing report should never mask a successful training run.
    """
    stage3_root = logs_dir.parent
    report_script = stage3_root / "scripts" / "experiment_report.py"
    report_out = stage3_root / "reports" / "experiment_report.html"

    if not report_script.exists():
        log.warning(
            "Report script not found at %s — skipping report refresh. "
            "Pass --no-report to silence this warning.",
            report_script,
        )
        return

    proc = subprocess.run(
        [sys.executable, str(report_script), "--logs-dir", str(logs_dir), "--out", str(report_out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        log.info("Report refreshed → %s", report_out)
    else:
        log.warning(
            "Report generation failed (exit %d):\n%s",
            proc.returncode,
            proc.stderr.strip(),
        )


def _run_training(
    config: ExperimentConfig,
    *,
    logs_dir: Path,
    device_str: str = "cpu",
    quiet: bool = False,
    no_report: bool = False,
) -> int:
    """Execute the full training pipeline for one experiment; return an exit code.

    Orchestrates data assembly, QC, cross-validation, run logging, and
    optionally report refresh. On any failure, writes a ``status="failed"``
    run log before re-raising so the results layer always has a record.

    Args:
        config:     Fully-validated ExperimentConfig for this run.
        logs_dir:   Directory to write the JSON run log.
        device_str: Torch device string (``"cpu"`` or ``"cuda"``).
        quiet:      Suppress per-fold progress lines.
        no_report:  Skip refreshing the HTML experiment report.

    Returns:
        0 on success. Raises on failure (caller handles the exit code).
    """
    study_dir = Path(config.study_data_dir)
    started_at = datetime.now(UTC)

    with track_run(config) as tracker:
        try:
            _validate_study_dir(study_dir)
            cohort = assemble_cohort(
                expression_path=study_dir / "data_mrna_seq_v2_rsem.txt",
                clinical_patient_path=study_dir / "data_clinical_patient.txt",
                clinical_sample_path=study_dir / "data_clinical_sample.txt",
                study_id=config.cohort,
            )
            log.info(
                "Cohort assembled — %s: %d patients, %d genes, event rate %.2f%%",
                config.cohort,
                cohort.n_patients,
                cohort.n_genes,
                cohort.event_rate * 100,
            )

            cohort = filter_zero_time_patients(cohort)
            cohort = remove_outlier_samples(cohort)
            cohort = clip_survival_time(cohort, max_months=120.0)
            log.info(
                "Post-QC cohort: %d patients, event rate %.2f%%",
                cohort.n_patients,
                cohort.event_rate * 100,
            )
            tracker.log_metric("cohort_n_patients", float(cohort.n_patients))
            tracker.log_metric("cohort_n_genes", float(cohort.n_genes))
            tracker.log_metric("cohort_event_rate", float(cohort.event_rate))

            device = torch.device(device_str)
            result = cross_validate(cohort, config, device=device, verbose=not quiet)
            tracker.log_cv_result(result)

            finished_at = datetime.now(UTC)
            log_path = write_run_log(
                config=config,
                result=result,
                logs_dir=logs_dir,
                status="success",
                error=None,
                started_at=started_at,
                finished_at=finished_at,
            )
            tracker.log_artifact(log_path)
            log.info(
                "DONE  C-index = %.4f +/- %.4f  (n_folds=%d)  log=%s",
                result.c_index_mean,
                result.c_index_std,
                config.n_folds,
                log_path,
            )

            if not no_report:
                _refresh_report(logs_dir)

            return 0

        except BaseException as exc:
            finished_at = datetime.now(UTC)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            try:
                log_path = write_run_log(
                    config=config,
                    result=None,
                    logs_dir=logs_dir,
                    status="failed",
                    error=tb,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                tracker.log_artifact(log_path)
                log.error("FAILED  %s: %s  log=%s", type(exc).__name__, exc, log_path)
            except Exception as log_exc:  # pragma: no cover - extremely defensive
                log.error(
                    "FAILED and could not write run log. "
                    "Original error: %s: %s. Log write error: %s: %s",
                    type(exc).__name__,
                    exc,
                    type(log_exc).__name__,
                    log_exc,
                )
            raise


# --------------------------------------------------------------------------- #
# Legacy argparse entry point (pathogems-train)
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return the populated namespace."""
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
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip refreshing the HTML experiment report after a successful run.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Legacy argparse entry point; reads a JSON experiment config.

    Orchestrates config loading, data assembly, QC, cross-validation, and
    run logging. On any failure past config load, writes a ``status="failed"``
    run log before re-raising so the results layer always has a record.
    """
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if not args.quiet else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = ExperimentConfig.from_json(args.config)
    return _run_training(
        config,
        logs_dir=args.logs_dir,
        device_str=args.device,
        quiet=args.quiet,
        no_report=args.no_report,
    )


# --------------------------------------------------------------------------- #
# Hydra entry point (pathogems-train-hydra)
# --------------------------------------------------------------------------- #

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(
        version_base=None,
        config_path="../../configs",  # relative to this source file
        config_name="base",
    )
    def hydra_main(cfg: DictConfig) -> None:
        """Hydra entry point; composes config from base.yaml + experiment override.

        The ``runtime`` sub-key carries the non-ExperimentConfig settings
        (logs_dir, device, quiet, no_report). Everything else is forwarded
        verbatim to ``ExperimentConfig.from_dict``.

        Usage (from the project root)::

            pathogems-train-hydra +experiment=brca_omics_baseline
            pathogems-train-hydra +experiment=brca_omics_baseline lr=1e-3
            pathogems-train-hydra -m +experiment=brca_omics_baseline,brca_omics_topk1000

        See ``configs/base.yaml`` for all available fields and their defaults.
        """
        # Extract runtime settings before passing the rest to ExperimentConfig.
        runtime = OmegaConf.to_container(cfg.runtime, resolve=True)
        if not isinstance(runtime, dict):
            runtime = {}

        logs_dir = Path(str(runtime.get("logs_dir", "stage3_experiments/logs")))
        device_str = str(runtime.get("device", "cpu"))
        quiet = bool(runtime.get("quiet", False))
        no_report = bool(runtime.get("no_report", False))

        # Configure logging early.
        logging.basicConfig(
            level=logging.DEBUG if not quiet else logging.WARNING,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

        # Convert DictConfig → plain dict, drop the runtime sub-key.
        # OmegaConf.to_container returns a wide union type; we assert str keys
        # because base.yaml only contains string-keyed mappings.
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict from OmegaConf, got {type(raw)}")
        cfg_dict: dict[str, Any] = {str(k): v for k, v in raw.items()}
        cfg_dict.pop("runtime", None)

        # Validate and construct the experiment config. from_dict raises
        # ValueError for unknown fields, wrong types, or validation failures.
        config = ExperimentConfig.from_dict(cfg_dict)

        _run_training(
            config,
            logs_dir=logs_dir,
            device_str=device_str,
            quiet=quiet,
            no_report=no_report,
        )

except ImportError:  # pragma: no cover

    def hydra_main() -> None:
        """Stub: hydra-core is not installed."""
        print(
            "hydra-core is required for pathogems-train-hydra.\n"
            "Install it with:  pip install hydra-core>=1.3",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

"""Command-line entry point: `pathogems-train --config <path>`.

The CLI is intentionally tiny: parse args, load config, load data, run CV,
write run log, print a one-line summary. Anything more belongs in the
library modules where it can be tested without spinning up a subprocess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .config import ExperimentConfig
from .data import assemble_cohort
from .run_log import write_run_log
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
    config = ExperimentConfig.from_json(args.config)

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

    device = torch.device(args.device)

    result = cross_validate(cohort, config, device=device, verbose=not args.quiet)

    log_path = write_run_log(
        config=config,
        result=result,
        logs_dir=args.logs_dir,
        status="success",
        error=None,
    )
    print(
        f"[cli] DONE  C-index = {result.c_index_mean:.4f} +/- {result.c_index_std:.4f}  "
        f"(n_folds={config.n_folds})  log={log_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

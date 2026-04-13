"""Structured JSON run-log writer — the Stage 3 -> Stage 4 contract.

Schema is versioned and documented in ADR 0005. This module is the single
place that schema is produced; Stage 4 reads the schema and fails loudly
on an unknown `schema_version`. All downstream consumers read from these
files, never from stdout.
"""

from __future__ import annotations

import json
import math
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .train import CVResult

SCHEMA_VERSION = 1


def _git_sha() -> str | None:
    """Best-effort `git rev-parse HEAD`. Returns None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _environment() -> dict[str, Any]:
    """Capture things that change C-index between runs-of-same-config."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:  # torch should always be installed, but don't crash the log writer
        pass
    return info


def _jsonable(x: Any) -> Any:
    """Coerce numpy / NaN / tuple into JSON-safe types.

    We explicitly convert `NaN` to `None` (JSON `null`) because a `nan`
    leaked into a JSON file is a hard-to-debug parsing error on the
    Stage 4 side (many JSON decoders reject `NaN` literal).
    """
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def build_run_log(
    config: ExperimentConfig,
    result: CVResult | None,
    *,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Assemble the run-log dict. Pure function; easy to unit-test."""
    if status not in {"success", "failed"}:
        raise ValueError(f"status must be 'success' or 'failed', got {status!r}")

    metrics: dict[str, Any] = {}
    if result is not None:
        metrics = {
            "c_index_mean": _jsonable(result.c_index_mean),
            "c_index_std": _jsonable(result.c_index_std),
            "c_index_folds": _jsonable(result.per_fold_c_index()),
            "final_loss_mean": _jsonable(result.final_loss_mean),
            "final_loss_folds": _jsonable(result.per_fold_final_loss()),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.name,
        "config": config.to_dict(),
        "git_sha": _git_sha(),
        "started_at": started_at.astimezone(UTC).isoformat(timespec="seconds"),
        "finished_at": finished_at.astimezone(UTC).isoformat(timespec="seconds"),
        "wall_clock_sec": (finished_at - started_at).total_seconds(),
        "status": status,
        "error": error,
        "metrics": metrics,
        "environment": _environment(),
        "notes": config.notes,
    }


def write_run_log(
    config: ExperimentConfig,
    result: CVResult | None,
    logs_dir: Path,
    *,
    status: str,
    error: str | None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> Path:
    """Build and write a run log; return the path to the file."""
    now = datetime.now(UTC)
    log = build_run_log(
        config=config,
        result=result,
        started_at=started_at or now,
        finished_at=finished_at or now,
        status=status,
        error=error,
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{config.name}_run.json"
    path.write_text(json.dumps(log, indent=2) + "\n")
    return path


def read_run_log(path: Path) -> dict[str, Any]:
    """Load a run log and assert schema_version matches what we know."""
    data = json.loads(path.read_text())
    v = data.get("schema_version")
    if v != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version={v}; this code only handles v{SCHEMA_VERSION}. "
            "Upgrade Stage 4 explicitly — do not silently accept unknown schemas."
        )
    return data

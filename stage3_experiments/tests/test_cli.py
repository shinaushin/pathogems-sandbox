"""Tests for the CLI failure-log contract.

We do not unit-test the happy path here — that's covered by
`test_smoke_end_to_end.py` which exercises the full training harness. This
test file exists to lock in the *failure* contract: if anything past the
config load raises, the CLI must leave behind a schema-compliant run log
with `status="failed"` and a traceback, and must re-raise so the shell
sees a non-zero exit.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from pathogems.cli import main
from pathogems.config import ExperimentConfig


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    cfg = ExperimentConfig(
        name="failing_run",
        study_data_dir=str(tmp_path / "nonexistent"),  # triggers load failure
    )
    p = tmp_path / "cfg.json"
    cfg.to_json(p)
    return p


class TestCLIFailurePath:
    def test_writes_failed_run_log_and_reraises(self, tmp_path: Path, tmp_config: Path) -> None:
        logs_dir = tmp_path / "logs"

        # The cohort-load path will fail because the directory doesn't
        # exist. assemble_cohort raises FileNotFoundError from pandas.
        with pytest.raises((FileNotFoundError, OSError)):
            main(["--config", str(tmp_config), "--logs-dir", str(logs_dir)])

        log_path = logs_dir / "failing_run_run.json"
        assert log_path.exists(), "CLI must write a run log even on failure."

        log = json.loads(log_path.read_text())
        assert log["status"] == "failed"
        assert log["error"], "error field must be populated with a traceback."
        assert "Traceback" in log["error"]
        assert log["metrics"] == {}, "No metrics on a failed run."

    def test_writes_failed_log_on_training_error(self, tmp_path: Path) -> None:
        """Even a failure mid-CV must produce a schema-compliant log."""
        cfg = ExperimentConfig(name="boom", study_data_dir="")
        cfg_path = tmp_path / "c.json"
        cfg.to_json(cfg_path)

        logs_dir = tmp_path / "logs"
        # Mock assemble_cohort to succeed but cross_validate to blow up.
        with (
            mock.patch("pathogems.cli.assemble_cohort") as fake_assemble,
            mock.patch("pathogems.cli.cross_validate", side_effect=RuntimeError("training blew up")),
        ):
            fake_assemble.return_value = mock.MagicMock(n_patients=10, n_genes=5, event_rate=0.3)
            with pytest.raises(RuntimeError, match="training blew up"):
                main(["--config", str(cfg_path), "--logs-dir", str(logs_dir)])

        log = json.loads((logs_dir / "boom_run.json").read_text())
        assert log["status"] == "failed"
        assert "training blew up" in log["error"]

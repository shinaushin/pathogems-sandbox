"""Tests for pathogems.run_log."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pathogems.config import ExperimentConfig
from pathogems.run_log import SCHEMA_VERSION, build_run_log, read_run_log, write_run_log
from pathogems.train import CVResult, FoldResult


def _make_result(c_indices: list[float]) -> CVResult:
    return CVResult(
        folds=[
            FoldResult(
                fold_id=i,
                c_index=c,
                final_train_loss=0.1,
                final_val_loss=0.2,
                epochs_trained=30,
                best_epoch=20,
                wall_clock_sec=1.0,
            )
            for i, c in enumerate(c_indices)
        ]
    )


class TestBuildRunLog:
    def test_all_required_fields_present(self) -> None:
        cfg = ExperimentConfig(name="unit")
        res = _make_result([0.65, 0.68, 0.70, 0.66, 0.69])
        t0 = datetime(2026, 4, 13, 12, 0, 0, tzinfo=UTC)
        t1 = datetime(2026, 4, 13, 12, 0, 10, tzinfo=UTC)
        log = build_run_log(cfg, res, started_at=t0, finished_at=t1, status="success", error=None)

        for key in (
            "schema_version",
            "run_name",
            "config",
            "git_sha",
            "started_at",
            "finished_at",
            "wall_clock_sec",
            "status",
            "error",
            "metrics",
            "environment",
            "notes",
        ):
            assert key in log, f"Missing required field: {key}"

        assert log["schema_version"] == SCHEMA_VERSION
        assert log["run_name"] == "unit"
        assert log["status"] == "success"
        assert log["metrics"]["c_index_mean"] == pytest.approx(0.676, abs=1e-3)
        assert len(log["metrics"]["c_index_folds"]) == 5

    def test_nan_c_index_becomes_none(self) -> None:
        cfg = ExperimentConfig(name="x")
        res = _make_result([float("nan"), 0.6, 0.7])
        t0 = datetime(2026, 4, 13, tzinfo=UTC)
        log = build_run_log(cfg, res, started_at=t0, finished_at=t0, status="success", error=None)
        # JSON has no NaN; the writer must coerce to None.
        assert log["metrics"]["c_index_folds"][0] is None

    def test_invalid_status_rejected(self) -> None:
        cfg = ExperimentConfig(name="x")
        t0 = datetime(2026, 4, 13, tzinfo=UTC)
        with pytest.raises(ValueError, match="status"):
            build_run_log(cfg, None, started_at=t0, finished_at=t0, status="unknown", error=None)


class TestWriteRoundTrip:
    def test_write_then_read(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="rt")
        res = _make_result([0.6, 0.65, 0.7])
        path = write_run_log(cfg, res, logs_dir=tmp_path, status="success", error=None)
        assert path.exists()
        loaded = read_run_log(path)
        assert loaded["run_name"] == "rt"
        assert len(loaded["metrics"]["c_index_folds"]) == 3
        # Ensure valid JSON with no NaN literal.
        text = path.read_text()
        assert "NaN" not in text

    def test_read_rejects_unknown_schema_version(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": 999}))
        with pytest.raises(ValueError, match="schema_version"):
            read_run_log(path)

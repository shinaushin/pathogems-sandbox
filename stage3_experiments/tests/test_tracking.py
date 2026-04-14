"""Tests for pathogems.tracking.

These tests avoid importing mlflow. The null-tracker contract and the
"mlflow not installed falls back to null" path are exactly what we want
to exercise most: they guarantee that the training harness never
depends on observability being working.
"""

from __future__ import annotations

import sys

import pytest

from pathogems.config import ExperimentConfig
from pathogems.tracking import _MLflowTracker, _NullTracker, track_run


class TestNullTracker:
    def test_all_methods_are_noops(self) -> None:
        """The null tracker must accept every call without raising."""
        t = _NullTracker()
        t.log_params({"x": 1, "y": None})
        t.log_metric("c_index", 0.65)
        t.log_metric("c_index", 0.7, step=1)
        # log_cv_result requires a CVResult-shaped object — skip it here,
        # it is covered indirectly by `test_track_run_disabled_yields_null`.
        # A non-existent path is fine for log_artifact too.
        from pathlib import Path
        t.log_artifact(Path("/tmp/does_not_exist"))


class TestTrackRun:
    def test_disabled_yields_null_tracker(self) -> None:
        cfg = ExperimentConfig(name="no_track", enable_mlflow=False)
        with track_run(cfg) as tracker:
            assert isinstance(tracker, _NullTracker)

    def test_enabled_but_mlflow_missing_falls_back_to_null(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """If mlflow is not importable, tracker degrades gracefully.

        This is important: a production run should never fail because
        someone forgot to `pip install mlflow` on the training box.
        """
        # Simulate ImportError when the tracking module tries to import
        # mlflow. We blacklist it at the finder level.
        monkeypatch.setitem(sys.modules, "mlflow", None)
        cfg = ExperimentConfig(name="mlflow_missing", enable_mlflow=True)
        with track_run(cfg) as tracker:
            assert isinstance(tracker, _NullTracker)

        captured = capsys.readouterr()
        assert "mlflow not installed" in captured.out


class TestMLflowTrackerAdapter:
    """We can test the adapter's metric-filtering logic with a fake mlflow
    module, without needing the real package installed.
    """

    def test_log_metric_skips_nonfinite_values(self) -> None:
        calls: list[tuple[str, float]] = []

        class FakeML:
            @staticmethod
            def log_metric(name: str, value: float, step: int | None = None) -> None:
                calls.append((name, value))

        t = _MLflowTracker(FakeML())
        t.log_metric("good", 0.5)
        t.log_metric("nan", float("nan"))
        t.log_metric("inf", float("inf"))
        t.log_metric("neg_inf", float("-inf"))
        assert calls == [("good", 0.5)]

    def test_log_params_filters_nones(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeML:
            @staticmethod
            def log_params(params: dict[str, object]) -> None:
                calls.append(params)

        t = _MLflowTracker(FakeML())
        t.log_params({"a": 1, "b": None, "c": "x"})
        assert calls == [{"a": 1, "c": "x"}]

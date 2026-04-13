"""Tests for pathogems.config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathogems.config import ExperimentConfig


class TestExperimentConfig:
    def test_round_trip_preserves_values(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="unit_test", hidden_dims=(64, 16), modalities=("RNA_seq",))
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        back = ExperimentConfig.from_json(p)
        assert back == cfg

    def test_json_has_lists_not_tuples(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(name="x", hidden_dims=(32, 8))
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        raw = json.loads(p.read_text())
        # JSON can't express tuples; round-tripping through to_dict must
        # produce plain lists to keep downstream diff tools happy.
        assert isinstance(raw["hidden_dims"], list)
        assert isinstance(raw["modalities"], list)

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config fields"):
            ExperimentConfig.from_dict({"name": "x", "futuristic_flag": True})

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown config_version"):
            ExperimentConfig.from_dict({"name": "x", "config_version": 999})

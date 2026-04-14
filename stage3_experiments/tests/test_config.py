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

    def test_batch_size_none_roundtrips_as_json_null(self, tmp_path: Path) -> None:
        """`batch_size=None` (full-batch) must survive a JSON round-trip.

        JSON's natural representation of None is `null`, and dataclass
        `from_dict` should accept it without any coercion. This test
        pins the contract so a future "make batch_size int only" change
        explicitly breaks the test instead of silently breaking configs.
        """
        cfg = ExperimentConfig(name="fullbatch", batch_size=None)
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        raw = json.loads(p.read_text())
        assert raw["batch_size"] is None
        back = ExperimentConfig.from_json(p)
        assert back.batch_size is None

    def test_batch_size_default_is_none(self) -> None:
        """The default is full-batch (ADR 0007)."""
        assert ExperimentConfig(name="x").batch_size is None

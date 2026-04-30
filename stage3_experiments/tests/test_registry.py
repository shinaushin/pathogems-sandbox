"""Tests for the generic Registry and the three concrete registries.

The concrete registries (model / loss / optimizer) are exercised end-to-end
by `test_smoke_end_to_end.py`; here we nail down the contract of the
generic class and the *error paths* that are easy to regress silently.
"""

from __future__ import annotations

import pytest

from pathogems.loss import LOSS_REGISTRY
from pathogems.model import MODEL_REGISTRY
from pathogems.optimizers import OPTIMIZER_REGISTRY
from pathogems.registry import Registry


class TestRegistry:
    """Contract tests for the generic `Registry[T]`."""

    def test_register_and_get(self) -> None:
        r: Registry[int] = Registry("thing")

        @r.register("answer")
        def _answer() -> int:
            return 42

        # `register` returns the decorated object unchanged.
        assert _answer() == 42
        # `get` returns the same object (callable in this case).
        assert r.get("answer") is _answer
        assert "answer" in r
        assert len(r) == 1
        assert r.names() == ["answer"]

    def test_duplicate_name_rejected(self) -> None:
        """Duplicate registrations are a loud error, not a silent overwrite.

        Silent overwrites were the original sin we're trying to prevent:
        two modules both registering "cox_ph" would turn "which one runs?"
        into an import-order dependency.
        """
        r: Registry[int] = Registry("thing")
        r.register("a")(1)
        with pytest.raises(ValueError, match="already registered"):
            r.register("a")(2)

    def test_get_unknown_suggests_close_match(self) -> None:
        """Typos produce a useful `Did you mean?` hint via difflib."""
        r: Registry[str] = Registry("optimizer")
        r.register("adam")("adam-impl")
        r.register("sgd")("sgd-impl")

        with pytest.raises(KeyError) as excinfo:
            r.get("adm")  # one-letter typo of "adam"

        msg = str(excinfo.value)
        assert "Unknown optimizer" in msg
        assert "'adm'" in msg
        # The close match is surfaced so the user doesn't have to grep.
        assert "adam" in msg

    def test_get_unknown_no_close_match(self) -> None:
        """Completely unrelated names still produce a useful error, just
        without the hint — better to say nothing than to suggest nonsense.
        """
        r: Registry[str] = Registry("optimizer")
        r.register("adam")("adam-impl")

        with pytest.raises(KeyError) as excinfo:
            r.get("xylophone")

        assert "Did you mean" not in str(excinfo.value)

    def test_repr_is_informative(self) -> None:
        r: Registry[int] = Registry("widget")
        r.register("foo")(1)
        r.register("bar")(2)
        # Names are sorted for a deterministic repr (handy in test failure
        # messages and in snapshot-style assertions).
        assert repr(r) == "Registry('widget', names=['bar', 'foo'])"


class TestConcreteRegistries:
    """The three concrete registries should have the entries we ship."""

    def test_model_registry_has_omics_mlp(self) -> None:
        assert "omics_mlp" in MODEL_REGISTRY

    def test_loss_registry_has_cox_ph(self) -> None:
        assert "cox_ph" in LOSS_REGISTRY

    def test_optimizer_registry_has_adam_and_sgd(self) -> None:
        assert "adam" in OPTIMIZER_REGISTRY
        assert "sgd" in OPTIMIZER_REGISTRY

    def test_optimizer_registry_has_adamw(self) -> None:
        assert "adamw" in OPTIMIZER_REGISTRY

    def test_adamw_factory_returns_adamw_optimizer(self) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        from pathogems.config import ExperimentConfig

        cfg = ExperimentConfig(
            name="adamw_test",
            cohort="synthetic",
            study_data_dir="",
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            notes="AdamW factory test.",
        )
        params = nn.Linear(10, 1).parameters()
        opt = OPTIMIZER_REGISTRY.get("adamw")(params, cfg)
        assert isinstance(opt, optim.AdamW)
        assert opt.defaults["lr"] == pytest.approx(1e-3)
        assert opt.defaults["weight_decay"] == pytest.approx(1e-4)

    def test_typo_in_optimizer_name_surfaces_hint(self) -> None:
        # This is the exact bug the registry was introduced to catch:
        # a config typo should tell the user what they likely meant rather
        # than just "unknown optimizer 'adm'".
        with pytest.raises(KeyError, match="adam"):
            OPTIMIZER_REGISTRY.get("adm")

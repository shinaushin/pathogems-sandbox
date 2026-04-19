"""Tests for pathogems.model."""

from __future__ import annotations

import pytest
import torch

from pathogems.model import LinearCox, OmicsMLP, OmicsMLPConfig


class TestOmicsMLPConfig:
    def test_rejects_nonpositive_in_features(self) -> None:
        with pytest.raises(ValueError, match="in_features"):
            OmicsMLPConfig(in_features=0)

    def test_rejects_empty_hidden_dims(self) -> None:
        with pytest.raises(ValueError, match="hidden_dims"):
            OmicsMLPConfig(in_features=10, hidden_dims=())

    def test_rejects_dropout_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            OmicsMLPConfig(in_features=10, dropout=1.0)


class TestOmicsMLP:
    def test_forward_shape(self) -> None:
        cfg = OmicsMLPConfig(in_features=50)
        model = OmicsMLP(cfg)
        x = torch.randn(8, 50)
        out = model(x)
        # Scalar risk per sample.
        assert out.shape == (8,)
        assert out.dtype == torch.float32

    def test_rejects_wrong_input_dim(self) -> None:
        cfg = OmicsMLPConfig(in_features=50)
        model = OmicsMLP(cfg)
        with pytest.raises(ValueError, match="Expected input"):
            model(torch.randn(8, 49))

    def test_weights_nonzero_after_init(self) -> None:
        """Kaiming init should leave weights non-zero; all-zeros is a classic stall."""
        cfg = OmicsMLPConfig(in_features=20)
        model = OmicsMLP(cfg)
        # Biases are zero by design. Check that at least one Linear weight is not.
        import torch.nn as nn

        linears = [m for m in model.net.modules() if isinstance(m, nn.Linear)]
        assert any(torch.any(m.weight != 0) for m in linears)

    def test_parameter_count_scales_with_hidden(self) -> None:
        small = OmicsMLP(OmicsMLPConfig(in_features=100, hidden_dims=(32,)))
        large = OmicsMLP(OmicsMLPConfig(in_features=100, hidden_dims=(256,)))
        assert small.num_parameters() < large.num_parameters()

    def test_eval_mode_deterministic(self) -> None:
        cfg = OmicsMLPConfig(in_features=30, dropout=0.5)
        model = OmicsMLP(cfg).eval()
        x = torch.randn(4, 30)
        # No dropout in eval mode, two calls must match exactly.
        torch.testing.assert_close(model(x), model(x))


class TestLinearCox:
    """The linear Cox model is deliberately boring; we just lock the
    shape contract and the "no hidden non-linearity" invariant.
    """

    def test_forward_shape(self) -> None:
        model = LinearCox(in_features=50)
        x = torch.randn(8, 50)
        out = model(x)
        assert out.shape == (8,)
        assert out.dtype == torch.float32

    def test_rejects_wrong_input_dim(self) -> None:
        model = LinearCox(in_features=50)
        with pytest.raises(ValueError, match="Expected input"):
            model(torch.randn(8, 49))

    def test_rejects_nonpositive_in_features(self) -> None:
        with pytest.raises(ValueError, match="in_features"):
            LinearCox(in_features=0)

    def test_is_exactly_linear(self) -> None:
        """f(a*x1 + b*x2) must equal a*f(x1) + b*f(x2) (up to the bias)."""
        model = LinearCox(in_features=10).eval()
        x1 = torch.randn(1, 10)
        x2 = torch.randn(1, 10)
        a, b = 0.7, -1.3

        # Subtract the bias to reduce to a strictly linear map.
        bias = model.linear.bias.detach()

        def f(x: torch.Tensor) -> torch.Tensor:
            return model(x) - bias

        torch.testing.assert_close(
            f(a * x1 + b * x2),
            a * f(x1) + b * f(x2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_parameter_count(self) -> None:
        """LinearCox has exactly in_features + 1 parameters (weight + bias)."""
        assert LinearCox(in_features=100).num_parameters() == 101

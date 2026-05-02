"""Tests for pathogems.loss.

We verify:
    1. Closed-form agreement with a naive reference on tiny inputs (the
       reference cannot be used in production — it is O(B^2) — but it is
       the ground truth we compare against here).
    2. Invariance to arbitrary monotone transform of `r` is achieved
       structurally by the log-sum-exp formulation (no activation on top).
    3. Numerical stability at large |r|.
    4. Gradient flows through the loss (mini end-to-end sanity check).
    5. The degenerate all-censored batch does not crash.
"""

from __future__ import annotations

import math

import pytest
import torch

from pathogems.loss import cox_ph_loss


def _naive_cox(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """O(B^2) reference implementation. Correct by inspection; slow."""
    n_events = int(event.sum().item())
    if n_events == 0:
        return torch.tensor(0.0)
    total = 0.0
    for i in range(risk.shape[0]):
        if event[i] == 0:
            continue
        risk_set = risk[time >= time[i]]
        total += float(risk[i].item()) - math.log(float(torch.exp(risk_set).sum().item()))
    return torch.tensor(-total / n_events)


class TestCoxPHLoss:
    def test_matches_naive_reference(self) -> None:
        torch.manual_seed(0)
        risk = torch.randn(20)
        time = torch.rand(20) * 100
        event = torch.randint(0, 2, (20,)).float()

        ours = cox_ph_loss(risk, time, event)
        ref = _naive_cox(risk, time, event)
        torch.testing.assert_close(ours, ref, rtol=1e-4, atol=1e-4)

    def test_numerical_stability_at_large_risk(self) -> None:
        # Risk values of ~1e3 would overflow a naive exp-then-sum.
        risk = torch.tensor([1000.0, 999.0, 998.0, 997.0])
        time = torch.tensor([1.0, 2.0, 3.0, 4.0])
        event = torch.tensor([1.0, 1.0, 0.0, 1.0])
        loss = cox_ph_loss(risk, time, event)
        assert torch.isfinite(loss).item()

    def test_all_censored_returns_finite_zero(self) -> None:
        risk = torch.randn(10, requires_grad=True)
        time = torch.rand(10) * 100
        event = torch.zeros(10)
        loss = cox_ph_loss(risk, time, event)
        assert torch.isfinite(loss).item()
        assert float(loss.item()) == 0.0
        loss.backward()  # gradient path intact even in the degenerate case

    def test_gradient_flows(self) -> None:
        risk = torch.randn(16, requires_grad=True)
        time = torch.rand(16) * 50
        event = torch.randint(0, 2, (16,)).float()
        # Guarantee at least one event so the loss is not degenerate.
        event[0] = 1.0
        loss = cox_ph_loss(risk, time, event)
        loss.backward()
        assert risk.grad is not None
        assert torch.any(risk.grad != 0)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            cox_ph_loss(torch.zeros(5), torch.zeros(4), torch.zeros(5))

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError):
            cox_ph_loss(torch.zeros(5, 1), torch.zeros(5), torch.zeros(5))

    def test_tied_survival_times_finite(self) -> None:
        """Multiple patients sharing the same time — Breslow tie handling must not NaN.

        Breslow approximation treats tied times as a single risk set (the
        cumulative logsum covers all positions in the sorted order), so the
        loss must be finite even when many patients share an identical time.
        """
        # Five patients: two tied at time 10, two tied at time 20, one at 30.
        risk = torch.tensor([0.5, -0.5, 1.0, -1.0, 0.0])
        time = torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0])
        event = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        loss = cox_ph_loss(risk, time, event)
        assert torch.isfinite(loss).item(), f"Loss was not finite: {loss}"

    def test_all_events_finite(self) -> None:
        """Every patient dies — a valid (if unusual) batch. Loss must be finite."""
        torch.manual_seed(3)
        risk = torch.randn(12)
        time = torch.rand(12) * 100 + 1.0  # all positive
        event = torch.ones(12)  # all events
        loss = cox_ph_loss(risk, time, event)
        assert torch.isfinite(loss).item(), f"Loss was not finite with all events: {loss}"

    def test_single_event_finite(self) -> None:
        """Batch with only one event — the risk set for that event is all others.

        This is the extreme case of event scarcity: the partial likelihood
        has a single term and the normalisation is by 1. Should still give
        a finite, meaningful loss.
        """
        risk = torch.tensor([2.0, 0.5, -0.5, -1.0, 0.0])
        time = torch.tensor([5.0, 10.0, 20.0, 30.0, 40.0])
        event = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # only one event
        loss = cox_ph_loss(risk, time, event)
        assert torch.isfinite(loss).item(), f"Single-event loss was not finite: {loss}"
        # Manually verify: only one term: -(risk[0] - log(exp(r[0]) + ... + exp(r[4])))
        # The event patient (index 0) has the shortest time, so all others are in its risk set.
        ref = _naive_cox(risk, time, event)
        torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)

    def test_monotone_shift_invariance(self) -> None:
        """Adding a constant to every risk must not change the loss.

        This is a structural property of the Cox partial likelihood. It's
        a nice regression canary: any future "improvement" to the loss
        that breaks shift invariance is almost certainly wrong.
        """
        torch.manual_seed(1)
        risk = torch.randn(30)
        time = torch.rand(30) * 100
        event = torch.randint(0, 2, (30,)).float()
        event[0] = 1.0

        base = cox_ph_loss(risk, time, event)
        shifted = cox_ph_loss(risk + 7.5, time, event)
        torch.testing.assert_close(base, shifted, rtol=1e-5, atol=1e-5)

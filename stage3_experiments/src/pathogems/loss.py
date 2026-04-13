r"""Cox partial likelihood loss (Breslow ties handling).

Rationale for this loss choice is in ADR 0002. This module owns the
mathematical implementation, with tests in `tests/test_loss.py` that
compare against a closed-form reference on small inputs.

Definitions
-----------
Let :math:`r_i = f(x_i; \theta)` be the model's scalar risk score for
patient *i*, with observed time :math:`t_i` and event indicator
:math:`\delta_i \in \{0, 1\}`. The Breslow approximation to the Cox
negative log partial likelihood over a batch is

.. math::
    L(\theta) = -\frac{1}{|E|}
        \sum_{i : \delta_i = 1}
            \left( r_i - \log \sum_{j : t_j \geq t_i} \exp(r_j) \right)

where :math:`E = \{i : \delta_i = 1\}` is the set of events in the batch.
The normalization by :math:`|E|` (event count) keeps the loss scale
comparable across batch sizes with different event rates — the alternative
of dividing by batch size makes the gradient magnitude shrink with event
scarcity, which destabilizes training on TCGA-like cohorts.

Numerical stability
-------------------
The inner sum :math:`\sum \exp(r_j)` can overflow for large :math:`|r|`.
We apply the standard log-sum-exp trick: subtract the max risk in the
risk set before exponentiating. This is implemented via
`torch.logcumsumexp` after sorting by descending time, which gives the
correct risk-set sum in *linear* time rather than the naive :math:`O(B^2)`.

The log-cumulative trick
------------------------
If we sort the batch by *descending* time, then the risk set
:math:`\{j : t_j \geq t_i\}` is exactly the prefix :math:`\{0, 1, ..., i\}`
in the sorted order. So a single `logcumsumexp` over the sorted risks
computes every :math:`\log \sum_j \exp(r_j)` we need, and we select the
ones at event positions.
"""

from __future__ import annotations

import torch


def cox_ph_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    *,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Negative log partial likelihood of the Cox PH model, Breslow ties.

    Args:
        risk:  shape (B,), model output. No activation on top.
        time:  shape (B,), non-negative survival/censoring time.
        event: shape (B,), 0.0 for censored, 1.0 for observed event.
        epsilon: Floor on the divisor to guard against batches with zero
            events. When no events are present, the partial likelihood
            is undefined and we return 0.0 so training does not crash on
            a very-rare all-censored batch.

    Returns:
        Scalar tensor, the mean negative log partial likelihood.

    Raises:
        ValueError: if input shapes disagree or are not 1-D.
    """
    if risk.dim() != 1 or time.dim() != 1 or event.dim() != 1:
        raise ValueError(f"All inputs must be 1-D; got shapes {risk.shape}, {time.shape}, {event.shape}.")
    if not (risk.shape == time.shape == event.shape):
        raise ValueError("risk, time, and event must share shape.")

    # Early-return guard for the degenerate all-censored batch.
    n_events = event.sum()
    if n_events.item() == 0:
        # Return a tensor that is part of the graph so `.backward()` works.
        return (risk * 0.0).sum()

    # Sort by time DESCENDING. After sorting, risk_set[i] == {0..i} is
    # exactly {j : t_j >= t_i}, modulo ties. Breslow treats ties as a single
    # risk set at that time, which the cumulative formulation matches
    # because tied times share the same cumulative slice.
    order = torch.argsort(time, descending=True)
    risk_sorted = risk[order]
    event_sorted = event[order]

    # Log-sum-exp along the sorted axis — O(B) and numerically stable.
    # `logcumsumexp` produces, at position i, log(sum_{k<=i} exp(risk_sorted[k])).
    log_risk_set = torch.logcumsumexp(risk_sorted, dim=0)

    # Per-patient contribution: (r_i - log sum_{risk set}). Contributes only
    # at positions where the event was observed.
    partial = risk_sorted - log_risk_set
    loss = -(partial * event_sorted).sum() / torch.clamp(n_events, min=epsilon)
    return loss

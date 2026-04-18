"""Harrell's concordance index for right-censored survival data.

The C-index answers: "Of all comparable patient pairs (i, j) where patient
i had the shorter observed time *and* event was observed at i, how often
does the model assign a higher risk to i than to j?" A C-index of 0.5 is
random; 1.0 is perfect ordering.

This module provides two implementations:

    * `concordance_index` — pure-numpy, O(N^2), the reference. Fast
      enough for TCGA-scale test folds (hundreds of patients) and has
      zero external dependencies beyond numpy. We use this as the main
      evaluation metric.

    * `concordance_index_scikit_survival` — thin wrapper around
      `sksurv.metrics.concordance_index_censored`. Kept for cross-checking
      in tests; NOT used for reporting (to avoid tying our metric numbers
      to a specific sksurv version).

Why implement it ourselves instead of just using sksurv? Two reasons:
    1. Reporting numbers should not depend silently on a third-party
       version bump.
    2. When the metric is 50 lines and unit-tested against sksurv, the
       reader gets to see exactly what is being computed — which matters
       given how much survival literature quotes C-index without defining
       ties handling.
"""

from __future__ import annotations

import numpy as np


def concordance_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    r"""Harrell's C-index for right-censored survival data.

    Definition (with tie handling):
        For each pair (i, j) with i != j:
          - i is *comparable* to j iff (event_i == 1 and time_i < time_j)
            or (event_j == 1 and time_j < time_i). In short: at least one
            of them had an observed event, and the one that did had the
            shorter time.
          - A comparable pair is *concordant* if the model assigns higher
            risk to the patient with the shorter time.
          - If the model assigns equal risk, the pair contributes 0.5 (tied).

        C-index = concordant_weight / comparable_count.

    Args:
        risk:  shape (N,). Higher = higher predicted hazard.
        time:  shape (N,).
        event: shape (N,). {0, 1}.

    Returns:
        The concordance index as a float in [0, 1], or `float('nan')` if
        no comparable pairs exist (a fold with zero events).

    Notes:
        - O(N^2) naive implementation. Fine for N up to ~5000. For
          larger test folds we would switch to the O(N log N) algorithm
          (Therneau), but TCGA cohorts never exceed that.
        - Ties in *time* between two events are handled correctly: neither
          is comparable against the other because neither time is strictly
          smaller. A censored patient tied on time with an event patient
          is also not comparable (strict inequality is required).
    """
    risk = np.asarray(risk)
    time = np.asarray(time)
    event = np.asarray(event)
    if not (risk.shape == time.shape == event.shape):
        raise ValueError("risk, time, event must share shape.")
    if risk.ndim != 1:
        raise ValueError("All inputs must be 1-D.")

    n = risk.shape[0]
    concordant = 0.0
    comparable = 0
    # Nested loop, only considering j > i (symmetry handled via explicit branches).
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                # No strict ordering -> not comparable under Harrell's definition.
                continue
            # Identify the patient with the shorter time and whether that
            # patient had an observed event.
            if time[i] < time[j]:
                shorter_i, longer_i = i, j
            else:
                shorter_i, longer_i = j, i
            if event[shorter_i] == 0:
                continue  # shorter-time patient is censored -> not comparable
            comparable += 1
            if risk[shorter_i] > risk[longer_i]:
                concordant += 1.0
            elif risk[shorter_i] == risk[longer_i]:
                concordant += 0.5

    if comparable == 0:
        return float("nan")
    return concordant / comparable


def concordance_index_scikit_survival(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Thin wrapper around `sksurv.metrics.concordance_index_censored` for testing."""
    from sksurv.metrics import concordance_index_censored  # local import keeps it optional

    cindex, *_ = concordance_index_censored(
        event.astype(bool), time.astype(float), risk.astype(float)
    )
    return float(cindex)

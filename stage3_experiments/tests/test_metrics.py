"""Tests for pathogems.metrics.

The most important tests here are `test_perfect_ordering_is_one` and
`test_matches_scikit_survival`: the first is the definition of C-index,
the second is cross-validation against a widely-used reference library.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathogems.metrics import concordance_index


class TestConcordanceIndex:
    def test_perfect_ordering_is_one(self) -> None:
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([1, 1, 1, 1, 1])
        # Highest risk dies first -> perfect concordance.
        assert concordance_index(risk, time, event) == pytest.approx(1.0)

    def test_reversed_ordering_is_zero(self) -> None:
        risk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = np.array([1, 1, 1, 1, 1])
        assert concordance_index(risk, time, event) == pytest.approx(0.0)

    def test_random_risk_is_around_half(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        risk = rng.standard_normal(n)
        time = rng.uniform(1, 100, size=n)
        event = rng.binomial(1, 0.3, size=n)
        c = concordance_index(risk, time, event)
        assert 0.4 < c < 0.6  # wide but statistically robust

    def test_all_censored_returns_nan(self) -> None:
        risk = np.array([1.0, 2.0, 3.0])
        time = np.array([1.0, 2.0, 3.0])
        event = np.zeros(3, dtype=int)
        assert np.isnan(concordance_index(risk, time, event))

    def test_ties_in_risk_score_half_point(self) -> None:
        # Two events, tied risks -> 0.5.
        risk = np.array([1.0, 1.0])
        time = np.array([1.0, 2.0])
        event = np.array([1, 1])
        assert concordance_index(risk, time, event) == pytest.approx(0.5)

    def test_censored_shorter_time_not_comparable(self) -> None:
        # Patient 0 is censored at time 1; patient 1 has an event at time 2.
        # The only comparable pair requires the *shorter-time* patient to have
        # had the event, which is not the case here -> no comparable pairs.
        risk = np.array([5.0, 1.0])
        time = np.array([1.0, 2.0])
        event = np.array([0, 1])
        # comparable == 0 -> nan
        assert np.isnan(concordance_index(risk, time, event))

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            concordance_index(np.zeros(5), np.zeros(4), np.zeros(5))

    def test_matches_scikit_survival(self) -> None:
        """Cross-validate against the widely-used reference implementation."""
        try:
            from pathogems.metrics import concordance_index_scikit_survival
        except ImportError:
            pytest.skip("scikit-survival not installed")
        rng = np.random.default_rng(1)
        n = 100
        for _ in range(5):
            risk = rng.standard_normal(n)
            time = rng.uniform(1, 100, size=n)
            event = rng.binomial(1, 0.4, size=n)
            ours = concordance_index(risk, time, event)
            ref = concordance_index_scikit_survival(risk, time, event)
            if np.isnan(ours):
                continue
            assert abs(ours - ref) < 1e-6, f"Ours={ours}, sksurv={ref}"

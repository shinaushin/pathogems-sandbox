"""Tests for pathogems.data.

The most important tests in this file are `test_preprocessor_no_leakage`
and `test_cv_splits_deterministic_and_stratified`: if either regresses,
every C-index number we report becomes untrustworthy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathogems.data import (
    Preprocessor,
    SurvivalCohort,
    assemble_cohort,
    build_fold_tensors,
    cv_splits,
    load_clinical_patient,
    load_clinical_sample,
    load_expression_matrix,
)


# --------------------------------------------------------------------------- #
# Tiny synthetic cBioPortal-shaped files built on the fly, keeps tests hermetic
# --------------------------------------------------------------------------- #
def _write_expr(path: Path, genes: list[str], samples: list[str], values: np.ndarray) -> None:
    df = pd.DataFrame(values, index=genes, columns=samples)
    df.index.name = "Hugo_Symbol"
    df.insert(0, "Entrez_Gene_Id", range(1, len(genes) + 1))
    df.reset_index().to_csv(path, sep="\t", index=False)


def _write_clinical_patient(path: Path, rows: list[dict[str, object]]) -> None:
    # cBioPortal real files prefix 4 header lines with `#` (display names,
    # descriptions, types, priority). We reproduce one to exercise the parser.
    with path.open("w") as f:
        f.write("#Patient Identifier\tOverall Survival Status\tOverall Survival (Months)\n")
        f.write("#Patient Identifier\tStatus\tMonths\n")
        f.write("#STRING\tSTRING\tNUMBER\n")
        f.write("#1\t1\t1\n")
        f.write("PATIENT_ID\tOS_STATUS\tOS_MONTHS\n")
        for r in rows:
            f.write(f"{r['PATIENT_ID']}\t{r['OS_STATUS']}\t{r['OS_MONTHS']}\n")


def _write_clinical_sample(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w") as f:
        f.write("#Sample Identifier\tPatient Identifier\n")
        f.write("#Sample\tPatient\n")
        f.write("#STRING\tSTRING\n")
        f.write("#1\t1\n")
        f.write("SAMPLE_ID\tPATIENT_ID\n")
        for r in rows:
            f.write(f"{r['SAMPLE_ID']}\t{r['PATIENT_ID']}\n")


@pytest.fixture
def tiny_cbioportal(tmp_path: Path) -> dict[str, Path]:
    """A 5-patient, 4-gene dataset shaped exactly like cBioPortal's output."""
    genes = ["TP53", "BRCA1", "MYC", "KRAS"]
    # Two samples for P1 (primary + normal) to exercise the primary-only filter.
    samples = [
        "TCGA-P1-0001-01",  # primary
        "TCGA-P1-0001-11",  # normal — should be dropped
        "TCGA-P2-0002-01",
        "TCGA-P3-0003-01",
        "TCGA-P4-0004-01",
        "TCGA-P5-0005-01",
    ]
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1000, size=(len(genes), len(samples))).astype(float)
    expr_path = tmp_path / "expr.tsv"
    _write_expr(expr_path, genes, samples, values)

    clin_patient_path = tmp_path / "patient.tsv"
    _write_clinical_patient(
        clin_patient_path,
        [
            {"PATIENT_ID": "TCGA-P1-0001", "OS_STATUS": "1:DECEASED", "OS_MONTHS": 12.0},
            {"PATIENT_ID": "TCGA-P2-0002", "OS_STATUS": "0:LIVING", "OS_MONTHS": 60.0},
            {"PATIENT_ID": "TCGA-P3-0003", "OS_STATUS": "1:DECEASED", "OS_MONTHS": 24.0},
            {"PATIENT_ID": "TCGA-P4-0004", "OS_STATUS": "0:LIVING", "OS_MONTHS": 36.0},
            {"PATIENT_ID": "TCGA-P5-0005", "OS_STATUS": "1:DECEASED", "OS_MONTHS": 48.0},
        ],
    )

    clin_sample_path = tmp_path / "sample.tsv"
    _write_clinical_sample(
        clin_sample_path,
        [{"SAMPLE_ID": s, "PATIENT_ID": s[:12]} for s in samples],
    )

    return {
        "expr": expr_path,
        "patient": clin_patient_path,
        "sample": clin_sample_path,
    }


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
class TestParsers:
    def test_expression_shape_and_transpose(self, tiny_cbioportal: dict[str, Path]) -> None:
        expr = load_expression_matrix(tiny_cbioportal["expr"])
        # Samples are rows, genes are columns (ML convention).
        assert expr.shape == (6, 4)
        assert set(expr.columns) == {"TP53", "BRCA1", "MYC", "KRAS"}

    def test_clinical_patient_strips_comment_header(self, tiny_cbioportal: dict[str, Path]) -> None:
        clin = load_clinical_patient(tiny_cbioportal["patient"])
        assert len(clin) == 5
        assert set(clin.columns) == {"PATIENT_ID", "OS_STATUS", "OS_MONTHS"}

    def test_clinical_sample_strips_comment_header(self, tiny_cbioportal: dict[str, Path]) -> None:
        samp = load_clinical_sample(tiny_cbioportal["sample"])
        assert len(samp) == 6  # 5 primary + 1 normal
        assert set(samp.columns) == {"SAMPLE_ID", "PATIENT_ID"}


# --------------------------------------------------------------------------- #
# Cohort assembly
# --------------------------------------------------------------------------- #
class TestAssembleCohort:
    def test_keeps_only_primary_tumor_samples(self, tiny_cbioportal: dict[str, Path]) -> None:
        cohort = assemble_cohort(
            tiny_cbioportal["expr"],
            tiny_cbioportal["patient"],
            tiny_cbioportal["sample"],
            study_id="tiny",
        )
        # P1 has a normal-tissue sample; it must not inflate the cohort size.
        assert cohort.n_patients == 5

    def test_index_alignment_is_strict(self, tiny_cbioportal: dict[str, Path]) -> None:
        cohort = assemble_cohort(
            tiny_cbioportal["expr"],
            tiny_cbioportal["patient"],
            tiny_cbioportal["sample"],
            study_id="tiny",
        )
        assert cohort.expression.index.equals(cohort.time.index)
        assert cohort.time.index.equals(cohort.event.index)

    def test_event_mapped_to_int_01(self, tiny_cbioportal: dict[str, Path]) -> None:
        cohort = assemble_cohort(
            tiny_cbioportal["expr"],
            tiny_cbioportal["patient"],
            tiny_cbioportal["sample"],
            study_id="tiny",
        )
        assert set(cohort.event.unique()).issubset({0, 1})
        # 3 deceased / 5 total in the fixture.
        assert int(cohort.event.sum()) == 3


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def _make_cohort(n_patients: int = 80, n_genes: int = 50, event_rate: float = 0.4, seed: int = 0) -> SurvivalCohort:
    """Synthetic cohort for preprocessing / CV tests (no file I/O)."""
    rng = np.random.default_rng(seed)
    patients = [f"P{i:03d}" for i in range(n_patients)]
    genes = [f"G{i:03d}" for i in range(n_genes)]
    expr = pd.DataFrame(rng.uniform(0, 500, size=(n_patients, n_genes)), index=patients, columns=genes)
    event = pd.Series(rng.binomial(1, event_rate, size=n_patients), index=patients)
    time = pd.Series(rng.uniform(1, 120, size=n_patients).astype(float), index=patients)
    return SurvivalCohort(expression=expr, time=time, event=event, study_id="synthetic")


class TestPreprocessor:
    def test_selects_top_k_genes(self) -> None:
        cohort = _make_cohort(n_patients=40, n_genes=20)
        pre = Preprocessor(top_k=5).fit(cohort.expression)
        assert len(pre.selected_genes) == 5
        assert all(g in cohort.expression.columns for g in pre.selected_genes)

    def test_transform_before_fit_raises(self) -> None:
        cohort = _make_cohort()
        with pytest.raises(RuntimeError, match="before fit"):
            Preprocessor(top_k=3).transform(cohort.expression)

    def test_transform_is_zero_mean_unit_var_on_training_data(self) -> None:
        cohort = _make_cohort()
        pre = Preprocessor(top_k=10).fit(cohort.expression)
        x = pre.transform(cohort.expression)
        # Training data transformed with training statistics -> zero mean, unit std per column.
        np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(x.std(axis=0), 1.0, atol=1e-5)

    def test_preprocessor_no_leakage(self) -> None:
        """Fit on train, apply to test -> test statistics NOT centered by design.

        This is the explicit guarantee the class exists to provide. If
        someone ever refactors the preprocessor to "normalize on the union",
        this test fails.
        """
        cohort = _make_cohort(seed=1)
        train = cohort.expression.iloc[:60]
        test = cohort.expression.iloc[60:]
        pre = Preprocessor(top_k=10).fit(train)

        x_train = pre.transform(train)
        x_test = pre.transform(test)

        # Train is centered; test is NOT exactly centered (it was standardized
        # with train's mean/std, not its own).
        np.testing.assert_allclose(x_train.mean(axis=0), 0.0, atol=1e-5)
        assert np.max(np.abs(x_test.mean(axis=0))) > 1e-3, (
            "Test set appears centered — preprocessor is leaking test-fold statistics."
        )

    def test_top_k_exceeds_genes_raises(self) -> None:
        cohort = _make_cohort(n_genes=5)
        with pytest.raises(ValueError, match="exceeds available genes"):
            Preprocessor(top_k=10).fit(cohort.expression)


# --------------------------------------------------------------------------- #
# CV splits and fold tensors
# --------------------------------------------------------------------------- #
class TestCVSplits:
    def test_cv_splits_deterministic_and_stratified(self) -> None:
        cohort = _make_cohort(n_patients=100, event_rate=0.3, seed=2)

        # Determinism: same seed -> same partition.
        a = cv_splits(cohort, n_folds=5, seed=42)
        b = cv_splits(cohort, n_folds=5, seed=42)
        for (ta, te), (tb, te2) in zip(a, b, strict=True):
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(te, te2)

        # Different seed -> different partition (at least one fold differs).
        c = cv_splits(cohort, n_folds=5, seed=43)
        assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c, strict=True))

        # Stratification: event rate in each test fold is close to overall rate.
        overall = cohort.event.mean()
        for _, test_idx in a:
            fold_rate = cohort.event.iloc[test_idx].mean()
            assert abs(fold_rate - overall) < 0.1, (
                f"Fold event rate {fold_rate:.3f} diverges from overall {overall:.3f}; "
                "stratification appears broken."
            )

    def test_each_patient_in_exactly_one_test_fold(self) -> None:
        cohort = _make_cohort(n_patients=50)
        splits = cv_splits(cohort, n_folds=5, seed=0)
        counts = Counter()
        for _, test_idx in splits:
            counts.update(test_idx.tolist())
        assert all(c == 1 for c in counts.values())
        assert len(counts) == cohort.n_patients

    def test_n_folds_too_small_raises(self) -> None:
        cohort = _make_cohort(n_patients=20)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            cv_splits(cohort, n_folds=1, seed=0)


class TestBuildFoldTensors:
    def test_shapes_and_dtype(self) -> None:
        cohort = _make_cohort(n_patients=80, n_genes=30)
        splits = cv_splits(cohort, n_folds=4, seed=0)
        train_idx, test_idx = splits[0]
        ft = build_fold_tensors(cohort, train_idx, test_idx, top_k=10)

        assert ft.x_train.shape == (len(train_idx), 10)
        assert ft.x_test.shape == (len(test_idx), 10)
        assert ft.x_train.dtype == np.float32
        assert ft.time_train.shape == (len(train_idx),)
        assert ft.event_train.shape == (len(train_idx),)
        assert set(np.unique(ft.event_train).tolist()).issubset({0.0, 1.0})
        assert len(ft.selected_genes) == 10

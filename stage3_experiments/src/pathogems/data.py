"""Data loading, preprocessing, and splitting for the omics-only baseline.

Design goals:
    1. **No preprocessing leakage between folds.** Feature selection (top-k
       most variable genes), log transform, and z-score standardization are
       fit on the *training* fold only, then applied to train and test. This
       is the single biggest correctness requirement for any CV-reported
       survival metric and the reason we expose a `Preprocessor` class with
       a scikit-learn-style `fit` / `transform` split rather than a single
       free-function pipeline.
    2. **Deterministic splits.** `cv_splits(..., seed=S)` produces the same
       partition of patients for the same `seed`, regardless of hardware,
       Python version, or number of CPUs. This makes every run log
       reproducible.
    3. **Typed, read-only returns.** Callers get frozen dataclasses, not
       tuples — so "did I pass times before events, or events before times?"
       is impossible. The Cox loss is unforgiving on that mistake.
    4. **Clinical parser tolerant of cBioPortal quirks.** cBioPortal
       clinical TSVs have four comment-prefixed metadata lines before the
       real header; `pandas.read_csv(comment='#')` handles that uniformly.

Scope: this module only handles omics-only inputs for ADR 0001's baseline.
WSI feature loading will arrive in a separate module when Stage 2's WSI
pipeline is ready, so this file never grows conditional multimodal paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Clinical column names as published by cBioPortal PanCancer Atlas 2018.
# These are hard-coded intentionally: they are contracts with an external
# dataset, and silent renames would introduce the exact kind of leakage bug
# that a survival pipeline cannot afford. If cBioPortal ever changes them
# we want a loud KeyError, not a silent zero-event training set.
COL_OS_STATUS = "OS_STATUS"
COL_OS_MONTHS = "OS_MONTHS"
COL_PATIENT_ID = "PATIENT_ID"
COL_SAMPLE_ID = "SAMPLE_ID"

# cBioPortal encodes event as a prefixed string like "1:DECEASED".
_EVENT_MAP = {"0:LIVING": 0, "1:DECEASED": 1}


# --------------------------------------------------------------------------- #
# Typed data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SurvivalCohort:
    """All the data needed to train an omics survival model on one cohort.

    Attributes:
        expression: DataFrame of shape (n_patients, n_genes). Index is
            patient ID (e.g., "TCGA-3C-AAAU"). Columns are HGNC gene
            symbols. Values are raw RSEM-normalized expression (not yet
            log-transformed or z-scored — that happens per-fold).
        time: Series of length n_patients. Values are `OS_MONTHS` floats.
            Index matches `expression.index` exactly, in the same order.
        event: Series of length n_patients. Values are {0, 1} int. `1` =
            death observed during follow-up, `0` = right-censored (alive
            at last follow-up).
        study_id: The cBioPortal study id, for provenance.
    """

    expression: pd.DataFrame
    time: pd.Series
    event: pd.Series
    study_id: str

    def __post_init__(self) -> None:
        # Defensive invariants — expensive bugs further down the pipeline
        # are cheaper to prevent than to debug.
        if not (self.expression.index.equals(self.time.index) and self.time.index.equals(self.event.index)):
            raise ValueError("expression / time / event must share an identical index in the same order.")
        if self.event.isin([0, 1]).sum() != len(self.event):
            raise ValueError("event must be 0/1 only.")
        if (self.time < 0).any():
            raise ValueError("time must be non-negative (months).")

    @property
    def n_patients(self) -> int:
        return len(self.time)

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]

    @property
    def event_rate(self) -> float:
        return float(self.event.mean())


@dataclass(frozen=True, slots=True)
class FoldTensors:
    """Numpy arrays ready to hand to PyTorch for one CV fold.

    All arrays are ordered consistently — row i of X_train corresponds to
    time_train[i] and event_train[i].
    """

    x_train: np.ndarray  # (n_train, n_features) float32
    time_train: np.ndarray  # (n_train,) float32
    event_train: np.ndarray  # (n_train,) float32 (0.0 / 1.0)
    x_test: np.ndarray  # (n_test, n_features) float32
    time_test: np.ndarray  # (n_test,) float32
    event_test: np.ndarray  # (n_test,) float32
    selected_genes: list[str]  # names of the n_features genes, in column order


# --------------------------------------------------------------------------- #
# cBioPortal file parsers
# --------------------------------------------------------------------------- #
def load_expression_matrix(path: Path) -> pd.DataFrame:
    """Load a cBioPortal RSEM expression matrix.

    File layout:
        Hugo_Symbol<TAB>Entrez_Gene_Id<TAB>TCGA-XX-XXXX-01<TAB>...
        TP53<TAB>7157<TAB>123.4<TAB>...
        ...

    Returns:
        DataFrame of shape (n_samples, n_genes). Rows are sample IDs
        (transposed from the on-disk layout, because every consumer in ML
        wants samples-as-rows). Columns are HGNC symbols.

    Notes:
        - Duplicated gene symbols (rare, but it happens with read-through
          transcripts) are collapsed by taking the mean — this is what
          every TCGA pipeline I have seen does, and the alternatives (first
          / last / error) are either arbitrary or block training unnecessarily.
        - Rows with a missing `Hugo_Symbol` (~10 such rows in BRCA) are
          dropped, since we key features by gene symbol.
    """
    df = pd.read_csv(path, sep="\t")
    if "Hugo_Symbol" not in df.columns:
        raise ValueError(f"{path} does not look like a cBioPortal RSEM matrix (no Hugo_Symbol column).")

    # Drop the non-feature metadata columns; everything else is a sample.
    meta_cols = [c for c in ("Hugo_Symbol", "Entrez_Gene_Id") if c in df.columns]
    df = df.dropna(subset=["Hugo_Symbol"])
    # Collapse duplicate symbols by taking the mean of the numeric columns.
    sample_cols = [c for c in df.columns if c not in meta_cols]
    df = df.groupby("Hugo_Symbol", as_index=True)[sample_cols].mean(numeric_only=True)
    # Transpose so rows are samples, columns are genes (ML convention).
    return df.T


def load_clinical_patient(path: Path) -> pd.DataFrame:
    """Load `data_clinical_patient.txt` and return a tidy patient-indexed frame.

    cBioPortal prefixes four metadata lines with `#` before the true
    column header. `read_csv(comment="#")` skips them. We return only the
    columns needed for survival; additional columns can be joined in later.
    """
    df = pd.read_csv(path, sep="\t", comment="#")
    required = {COL_PATIENT_ID, COL_OS_STATUS, COL_OS_MONTHS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required clinical columns: {sorted(missing)}")
    return df[[COL_PATIENT_ID, COL_OS_STATUS, COL_OS_MONTHS]].copy()


def load_clinical_sample(path: Path) -> pd.DataFrame:
    """Load `data_clinical_sample.txt` and return a sample -> patient mapping."""
    df = pd.read_csv(path, sep="\t", comment="#")
    required = {COL_SAMPLE_ID, COL_PATIENT_ID}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required mapping columns: {sorted(missing)}")
    return df[[COL_SAMPLE_ID, COL_PATIENT_ID]].copy()


# --------------------------------------------------------------------------- #
# Cohort assembly
# --------------------------------------------------------------------------- #
def assemble_cohort(
    expression_path: Path,
    clinical_patient_path: Path,
    clinical_sample_path: Path,
    study_id: str,
) -> SurvivalCohort:
    """Join expression + clinical into a single per-patient training table.

    Steps:
        1. Load the three files.
        2. Collapse expression from one-row-per-sample to one-row-per-patient
           by averaging technical/biological replicates. TCGA BRCA has a
           small number of patients with both primary-tumor and
           normal-tissue samples; for the omics-only baseline we keep only
           primary-tumor samples (sample-type code "01" in the barcode's
           4th block). This is the convention used across TCGA-BRCA
           survival papers.
        3. Join to clinical on patient id.
        4. Drop patients missing either `OS_STATUS` or `OS_MONTHS`.
        5. Map `OS_STATUS` strings to 0/1 and `OS_MONTHS` to float.
    """
    expr = load_expression_matrix(expression_path)
    patient_clin = load_clinical_patient(clinical_patient_path)
    sample_map = load_clinical_sample(clinical_sample_path)

    # Keep only primary-tumor samples. TCGA barcodes embed a sample-type
    # code at position 13-14 of the sample id (e.g. "-01" = primary tumor,
    # "-11" = solid normal). For safety we parse the full barcode rather
    # than slicing by index.
    is_primary = sample_map[COL_SAMPLE_ID].str.extract(r"-(\d{2})[A-Z]?$")[0] == "01"
    primary_samples = sample_map.loc[is_primary, [COL_SAMPLE_ID, COL_PATIENT_ID]]

    # Restrict expression to primary samples we have mappings for.
    expr = expr.loc[expr.index.intersection(primary_samples[COL_SAMPLE_ID])]
    # Attach patient id to each expression row and average per patient.
    sample_to_patient = primary_samples.set_index(COL_SAMPLE_ID)[COL_PATIENT_ID]
    expr = expr.assign(_patient=sample_to_patient.loc[expr.index].to_numpy())
    patient_expr = expr.groupby("_patient", sort=True).mean(numeric_only=True)
    patient_expr.index.name = COL_PATIENT_ID

    # Join on patient id; drop rows missing survival.
    clin = patient_clin.set_index(COL_PATIENT_ID)
    clin = clin.loc[clin[COL_OS_STATUS].isin(_EVENT_MAP.keys())]
    clin = clin.dropna(subset=[COL_OS_MONTHS])

    # Align indices.
    common = patient_expr.index.intersection(clin.index)
    patient_expr = patient_expr.loc[common].sort_index()
    clin = clin.loc[common].sort_index()

    event = clin[COL_OS_STATUS].map(_EVENT_MAP).astype(int)
    time = clin[COL_OS_MONTHS].astype(float)

    return SurvivalCohort(expression=patient_expr, time=time, event=event, study_id=study_id)


# --------------------------------------------------------------------------- #
# Preprocessing (fit-on-train-only)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Preprocessor:
    """Top-k variable-gene selection + log transform + per-gene z-score.

    `fit` must be called on training data only. `transform` then applies
    the *training-fold* statistics to arbitrary data. Calling `transform`
    before `fit` raises.

    This is the mechanism that prevents preprocessing leakage between CV
    folds (ADR 0004). Every test in tests/test_data.py that touches
    preprocessing checks this property.
    """

    top_k: int
    _selected_genes: list[str] | None = None
    _mean: np.ndarray | None = None  # shape (top_k,), in selected-gene order
    _std: np.ndarray | None = None

    def fit(self, expression: pd.DataFrame) -> Self:
        """Choose the top-k most variable genes on log-transformed expression.

        We compute gene variance on `log2(x + 1)` (not on raw RSEM) because
        raw RSEM variance is dominated by a handful of ultra-high-expression
        housekeeping genes and ends up selecting the same uninformative
        features every time.
        """
        log_expr = np.log2(expression.to_numpy(dtype=np.float64) + 1.0)
        gene_var = log_expr.var(axis=0)  # shape (n_genes,)
        n_genes = log_expr.shape[1]
        if self.top_k > n_genes:
            raise ValueError(f"top_k={self.top_k} exceeds available genes ({n_genes}).")
        top_idx = np.argsort(gene_var)[-self.top_k :]
        # argsort returns ascending; flip so the most variable gene is first.
        top_idx = top_idx[::-1]
        self._selected_genes = [str(g) for g in expression.columns[top_idx]]

        # Compute per-gene mean / std on the selected genes, again on log scale.
        selected_log = log_expr[:, top_idx]
        self._mean = selected_log.mean(axis=0)
        # Guard against zero-variance genes after the selection step (can't
        # happen with this selector, but a future selector might allow it).
        self._std = selected_log.std(axis=0)
        self._std = np.where(self._std < 1e-8, 1.0, self._std)
        return self

    def transform(self, expression: pd.DataFrame) -> np.ndarray:
        if self._selected_genes is None or self._mean is None or self._std is None:
            raise RuntimeError("Preprocessor.transform called before fit().")
        # Select the training-fold genes. Missing genes -> KeyError rather
        # than silently filling zeros; loudness beats silence in a survival
        # pipeline.
        selected = expression[self._selected_genes].to_numpy(dtype=np.float64)
        selected = np.log2(selected + 1.0)
        return ((selected - self._mean) / self._std).astype(np.float32)

    @property
    def selected_genes(self) -> list[str]:
        if self._selected_genes is None:
            raise RuntimeError("Preprocessor has not been fit yet.")
        return list(self._selected_genes)


# --------------------------------------------------------------------------- #
# Cross-validation splitter
# --------------------------------------------------------------------------- #
def cv_splits(
    cohort: SurvivalCohort,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified-by-event k-fold CV indices (fold_id -> (train_idx, test_idx)).

    Stratification is on the event indicator, not on time — this keeps the
    event rate roughly constant across folds (see ADR 0004). We use
    scikit-learn's `StratifiedKFold` with `shuffle=True` and a fixed
    `random_state` so the split is a pure function of `(cohort, seed)`.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (got {n_folds}).")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = cohort.event.to_numpy()
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(np.zeros_like(y), y)]


def build_fold_tensors(
    cohort: SurvivalCohort,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    top_k: int,
) -> FoldTensors:
    """Fit preprocessing on `train_idx`, apply to both, return numpy tensors.

    This is the single entry point the trainer uses per fold — keeping it
    here (rather than inline in the training loop) means leakage is
    impossible from inside `train.py`.
    """
    expr_train = cohort.expression.iloc[train_idx]
    expr_test = cohort.expression.iloc[test_idx]
    pre = Preprocessor(top_k=top_k).fit(expr_train)

    return FoldTensors(
        x_train=pre.transform(expr_train),
        time_train=cohort.time.iloc[train_idx].to_numpy(dtype=np.float32),
        event_train=cohort.event.iloc[train_idx].to_numpy(dtype=np.float32),
        x_test=pre.transform(expr_test),
        time_test=cohort.time.iloc[test_idx].to_numpy(dtype=np.float32),
        event_test=cohort.event.iloc[test_idx].to_numpy(dtype=np.float32),
        selected_genes=pre.selected_genes,
    )

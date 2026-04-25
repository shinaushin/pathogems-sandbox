"""Data loading, preprocessing, and splitting for the omics-only baseline.

Design goals:
    1. **No preprocessing leakage between folds.** Feature selection (top-k
       most variable genes), log transform, and robust z-score normalization
       are fit on the *training* fold only, then applied to train and test.
       This is the single biggest correctness requirement for any CV-reported
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

Cohort-level QC pipeline (apply in this order before cv_splits):
    1. assemble_cohort()           — raw join of expression + clinical
    2. filter_zero_time_patients() — drop OS_MONTHS == 0 (data artefacts)
    3. remove_outlier_samples()    — PCA-based MAD outlier removal
    4. clip_survival_time()        — administrative censoring at max_months

Per-fold preprocessing (inside build_fold_tensors, fit on training fold only):
    5. Minimum-expression filter   — keep genes expressed in ≥ fraction of samples
    6. Variance-based gene selection — top-k most variable genes (log scale)
    7. Robust z-score              — (x - median) / MAD, per gene, per fold

Scope: this module only handles omics-only inputs for ADR 0001's baseline.
WSI feature loading will arrive in a separate module when Stage 2's WSI
pipeline is ready, so this file never grows conditional multimodal paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

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
        if not (
            self.expression.index.equals(self.time.index)
            and self.time.index.equals(self.event.index)
        ):
            raise ValueError(
                "expression / time / event must share an identical index in the same order."
            )
        if self.event.isin([0, 1]).sum() != len(self.event):
            raise ValueError("event must be 0/1 only.")
        if (self.time < 0).any():
            raise ValueError("time must be non-negative (months).")

    @property
    def n_patients(self) -> int:
        """Number of patients in the cohort."""
        return len(self.time)

    @property
    def n_genes(self) -> int:
        """Number of gene features in the expression matrix."""
        return int(self.expression.shape[1])

    @property
    def event_rate(self) -> float:
        """Fraction of patients who experienced the event (died)."""
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
        raise ValueError(
            f"{path} does not look like a cBioPortal RSEM matrix (no Hugo_Symbol column)."
        )

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

    Call the cohort-level QC helpers after this function to apply further
    quality filters before CV splitting:
        filter_zero_time_patients() → remove_outlier_samples() → clip_survival_time()
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
    expr = expr.loc[expr.index.intersection(primary_samples[COL_SAMPLE_ID].tolist())]
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
# Cohort-level QC helpers (apply before cv_splits)
# --------------------------------------------------------------------------- #
def filter_zero_time_patients(cohort: SurvivalCohort) -> SurvivalCohort:
    """Remove patients whose recorded survival time is zero (QC suggestion #2).

    OS_MONTHS == 0 is a known artefact in TCGA data — typically caused by
    administrative rounding or data-entry errors rather than patients who
    genuinely survived zero months. These rows can destabilise the Cox
    partial likelihood (which computes risk sets at each observed death time;
    time-zero events collapse the first risk set to a single patient, making
    the log-likelihood undefined or numerically extreme).

    Returns a new SurvivalCohort with zero-time patients removed.
    """
    mask = cohort.time > 0.0
    n_removed = int((~mask).sum())
    if n_removed > 0:
        log.info(
            "filter_zero_time_patients: removed %d patient(s) with OS_MONTHS == 0 "
            "(data artefacts, not genuine zero-month survivors). %d remain.",
            n_removed,
            int(mask.sum()),
        )
    idx = cohort.expression.index[mask]
    return SurvivalCohort(
        expression=cohort.expression.loc[idx],
        time=cohort.time.loc[idx],
        event=cohort.event.loc[idx],
        study_id=cohort.study_id,
    )


def remove_outlier_samples(
    cohort: SurvivalCohort,
    n_components: int = 10,
    mad_threshold: float = 5.0,
) -> SurvivalCohort:
    """Flag and remove expression outlier patients using PCA + MAD filtering (QC suggestion #1).

    Algorithm:
        1. Log-transform the full expression matrix: log₂(RSEM + 1).
        2. Fit PCA on all patients, keeping `n_components` principal components.
        3. For each PC, compute the median and MAD (median absolute deviation)
           of the patient scores across that component.
        4. A patient is flagged as an outlier if *any* of their PC scores lies
           more than `mad_threshold` MADs from the PC median.
        5. Flagged patients are removed from the cohort.

    Why PCA + MAD rather than raw expression + z-score:
        Raw high-dimensional expression distances are dominated by a handful
        of ultra-high-variance genes, which can mask globally aberrant samples.
        Projecting into PCA space first compresses the information into a small
        number of orthogonal axes and normalises the influence of individual
        genes. MAD (rather than standard deviation) is then used because it is
        itself robust to the outliers we are trying to detect — standard
        deviation is pulled upward by outlier values, which would lower the
        z-score of the very samples we want to flag.

    Args:
        cohort: Input cohort.
        n_components: Number of PCA components to inspect. Default 10 is
            sufficient to capture the major axes of expression variation in
            TCGA-BRCA while remaining sensitive to global outliers.
        mad_threshold: Number of MADs beyond which a sample is flagged.
            5.0 is deliberately conservative — equivalent to roughly ±7s
            for Gaussian data — to avoid removing genuine biological
            extremes. Only clear technical artefacts should be caught.

    Returns:
        New SurvivalCohort with outlier patients removed.
    """
    log_expr = np.log2(cohort.expression.to_numpy(dtype=np.float64) + 1.0)
    n_patients, n_genes = log_expr.shape
    n_comp = min(n_components, n_patients - 1, n_genes)

    # Compact PCA via truncated SVD (no sklearn dependency).
    # Centre the matrix, then project onto the top n_comp right singular vectors.
    X_c = log_expr - log_expr.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    scores = X_c @ Vt[:n_comp].T  # (n_patients, n_comp)

    # Per-PC robust statistics.
    pc_medians = np.median(scores, axis=0)  # (n_comp,)
    pc_mads = np.median(np.abs(scores - pc_medians), axis=0)  # (n_comp,)
    # Guard against zero-MAD PCs (e.g. a constant PC due to rank deficiency).
    pc_mads = np.where(pc_mads < 1e-8, 1.0, pc_mads)

    # Flag patients with any PC score > mad_threshold MADs from the median.
    robust_z = np.abs(scores - pc_medians) / pc_mads  # (n_patients, n_comp)
    is_outlier = (robust_z > mad_threshold).any(axis=1)
    n_outliers = int(is_outlier.sum())

    if n_outliers > 0:
        outlier_ids = cohort.expression.index[is_outlier].tolist()
        log.info(
            "remove_outlier_samples: flagged %d patient(s) with a PC score "
            "> %.1f MADs from the median (PCA components 1-%d). "
            "Removed: %s. %d remain.",
            n_outliers,
            mad_threshold,
            n_comp,
            outlier_ids,
            n_patients - n_outliers,
        )
    else:
        log.info(
            "remove_outlier_samples: no outlier patients detected "
            "(threshold=%.1f MADs, %d components). All %d patients retained.",
            mad_threshold,
            n_comp,
            n_patients,
        )

    keep_idx = cohort.expression.index[~is_outlier]
    return SurvivalCohort(
        expression=cohort.expression.loc[keep_idx],
        time=cohort.time.loc[keep_idx],
        event=cohort.event.loc[keep_idx],
        study_id=cohort.study_id,
    )


def clip_survival_time(
    cohort: SurvivalCohort,
    max_months: float = 120.0,
) -> SurvivalCohort:
    """Administratively censor patients whose follow-up exceeds max_months (QC suggestion #5).

    Why clip at all:
        In TCGA-BRCA, a small fraction of patients have follow-up times
        exceeding 10 years (120 months). These long-tail observations exert
        disproportionate leverage on neural network training — the model can
        dedicate capacity to distinguishing patients at 130 months from patients
        at 140 months, which is clinically irrelevant and statistically noisy
        (very few patients survive that long, so the estimates are unreliable).
        Clipping at 10 years is standard in TCGA-BRCA survival papers and focuses
        the model's attention on clinically actionable timeframes.

    How censoring works:
        A patient whose death was observed at month 150 is *not* treated as if
        they survived 120 months — that would be falsifying data. Instead, we
        convert them to a censored observation at 120 months: their event
        indicator becomes 0 (we stop observing them at the cut-off, just as if
        the study had ended then). This is proper administrative censoring and
        is consistent with how Cox models handle right-censoring.

    Args:
        cohort: Input cohort.
        max_months: Upper limit on follow-up time. Default 120 (10 years).

    Returns:
        New SurvivalCohort with times and events adjusted.
    """
    time = cohort.time.copy()
    event = cohort.event.copy()

    beyond = time > max_months
    n_clipped = int(beyond.sum())
    n_event_converted = int((beyond & (event == 1)).sum())

    if n_clipped > 0:
        log.info(
            "clip_survival_time: clipped %d patient(s) with OS_MONTHS > %.0f to %.0f "
            "months (administrative censoring). %d of those had event=1 converted to 0.",
            n_clipped,
            max_months,
            max_months,
            n_event_converted,
        )

    event.loc[beyond] = 0
    time.loc[beyond] = max_months

    return SurvivalCohort(
        expression=cohort.expression,
        time=time,
        event=event,
        study_id=cohort.study_id,
    )


# --------------------------------------------------------------------------- #
# Preprocessing (fit-on-train-only)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Preprocessor:
    """Min-expression filter + top-k variable-gene selection + robust z-score.

    `fit` must be called on training data only. `transform` then applies
    the *training-fold* statistics to arbitrary data. Calling `transform`
    before `fit` raises.

    This is the mechanism that prevents preprocessing leakage between CV
    folds (ADR 0004). Every test in tests/test_data.py that touches
    preprocessing checks this property.

    Per-fold steps (all fit on training data only):

    Step 1 — Minimum expression filter (suggestion #3):
        Genes expressed in fewer than `min_expressed_fraction` of training
        samples (where "expressed" means log₂(RSEM + 1) > 1, i.e. RSEM > 1)
        are excluded from the candidate gene universe before variance-based
        selection. These genes are near-zero across most patients; they add
        noise to variance calculations and tend to select uninformative
        low-expression genes that happen to be non-zero in a small subset.

    Step 2 — Log transform + variance-based gene selection:
        Among expressed genes, variance is computed on log₂(RSEM + 1) and
        the top-k most variable genes are selected. Log scale is used because
        raw RSEM variance is dominated by ultra-high-expression housekeeping
        genes, ending up selecting the same uninformative features every fold.

    Step 3 — Robust z-score (suggestion #4):
        Each selected gene is centred by its training-fold median and scaled by
        its training-fold MAD (median absolute deviation). Compared to standard
        mean/std z-scoring, median/MAD is robust to outlier samples within a
        fold: a single patient with an aberrant expression value cannot pull the
        centre or scale estimate far from the bulk of the data.
    """

    top_k: int
    min_expressed_fraction: float = 0.20
    _selected_genes: list[str] | None = None
    _center: np.ndarray | None = None  # shape (top_k,) — training-fold median per gene
    _scale: np.ndarray | None = None  # shape (top_k,) — training-fold MAD per gene

    def fit(self, expression: pd.DataFrame) -> Self:
        """Fit min-expression filter, gene selector, and robust scaler on training data.

        Args:
            expression: DataFrame of raw RSEM values, shape (n_train, n_genes).

        Returns:
            self (for chaining: Preprocessor(top_k=500).fit(expr_train))
        """
        raw = expression.to_numpy(dtype=np.float64)
        log_expr = np.log2(raw + 1.0)  # shape (n_train, n_genes)

        # ---- Step 1: Minimum expression filter ----
        # A gene is "expressed" in a sample if log₂(RSEM+1) > 1, i.e. RSEM > 1.
        expressed_frac = (log_expr > 1.0).mean(axis=0)  # shape (n_genes,)
        expressed_mask = expressed_frac >= self.min_expressed_fraction
        n_expressed = int(expressed_mask.sum())
        n_total = log_expr.shape[1]

        if n_expressed == 0:
            raise ValueError(
                f"No genes passed the minimum-expression filter "
                f"(min_expressed_fraction={self.min_expressed_fraction}). "
                f"Check that the expression matrix contains raw RSEM values."
            )
        if n_expressed < self.top_k:
            raise ValueError(
                f"top_k={self.top_k} exceeds available genes after minimum-expression "
                f"filtering ({n_expressed} genes passed out of {n_total} total)."
            )

        log_expressed = log_expr[:, expressed_mask]  # (n_train, n_expressed)
        expressed_gene_names = expression.columns[expressed_mask]

        # ---- Step 2: Variance-based gene selection ----
        gene_var = log_expressed.var(axis=0)  # shape (n_expressed,)
        top_idx = np.argsort(gene_var)[-self.top_k :][::-1]  # descending variance
        self._selected_genes = [str(g) for g in expressed_gene_names[top_idx]]

        # ---- Step 3: Robust z-score statistics (median / MAD) ----
        selected_log = log_expressed[:, top_idx]  # (n_train, top_k)
        self._center = np.median(selected_log, axis=0)  # training median per gene
        abs_dev = np.abs(selected_log - self._center)
        self._scale = np.median(abs_dev, axis=0)  # training MAD per gene
        # Guard against zero-MAD genes (fully constant expression in training fold).
        self._scale = np.where(self._scale < 1e-8, 1.0, self._scale)

        return self

    def transform(self, expression: pd.DataFrame) -> np.ndarray:
        """Apply training-fold preprocessing to expression data.

        Selects the training-fold genes, log-transforms, and applies the
        training-fold median/MAD normalization.

        Args:
            expression: DataFrame of raw RSEM values. Must contain all genes
                in `selected_genes` as columns (KeyError otherwise — loud
                beats silent in a survival pipeline).

        Returns:
            Float32 array of shape (n_samples, top_k), robust z-scored.
        """
        if self._selected_genes is None or self._center is None or self._scale is None:
            raise RuntimeError("Preprocessor.transform called before fit().")
        # Select the training-fold genes. Missing genes -> KeyError rather
        # than silently filling zeros; loudness beats silence in a survival
        # pipeline.
        selected = expression[self._selected_genes].to_numpy(dtype=np.float64)
        selected = np.log2(selected + 1.0)
        return ((selected - self._center) / self._scale).astype(np.float32)

    @property
    def selected_genes(self) -> list[str]:
        """Ordered list of gene names chosen by ``fit``.  Raises if not yet fit."""
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

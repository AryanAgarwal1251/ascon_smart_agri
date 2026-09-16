"""Dataset characterisation report (Phase 1, Section III-B).

Produces the ground-truth facts every later document must cite: actual column names/types,
per-column null and zero-variance checks, the *exact* duplicate count, the label vocabulary
with counts, and the numeric correlation matrix. Figures quoted from the literature are
treated as hypotheses to confirm --- nothing enters a later doc unless this script produced it.

Exit criterion (Phase 1): a reproducible report exists and its numbers are used downstream.

Scope note: ``dataset_root`` is scanned recursively for every ``*.csv`` file and all of them are
pooled into one corpus before any statistic is computed. This is deliberate: CICIoT2023 is
natively distributed as many per-capture CSV parts, and even when a mirror arrives already split
into train/test/validation files (as happened in this project), Section III-B2 requires
deduplication to run on the pooled corpus *before* our own split is drawn (R3) -- consuming a
pre-existing external split without re-pooling would make that guarantee unverifiable. This
function only reports facts; it does not split or deduplicate (that is Phase 2, ``data/dedup.py``
and ``data/split.py``).

Computed in a single streaming pass with ``pandas.read_csv(chunksize=...)`` so peak memory stays
bounded regardless of corpus size (R1):
    * null counts and zero-variance -- exact, dtype-agnostic (a column is zero-variance iff its
      running min == running max, which holds iff it has at most one distinct value).
    * exact duplicate count -- rows are hashed with ``pandas.util.hash_pandas_object`` (64-bit,
      order-sensitive over all columns including the label). The report is the count of rows
      beyond each hash's first occurrence, i.e. how many rows ``dedup.deduplicate`` would drop.
      Collision probability at corpus scale (~10 million rows, 64-bit hash) is on the order of
      1e-6 by the birthday bound and is treated as negligible for a reporting statistic; the
      leakage-critical hash used by the R3 gate (``data/dedup.record_hash``) is a separate,
      Phase-2 concern.
    * label vocabulary + counts -- accumulated with a running ``collections.Counter``.
    * numeric correlation matrix -- Pearson, accumulated via running sum/sum-of-squares/
      sum-of-products (exact, one pass, memory O(k^2) for k numeric columns). Rows with a null OR
      a non-finite value (+/-inf) in any numeric column are excluded from this accumulation only
      (both are already reported separately, via ``null_counts`` and ``infinite_counts``). This
      is a reporting statistic distinct from Phase 2's Spearman correlation *pruning*
      (Section III-C, Stage 2, tau=0.95), which runs later, on training blocks only, for feature
      selection rather than characterisation.
    * infinite counts -- some CICIoT2023 flow-rate features (e.g. ``Rate``) are computed as a
      division by flow duration and are +/-inf for near-zero-duration flows; this is a genuine
      property of the raw data (confirmed present in the official UNB distribution, absent from
      at least one third-party mirror seen in this project), not a parsing artefact, so it is
      counted per column rather than silently dropped or coerced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class CharacterizationReport:
    """Structured result of characterisation; serialised into the run manifest.

    Field mappings:
        n_records              -- total row count
        columns                -- column name -> dtype string
        null_counts            -- column name -> null count
        infinite_counts        -- numeric column name -> +/-inf count
        zero_variance_columns  -- list of column names with same data
        exact_duplicate_count  -- count of duplicate rows
        label_counts           -- label string -> occurrence count
        imbalance_ratio        -- max(label_counts) / min(label_counts)
        correlation_matrix     -- column name -> {column name -> Pearson r}
    """

    n_records: int
    columns: dict[str, str]  # column name -> dtype string
    null_counts: dict[str, int]
    infinite_counts: dict[str, int]  # per numeric column, +/-inf value count
    zero_variance_columns: list[str]
    exact_duplicate_count: int
    label_counts: dict[str, int]
    imbalance_ratio: float  # max class count / min class count (Section II-C)
    correlation_matrix: dict[str, dict[str, float]]  # Pearson, numeric columns only


def _discover_csv_parts(dataset_root: Path) -> list[Path]:
    parts = sorted(dataset_root.rglob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"no *.csv files found under {dataset_root}")
    return parts


def characterize_dataset(dataset_root: Path, chunk_size: int) -> CharacterizationReport:
    """Scan the raw CICIoT2023 CSV parts in fixed-size chunks and report the facts above."""
    parts = _discover_csv_parts(dataset_root)

    columns: dict[str, str] | None = None
    numeric_cols: list[str] = []
    null_counts: Counter[str] = Counter()
    infinite_counts: Counter[str] = Counter()
    col_min: dict[str, object] = {}
    col_max: dict[str, object] = {}
    label_counts: Counter[str] = Counter()
    hash_chunks: list[npt.NDArray[np.uint64]] = []
    n_records = 0

    n_valid = 0
    sum_vec: npt.NDArray[np.float64] | None = None
    sumsq_vec: npt.NDArray[np.float64] | None = None
    sumprod_mat: npt.NDArray[np.float64] | None = None

    for part in parts:
        for chunk in pd.read_csv(part, chunksize=chunk_size):
            if columns is None:
                columns = {str(name): str(dtype) for name, dtype in chunk.dtypes.items()}
                numeric_cols = [c for c in chunk.columns if pd.api.types.is_numeric_dtype(chunk[c])]
                sum_vec = np.zeros(len(numeric_cols), dtype=np.float64)
                sumsq_vec = np.zeros(len(numeric_cols), dtype=np.float64)
                sumprod_mat = np.zeros((len(numeric_cols), len(numeric_cols)), dtype=np.float64)
            elif list(chunk.columns) != list(columns.keys()):
                raise ValueError(
                    f"{part}: column layout {list(chunk.columns)} does not match "
                    f"the first file's {list(columns.keys())}"
                )

            n_records += len(chunk)

            for col in chunk.columns:
                series = chunk[col]
                null_counts[col] += int(series.isna().sum())
                non_null = series.dropna()
                if non_null.empty:
                    continue
                c_min, c_max = non_null.min(), non_null.max()
                col_min[col] = c_min if col not in col_min else min(col_min[col], c_min)
                col_max[col] = c_max if col not in col_max else max(col_max[col], c_max)
                if col in numeric_cols:
                    infinite_counts[col] += int(np.isinf(non_null.to_numpy(dtype=np.float64)).sum())

            if "label" in chunk.columns:
                label_counts.update(chunk["label"].astype(str))

            row_hashes: npt.NDArray[np.uint64] = np.asarray(
                pd.util.hash_pandas_object(chunk, index=False), dtype=np.uint64
            )
            hash_chunks.append(row_hashes)

            if numeric_cols:
                block = chunk[numeric_cols].to_numpy(dtype=np.float64)
                mask = np.isfinite(block).all(axis=1)
                block = block[mask]
                if block.shape[0]:
                    n_valid += block.shape[0]
                    assert sum_vec is not None and sumsq_vec is not None and sumprod_mat is not None
                    sum_vec += block.sum(axis=0)
                    sumsq_vec += (block**2).sum(axis=0)
                    sumprod_mat += block.T @ block

    assert columns is not None  # at least one chunk was read, else the glob above would have raised

    all_hashes = np.concatenate(hash_chunks) if hash_chunks else np.array([], dtype=np.uint64)
    exact_duplicate_count = n_records - int(np.unique(all_hashes).shape[0])

    zero_variance_columns = [c for c in columns if c in col_min and col_min[c] == col_max[c]]

    correlation_matrix: dict[str, dict[str, float]] = {}
    if numeric_cols and n_valid > 0:
        assert sum_vec is not None and sumsq_vec is not None and sumprod_mat is not None
        mean = sum_vec / n_valid
        cov = sumprod_mat / n_valid - np.outer(mean, mean)
        var = np.diag(cov).copy()
        std = np.sqrt(np.maximum(var, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(std, std)
        # Zero-variance columns have undefined correlation; report them as 0.0 rather than NaN.
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        correlation_matrix = {
            row_name: {col_name: float(corr[i, j]) for j, col_name in enumerate(numeric_cols)}
            for i, row_name in enumerate(numeric_cols)
        }

    if label_counts:
        counts = label_counts.values()
        imbalance_ratio = max(counts) / min(counts)
    else:
        imbalance_ratio = float("nan")

    return CharacterizationReport(
        n_records=n_records,
        columns=columns,
        null_counts=dict(null_counts),
        infinite_counts=dict(infinite_counts),
        zero_variance_columns=zero_variance_columns,
        exact_duplicate_count=exact_duplicate_count,
        label_counts=dict(label_counts),
        imbalance_ratio=imbalance_ratio,
        correlation_matrix=correlation_matrix,
    )

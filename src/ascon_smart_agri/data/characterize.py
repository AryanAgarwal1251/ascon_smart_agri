"""Dataset characterisation report (Phase 1, Section III-B).

Produces the ground-truth facts every later document must cite: actual column names/types,
per-column null and zero-variance checks, the *exact* duplicate count, the label vocabulary
with counts, and the numeric correlation matrix. Figures quoted from the literature are
treated as hypotheses to confirm --- nothing enters a later doc unless this script produced it.

Exit criterion (Phase 1): a reproducible report exists and its numbers are used downstream.

TODO(Phase 1): implement chunked characterisation over the raw CSV parts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterizationReport:
    """Structured result of characterisation; serialised into the run manifest."""

    n_records: int
    columns: dict[str, str]  # column name -> dtype string
    null_counts: dict[str, int]
    zero_variance_columns: list[str]
    exact_duplicate_count: int
    label_counts: dict[str, int]
    imbalance_ratio: float  # max class count / min class count (Section II-C)


def characterize_dataset(dataset_root: Path, chunk_size: int) -> CharacterizationReport:
    """Scan the raw CICIoT2023 CSV parts in fixed-size chunks and report the facts above."""
    del dataset_root, chunk_size
    raise NotImplementedError("Phase 1: dataset characterisation not implemented yet.")

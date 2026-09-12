"""Stratified, per-class-capped subsampler (Phase 1/2, Section III-B1, risk R1).

The full corpus (~46.7M records) does not fit in workstation memory, so we read fixed-size
CSV parts and apply a per-class cap kappa_c that compresses the dominant DDoS classes while
retaining every available instance of the rare families. The seed and the resulting per-class
counts are recorded in the run manifest.

TODO(Phase 1/2): chunked reader + reservoir/streaming per-class cap; emit counts to manifest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def stratified_capped_subsample(
    dataset_root: Path,
    *,
    target: int,
    per_class_cap: int,
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the capped subsample and its realised per-class counts.

    Returns:
        A ``(frame, per_class_counts)`` pair; ``per_class_counts`` goes into the manifest.
    """
    del dataset_root, target, per_class_cap, chunk_size, seed
    raise NotImplementedError("Phase 1/2: capped subsampling not implemented yet.")

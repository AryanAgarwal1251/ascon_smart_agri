"""Stratified, per-class-capped subsampler (Phase 2, Section III-B1, risk R1).

The full corpus (~46.7M records) does not fit in workstation memory, so we read whole CSV
parts, in a fixed order, and apply a per-class cap kappa_c that compresses the dominant DDoS
classes while retaining every available instance of the rare families. The seed and the
resulting per-class counts are recorded in the run manifest.

**Dataset root decision (flagged per Golden Rule 1, confirmed with the user):** this module
targets the official UNB raw distribution (``data/ciciot2023_raw/``, 309 per-capture CSVs, one
directory per attack type), not the pre-merged Kaggle mirror (``data/ciciot2023/``) that
``configs/default.yaml`` pointed at previously. Three things in the paper text make this the
only choice consistent with Section III-B/D: (1) Section III-B1 says the subsample is drawn "by
reading whole CSV parts in fixed order" -- a description of many per-capture files, which only
the raw distribution has (the Kaggle mirror is three already-merged files); (2) Section III-D
requires windows to stay "within a single source file", which is only recoverable when each CSV
*is* one capture, as in the raw distribution -- the Kaggle mirror destroyed that identity when
it was merged; (3) the raw distribution's record count (46,776,700) matches the paper's stated
~46.7M almost exactly, confirming it -- not the Kaggle mirror's ~17% subsample -- is the
benchmark the paper describes. See ``CHANGELOG.md`` for the full comparison.

**Label derivation (flagged per Golden Rule 1):** the raw distribution has no ``label`` column
and no per-row identifier; the label lives only in each CSV part's *filename*, not its parent
directory. The two disagree for benign traffic (directory ``Benign_Final/`` contains files named
``BenignTraffic*.pcap.csv``), so labels are derived from the filename, not the directory: strip
an optional trailing ``.pcap.csv``/``.csv`` suffix, then a trailing run of digits (the part
number), then one trailing ``-``/``_`` separator left behind. This reproduces all 34 of the raw
distribution's implied classes and matches the Kaggle mirror's own 34-label vocabulary almost
exactly (e.g. ``DDoS-PSHACK_Flood``, ``DDoS-RSTFINFlood``), which is a useful cross-check that
the derivation is correct, even though the Kaggle mirror itself is not used as data.

**Per-class cap mechanic (flagged per Golden Rule 1):** parts are read in natural part-number
order (``Foo.pcap.csv`` before ``Foo1.pcap.csv`` before ``Foo2.pcap.csv``, ...) and a part, once
started, is always read to completion -- ``chunk_size`` only bounds peak memory *within* a
single part's read, it never stops a read mid-part. Reading stops once a class's accumulated
row count reaches or exceeds ``per_class_cap``. Because some CICIoT2023 attack-type directories
hold parts far larger than a typical ``per_class_cap`` (a single DDoS-ICMP_Flood part alone is
~268k rows against a default cap of 200k), reading whole parts alone would let the cap overshoot
by up to one whole part per class -- which would make "cap" a misnomer. To keep it a real
ceiling, once the accumulated rows for a class exceed the cap, a *seeded, order-preserving*
random trim (no shuffling: kept rows keep their original relative order) drops the excess down
to exactly ``per_class_cap``. This is the only place ``seed`` is used, since the read order and
per-part inclusion are otherwise fully deterministic.

``target`` is not used to derive ``per_class_cap`` -- the paper does not specify that
relationship, and the two are independent knobs in ``configs/base.py``. It is only a sanity
check: a warning is raised if the realised total deviates from it by more than 50%, so a
badly-tuned cap does not silently produce a wrong-sized subsample.

**Source-file identity (resolves a gap flagged in ``data/dedup.py``):** every row carries a
``source_file`` column (the part's path relative to ``dataset_root``), so Section III-D's "never
window across a source file" rule is enforceable downstream, and a ``label`` column, so
``data/dedup.py`` and ``data/split.py`` can operate on the result unchanged.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_CSV_SUFFIX = re.compile(r"(?:\.pcap)?\.csv$", re.IGNORECASE)
_TRAILING_DIGITS = re.compile(r"\d+$")


def _label_from_filename(path: Path) -> str:
    """Derive a CICIoT2023 label from a part's filename (see the module docstring)."""
    stem = _CSV_SUFFIX.sub("", path.name)
    stem = _TRAILING_DIGITS.sub("", stem)
    return stem.rstrip("-_")


def _part_number(path: Path) -> int:
    """Natural sort key: the trailing digit run in the filename, or 0 if there is none.
    eg: BenignTraffic.csv -> 0
        BenignTraffic001.csv -> 1
        BenignTraffic123.csv -> 123
    """
    match = re.search(r"(\d+)(?:\.pcap)?\.csv$", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _discover_parts_by_label(dataset_root: Path) -> dict[str, list[Path]]:
    """Return, per attack type, the paths of every CSV part that belongs to it."""
    groups: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*.csv"):
        groups.setdefault(_label_from_filename(path), []).append(path)
    if not groups:
        raise FileNotFoundError(f"no *.csv files found under {dataset_root}")
    for parts in groups.values():
        parts.sort(key=_part_number)
    return groups


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

    # Validation
    if per_class_cap <= 0:
        raise ValueError(f"per_class_cap must be positive, got {per_class_cap}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")

    parts_by_label = _discover_parts_by_label(dataset_root)
    rng = np.random.default_rng(seed)

    class_frames: list[pd.DataFrame] = []
    per_class_counts: dict[str, int] = {}

    for label in sorted(parts_by_label):
        collected: list[pd.DataFrame] = []
        n_rows = 0
        for path in parts_by_label[label]:
            if n_rows >= per_class_cap:
                break
            rel_path = str(path.relative_to(dataset_root))
            for chunk in pd.read_csv(path, chunksize=chunk_size):
                chunk = chunk.copy()
                chunk["source_file"] = rel_path
                collected.append(chunk)
                n_rows += len(chunk)
            # The part above is always read to completion, even if it pushes n_rows past
            # per_class_cap -- only the *next* part is skipped (see the module docstring).

        class_frame = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
        if len(class_frame) > per_class_cap:
            keep = np.sort(rng.choice(len(class_frame), size=per_class_cap, replace=False))
            class_frame = class_frame.iloc[keep].reset_index(drop=True)
        class_frame["label"] = label

        per_class_counts[label] = len(class_frame)
        class_frames.append(class_frame)

    frame = pd.concat(class_frames, ignore_index=True)

    total = len(frame)
    if not (0.5 * target <= total <= 1.5 * target):
        warnings.warn(
            f"realised subsample total {total} deviates from target {target} by more than "
            "50%; check per_class_cap against this dataset's actual per-class part sizes",
            stacklevel=2,
        )

    return frame, per_class_counts

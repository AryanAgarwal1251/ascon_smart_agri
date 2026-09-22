"""One registered corpus -> one harmonised, leaf-labelled frame (Phase 9 plan, §3 step 2).

The three corpora enter the pipeline through this module and leave it in the same shape:
canonical feature columns (only those the corpus actually supplies -- nothing invented), a
``label`` column holding the corpus's *leaf* label (the taxonomy maps it to the C = 8 family
later, so per-leaf counts stay reportable), and a ``source_file`` column so Section III-D's
"never window across a source file" rule is enforceable downstream. The result drops straight
into ``dedup -> make_blocks -> stratified_block_split``, exactly as Phase 2 left them.

Two roads in, by ``DatasetSpec.granularity``:

* **window** corpora (CICIoT2023, CICIoMT2024): rows are already extractor windows, so the
  Phase 2 capped subsample is reused verbatim, with the corpus's own filename-label rule,
  then ``harmonise_columns`` renames into the canonical vocabulary.
* **packet** corpora (Edge-IIoTset): the file is read as text (only the columns the packet
  derivation needs), placeholders normalised, packets derived, windows of ``window_packets``
  aggregated within label runs, and *then* the per-class cap applied to window rows so the
  cap means the same thing -- rows the model sees -- for every corpus.

The per-class cap is the same one Phase 2 calibrated for CICIoT2023, applied per leaf.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import (
    CANONICAL_COLUMNS,
    DatasetSpec,
    discover_parts,
    harmonise_columns,
    normalise_placeholders,
)
from .packet_windows import (
    EDGE_IIOTSET_SOURCE_COLUMNS,
    aggregate_packet_windows,
    edge_iiotset_packets,
)
from .subsample import stratified_capped_subsample

_KEEP = ("label", "source_file")


def _canonical_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical feature columns the frame has, plus label and source_file, in that order."""
    features = [c for c in CANONICAL_COLUMNS if c in frame.columns]
    return frame[[*features, *_KEEP]]


def _cap_per_leaf(frame: pd.DataFrame, per_class_cap: int, seed: int) -> pd.DataFrame:
    """Seeded draw of at most ``per_class_cap`` rows per leaf, keeping the rows' file order."""
    rng = np.random.default_rng(seed)
    keep_positions: list[np.ndarray] = []
    for _, positions in frame.groupby("label", sort=True).indices.items():
        pos = np.asarray(positions)
        if len(pos) > per_class_cap:
            pos = rng.choice(pos, size=per_class_cap, replace=False)
        keep_positions.append(pos)
    kept = np.sort(np.concatenate(keep_positions))
    return frame.iloc[kept].reset_index(drop=True)


def load_window_corpus(
    spec: DatasetSpec,
    root: Path,
    *,
    per_class_cap: int,
    chunk_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """A window-granularity corpus through the Phase 2 capped subsample and harmonisation."""
    if spec.granularity != "window":
        raise ValueError(f"{spec.name} is a {spec.granularity} corpus, not a window corpus")
    parts = discover_parts(spec, root)
    label_from_path = spec.label_from_filename
    kwargs = {} if label_from_path is None else {"label_from_path": label_from_path}
    with warnings.catch_warnings():
        # The subsampler's target check is Phase 2's CICIoT2023 calibration; per-leaf caps
        # on another corpus land where its class sizes put them, and that is reported.
        warnings.simplefilter("ignore")
        frame, per_leaf = stratified_capped_subsample(
            root,
            target=per_class_cap,
            per_class_cap=per_class_cap,
            chunk_size=chunk_size,
            seed=seed,
            parts=parts,
            **kwargs,
        )
    harmonised = harmonise_columns(frame, spec)
    return _canonical_only(harmonised.frame), per_leaf


def load_packet_corpus(
    spec: DatasetSpec,
    root: Path,
    *,
    per_class_cap: int,
    window_packets: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """A packet-granularity corpus: placeholders -> packets -> windows -> per-leaf cap."""
    if spec.granularity != "packet":
        raise ValueError(f"{spec.name} is a {spec.granularity} corpus, not a packet corpus")
    if spec.label_column is None:
        raise ValueError(f"{spec.name} has no label column; a packet corpus needs one")
    windows: list[pd.DataFrame] = []
    wanted = {*EDGE_IIOTSET_SOURCE_COLUMNS, spec.label_column}
    for part in discover_parts(spec, root):
        raw = pd.read_csv(part, dtype=str, usecols=lambda c: c in wanted)
        raw["label"] = raw[spec.label_column].astype(str)
        raw["source_file"] = str(part.relative_to(root))
        packets, cols = edge_iiotset_packets(normalise_placeholders(raw))
        windows.append(aggregate_packet_windows(packets, cols, window_packets=window_packets))
    frame = _cap_per_leaf(pd.concat(windows, ignore_index=True), per_class_cap, seed)
    harmonised = harmonise_columns(frame, spec)
    out = _canonical_only(harmonised.frame)
    per_leaf = {str(k): int(v) for k, v in out["label"].value_counts().sort_index().items()}
    return out, per_leaf


def load_corpus(
    spec: DatasetSpec,
    root: Path | None = None,
    *,
    per_class_cap: int,
    chunk_size: int,
    window_packets: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Dispatch on granularity; returns ``(frame, per_leaf_counts)`` for the manifest."""
    base = root if root is not None else spec.default_root
    if spec.granularity == "window":
        return load_window_corpus(
            spec, base, per_class_cap=per_class_cap, chunk_size=chunk_size, seed=seed
        )
    return load_packet_corpus(
        spec, base, per_class_cap=per_class_cap, window_packets=window_packets, seed=seed
    )

"""One client's share of the Phase 4 cache, for a node process (Phase 8, plan §5, §7).

A node on a Raspberry Pi needs only *its* partition, unscaled, plus the means to window it
once the sealed scaler round has returned ``(mean, std)``. This module carves that out of
``artifacts/phase4_cache.npz`` deterministically: every node loads the same cache and draws
the same Dirichlet assignment from the same seed (``federated/partition.py``), then keeps the
rows of its own client index. That is a simulation convenience --- on real farms each node
would hold its own capture and no shared file would exist --- and is documented as such.

``sequence_cap`` exists because Phase 4's partitions hold 284k-458k sequences each and a Pi 4
is an order of magnitude slower than the laptop that ran them (plan §7). The cap keeps the
first ``sequence_cap`` windows *in capture order* (no shuffle, Section III-D) so the demo's
per-round time is bounded; the manifest records the cap, and the hardware run is never
presented as reproducing Phase 4's headline figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._types import Array
from ..sequences.windowing import build_windows, contiguity_segments
from .client import FederatedClient
from .partition import dirichlet_block_partition
from .scaler_stats import FeatureStats, local_sufficient_stats


@dataclass(frozen=True)
class ClientPartition:
    """Unscaled rows of one client, with the segment ids windowing needs."""

    client_index: int
    features: Array  # (rows, F) float64, UNSCALED
    labels: Array  # (rows,) int64 class indices
    segments: Array  # from contiguity_segments: windows never cross a segment boundary
    columns: list[str]

    @property
    def stats(self) -> FeatureStats:
        """The only thing that leaves this node before the weights: count/mean/M2."""
        return local_sufficient_stats(self.features)


def _block_strata(block_ids: Array, labels: Array) -> tuple[Array, Array]:
    """``(train_blocks, block_labels)``; mirrors ``scripts/run_phase4.py``'s block_strata."""
    train_blocks = np.unique(block_ids)
    first = np.array([np.flatnonzero(block_ids == b)[0] for b in train_blocks])
    return train_blocks, labels[first]


def load_client_partition(
    cache_path: Path, *, client_index: int, n_clients: int, alpha: float, seed: int
) -> ClientPartition:
    """Load the cache and return client ``client_index``'s unscaled rows."""
    if not 0 <= client_index < n_clients:
        raise ValueError(f"client_index must be in [0, {n_clients}), got {client_index}")
    blob = np.load(cache_path, allow_pickle=False)
    x_tr, y_tr, src_tr, idx_tr, blk_tr = (
        blob["Xtr"],
        blob["ytr"],
        blob["str_"],
        blob["itr"],
        blob["btr"],
    )
    train_blocks, block_labels = _block_strata(blk_tr, y_tr)
    assignment = dirichlet_block_partition(
        block_labels, n_clients=n_clients, alpha=alpha, seed=seed
    )
    mine = train_blocks[np.asarray(assignment[client_index], dtype=int)]
    mask = np.isin(blk_tr, mine)
    if not mask.any():
        raise ValueError(f"client {client_index} received no blocks at alpha={alpha}, seed={seed}")
    return ClientPartition(
        client_index=client_index,
        features=np.asarray(x_tr[mask], dtype=np.float64),
        labels=np.asarray(y_tr[mask], dtype=np.int64),
        segments=contiguity_segments(src_tr[mask], idx_tr[mask]),
        columns=[str(c) for c in blob["cols"]],
    )


def make_federated_client(
    partition: ClientPartition,
    mean: Array,
    std: Array,
    *,
    window: int,
    seed: int,
    hidden_size: int,
    n_classes: int,
    sequence_cap: int | None = None,
    batch_size: int = 1024,
) -> FederatedClient:
    """Scale with the federated ``(mean, std)``, window, cap, and wrap as a client."""
    features = np.asarray(partition.features, dtype=np.float64)
    scaled = ((features - np.asarray(mean, np.float64)) / np.asarray(std, np.float64)).astype(
        np.float32
    )
    seqs, labels = build_windows(scaled, partition.labels, partition.segments, window)
    if sequence_cap is not None and len(seqs) > sequence_cap:
        seqs, labels = seqs[:sequence_cap], labels[:sequence_cap]  # capture order, no shuffle
    return FederatedClient(
        partition.client_index,
        seqs,
        labels,
        seed=seed,
        hidden_size=hidden_size,
        n_classes=n_classes,
        batch_size=batch_size,
    )

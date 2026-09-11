"""Centralised training loop (Phase 3, Section III-E).

Phase 3 is a HARD GATE (III-J4): federation does not begin until single-client detection is
proven, because debugging an aggregation fault and a modelling fault at once is much harder
than either alone. Device-agnostic (CPU-only must suffice; a GPU may be used if present).

TODO(Phase 3): class-weighted CE (w_c = n / (C * n_c), Eq. 18, training data only), training
loop, checkpointing via safetensors, and full evaluation hookup.
"""

from __future__ import annotations

from .._types import Array


def class_weights(label_counts: dict[str, int]) -> dict[str, float]:
    """Class weights w_c = n / (C * n_c) from TRAINING counts only (Eq. 18)."""
    del label_counts
    raise NotImplementedError("Phase 3: class weighting not implemented yet.")


def train_centralized(
    sequences: Array,
    labels: Array,
    *,
    hidden_size: int,
    n_classes: int,
    epochs: int,
    seed: int,
) -> object:
    """Train the centralised GRU reference model and return the trained model."""
    del sequences, labels, hidden_size, n_classes, epochs, seed
    raise NotImplementedError("Phase 3: centralised training not implemented yet.")

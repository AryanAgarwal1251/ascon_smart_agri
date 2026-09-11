"""Dirichlet partitioning at BLOCK level (Phase 4, Section III-F1, gap G3/L2).

For each class c, draw p_c ~ Dir(alpha * 1_K) (Eq. 20) and allocate class-c *blocks* to the K
clients in those proportions. Small alpha => strong heterogeneity; large alpha => near-IID.
alpha is an experimental variable (alpha in {0.1, 0.5, 100}); per-client class histograms are
published so the partition can be inspected. Partitioning blocks (not rows) preserves temporal
locality into each client's sequence construction.

TODO(Phase 4): implement the block-level Dirichlet allocation + histogram export.
"""

from __future__ import annotations

from .._types import Array


def dirichlet_block_partition(
    block_labels: Array, *, n_clients: int, alpha: float, seed: int
) -> list[list[int]]:
    """Return, per client, the list of block ids assigned to it."""
    del block_labels, n_clients, alpha, seed
    raise NotImplementedError("Phase 4: Dirichlet block partition not implemented yet.")


def per_client_class_histograms(
    client_block_ids: list[list[int]], block_labels: Array
) -> list[dict[str, int]]:
    """Per-client class counts, published so the partition can be inspected (G3)."""
    del client_block_ids, block_labels
    raise NotImplementedError("Phase 4: class histograms not implemented yet.")

"""Dirichlet partitioning at BLOCK level (Phase 4, Section III-F1, gap G3/L2).

For each class c, draw p_c ~ Dir(alpha * 1_K) (Eq. 20) and allocate class-c *blocks* to the K
clients in those proportions. Small alpha => strong heterogeneity; large alpha => near-IID.
alpha is an experimental variable (alpha in {0.1, 0.5, 100}); per-client class histograms are
published so the partition can be inspected. Partitioning blocks (not rows) preserves temporal
locality into each client's sequence construction.

Implementation notes, and the decisions the paper leaves open (flagged per Golden Rule 1):

* **A fresh Dirichlet draw per class, not one draw reused across classes.** Eq. (20) is written
  per class (``p_c``), and drawing once would make every client's share of every class
  identical -- which is a *uniform* partition wearing a Dirichlet's clothes, and would silently
  defeat the point of sweeping alpha. This is the L2 failure the section exists to avoid.
* **Blocks are allocated by a seeded shuffle then a proportional cut**, so each block goes to
  exactly one client and no block is duplicated or dropped. The alternative -- sampling a client
  per block independently -- gives only the right *expected* proportions and adds a second
  source of variance on top of the Dirichlet draw.
* **A class with fewer blocks than clients cannot reach every client.** With 3 clients and 2
  blocks of a rare family, at least one client gets none of it; at alpha = 0.1 that is the
  intended behaviour rather than a defect, and it is exactly what the published per-client
  histograms are for. No block is invented to pad a client.
* **A client may receive zero blocks overall** under strong heterogeneity. That is a legitimate
  configuration, not an error: ``weighted_fedavg`` gives such a client weight 0, and the
  histogram shows it. Raising here would make alpha = 0.1 unrunnable.
"""

from __future__ import annotations

import numpy as np

from .._types import Array


def dirichlet_block_partition(
    block_labels: Array, *, n_clients: int, alpha: float, seed: int
) -> list[list[int]]:
    """Return, per client, the list of block ids assigned to it.

    ``block_labels[i]`` is the label of block ``i``; blocks are single-label by construction
    (``data/split.make_blocks``). The returned lists partition ``range(len(block_labels))``
    exactly: every block appears once, none is duplicated or dropped.
    """
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    labels = np.asarray(block_labels)
    if labels.ndim != 1:
        raise ValueError(f"block_labels must be 1-D, got shape {labels.shape}")

    rng = np.random.default_rng(seed)
    assignment: list[list[int]] = [[] for _ in range(n_clients)]

    # Sorted for determinism: the class iteration order fixes the RNG draw order.
    for label in sorted(np.unique(labels), key=str):
        blocks = np.flatnonzero(labels == label)
        rng.shuffle(blocks)

        # Eq. (20): a fresh draw for THIS class (see the module docstring).
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        # Cut points from the cumulative proportions; np.split handles empty shares naturally.
        cuts = (np.cumsum(proportions)[:-1] * len(blocks)).astype(int)
        for client, share in enumerate(np.split(blocks, cuts)):
            assignment[client].extend(int(b) for b in share)

    return [sorted(blocks) for blocks in assignment]


def per_client_class_histograms(
    client_block_ids: list[list[int]], block_labels: Array
) -> list[dict[str, int]]:
    """Per-client class counts, published so the partition can be inspected (G3).

    Counts are in BLOCKS, the unit the partition actually allocates. Every class in the corpus
    appears in every client's histogram, including with a count of zero -- an absent class is
    the most informative thing a heterogeneous partition can tell you, so it must not be
    omitted from the report.
    """
    labels = np.asarray(block_labels)
    vocabulary = [str(label) for label in sorted(np.unique(labels), key=str)]

    histograms: list[dict[str, int]] = []
    for blocks in client_block_ids:
        counts = dict.fromkeys(vocabulary, 0)
        for block in blocks:
            if not 0 <= block < len(labels):
                raise ValueError(f"block id {block} is outside the corpus")
            counts[str(labels[block])] += 1
        histograms.append(counts)
    return histograms

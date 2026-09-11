"""Federated aggregator / server (Phase 4, Section III-F, Algorithm 1).

One round: broadcast global theta -> each client trains locally and returns (theta_k, n_k) ->
server aggregates by weighted FedAvg (Eq. 21) -> broadcast the new global theta. The global
scaler (Section III-F4) is distributed once, before round one. The server never receives raw
records --- only parameter vectors and sequence counts cross the boundary.

TODO(Phase 4): implement the round loop, aggregation call, and convergence tracking.
"""

from __future__ import annotations

from .client import FederatedClient
from .serialization import StateDict


class FederatedServer:
    """Coordinates federated rounds over a fixed set of independent clients."""

    def __init__(self, clients: list[FederatedClient], *, aggregation: str = "weighted") -> None:
        self.clients = clients
        self.aggregation = aggregation

    def run_round(self, global_state: StateDict, *, local_epochs: int) -> StateDict:
        """Execute one federated round (Algorithm 1) and return the new global state."""
        del global_state, local_epochs
        raise NotImplementedError("Phase 4: federated round not implemented yet.")

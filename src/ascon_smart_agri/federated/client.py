"""Federated client (Phase 4, Section III-F, assumption A1).

K = 3 clients are simulated as INDEPENDENT scopes/processes; independence is enforced in code,
not merely documented (A1). A client trains locally from the broadcast global parameters with
a FRESH optimiser each round, and returns only ``(theta_k, n_k)`` --- no record of its local
data D_k is ever sent (Algorithm 1).

TODO(Phase 4): implement local training, fresh-optimiser-per-round, and (theta_k, n_k) return.
"""

from __future__ import annotations

from .serialization import StateDict


class FederatedClient:
    """A single simulated edge client holding a private, non-transmitted partition."""

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id

    def local_train(self, global_state: StateDict, *, local_epochs: int) -> tuple[StateDict, int]:
        """Train locally from ``global_state``; return ``(theta_k, n_k)``.

        ``n_k`` is the client's number of training SEQUENCES (the FedAvg weight, Eq. 21).
        """
        del global_state, local_epochs
        raise NotImplementedError("Phase 4: client local training not implemented yet.")

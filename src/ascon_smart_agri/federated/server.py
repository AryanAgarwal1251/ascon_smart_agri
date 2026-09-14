"""Federated aggregator / server (Phase 4, Section III-F, Algorithm 1).

One round: broadcast global theta -> each client trains locally and returns (theta_k, n_k) ->
server aggregates by weighted FedAvg (Eq. 21) -> broadcast the new global theta. The global
scaler (Section III-F4) is distributed once, before round one. The server never receives raw
records --- only parameter vectors and sequence counts cross the boundary.

Implementation notes:

* **Every client update crosses the boundary through safetensors**, even in-process. Calling
  ``serialize_state``/``deserialize_state`` on a simulated round costs little and makes the
  no-pickle rule (Section III-F2) structurally true rather than a claim about code that would
  transmit differently if it were ever wired to a socket. It also means ``n_k`` travels with
  the parameters instead of alongside them, where the two could drift apart.
* **Communication cost is measured, not assumed** (:meth:`bytes_per_round`): the serialised
  blobs are counted, so Eq. (22)'s 0.77 MiB/round figure is checked against what this
  implementation actually sends.
* ``aggregation="unweighted"`` selects the Section III-F2 ablation.
"""

from __future__ import annotations

import torch

from .aggregation import unweighted_average, weighted_fedavg
from .client import FederatedClient
from .serialization import StateDict, deserialize_state, serialize_state


class FederatedServer:
    """Coordinates federated rounds over a fixed set of independent clients."""

    def __init__(self, clients: list[FederatedClient], *, aggregation: str = "weighted") -> None:
        if not clients:
            raise ValueError("a federated server needs at least one client")
        if aggregation not in {"weighted", "unweighted"}:
            raise ValueError(f"aggregation must be 'weighted' or 'unweighted', got {aggregation!r}")
        self.clients = clients
        self.aggregation = aggregation
        self.last_round_bytes = 0

    def run_round(self, global_state: StateDict, *, local_epochs: int) -> dict[str, torch.Tensor]:
        """Execute one federated round (Algorithm 1) and return the new global state."""
        states: list[StateDict] = []
        counts: list[int] = []
        uploaded = 0

        for client in self.clients:
            # Broadcast: each client gets its own copy, never a shared reference.
            broadcast = {name: tensor.clone() for name, tensor in global_state.items()}
            state, n_k = client.local_train(broadcast, local_epochs=local_epochs)

            # The boundary. Nothing but these bytes moves from client to server.
            blob = serialize_state(state, n_k)
            uploaded += len(blob)
            received_state, received_n_k = deserialize_state(blob)

            states.append(received_state)
            counts.append(received_n_k)

        if self.aggregation == "weighted":
            new_global = weighted_fedavg(states, counts)
        else:
            new_global = unweighted_average(states)

        # Eq. (22) counts upload AND download: the new global goes back to every client.
        downloaded = len(serialize_state(new_global, 0)) * len(self.clients)
        self.last_round_bytes = uploaded + downloaded
        return new_global

    def sequence_counts(self) -> list[int]:
        """Per-client n_k, published so the FedAvg weighting can be inspected (Eq. 21)."""
        return [client.n_sequences for client in self.clients]


def theoretical_bytes_per_round(
    n_clients: int, n_parameters: int, *, bytes_per_parameter: int = 4
) -> int:
    """Eq. (22): ``B_round = 2 * K * |theta| * b``.

    For K = 3 and |theta| = 33,800 at single precision this is 811,200 bytes (~0.77 MiB), the
    figure Section III-F3 quotes against ~276 MB for pooling the records centrally.
    """
    if min(n_clients, n_parameters, bytes_per_parameter) <= 0:
        raise ValueError("n_clients, n_parameters and bytes_per_parameter must be positive")
    return 2 * n_clients * n_parameters * bytes_per_parameter

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
* :class:`SealedFederatedServer` (Phase 9) runs the same round with every update and every
  global download crossing as an **Ascon-AEAD128 frame** under the Phase 8 per-client,
  per-direction keys (``federated/transport.py``), so an in-process federation exercises the
  exact channel the hardware nodes use. A frame that opens to bottom is *dropped*: the client
  contributes nothing to that round (weight 0) and keeps its last good global state.
"""

from __future__ import annotations

import torch

from .aggregation import unweighted_average, weighted_fedavg
from .client import FederatedClient
from .serialization import StateDict, deserialize_state, serialize_state
from .transport import AggregatorChannel, ChannelKeys, ClientChannel


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


class SealedFederatedServer(FederatedServer):
    """:class:`FederatedServer` with the Phase 8 sealed weight channel on every exchange.

    ``client_ids`` name the clients on the wire (the ``client_id`` in every frame's associated
    data); ``keys`` are their up/down channel keys. Each client owns a :class:`ClientChannel`
    (seals updates under ``k_up``, opens globals under ``k_down``) and the aggregator owns one
    :class:`AggregatorChannel` per client; no key is shared between two encrypting parties.
    Round numbering starts at 1 and every frame is bound to its round, so a replayed or
    reordered frame opens to bottom just as it would across the network.
    """

    def __init__(
        self,
        clients: list[FederatedClient],
        client_ids: list[str],
        keys: dict[str, ChannelKeys],
        *,
        aggregation: str = "weighted",
    ) -> None:
        super().__init__(clients, aggregation=aggregation)
        if len(client_ids) != len(clients) or len(set(client_ids)) != len(client_ids):
            raise ValueError("client_ids must name every client exactly once")
        missing = [c for c in client_ids if c not in keys]
        if missing:
            raise ValueError(f"no channel keys for clients {missing}")
        self.client_ids = client_ids
        self._client_channels = {c: ClientChannel(c, keys[c]) for c in client_ids}
        self._aggregator_channels = {c: AggregatorChannel(c, keys[c]) for c in client_ids}
        # Each client's view of the global state: what it last opened successfully. A client
        # whose download frame failed to open trains from its previous view, never from a
        # frame it could not authenticate.
        self._client_view: dict[str, StateDict] = {}
        self._round = 0
        self.frames_dropped = 0
        self.frames_rejected_by_client: dict[str, int] = dict.fromkeys(client_ids, 0)

    def run_round(self, global_state: StateDict, *, local_epochs: int) -> dict[str, torch.Tensor]:
        """One round, every parameter vector Ascon-sealed in both directions."""
        self._round += 1
        states: list[StateDict] = []
        counts: list[int] = []
        uploaded = 0

        for client, client_id in zip(self.clients, self.client_ids, strict=True):
            # Broadcast: what this client last authenticated (the seed state before round 1).
            view = self._client_view.get(client_id, global_state)
            broadcast = {name: tensor.clone() for name, tensor in view.items()}
            state, n_k = client.local_train(broadcast, local_epochs=local_epochs)

            # Client -> aggregator: one sealed frame. This is the whole boundary.
            frame = self._client_channels[client_id].seal_update(state, n_k, self._round)
            uploaded += len(frame)
            opened = self._aggregator_channels[client_id].open_update(frame, self._round)
            if opened is None:
                self.frames_dropped += 1
                continue  # nothing applied; the client has weight 0 this round
            received_state, received_n_k = opened
            states.append(received_state)
            counts.append(received_n_k)

        if not states:
            raise RuntimeError("every client's update frame opened to bottom; nothing to aggregate")
        if self.aggregation == "weighted":
            new_global = weighted_fedavg(states, counts)
        else:
            new_global = unweighted_average(states)

        # Aggregator -> every client: one sealed frame each, opened at the client end.
        downloaded = 0
        for client_id in self.client_ids:
            frame = self._aggregator_channels[client_id].seal_global(new_global, self._round)
            downloaded += len(frame)
            view_state = self._client_channels[client_id].open_global(frame, self._round)
            if view_state is None:
                self.frames_rejected_by_client[client_id] += 1
                continue  # the client keeps its previous view
            self._client_view[client_id] = view_state
        self.last_round_bytes = uploaded + downloaded
        return new_global


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

"""Federated aggregator / server (Phase 4, Section III-F, Algorithm 1).

One round: broadcast global theta -> each client trains locally and returns (theta_k, n_k) ->
server aggregates by weighted FedAvg (Eq. 21) -> broadcast the new global theta. The global
scaler (Section III-F4) is distributed once, before round one. The server never receives raw
records --- only parameter vectors and sequence counts cross the boundary.

    IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (flagged per CLAUDE.md golden rule 1; see
    ``docs/design_paper.md``'s "Implementation deviation" section and ``federated/crypto.py``):
    every weight blob crossing the client<->server boundary, in BOTH directions, is now
    Ascon-AEAD128 encrypted and decrypted -- this protects Channel 3 (Section I-B), which the
    paper leaves unsolved, using the AEAD primitive the paper originally scoped to Channel 2.

Implementation notes:

* **Every client update crosses the boundary through safetensors, then through Ascon**, even
  in-process. This was already true of the upload leg before the crypto was added (see
  ``federated/serialization.py``); the broadcast leg now gets the same real
  serialize-encrypt-decrypt-deserialize round trip per client (previously it was just a
  ``tensor.clone()``, since only the upload leg's byte count mattered for Eq. 22). Making both
  legs cross a real wire-shaped boundary keeps the no-pickle rule and the "keys never in version
  control" rule structurally true rather than a claim about code that would behave differently if
  it were ever wired to a socket.
* **Communication cost is measured, not assumed** (:attr:`last_round_bytes`): the serialised AND
  encrypted blobs are counted, so Eq. (22)'s 0.77 MiB/round figure -- now plus a constant
  per-round AEAD overhead (a 16-byte nonce + 16-byte tag per blob) -- is checked against what
  this implementation actually sends.
* ``aggregation="unweighted"`` selects the Section III-F2 ablation.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..crypto.ascon_aead import AsconAEAD128, NonceRegistry
from .aggregation import unweighted_average, weighted_fedavg
from .client import FederatedClient
from .crypto import WeightAssociatedData, protect_state, unprotect_state
from .serialization import StateDict


class FederatedServer:
    """Coordinates federated rounds over a fixed set of independent clients."""

    def __init__(
        self,
        clients: list[FederatedClient],
        client_keys: Mapping[int, bytes],
        *,
        aggregation: str = "weighted",
        schema_version: str = "1",
    ) -> None:
        if not clients:
            raise ValueError("a federated server needs at least one client")
        if aggregation not in {"weighted", "unweighted"}:
            raise ValueError(f"aggregation must be 'weighted' or 'unweighted', got {aggregation!r}")
        missing = [client.client_id for client in clients if client.client_id not in client_keys]
        if missing:
            raise ValueError(f"no weight-transport key provided for client(s) {missing}")
        self.clients = clients
        self._client_keys = dict(client_keys)
        self.aggregation = aggregation
        self.schema_version = schema_version
        self._nonces = NonceRegistry()
        self.last_round_bytes = 0

    def run_round(
        self, global_state: StateDict, *, local_epochs: int, round_index: int
    ) -> dict[str, torch.Tensor]:
        """Execute one federated round (Algorithm 1) and return the new global state.

        ``round_index`` is authenticated as part of the weight-transport associated data
        (``federated/crypto.py``), giving Channel 3 the same replay/ordering protection Eq. (27)
        gives Channel 2's monotonic counter.
        """
        states: list[StateDict] = []
        counts: list[int] = []
        wire_bytes = 0

        for client in self.clients:
            cipher = AsconAEAD128(self._client_keys[client.client_id])

            # Broadcast: encrypt the global state individually per client, then decrypt it back
            # -- a client never receives a shared reference to the server's tensors, and (since
            # the "wire" is really just this call) never receives another client's ciphertext.
            broadcast_ad = WeightAssociatedData(
                client.client_id, round_index, "broadcast", self.schema_version
            )
            nonce, ad_bytes, ciphertext = protect_state(
                cipher, self._nonces, global_state, 0, ad=broadcast_ad
            )
            wire_bytes += len(ciphertext) + len(nonce)
            broadcast, _ = unprotect_state(cipher, nonce, ad_bytes, ciphertext)

            state, n_k = client.local_train(broadcast, local_epochs=local_epochs)

            # Upload: the client encrypts its own update; the server decrypts and verifies it.
            upload_ad = WeightAssociatedData(
                client.client_id, round_index, "upload", self.schema_version
            )
            nonce, ad_bytes, ciphertext = protect_state(
                cipher, self._nonces, state, n_k, ad=upload_ad
            )
            wire_bytes += len(ciphertext) + len(nonce)
            received_state, received_n_k = unprotect_state(cipher, nonce, ad_bytes, ciphertext)

            states.append(received_state)
            counts.append(received_n_k)

        if self.aggregation == "weighted":
            new_global = weighted_fedavg(states, counts)
        else:
            new_global = unweighted_average(states)

        self.last_round_bytes = wire_bytes
        return new_global

    def sequence_counts(self) -> list[int]:
        """Per-client n_k, published so the FedAvg weighting can be inspected (Eq. 21)."""
        return [client.n_sequences for client in self.clients]


def theoretical_bytes_per_round(
    n_clients: int, n_parameters: int, *, bytes_per_parameter: int = 4
) -> int:
    """Eq. (22): ``B_round = 2 * K * |theta| * b``, the paper's UNENCRYPTED formula.

    For K = 3 and |theta| = 33,800 at single precision this is 811,200 bytes (~0.77 MiB), the
    figure Section III-F3 quotes against ~276 MB for pooling the records centrally. This function
    intentionally does not model the AEAD overhead the implementation deviation adds -- see
    :func:`aead_overhead_bytes_per_round` for that -- so it stays a direct, checkable
    transcription of the paper's own equation.
    """
    if min(n_clients, n_parameters, bytes_per_parameter) <= 0:
        raise ValueError("n_clients, n_parameters and bytes_per_parameter must be positive")
    return 2 * n_clients * n_parameters * bytes_per_parameter


def aead_overhead_bytes_per_round(n_clients: int) -> int:
    """The per-round byte cost the implementation deviation adds on top of Eq. (22).

    Two legs (broadcast, upload) per client, each carrying a 16-byte nonce and a 16-byte tag
    (Eq. 29's ``+32 bytes``, applied here to weight blobs instead of telemetry payloads).
    """
    if n_clients <= 0:
        raise ValueError(f"n_clients must be positive, got {n_clients}")
    return 2 * n_clients * 32

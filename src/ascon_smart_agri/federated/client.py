"""Federated client (Phase 4, Section III-F, assumption A1).

K = 3 clients are simulated as INDEPENDENT scopes/processes; independence is enforced in code,
not merely documented (A1). A client trains locally from the broadcast global parameters with
a FRESH optimiser each round, and returns only ``(theta_k, n_k)`` --- no record of its local
data D_k is ever sent (Algorithm 1).

How independence is enforced rather than asserted:

* The client's data is held privately and is reachable through **no public attribute**; the
  only things :meth:`local_train` returns are a parameter state dict and a sequence count.
* The broadcast global state is **copied before use**, so a client cannot mutate the server's
  tensors in place and leak an effect sideways into its peers.
* A **fresh optimiser is constructed every round** (Section III-F2). Carrying Adam moments
  across rounds -- or worse, aggregating them -- would make the reported method something other
  than FedAvg while still looking like it converged.
* ``n_k`` is the number of training **sequences**, taken from the client's windowed tensor, so
  the FedAvg weight (Eq. 21) cannot accidentally become a row count.
"""

from __future__ import annotations

import torch
from torch import nn

from .._types import Array
from ..model.gru import build_detector
from ..model.train import train_module
from .serialization import StateDict


class FederatedClient:
    """A single simulated edge client holding a private, non-transmitted partition."""

    def __init__(
        self,
        client_id: int,
        sequences: Array | None = None,
        labels: Array | None = None,
        *,
        hidden_size: int = 96,
        n_classes: int = 8,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        # Private: named with a leading underscore and never returned or logged. The server
        # receives (theta_k, n_k) and nothing else.
        self._sequences = sequences
        self._labels = labels
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

    @property
    def n_sequences(self) -> int:
        """n_k -- the client's training SEQUENCE count, the FedAvg weight of Eq. (21)."""
        return 0 if self._sequences is None else len(self._sequences)

    def local_train(self, global_state: StateDict, *, local_epochs: int) -> tuple[StateDict, int]:
        """Train locally from ``global_state``; return ``(theta_k, n_k)``.

        ``n_k`` is the client's number of training SEQUENCES (the FedAvg weight, Eq. 21).
        """
        if local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {local_epochs}")

        # A client with no sequences returns the global parameters untouched and n_k = 0, so
        # weighted_fedavg gives it weight zero. This happens legitimately at alpha = 0.1.
        if self._sequences is None or self._labels is None or self.n_sequences == 0:
            return {name: tensor.clone() for name, tensor in global_state.items()}, 0

        n_features = self._sequences.shape[2]
        model = build_detector(n_features, self.hidden_size, self.n_classes)
        # Copy before loading: never hold a reference to the server's tensors.
        model.load_state_dict({name: tensor.clone() for name, tensor in global_state.items()})

        # train_module builds a FRESH optimiser on every call -- Section III-F2 requires it.
        trained: nn.Module = train_module(
            model,
            self._sequences,
            self._labels,
            n_classes=self.n_classes,
            epochs=local_epochs,
            seed=self.client_id,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        state = {name: tensor.detach().clone() for name, tensor in trained.state_dict().items()}
        return state, self.n_sequences

    def local_sufficient_statistics(self) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        """Count/mean/M2 over this client's rows, for the global scaler (Section III-F4)."""
        if self._sequences is None or self.n_sequences == 0:
            return None
        # The final record of each window, matching how single-record statistics are defined.
        rows = torch.as_tensor(self._sequences[:, -1, :], dtype=torch.float64)
        mean = rows.mean(dim=0)
        return len(rows), mean, ((rows - mean) ** 2).sum(dim=0)

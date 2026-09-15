"""Centralised training loop (Phase 3, Section III-E).

Phase 3 is a HARD GATE (III-J4): federation does not begin until single-client detection is
proven, because debugging an aggregation fault and a modelling fault at once is much harder
than either alone. Device-agnostic (CPU-only must suffice; a GPU may be used if present).

Implementation notes:

* **Class-weighted cross-entropy, weights from TRAINING data only** (Eq. 18,
  w_c = n / (C * n_c)). This is the first line of defence against the imbalance of Section
  II-C. There is deliberately no synthetic oversampling: interpolating flow records fabricates
  temporal structure that never occurred, and the windows built from it would be fiction.
* **Complete sequences are shuffled at batch time, never rows** (Section III-D). The shuffle
  happens over the first axis of an already-built ``(N, W, F)`` array, so within-window ordering
  is untouched.
* **A class absent from the training split gets weight 0**, not an infinite one. Eq. (18) divides
  by n_c, which is undefined at n_c = 0; weighting an unobservable class infinitely would let a
  single stray prediction dominate the loss. The absence is instead visible in the per-class F1
  of the evaluation report.
* Training is seeded end to end (torch, numpy and the batch permutation) so a run is
  reproducible from the manifest, as Section III-I4 requires.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .._types import Array
from ..federated.aggregation import fedprox_proximal_term
from .gru import build_detector


def class_weights(label_counts: dict[str, int]) -> dict[str, float]:
    """Class weights w_c = n / (C * n_c) from TRAINING counts only (Eq. 18)."""
    if not label_counts:
        raise ValueError("cannot compute class weights from an empty count map")
    if any(count < 0 for count in label_counts.values()):
        raise ValueError(f"negative class counts are not meaningful: {label_counts}")

    n = sum(label_counts.values())
    if n == 0:
        raise ValueError("cannot compute class weights when every class count is zero")
    n_classes = len(label_counts)
    return {
        name: (n / (n_classes * count) if count > 0 else 0.0)
        for name, count in label_counts.items()
    }


def train_module(
    model: nn.Module,
    sequences: Array,
    labels: Array,
    *,
    n_classes: int,
    epochs: int,
    seed: int,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
    fedprox_mu: float | None = None,
    fedprox_reference: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Train ANY ``(B, W, F) -> (B, C)`` module under the Section III-E regime.

    Factored out of :func:`train_centralized` so the Section III-I1 baselines train under an
    identical regime -- same optimiser, same class-weighted CE, same seeding, same batching,
    same number of epochs. When only the architecture differs, a difference in the result is
    attributable to the architecture, which is the entire point of the MLP baseline ("isolates
    the contribution of recurrence").

    ``fedprox_mu``/``fedprox_reference`` wire in the Section III-F2 R4 fallback: when both are
    given, every step's loss gains ``mu/2 * ||theta - theta_global||^2`` against the FIXED
    ``fedprox_reference`` (the parameters this client started the round from), computed live
    from the model's current parameters rather than a state-dict snapshot taken each step. Plain
    FedAvg is ``fedprox_mu=None`` (the default), matching how the scaffold always described
    FedProx as a fallback, not the default regime.
    """
    if (fedprox_mu is None) != (fedprox_reference is None):
        raise ValueError("fedprox_mu and fedprox_reference must be given together, or not at all")
    if fedprox_mu is not None and fedprox_mu < 0:
        raise ValueError(f"fedprox_mu must be non-negative, got {fedprox_mu}")
    if sequences.ndim != 3:
        raise ValueError(f"sequences must be (N, W, F), got shape {sequences.shape}")
    if len(sequences) != len(labels):
        raise ValueError(f"sequences/labels lengths disagree: {len(sequences)}, {len(labels)}")
    if len(sequences) == 0:
        raise ValueError("cannot train on zero sequences")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)

    y = np.asarray(labels).astype(np.int64)
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must lie in [0, {n_classes}), got [{y.min()}, {y.max()}]")

    # Eq. (18), on training data only.
    counts = {index: int(np.sum(y == index)) for index in range(n_classes)}
    weights = class_weights({str(k): v for k, v in counts.items()})
    weight_tensor = torch.tensor(
        [weights[str(index)] for index in range(n_classes)], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_all = torch.as_tensor(np.asarray(sequences, dtype=np.float32))
    y_all = torch.as_tensor(y)

    model.train()
    for epoch in range(epochs):
        # Shuffle COMPLETE SEQUENCES only -- never the rows inside a window (Section III-D).
        order = rng.permutation(len(x_all))
        running_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), batch_size):
            index = torch.as_tensor(order[start : start + batch_size])
            x_batch = x_all[index].to(device)
            y_batch = y_all[index].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            if fedprox_mu is not None and fedprox_reference is not None:
                # Live parameters, not a state_dict() snapshot: this must stay in the autograd
                # graph so the proximal term's gradient actually pulls training back toward
                # fedprox_reference, rather than only being logged.
                current = dict(model.named_parameters())
                loss = loss + fedprox_proximal_term(current, fedprox_reference, fedprox_mu)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())
            n_batches += 1
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  weighted CE {running_loss / n_batches:.4f}")

    model.eval()
    return model


def train_centralized(
    sequences: Array,
    labels: Array,
    *,
    hidden_size: int,
    n_classes: int,
    epochs: int,
    seed: int,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> nn.Module:
    """Train the centralised GRU reference model and return the trained model."""
    if sequences.ndim != 3:
        raise ValueError(f"sequences must be (N, W, F), got shape {sequences.shape}")
    torch.manual_seed(seed)  # seed before construction so the init is reproducible too
    model = build_detector(sequences.shape[2], hidden_size, n_classes)
    return train_module(
        model,
        sequences,
        labels,
        n_classes=n_classes,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        verbose=verbose,
    )


def predict(
    model: nn.Module, sequences: Array, *, batch_size: int = 4096, device: str = "cpu"
) -> tuple[Array, Array]:
    """Return ``(predicted_class, class_probabilities)`` for a batch of sequences."""
    if sequences.ndim != 3:
        raise ValueError(f"sequences must be (N, W, F), got shape {sequences.shape}")

    model.eval()
    x_all = torch.as_tensor(np.asarray(sequences, dtype=np.float32))
    probabilities = []
    # The context-manager form, not the @torch.no_grad() decorator: the decorator is untyped
    # in the pre-commit mypy environment (which has no torch), and an untyped decorator would
    # silently make this whole function untyped.
    with torch.no_grad():
        for start in range(0, len(x_all), batch_size):
            logits = model(x_all[start : start + batch_size].to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    probability = np.concatenate(probabilities) if probabilities else np.empty((0, 0))
    return probability.argmax(axis=1), probability

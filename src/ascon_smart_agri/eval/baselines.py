"""The five required baselines (Phase 3/4, Section III-I1).

Reported for EVERY configuration:
    1. random forest on single records      (motivated by S13's dissenting result)
    2. MLP on single records                 (isolates the contribution of recurrence)
    3. centralised GRU                        (upper bound attainable by pooling)
    4. three local-only GRUs                  (lower bound attainable without federating)
    5. federated global GRU                   (the method under test)

Baselines 3 and 4 bracket the federated result and together answer gap G4 (local-only is what
an operator gets by declining to federate). Every headline number is mean +/- std over >= 3
seeds (III-I4).

Design decisions, flagged per Golden Rule 1:

* **"Single records" is implemented as the FINAL record of each window**, and the single-record
  baselines take the same ``(N, W, F)`` tensors the GRU takes, using only ``x[:, -1, :]``. Since
  a window's label is its final record's (y_i = y_{i+W-1}, Section III-D), this makes the
  baselines predict *exactly the same targets from exactly the same test set* as the GRU. Giving
  them a separately-derived record set would leave any difference confounded with a difference
  in evaluation population; this way the only thing that varies is whether the model can see
  history.
* **The MLP is parameter-matched to the GRU, not width-matched.** To "isolate the contribution
  of recurrence" the two models must differ in architecture and not in capacity, so the hidden
  width is solved for from the GRU's Eq. (19) budget (33,800 at the reference sizing) rather
  than copied from H=96, which would give the MLP ~2.4k parameters and confound recurrence with
  capacity. It also trains through the *same* ``train_module`` loop as the GRU -- same optimiser,
  class-weighted CE, seeding, batching and epochs.
* **The scaffold's baseline signatures took only ``(x, y, seed)``**, which cannot express a
  train/test split and so could not return test metrics at all. They are widened here to take
  train and test explicitly. Flagged as a scaffold correction rather than worked around.
* The random forest is the one baseline that does **not** share the training regime, because it
  is not a gradient model; it gets ``class_weight="balanced"``, the tree analogue of Eq. (18)'s
  reweighting, and no oversampling.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

from .._types import Array
from ..model.gru import expected_param_count
from ..model.train import predict, train_centralized, train_module
from .metrics import MulticlassMetrics, multiclass_metrics


def _final_records(sequences: Array) -> Array:
    """The single record each window's label refers to: ``x[:, -1, :]`` (Section III-D)."""
    if sequences.ndim != 3:
        raise ValueError(f"sequences must be (N, W, F), got shape {sequences.shape}")
    return np.asarray(sequences)[:, -1, :]


def mlp_hidden_for_parameter_budget(n_features: int, n_classes: int, budget: int) -> int:
    """Hidden width whose one-layer MLP lands closest to ``budget`` parameters, at least 1.

    A single-hidden-layer MLP holds ``(F*H + H) + (H*C + C)`` parameters, so the width that
    matches a given budget is ``(budget - C) / (F + 1 + C)``.
    """
    if min(n_features, n_classes, budget) <= 0:
        raise ValueError("n_features, n_classes and budget must all be positive")
    return max(1, round((budget - n_classes) / (n_features + 1 + n_classes)))


class MLPDetector(nn.Module):
    """Single-record MLP baseline: reads only ``x[:, -1, :]``, so it cannot use history.

    Lives here rather than in ``model/`` deliberately: it exists only as the Section III-I1
    comparison, and is not part of the architecture under test.
    """

    def __init__(self, n_features: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        if min(n_features, hidden_size, n_classes) <= 0:
            raise ValueError("n_features, hidden_size and n_classes must all be positive")
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, W, F)`` to ``(B, C)`` using the final step only -- no recurrence."""
        if x.ndim != 3:
            raise ValueError(f"expected a (B, W, F) batch, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features per step, got {x.shape[-1]}")
        logits: torch.Tensor = self.net(x[:, -1, :])
        return logits


def random_forest_single_record(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    *,
    seed: int,
    class_names: list[str],
    n_estimators: int = 200,
    max_train_rows: int | None = 300_000,
) -> MulticlassMetrics:
    """Baseline 1: random forest on single records.

    ``max_train_rows`` caps the fit with a seeded draw; a forest over the full split costs far
    more than the comparison is worth, and the cap is recorded in the manifest.
    """
    x = _final_records(x_train)
    y = np.asarray(y_train)
    if max_train_rows is not None and len(x) > max_train_rows:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(x), size=max_train_rows, replace=False)
        x, y = x[take], y[take]

    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",  # the tree analogue of Eq. (18); never oversampling
        random_state=seed,
        n_jobs=-1,
    )
    forest.fit(x, y)
    return multiclass_metrics(y_test, forest.predict(_final_records(x_test)), class_names)


def mlp_single_record(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    *,
    seed: int,
    class_names: list[str],
    n_classes: int,
    epochs: int,
    gru_hidden_size: int = 96,
    batch_size: int = 1024,
    verbose: bool = False,
) -> MulticlassMetrics:
    """Baseline 2: MLP on single records (no recurrence), parameter-matched to the GRU."""
    n_features = np.asarray(x_train).shape[2]
    budget = expected_param_count(n_features, gru_hidden_size, n_classes)
    hidden = mlp_hidden_for_parameter_budget(n_features, n_classes, budget)

    torch.manual_seed(seed)
    trained = train_module(
        MLPDetector(n_features, hidden, n_classes),
        x_train,
        y_train,
        n_classes=n_classes,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        verbose=verbose,
    )
    y_pred, _ = predict(trained, x_test)
    return multiclass_metrics(y_test, y_pred, class_names)


def centralized_gru(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    *,
    seed: int,
    class_names: list[str],
    n_classes: int,
    epochs: int,
    hidden_size: int = 96,
    batch_size: int = 1024,
    verbose: bool = False,
) -> MulticlassMetrics:
    """Baseline 3: centralised GRU (upper bound attainable by pooling)."""
    model = train_centralized(
        x_train,
        y_train,
        hidden_size=hidden_size,
        n_classes=n_classes,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        verbose=verbose,
    )
    y_pred, _ = predict(model, x_test)
    return multiclass_metrics(y_test, y_pred, class_names)


def local_only_grus(
    client_seqs: list[Array], client_y: list[Array], *, seed: int
) -> list[MulticlassMetrics]:
    """Baseline 4: one GRU per client, no federation (lower bound)."""
    del client_seqs, client_y, seed
    raise NotImplementedError("Phase 4: local-only baseline not implemented yet.")


def federated_global_gru(
    client_seqs: list[Array], client_y: list[Array], *, seed: int
) -> MulticlassMetrics:
    """Baseline 5: the federated global GRU (method under test)."""
    del client_seqs, client_y, seed
    raise NotImplementedError("Phase 4: federated-global baseline not implemented yet.")

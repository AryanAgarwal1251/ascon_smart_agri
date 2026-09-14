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

Baselines 4-5 (Phase 4) add three more, flagged the same way:

* **Every baseline is scored on the SAME shared global test set** (Section III-B3), including
  the local-only clients. A local model evaluated on its own partition's held-out slice would
  be measuring a different, easier question -- each client's partition is class-skewed by the
  Dirichlet draw (Eq. 20), so a client that never saw a rare family would never be asked about
  it. Scoring every model on the shared test set is what makes baselines 3, 4 and 5 a bracket
  rather than three unrelated numbers.
* **A client with zero sequences yields ``None``, not a zero-filled metric bundle.** Under
  alpha = 0.1 a client can legitimately receive no blocks at all (see ``federated/partition.py``),
  and it cannot train a detector. Returning a bundle of zeros would silently drag a reported
  mean downward as if the client had trained and failed; returning ``None`` forces the caller to
  say what it did. This is the one place baseline 4's return type widens, and it is deliberate.
* **Local-only clients are seeded from a ``SeedSequence`` spawned off the run seed**, not from
  ``seed + client_id``, so client 0 at seed 1 and client 1 at seed 0 are not the same run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

from .._types import Array
from ..federated.client import FederatedClient
from ..federated.serialization import StateDict
from ..federated.server import FederatedServer
from ..model.gru import build_detector, expected_param_count
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


def _check_partitions(client_seqs: list[Array], client_y: list[Array]) -> None:
    """Validate a per-client partition list before anything expensive happens."""
    if not client_seqs:
        raise ValueError("need at least one client partition")
    if len(client_seqs) != len(client_y):
        raise ValueError(
            f"partition lengths disagree: {len(client_seqs)} feature sets, {len(client_y)} labels"
        )
    for index, (seqs, labels) in enumerate(zip(client_seqs, client_y, strict=True)):
        if len(seqs) != len(labels):
            raise ValueError(f"client {index}: {len(seqs)} sequences but {len(labels)} labels")
        if len(seqs) > 0 and np.asarray(seqs).ndim != 3:
            raise ValueError(f"client {index}: sequences must be (N, W, F)")


def local_only_grus(
    client_seqs: list[Array],
    client_y: list[Array],
    x_test: Array,
    y_test: Array,
    *,
    seed: int,
    class_names: list[str],
    n_classes: int,
    epochs: int,
    hidden_size: int = 96,
    batch_size: int = 1024,
    device: str = "cpu",
    verbose: bool = False,
) -> list[MulticlassMetrics | None]:
    """Baseline 4: one GRU per client, no federation (lower bound).

    This is what an operator gets by declining to federate: each client trains on its own
    partition alone and is scored on the shared global test set. Together with baseline 3 it
    brackets the federated result and answers gap G4.

    Returns one entry per client, in client order. An entry is ``None`` where the client holds
    zero sequences and therefore has no detector to report -- see the module docstring.
    """
    _check_partitions(client_seqs, client_y)

    # Independent, collision-free per-client seeds derived from the run seed.
    client_seeds = np.random.SeedSequence(seed).generate_state(len(client_seqs))

    results: list[MulticlassMetrics | None] = []
    for client_id, (seqs, labels) in enumerate(zip(client_seqs, client_y, strict=True)):
        if len(seqs) == 0:
            # No data, so no model. Reported as absent rather than as a zero score.
            results.append(None)
            continue
        model = train_centralized(
            np.asarray(seqs),
            np.asarray(labels),
            hidden_size=hidden_size,
            n_classes=n_classes,
            epochs=epochs,
            seed=int(client_seeds[client_id]),
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
        y_pred, _ = predict(model, x_test, device=device)
        results.append(multiclass_metrics(y_test, y_pred, class_names))
    return results


@dataclass(frozen=True)
class FederatedRun:
    """One federated run's outcome, carrying what Sections III-F and III-I want recorded.

    ``federated_global_gru`` returns only :attr:`metrics`, matching the Section III-I1 baseline
    contract; the runner script uses the rest for the manifest (Eq. 22's measured cost, the
    Eq. 21 weights actually applied, and the convergence curve over R rounds).
    """

    metrics: MulticlassMetrics
    sequence_counts: list[int]  # n_k per client -- the Eq. (21) weights
    bytes_per_round: list[int]  # measured, not assumed (Eq. 22)
    per_round_macro_f1: list[float]  # empty unless evaluate_each_round was set


def _evaluate_state(
    state: StateDict,
    x_test: Array,
    y_test: Array,
    *,
    n_features: int,
    hidden_size: int,
    n_classes: int,
    class_names: list[str],
    device: str,
) -> MulticlassMetrics:
    """Load a global parameter vector into a fresh detector and score it on the test set."""
    model = build_detector(n_features, hidden_size, n_classes)
    model.load_state_dict({name: tensor.clone() for name, tensor in state.items()})
    y_pred, _ = predict(model, x_test, device=device)
    return multiclass_metrics(y_test, y_pred, class_names)


def run_federation(
    client_seqs: list[Array],
    client_y: list[Array],
    x_test: Array,
    y_test: Array,
    *,
    seed: int,
    class_names: list[str],
    n_classes: int,
    rounds: int,
    local_epochs: int,
    hidden_size: int = 96,
    batch_size: int = 1024,
    aggregation: str = "weighted",
    device: str = "cpu",
    evaluate_each_round: bool = False,
    verbose: bool = False,
) -> FederatedRun:
    """Run R rounds of Algorithm 1 over K clients and score the final global model.

    ``evaluate_each_round`` costs one extra prediction pass per round and is off by default;
    the runner turns it on to record the convergence curve.
    """
    _check_partitions(client_seqs, client_y)
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")

    n_features = int(np.asarray(x_test).shape[2])

    # Seed before construction so the initial broadcast theta is reproducible (III-I4).
    torch.manual_seed(seed)
    initial = build_detector(n_features, hidden_size, n_classes)
    global_state: StateDict = {
        name: tensor.detach().clone() for name, tensor in initial.state_dict().items()
    }

    clients = [
        FederatedClient(
            client_id,
            np.asarray(seqs) if len(seqs) > 0 else None,
            np.asarray(labels) if len(labels) > 0 else None,
            hidden_size=hidden_size,
            n_classes=n_classes,
            batch_size=batch_size,
            device=device,
        )
        for client_id, (seqs, labels) in enumerate(zip(client_seqs, client_y, strict=True))
    ]
    server = FederatedServer(clients, aggregation=aggregation)

    def evaluate(state: StateDict) -> MulticlassMetrics:
        return _evaluate_state(
            state,
            x_test,
            y_test,
            n_features=n_features,
            hidden_size=hidden_size,
            n_classes=n_classes,
            class_names=class_names,
            device=device,
        )

    bytes_per_round: list[int] = []
    per_round_macro_f1: list[float] = []

    for round_index in range(rounds):
        global_state = server.run_round(global_state, local_epochs=local_epochs)
        bytes_per_round.append(server.last_round_bytes)
        if evaluate_each_round:
            macro_f1 = evaluate(global_state).macro_f1
            per_round_macro_f1.append(macro_f1)
            if verbose:
                print(f"  round {round_index + 1}/{rounds}  macro-F1 {macro_f1:.4f}")

    return FederatedRun(
        metrics=evaluate(global_state),
        sequence_counts=server.sequence_counts(),
        bytes_per_round=bytes_per_round,
        per_round_macro_f1=per_round_macro_f1,
    )


def federated_global_gru(
    client_seqs: list[Array],
    client_y: list[Array],
    x_test: Array,
    y_test: Array,
    *,
    seed: int,
    class_names: list[str],
    n_classes: int,
    rounds: int,
    local_epochs: int,
    hidden_size: int = 96,
    batch_size: int = 1024,
    aggregation: str = "weighted",
    device: str = "cpu",
    verbose: bool = False,
) -> MulticlassMetrics:
    """Baseline 5: the federated global GRU (method under test)."""
    return run_federation(
        client_seqs,
        client_y,
        x_test,
        y_test,
        seed=seed,
        class_names=class_names,
        n_classes=n_classes,
        rounds=rounds,
        local_epochs=local_epochs,
        hidden_size=hidden_size,
        batch_size=batch_size,
        aggregation=aggregation,
        device=device,
        verbose=verbose,
    ).metrics

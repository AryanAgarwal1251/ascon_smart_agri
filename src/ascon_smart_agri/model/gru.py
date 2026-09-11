"""GRU detector (Phase 3, Section III-E).

Architecture: single GRU layer -> layer normalisation -> linear head -> class logits.

Layer normalisation is chosen over batch normalisation *because* of the federated setting:
batch statistics are running estimates of one client's input distribution, so averaging them
across clients with different distributions describes no client's data (III-E). LayerNorm has
no such cross-client state.

Training uses class-weighted cross-entropy with weights computed on training data only; NO
synthetic oversampling (interpolating flow records fabricates temporal structure that never
occurred). The architecture is implemented generically, but the default sizing
(F=16, H=96, C=8) must reproduce the Eq. (19) parameter count of 33,800 --- asserted by
``tests/test_model_param_count.py``.

TODO(Phase 3): implement the nn.Module (GRU -> LayerNorm(h_W) -> Linear) and forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep torch out of import time during scaffolding
    import torch
    from torch import nn


def expected_param_count(n_features: int, hidden_size: int, n_classes: int) -> int:
    """Closed-form parameter count from Eq. (19); the model must match this exactly.

        3(FH + H^2 + 2H)  [GRU]  +  2H  [LayerNorm]  +  (HC + C)  [linear head]

    For F=16, H=96, C=8 this is 32,832 + 192 + 776 = 33,800.
    """
    f, h, c = n_features, hidden_size, n_classes
    gru = 3 * (f * h + h * h + 2 * h)
    layer_norm = 2 * h
    head = h * c + c
    return gru + layer_norm + head


def build_detector(n_features: int, hidden_size: int, n_classes: int) -> nn.Module:
    """Construct the GRU detector (GRU -> LayerNorm -> Linear)."""
    del n_features, hidden_size, n_classes
    raise NotImplementedError("Phase 3: GRU detector not implemented yet.")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters of an instantiated model (validated against Eq. 19)."""
    del model
    raise NotImplementedError("Phase 3: parameter counting not implemented yet.")

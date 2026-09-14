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

Implementation notes:

* ``torch`` is now imported at module scope. The scaffold deferred it behind ``TYPE_CHECKING``
  so the package stayed importable before the scientific stack was wired up; Phase 3 needs the
  real module, and ``torch`` is a declared dependency.
* **The classification state is h_W, the final hidden state**, matching Section III-D's
  y_i = y_{i+W-1}: the verdict is about what is happening at the end of the window given the
  recent history, so the head reads the last step's hidden state rather than pooling over the
  window. LayerNorm is applied to that vector before the head, per the architecture line above.
* PyTorch's ``nn.GRU`` with ``bias=True`` carries exactly the 3(FH + H^2 + 2H) parameters
  Eq. (19) counts (``weight_ih`` 3H x F, ``weight_hh`` 3H x H, and both 3H bias vectors), so the
  reference sizing lands on 33,800 without any adjustment. The equality is not assumed --
  ``count_parameters`` is checked against ``expected_param_count`` in the test suite.
"""

from __future__ import annotations

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


class GRUDetector(nn.Module):
    """Single-layer GRU -> LayerNorm(h_W) -> linear head (Section III-E, Eqs. 13-17)."""

    def __init__(self, n_features: int, hidden_size: int, n_classes: int) -> None:
        super().__init__()
        if min(n_features, hidden_size, n_classes) <= 0:
            raise ValueError(
                f"n_features/hidden_size/n_classes must all be positive, got "
                f"{n_features}, {hidden_size}, {n_classes}"
            )
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.gru = nn.GRU(n_features, hidden_size, num_layers=1, batch_first=True)
        # LayerNorm, never BatchNorm: batch statistics do not aggregate across federated
        # clients (see the module docstring).
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``(B, W, F)`` batch of sequences to ``(B, C)`` class logits."""
        if x.ndim != 3:
            raise ValueError(f"expected a (B, W, F) batch, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features per step, got {x.shape[-1]}")
        hidden_states, _ = self.gru(x)
        h_final = hidden_states[:, -1, :]  # h_W -- the window's last step (Section III-D)
        logits: torch.Tensor = self.head(self.norm(h_final))
        return logits


def build_detector(n_features: int, hidden_size: int, n_classes: int) -> nn.Module:
    """Construct the GRU detector (GRU -> LayerNorm -> Linear)."""
    return GRUDetector(n_features, hidden_size, n_classes)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters of an instantiated model (validated against Eq. 19)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

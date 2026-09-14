"""Unit tests for the GRU detector's architecture and forward pass (Section III-E).

The parameter-count invariant lives in ``tests/test_model_param_count.py``; this file covers
the structural choices the paper is explicit about (LayerNorm not BatchNorm, the head reading
h_W) and the shape contract the training loop depends on.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ascon_smart_agri.model.gru import GRUDetector, build_detector


def test_forward_maps_batch_of_windows_to_class_logits() -> None:
    model = build_detector(16, 96, 8)

    logits = model(torch.randn(4, 16, 16))  # (B=4, W=16, F=16)

    assert logits.shape == (4, 8)  # (B, C)
    assert torch.isfinite(logits).all()


def test_uses_layernorm_and_never_batchnorm() -> None:
    """Batch statistics do not aggregate across federated clients (Section III-E)."""
    model = build_detector(16, 96, 8)
    module_types = {type(m) for m in model.modules()}

    assert nn.LayerNorm in module_types
    assert not any(issubclass(t, nn.modules.batchnorm._BatchNorm) for t in module_types)


def test_architecture_is_gru_then_norm_then_linear() -> None:
    model = build_detector(16, 96, 8)

    assert isinstance(model.gru, nn.GRU)
    assert model.gru.num_layers == 1  # a SINGLE GRU layer, per Section III-E
    assert model.gru.batch_first is True
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.head, nn.Linear)
    assert model.head.out_features == 8


def test_verdict_depends_on_the_final_step_of_the_window() -> None:
    """The head reads h_W, so changing the LAST record must move the logits (y_i = y_{i+W-1})."""
    torch.manual_seed(0)
    model = build_detector(4, 8, 3).eval()
    x = torch.randn(1, 5, 4)
    modified = x.clone()
    modified[0, -1, :] += 5.0

    with torch.no_grad():
        assert not torch.allclose(model(x), model(modified))


def test_a_single_row_window_is_accepted() -> None:
    # W=1 is a declared ablation ("does recurrence earn its place?", Section III-D).
    model = build_detector(16, 96, 8)

    assert model(torch.randn(2, 1, 16)).shape == (2, 8)


def test_gradients_flow_to_every_parameter() -> None:
    model = build_detector(4, 8, 3)
    logits = model(torch.randn(6, 5, 4))
    loss = nn.functional.cross_entropy(logits, torch.randint(0, 3, (6,)))

    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient at {name}"


def test_forward_rejects_a_non_batched_input() -> None:
    model = build_detector(16, 96, 8)

    with pytest.raises(ValueError, match=r"\(B, W, F\)"):
        model(torch.randn(16, 16))  # missing the batch dimension


def test_forward_rejects_the_wrong_feature_count() -> None:
    model = build_detector(16, 96, 8)

    with pytest.raises(ValueError, match="expected 16 features"):
        model(torch.randn(2, 16, 12))


@pytest.mark.parametrize(("f", "h", "c"), [(0, 96, 8), (16, 0, 8), (16, 96, 0), (-1, 96, 8)])
def test_construction_rejects_non_positive_dimensions(f: int, h: int, c: int) -> None:
    with pytest.raises(ValueError, match="must all be positive"):
        GRUDetector(f, h, c)

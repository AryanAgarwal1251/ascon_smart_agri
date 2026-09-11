"""Model parameter-count sanity (Eq. 19). The formula check is active now; the built-model
check activates in Phase 3."""

from __future__ import annotations

import pytest

from ascon_smart_agri.model.gru import build_detector, count_parameters, expected_param_count


def test_reference_sizing_is_33800() -> None:
    # F=16, H=96, C=8 -> 32,832 (GRU) + 192 (LN) + 776 (head) = 33,800 (Section III-E).
    assert expected_param_count(16, 96, 8) == 33_800


def test_formula_components() -> None:
    # Cross-check the three additive terms of Eq. (19) independently.
    f, h, c = 16, 96, 8
    gru = 3 * (f * h + h * h + 2 * h)
    layer_norm = 2 * h
    head = h * c + c
    assert (gru, layer_norm, head) == (32_832, 192, 776)
    assert gru + layer_norm + head == expected_param_count(f, h, c)


@pytest.mark.skip(reason="pending Phase 3: build_detector() not implemented yet")
def test_built_model_matches_formula() -> None:
    model = build_detector(16, 96, 8)
    assert count_parameters(model) == expected_param_count(16, 96, 8)

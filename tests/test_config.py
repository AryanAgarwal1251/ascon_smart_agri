"""Config schema + default YAML sanity (active now)."""

from __future__ import annotations

from pathlib import Path

import pytest
from configs.base import RunConfig, load_run_config
from pydantic import ValidationError


def test_default_config_values(default_config: RunConfig) -> None:
    cfg = default_config
    assert cfg.features.correlation_tau == 0.95  # Stage-2 tau (III-C)
    assert cfg.sequence.w_sweep == [1, 8, 16, 32]  # W ablation incl. W=1 (III-D)
    assert cfg.model.n_classes == 8  # benign + 7 families (III-B3)
    assert cfg.federated.n_clients == 3  # K (A1)
    assert cfg.federated.alpha_sweep == [0.1, 0.5, 100.0]  # heterogeneity (III-I3)
    assert cfg.federated.e_sweep == [1, 3, 5]  # local-epoch ablation (III-I3)
    assert cfg.crypto.ad_fields == ["edge_id", "device_id", "counter", "schema_version"]  # Eq.27


def test_default_yaml_loads_and_matches() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = load_run_config(yaml_path)
    assert cfg.model.n_features == 16
    assert cfg.model.hidden_size == 96
    assert cfg.evaluation.seeds == [0, 1, 2]


def test_fewer_than_three_seeds_rejected() -> None:
    # III-I4: single/double-seed runs are not findings.
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"evaluation": {"seeds": [0]}})


def test_non_eight_class_rejected() -> None:
    # III-B3: the primary task is fixed at C=8.
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"model": {"n_classes": 33}})

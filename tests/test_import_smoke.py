"""Import smoke test: every module imports cleanly (catches syntax/typo/circular imports)."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "ascon_smart_agri",
    "ascon_smart_agri.cli",
    "ascon_smart_agri.data.characterize",
    "ascon_smart_agri.data.subsample",
    "ascon_smart_agri.data.dedup",
    "ascon_smart_agri.data.split",
    "ascon_smart_agri.features.selection",
    "ascon_smart_agri.sequences.windowing",
    "ascon_smart_agri.model.gru",
    "ascon_smart_agri.model.train",
    "ascon_smart_agri.federated.partition",
    "ascon_smart_agri.federated.aggregation",
    "ascon_smart_agri.federated.scaler_stats",
    "ascon_smart_agri.federated.serialization",
    "ascon_smart_agri.federated.client",
    "ascon_smart_agri.federated.server",
    "ascon_smart_agri.telemetry.simulate",
    "ascon_smart_agri.telemetry.provenance",
    "ascon_smart_agri.crypto.ascon_aead",
    "ascon_smart_agri.routing.router",
    "ascon_smart_agri.routing.cloud_sink",
    "ascon_smart_agri.routing.alert_sink",
    "ascon_smart_agri.eval.metrics",
    "ascon_smart_agri.eval.baselines",
    "ascon_smart_agri.eval.report",
    "ascon_smart_agri.eval.manifest",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)

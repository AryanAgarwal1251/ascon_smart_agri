"""Unit tests for the per-run manifest (Section III-I4).

No headline number is reported without a manifest, so the manifest has to survive a round trip
and has to record the things that make a number reproducible: seeds, config, versions, commit.
"""

from __future__ import annotations

import json
from pathlib import Path

from ascon_smart_agri.eval.manifest import RunManifest, collect_environment


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = RunManifest(
        run_name="unit",
        seeds=[0, 1, 2],
        library_versions={"torch": "2.7.0"},
        hardware={"platform": "test"},
        config_snapshot={"data": {"seed": 0}},
        subsample_per_class_counts={"BenignTraffic": 10},
        results={"macro_f1_mean": 0.5},
    )

    path = manifest.write(tmp_path)
    loaded = json.loads(path.read_text())

    assert path.name == "manifest_unit.json"
    assert loaded["seeds"] == [0, 1, 2]
    assert loaded["config_snapshot"]["data"]["seed"] == 0
    assert loaded["results"]["macro_f1_mean"] == 0.5
    assert loaded["subsample_per_class_counts"]["BenignTraffic"] == 10


def test_manifest_creates_a_missing_output_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "run"

    path = RunManifest(run_name="r", seeds=[0, 1, 2]).write(target)

    assert path.exists()


def test_manifest_records_at_least_three_seeds_when_given_them() -> None:
    # Section III-I4: a single-run number is not a finding. The manifest carries the seed list
    # so a reader can check that rule was followed.
    manifest = RunManifest(run_name="r", seeds=[0, 1, 2])

    assert len(manifest.seeds) >= 3


def test_collect_environment_reports_versions_and_hardware() -> None:
    versions, hardware = collect_environment()

    assert "python" in versions
    assert "torch" in versions
    assert "numpy" in versions
    assert "platform" in hardware
    assert "git_commit" in hardware
    assert "git_dirty" in hardware

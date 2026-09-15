"""The `asa` entry point dispatches to real drivers or says plainly why it cannot."""

from __future__ import annotations

import runpy
import sys

import pytest

from ascon_smart_agri import cli


def test_every_phase_has_a_driver_or_an_explanation() -> None:
    for phase, (_, script) in cli.PHASES.items():
        if script is None:
            assert phase in cli.LIBRARY_ONLY
        else:
            assert (cli.SCRIPTS / script).exists(), f"{phase} -> {script} missing"


def test_library_only_phase_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["telemetry"]) == 2
    assert "no standalone driver" in capsys.readouterr().err


def test_arguments_pass_through_to_the_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_path(path: str, run_name: str) -> None:
        seen["path"], seen["argv"] = path, list(sys.argv)

    monkeypatch.setattr(runpy, "run_path", fake_run_path)
    assert cli.main(["--config", "c.yaml", "federate", "--rounds", "2"]) == 0
    assert str(seen["path"]).endswith("run_phase4.py")
    assert seen["argv"][1:] == ["--config", "c.yaml", "--rounds", "2"]

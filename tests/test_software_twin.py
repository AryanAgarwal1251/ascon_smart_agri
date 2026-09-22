"""The Phase 8 software twin, headless (docs/plans/phase8-software-twin.md).

Two always-on checks keep the compose topology honest against the CLI, and one end-to-end
smoke runs the whole thing as local processes --- sealed federation, cloud receiver, Pi
runtime with a scripted attack scenario --- whenever the Phase 4 cache is on disk (it is
gitignored, so the smoke skips on a checkout without it rather than failing).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from ascon_smart_agri import cli

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "docker-compose.yml"
CACHE = REPO / "artifacts" / "phase4_cache.npz"
ENV = {**os.environ, "PYTHONPATH": str(REPO)}


def _asa_commands_in(service: dict[str, object]) -> list[str]:
    """Every `asa <subcommand>` a compose service invokes, in list or shell-string form."""
    command = service.get("command")
    if command is None:
        return []
    if isinstance(command, list) and command and not str(command[0]).startswith("asa "):
        return [str(command[0])]
    text = " ".join(command) if isinstance(command, list) else str(command)
    return [tok.split()[1] for tok in text.split("&&") if tok.split() and tok.split()[0] == "asa"]


def test_compose_services_only_invoke_registered_subcommands() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    for name, service in services.items():
        if (
            "image" in service
            and "build" not in service
            and service.get("image") != "ascon-smart-agri:twin"
        ):
            continue  # mosquitto
        if service.get("entrypoint", [""])[0] == "/bin/sh" and "init-secrets" in name:
            continue
        for sub in _asa_commands_in(service):
            assert sub in cli.PHASES, f"{name} runs `asa {sub}` which the CLI does not know"


def test_compose_topology_matches_the_plan() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    assert {"aggregator", "cloud-receiver", "pi-1", "pi-2", "sim-3"} <= set(services)
    assert {"broker-farm1", "broker-farm2", "sensors-farm1", "sensors-farm2"} <= set(services)
    # Each farm's Pi talks to ITS OWN broker; readings never traverse the laptop.
    assert "broker-farm1" in services["pi-1"]["command"][0]
    assert "broker-farm2" in services["pi-2"]["command"][0]
    # The cloud hop is TLS; the weight channel keys are mounted read-only.
    assert "https://cloud-receiver" in services["pi-1"]["command"][0]
    assert any(v.startswith("keys:/keys:ro") for v in services["pi-1"]["volumes"])


def test_no_document_tells_anyone_to_abort_on_container_exit() -> None:
    """``--abort-on-container-exit`` breaks this topology, so no doc may recommend it.

    Compose treats ANY container exit as the abort signal, and ``init-secrets`` is a one-shot
    that exits 0 by design. Measured on 2026-09-22: init-secrets finished 4.2 s into the run,
    compose stopped the brokers and SIGKILLed the aggregator 3.4 s after that, and the Pis then
    spent 28 minutes loading their partitions before dying on ``Name or service not known``.
    ``--exit-code-from`` implies the same flag and fails the same way.
    """
    offenders = []
    for path in (COMPOSE, REPO / "CLAUDE.md", REPO / "docs" / "plans" / "phase8-software-twin.md"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            # Drop a YAML comment marker, then anything after a trailing "#" -- a note *about*
            # the flag is exactly what this fix added, and must not count as recommending it.
            command = line.lstrip("# ").split("#", 1)[0].strip()
            if command.startswith(("docker compose", "$ docker compose")) and (
                "--abort-on-container-exit" in command or "--exit-code-from" in command
            ):
                offenders.append(f"{path.name}:{number}: {command}")
    assert not offenders, "these recommend a flag that tears the topology down:\n" + "\n".join(
        offenders
    )


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_file_is_valid_for_docker() -> None:
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--quiet"],
        capture_output=True, text=True, cwd=REPO,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr


def _run(args: list[str], **kw: object) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-u", *args],
        cwd=REPO,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **kw,  # type: ignore[arg-type]
    )


@pytest.mark.skipif(not CACHE.exists(), reason="artifacts/phase4_cache.npz not present")
def test_headless_twin_end_to_end(tmp_path: Path) -> None:
    keys = tmp_path / "twin.demo.key"
    subprocess.run(
        [sys.executable, "scripts/run_aggregator.py", "--generate-keys", "--keys", str(keys),
         "--clients", "pi-1,pi-2,sim-3"],
        cwd=REPO, env=ENV, check=True, capture_output=True,
    )  # fmt: skip
    out_cfg = tmp_path / "cfg.yaml"
    out_cfg.write_text(f"output_dir: {tmp_path / 'artifacts'}\n")
    common = ["--config", str(out_cfg), "--keys", str(keys)]

    # --- training plane: sealed federation, one round, tiny caps -------------------------
    agg = _run(["scripts/run_aggregator.py", *common, "--host", "127.0.0.1", "--port", "7733",
                "--rounds", "1", "--checkpoint", str(tmp_path / "g.safetensors")])  # fmt: skip
    time.sleep(2)
    client_args = ["--server", "127.0.0.1:7733", "--rounds", "1", "--local-epochs", "1",
                   "--sequence-cap", "300", "--platform", "pytest-headless-twin"]  # fmt: skip
    clients = []
    for i, cid in enumerate(["pi-1", "pi-2", "sim-3"]):
        node = ["scripts/run_client_node.py", *common, "--client-id", cid, "--client-index", str(i)]
        clients.append(_run([*node, *client_args]))
    for proc in (*clients, agg):
        out, _ = proc.communicate(timeout=600)
        assert proc.returncode == 0, out

    # --- runtime plane: cloud receiver + pi-1 runtime with a scripted attack switch ------
    cloud = _run(["scripts/run_cloud_receiver.py", "--config", str(out_cfg), "--host", "127.0.0.1",
                  "--port", "8447", "--edges", "pi-1", "--seconds", "120",
                  "--platform", "pytest-headless-twin"])  # fmt: skip
    time.sleep(1.5)
    runtime = _run(["scripts/run_pi_runtime.py", "--config", str(out_cfg), "--client-id", "pi-1",
                    "--source", "sim", "--messages", "90", "--attack-after", "45",
                    "--cloud", "http://127.0.0.1:8447",
                    "--platform", "pytest-headless-twin"])  # fmt: skip
    out, _ = runtime.communicate(timeout=300)
    assert runtime.returncode == 0, out
    cloud.send_signal(signal.SIGINT)  # Ctrl-C: the receiver writes its manifest and exits 0
    cloud_out, _ = cloud.communicate(timeout=60)
    assert cloud.returncode == 0, cloud_out

    r = json.loads((tmp_path / "artifacts" / "manifest_phase8_runtime_pi-1.json").read_text())[
        "results"
    ]
    c = json.loads((tmp_path / "artifacts" / "manifest_phase8_cloud_receiver.json").read_text())[
        "results"
    ]
    assert r["received"] == 90 and r["platform"] == "pytest-headless-twin"
    assert r["benign_verdicts"] + r["malicious_verdicts"] + r["buffering"] == 90
    # G1 on the wire: the cloud accepted exactly the benign verdicts and nothing else.
    assert r["cloud_sent"] == r["benign_verdicts"] == c["accepted"]
    assert r["cloud_failed"] == 0 and c["rejected"] == 0
    assert r["alerts"] == r["malicious_verdicts"]
    assert r["scenario_agreement"]["attack"][1] > 0  # the scenario switch reached the adapter

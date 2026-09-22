"""Phase 8: the local-GRU runtime on a Raspberry Pi (plan §2 runtime plane; §8 option A).

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_pi_runtime.py \\
        --client-id pi-1 --farm farm1 --broker localhost:1883 \\
        --cloud https://<laptop-ip>:8443 --ca-cert keys/cloud.demo.pem --messages 200

The runtime plane, one message at a time, exactly as Phase 7 ran it but fed by the farm's
MQTT broker and routed over TLS:

    MqttSensorSource (farm broker; ESP32, Wokwi or virtual sensors -- indistinguishable)
        -> FeatureProvenanceAdapter (G6: held-out records only; the device's SCENARIO picks
           the benign or attack pool -- a selection over real records, never synthesis)
        -> DeviceWindowBuffer (W readings per device)
        -> the global GRU this node received back from the master (run_client_node.py)
        -> Eq. (5): benign  -> TlsVerdictRouter -> HTTPS cloud receiver
                    malicious -> AlertSink (holds no cloud reference)

The held-out pool is the Phase 4 cache's test split (never trained on), scaled with the
federated scaler this node received in the sealed round 0.

``--source sim`` replaces the broker with ``telemetry.simulate.simulate_stream`` and a scripted
scenario switch (``--attack-after``), so the whole runtime can run headlessly in a test with
no broker at all. The manifest records ``platform`` so a simulation is never mistaken for a
Pi result, and the same declared limitation as Phase 7 applies: this shows the architecture
works end to end, not that the detector would catch attacks on a live farm network.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import time
from pathlib import Path

import numpy as np
import torch
from configs.base import load_run_config

from ascon_smart_agri.data.taxonomy import BENIGN_CLASS_INDEX, CLASS_NAMES
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.model.checkpoint import load_model
from ascon_smart_agri.routing.alert_sink import AlertSink
from ascon_smart_agri.routing.tls_path import HttpsCloudClient, ReadingEnvelope, TlsVerdictRouter
from ascon_smart_agri.sequences.streaming import DeviceWindowBuffer
from ascon_smart_agri.telemetry.mqtt_source import MqttSensorSource, SensorReading
from ascon_smart_agri.telemetry.provenance import FeatureProvenanceAdapter
from ascon_smart_agri.telemetry.simulate import simulate_stream

print = functools.partial(builtins.print, flush=True)


def held_out_adapter(cache: Path, scaler: Path, seed: int) -> FeatureProvenanceAdapter:
    """The G6 pool: the cache's TEST split, scaled with the node's federated scaler."""
    blob = np.load(cache, allow_pickle=False)
    sc = np.load(scaler, allow_pickle=False)
    x = np.asarray(blob["Xte"], dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    scaled = ((x[finite] - sc["mean"]) / sc["std"]).astype(np.float32)
    refs = [f"{s}:{i}" for s, i in zip(blob["ste"][finite], blob["ite"][finite], strict=True)]
    labels = [CLASS_NAMES[int(y)] for y in blob["yte"][finite]]
    return FeatureProvenanceAdapter(scaled, refs, labels=labels, seed=seed)


def simulated_readings(
    device_ids: list[str], n: int, seed: int, attack_after: int
) -> list[SensorReading]:
    """Broker-free stand-in: the Phase 5 simulator plus a scripted scenario switch."""
    out = []
    for m in simulate_stream(device_ids, n_messages=n, seed=seed):
        scenario = "attack" if attack_after and m.stream_index >= attack_after else "benign"
        out.append(
            SensorReading(
                farm_id="sim",
                device_id=m.device_id,
                payload={**m.payload, "seq": m.counter},
                counter=m.counter,
                stream_index=m.stream_index,
                scenario=scenario,
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--farm", default="farm1", help="farm id in the MQTT topic contract")
    parser.add_argument("--source", choices=["mqtt", "sim"], default="mqtt")
    parser.add_argument("--broker", default="localhost:1883")
    parser.add_argument("--cloud", default="http://localhost:8443", help="receiver base URL")
    parser.add_argument("--ca-cert", default="", help="the receiver's demo certificate (PEM)")
    parser.add_argument(
        "--checkpoint", default="", help="default: artifacts/phase8_client_<id>_model"
    )
    parser.add_argument("--scaler", default="", help="default: artifacts/phase8_client_<id>_scaler")
    parser.add_argument("--cache", default="artifacts/phase4_cache.npz")
    parser.add_argument("--messages", type=int, default=200, help="stop after this many readings")
    parser.add_argument("--idle-timeout", type=float, default=60.0, help="mqtt: stop after silence")
    parser.add_argument("--attack-after", type=int, default=0, help="sim: switch scenario at index")
    parser.add_argument("--platform", default="unspecified")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    out_dir = Path(cfg.output_dir)
    checkpoint = Path(
        args.checkpoint or out_dir / f"phase8_client_{args.client_id}_model.safetensors"
    )
    scaler = Path(args.scaler or out_dir / f"phase8_client_{args.client_id}_scaler.npz")
    for path in (checkpoint, scaler, Path(args.cache)):
        if not path.exists():
            raise SystemExit(f"missing {path}; run `asa client-node` for {args.client_id} first")

    adapter = held_out_adapter(Path(args.cache), scaler, cfg.telemetry.seed)
    model = load_model(
        checkpoint,
        n_features=cfg.model.n_features,
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
    )
    buffer = DeviceWindowBuffer(window=cfg.sequence.window)
    cloud = HttpsCloudClient(args.cloud, ca_cert=Path(args.ca_cert) if args.ca_cert else None)
    alert = AlertSink()
    router = TlsVerdictRouter(cloud, alert)
    print(
        f"[{args.client_id}] runtime | model {checkpoint.name} | W={cfg.sequence.window}"
        f" | cloud {args.cloud} ({'TLS' if args.cloud.startswith('https') else 'plain'})"
        f" | source {args.source}"
    )

    source: MqttSensorSource | None = None
    if args.source == "mqtt":
        host, _, port = args.broker.rpartition(":")
        source = MqttSensorSource(args.farm)
        source.connect(host, int(port))
        readings = source.readings(timeout_s=args.idle_timeout)
        print(f"[{args.client_id}] subscribed to farm/{args.farm}/sensor/# on {args.broker}")
    else:
        readings = iter(
            simulated_readings(
                cfg.telemetry.device_ids, args.messages, cfg.telemetry.seed, args.attack_after
            )
        )

    latency_us: dict[str, list[float]] = {
        "provenance": [],
        "window": [],
        "inference": [],
        "routing": [],
    }
    n_received = n_buffering = n_benign = n_malicious = 0
    scenario_agreement = {"benign": [0, 0], "attack": [0, 0]}  # [agree, total], informal only
    started = time.time()

    for reading in readings:
        n_received += 1
        t0 = time.perf_counter()
        provenanced = adapter.network_features_for(reading.stream_index, scenario=reading.scenario)
        latency_us["provenance"].append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        window = buffer.push(reading.device_id, provenanced.features)
        latency_us["window"].append((time.perf_counter() - t0) * 1e6)
        if window is None:
            n_buffering += 1
            if n_received >= args.messages:
                break
            continue

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(torch.as_tensor(window[None, :, :], dtype=torch.float32))
            predicted = int(logits.argmax(dim=1).item())
        latency_us["inference"].append((time.perf_counter() - t0) * 1e6)
        benign = predicted == BENIGN_CLASS_INDEX  # Eq. (5)
        n_benign += benign
        n_malicious += not benign
        tally = scenario_agreement[reading.scenario]
        tally[0] += benign == (reading.scenario == "benign")
        tally[1] += 1

        t0 = time.perf_counter()
        router.route(
            verdict_benign=benign,
            envelope=ReadingEnvelope(
                args.client_id,
                reading.device_id,
                reading.counter,
                cfg.telemetry.schema_version,
                reading.payload,
            ),
        )
        latency_us["routing"].append((time.perf_counter() - t0) * 1e6)
        if n_received >= args.messages:
            break

    if source is not None:
        source.close()
    elapsed = time.time() - started

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values)
        return {"median_us": float(np.median(arr)), "mean_us": float(arr.mean()), "n": len(arr)}

    print(
        f"\n[{args.client_id}] received {n_received} | buffering {n_buffering}"
        f" | benign {n_benign} -> cloud (sent {cloud.sent}, failed {cloud.failed})"
        f" | malicious {n_malicious} -> alerts {alert.alert_count}"
    )
    print(
        f"[{args.client_id}] G1: malicious readings reaching the cloud = 0 by construction;"
        f" cloud accepted {cloud.sent} == benign verdicts {n_benign}: {cloud.sent == n_benign}"
    )
    for scenario, (agree, total) in scenario_agreement.items():
        if total:
            print(
                f"[{args.client_id}] scenario {scenario}: verdict agreed {agree}/{total} (informal)"
            )

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase8_runtime_{args.client_id}",
        seeds=[cfg.telemetry.seed],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=cfg.model_dump(mode="json"),
        results={
            "client_id": args.client_id,
            "farm": args.farm,
            "source": args.source,
            "checkpoint": str(checkpoint),
            "received": n_received,
            "buffering": n_buffering,
            "benign_verdicts": n_benign,
            "malicious_verdicts": n_malicious,
            "cloud_sent": cloud.sent,
            "cloud_failed": cloud.failed,
            "cloud_last_error": cloud.last_error,
            "alerts": alert.alert_count,
            "scenario_agreement": scenario_agreement,
            "dropped_by_source": source.dropped if source is not None else [],
            "stage_latency": {k: summary(v) for k, v in latency_us.items()},
            "elapsed_seconds": round(elapsed, 1),
            "platform": args.platform,
            "declared_limitation": (
                "architectural correctness only; held-out records, not live capture"
            ),
        },
    )
    out = manifest.write(out_dir)
    print(f"[{args.client_id}] manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

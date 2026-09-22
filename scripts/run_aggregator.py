"""Phase 8: the master-GRU aggregator node (plan §2, §4). Runs on the laptop.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_aggregator.py \\
        --keys keys/phase8.demo.key --clients pi-1,pi-2,sim-3 --host 0.0.0.0 --port 7700

Serves round 0 (the sealed scaler exchange) and then ``--rounds`` weight rounds, aggregating
by weighted FedAvg (Eq. 21) in the order ``--clients`` lists them. Every frame in and out is
Ascon-sealed (``federated/transport.py``); a frame that fails to open is dropped without a
reply and logged. When the last round is answered the node saves the global model as a
safetensors checkpoint and writes ``artifacts/manifest_phase8_aggregator.json`` with the
per-round byte counts, timings and rejection log.

``--generate-keys`` writes a fresh labelled demo key file for the listed clients and exits;
copy it to each Pi by hand (it is gitignored: ``keys/`` and ``*.demo.key``).
"""

from __future__ import annotations

import argparse
import builtins
import dataclasses
import functools
import time
from pathlib import Path

import torch
from configs.base import load_run_config

from ascon_smart_agri.crypto.ascon_aead import backend_provenance, verify_kat_conformance
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.federated.node import AggregatorNode
from ascon_smart_agri.federated.transport import generate_demo_keys, load_keys, write_keys
from ascon_smart_agri.model.checkpoint import save_model
from ascon_smart_agri.model.gru import build_detector

print = functools.partial(builtins.print, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--keys", default="keys/phase8.demo.key", help="labelled demo key file")
    parser.add_argument("--clients", default="pi-1,pi-2,sim-3", help="client ids, in order")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7700)
    parser.add_argument("--rounds", type=int, default=0, help="R; default: config value")
    parser.add_argument("--aggregation", default="", help="weighted|unweighted")
    parser.add_argument("--round-timeout", type=float, default=3600.0, help="seconds")
    parser.add_argument("--checkpoint", default="artifacts/phase8_global_model.safetensors")
    parser.add_argument("--generate-keys", action="store_true")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    client_ids = [c.strip() for c in args.clients.split(",") if c.strip()]
    keys_path = Path(args.keys)

    if args.generate_keys:
        write_keys(keys_path, generate_demo_keys(client_ids))
        print(f"[keys] wrote labelled DEMO keys for {client_ids} -> {keys_path} (not in git)")
        return 0

    if not verify_kat_conformance():
        print("[crypto] Ascon KAT conformance FAILED; refusing to serve (R5)")
        return 1
    keys = load_keys(keys_path)
    rounds = args.rounds or cfg.federated.rounds
    aggregation = args.aggregation or cfg.federated.aggregation

    node = AggregatorNode(
        client_ids,
        keys,
        rounds=rounds,
        aggregation=aggregation,
        host=args.host,
        port=args.port,
        round_timeout_s=args.round_timeout,
    )
    print(
        f"[aggregator] {node.host}:{node.port} | clients {client_ids} | R={rounds}"
        f" | {aggregation} FedAvg | scaler round 0 first"
    )
    started = time.time()
    node.start()
    answered = -1
    try:
        while not node.finished:
            time.sleep(1.0)
            if node.records and node.records[-1].round_index != answered:
                rec = node.records[-1]
                answered = rec.round_index
                what = "scaler" if rec.round_index == 0 else f"round {rec.round_index}/{rounds}"
                print(
                    f"  [{what}] n_k={rec.sequence_counts} | up {rec.bytes_up:,} B"
                    f" | down {rec.bytes_down:,} B | {rec.seconds:.1f}s"
                    f" | rejected so far {rec.rejections}"
                )
    finally:
        node.stop()

    assert node.global_state is not None
    model = build_detector(cfg.model.n_features, cfg.model.hidden_size, cfg.model.n_classes)
    model.load_state_dict({k: v.clone() for k, v in node.global_state.items()})
    checkpoint = save_model(
        model,
        Path(args.checkpoint),
        metadata={"phase": "8", "rounds": str(rounds), "clients": ",".join(client_ids)},
    )

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name="phase8_aggregator",
        seeds=[cfg.data.seed],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=cfg.model_dump(mode="json"),
        ascon_backend={**backend_provenance(), "kat_passed": "true"},
        results={
            "clients": client_ids,
            "rounds": rounds,
            "aggregation": aggregation,
            "key_file": str(keys_path),
            "key_label": "DEMO",
            "per_round": [dataclasses.asdict(r) for r in node.records],
            "rejections": node.rejections,
            "checkpoint": str(checkpoint),
            "elapsed_seconds": round(time.time() - started, 1),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "torch_threads": torch.get_num_threads(),
        },
    )
    out = manifest.write(Path(cfg.output_dir))
    print(f"[aggregator] done | checkpoint {checkpoint} | manifest {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

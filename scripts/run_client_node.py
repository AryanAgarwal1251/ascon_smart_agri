"""Phase 8: a local-GRU client node (plan §2, §4). Runs on each Raspberry Pi (or on the laptop
as the simulated third client).

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_client_node.py \\
        --client-id pi-1 --client-index 0 --server 192.168.1.10:7700 \\
        --keys keys/phase8.demo.key --cache artifacts/phase4_cache.npz --sequence-cap 40000

The node carves client ``--client-index`` out of the Phase 4 cache with the same Dirichlet
draw every node uses (``federated/cache_partition.py``), sends its scaler statistics sealed
(round 0), receives the global ``(mean, std)`` sealed, windows its own rows, then trains
locally for ``--rounds`` rounds, sealing each ``(theta_k, n_k)`` up and opening the global
that comes back. Nothing but sealed frames leaves the process. When finished it writes
``artifacts/manifest_phase8_client_<id>.json`` with its byte counts and per-round timings ---
on a Pi those timings are the feasibility numbers plan §7 asks for --- and saves what the
runtime needs next: the final global model (``artifacts/phase8_client_<id>_model.safetensors``)
and the federated scaler (``..._scaler.npz``), both consumed by ``run_pi_runtime.py``.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import time
from pathlib import Path

import numpy as np
from configs.base import load_run_config

from ascon_smart_agri.crypto.ascon_aead import backend_provenance, verify_kat_conformance
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.federated.cache_partition import (
    load_client_partition,
    make_federated_client,
)
from ascon_smart_agri.federated.node import ClientNode, initial_global_state
from ascon_smart_agri.federated.transport import load_keys
from ascon_smart_agri.model.checkpoint import save_model
from ascon_smart_agri.model.gru import build_detector

print = functools.partial(builtins.print, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--client-id", required=True, help="e.g. pi-1; must match the key file")
    parser.add_argument("--client-index", type=int, required=True, help="0..K-1 in the partition")
    parser.add_argument("--server", default="127.0.0.1:7700", help="aggregator host:port")
    parser.add_argument("--keys", default="keys/phase8.demo.key")
    parser.add_argument("--cache", default="artifacts/phase4_cache.npz")
    parser.add_argument("--rounds", type=int, default=0, help="R; default: config value")
    parser.add_argument("--local-epochs", type=int, default=0, help="E; default: config value")
    parser.add_argument("--alpha", type=float, default=0.0, help="default: config value")
    parser.add_argument("--sequence-cap", type=int, default=0, help="0 = no cap (plan §7)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--connect-retries", type=int, default=120, help="x0.5s; the aggregator may start later"
    )
    parser.add_argument(
        "--platform",
        default="unspecified",
        help="recorded verbatim, e.g. raspberry-pi-5 or docker-arm64-simulation; timings "
        "from a simulation are never quoted as Pi numbers",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if not verify_kat_conformance():
        print("[crypto] Ascon KAT conformance FAILED; refusing to run (R5)")
        return 1
    keys = load_keys(Path(args.keys))
    if args.client_id not in keys:
        print(f"[keys] no channel keys for {args.client_id!r} in {args.keys}")
        return 1
    host, _, port = args.server.rpartition(":")
    rounds = args.rounds or cfg.federated.rounds
    local_epochs = args.local_epochs or cfg.federated.local_epochs
    alpha = args.alpha or cfg.federated.dirichlet_alpha
    cap = args.sequence_cap or None
    seed = cfg.data.seed

    t0 = time.time()
    partition = load_client_partition(
        Path(args.cache),
        client_index=args.client_index,
        n_clients=cfg.federated.n_clients,
        alpha=alpha,
        seed=seed,
    )
    print(
        f"[{args.client_id}] partition {args.client_index}: {len(partition.features):,} rows"
        f" | F={len(partition.columns)} | loaded in {time.time() - t0:.1f}s"
    )

    def make_client(mean: np.ndarray, std: np.ndarray):  # type: ignore[no-untyped-def]
        client = make_federated_client(
            partition,
            mean,
            std,
            window=cfg.sequence.window,
            seed=seed,
            hidden_size=cfg.model.hidden_size,
            n_classes=cfg.model.n_classes,
            sequence_cap=cap,
            batch_size=args.batch_size,
        )
        print(f"[{args.client_id}] {client.n_sequences:,} training sequences (cap={cap})")
        return client

    node = ClientNode(
        args.client_id,
        keys[args.client_id],
        server=(host, int(port)),
        local_stats=partition.stats,
        make_client=make_client,
        initial_state=initial_global_state(
            seed, cfg.model.n_features, cfg.model.hidden_size, cfg.model.n_classes
        ),
        rounds=rounds,
        local_epochs=local_epochs,
        retries=args.connect_retries,
    )
    print(f"[{args.client_id}] -> {host}:{port} | R={rounds} | E={local_epochs} | sealed both ways")
    started = time.time()
    node.run()
    elapsed = time.time() - started
    assert node.federated_client is not None
    assert node.global_state is not None and node.scaler is not None

    # What the Pi runtime needs: the weights it received back, and the scaler it was given.
    model = build_detector(cfg.model.n_features, cfg.model.hidden_size, cfg.model.n_classes)
    model.load_state_dict({k: v.clone() for k, v in node.global_state.items()})
    out_dir = Path(cfg.output_dir)
    checkpoint = save_model(
        model,
        out_dir / f"phase8_client_{args.client_id}_model.safetensors",
        metadata={"phase": "8", "client_id": args.client_id, "rounds": str(rounds)},
    )
    scaler_path = out_dir / f"phase8_client_{args.client_id}_scaler.npz"
    np.savez(scaler_path, mean=node.scaler[0], std=node.scaler[1], columns=partition.columns)

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase8_client_{args.client_id}",
        seeds=[seed],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=cfg.model_dump(mode="json"),
        ascon_backend={**backend_provenance(), "kat_passed": "true"},
        results={
            "client_id": args.client_id,
            "client_index": args.client_index,
            "rows": len(partition.features),
            "n_sequences": node.federated_client.n_sequences,
            "sequence_cap": cap,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "bytes_up": node.bytes_up,
            "bytes_down": node.bytes_down,
            "round_seconds": [round(s, 2) for s in node.round_seconds],
            "elapsed_seconds": round(elapsed, 1),
            "checkpoint": str(checkpoint),
            "scaler": str(scaler_path),
            "platform": args.platform,
        },
    )
    out = manifest.write(Path(cfg.output_dir))
    print(
        f"[{args.client_id}] done in {elapsed:.0f}s | up {node.bytes_up:,} B"
        f" | down {node.bytes_down:,} B | manifest {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

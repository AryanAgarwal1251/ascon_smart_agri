"""Phase 7: the real end-to-end runtime pipeline (Section III-A, III-H, Eq. 5).

Assembles every piece built across Phases 1-6 into the actual runtime plane the paper
describes: "telemetry arrives, is classified using the current global model, and is routed
according to Equation (5)." This is the first script in the project where that whole sentence
is executed for real, in order, on one message at a time -- not simulated piecewise.

    simulate_stream (Phase 5)
        -> FeatureProvenanceAdapter (Phase 5, G6: held-out features, never payload-derived)
        -> DeviceWindowBuffer (Phase 7: streaming window assembly)
        -> the REAL trained federated model (Phase 7: train_federated_model.py's checkpoint)
        -> Eq. (5) binary projection (y > 0, Benign is class index 0)
        -> VerdictRouter (Phase 6): benign -> MockCloudReceiver (plaintext)
                                     malicious -> AlertSink (no cloud reference, ever)

    IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER (flagged per CLAUDE.md golden rule 1; see
    ``docs/design_paper.md``'s "Implementation deviation" section): the cloud leg here no longer
    encrypts -- cloud-payload protection is assumed handled outside this codebase. Ascon-AEAD128
    now protects Channel 3 (the federated weight transport, ``federated/crypto.py``) instead,
    which is exercised during training (``train_federated_model.py``), not in this runtime demo.

Per-stage latency (Section III-I2's "per-stage runtime latency") is measured for each message:
provenance lookup, window buffering, model inference, and routing (send or alert).

DECLARED LIMITATION, same as telemetry/provenance.py's own: this demonstrates architectural
correctness -- the pipeline composes, the planes stay separate, and routing is correct. It does
NOT demonstrate that this model would detect attacks against a live agricultural deployment; the
held-out records are historical CICIoT2023 rows, not live capture.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase7.py \
        --checkpoint artifacts/federated_global_model.safetensors [--cache path.npz]
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path

import numpy as np
import torch
from configs.base import load_run_config

from ascon_smart_agri.crypto.ascon_aead import AssociatedData
from ascon_smart_agri.data.dedup import deduplicate
from ascon_smart_agri.data.scaling import fit_scaler
from ascon_smart_agri.data.split import make_blocks, stratified_block_split
from ascon_smart_agri.data.subsample import stratified_capped_subsample
from ascon_smart_agri.data.taxonomy import BENIGN_CLASS_INDEX
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.report import mean_std
from ascon_smart_agri.features.selection import FeatureSelector
from ascon_smart_agri.model.checkpoint import load_model, read_metadata
from ascon_smart_agri.routing.alert_sink import AlertSink
from ascon_smart_agri.routing.cloud_sink import MockCloudReceiver
from ascon_smart_agri.routing.router import VerdictRouter
from ascon_smart_agri.sequences.streaming import DeviceWindowBuffer
from ascon_smart_agri.telemetry.provenance import FeatureProvenanceAdapter
from ascon_smart_agri.telemetry.simulate import simulate_stream

print = functools.partial(builtins.print, flush=True)

EDGE_ID = "gateway01"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="artifacts/federated_global_model.safetensors")
    parser.add_argument(
        "--cache",
        default="",
        help="unused; the runtime pipeline needs the "
        "held-out frame itself (source_file, label), not flattened arrays",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint at {checkpoint_path}. Run train_federated_model.py first.")
    checkpoint_meta = read_metadata(checkpoint_path)
    print(f"[checkpoint] {checkpoint_path}  metadata={checkpoint_meta}")

    # ---- Rebuild the SAME held-out test split the checkpoint was evaluated against ----------
    frame, _ = stratified_capped_subsample(
        cfg.data.dataset_root,
        target=cfg.data.subsample_target,
        per_class_cap=cfg.data.per_class_cap,
        chunk_size=cfg.data.chunk_size,
        seed=cfg.data.seed,
    )
    deduped = deduplicate(frame)
    block_ids = make_blocks(deduped, block_size=cfg.data.block_size)
    split = stratified_block_split(
        deduped, block_ids, test_fraction=cfg.data.test_fraction, seed=cfg.data.seed
    )
    train_frame = deduped.loc[block_ids.isin(split.train_blocks)]
    test_frame = deduped.loc[block_ids.isin(split.test_blocks)]  # THE HELD-OUT SET (G6)

    selection = FeatureSelector().fit(
        train_frame.drop(columns=["label"]),
        train_frame["label"],
        tau=cfg.features.correlation_tau,
        f=cfg.features.selected_f,
        f_sweep=cfg.features.f_sweep,
        rf_n_estimators=cfg.features.rf_n_estimators,
        rrf_k=cfg.features.rrf_k,
        sample_size=cfg.features.selection_sample_size,
        seed=cfg.data.seed,
    )
    columns = selection.selected_columns
    train_values = train_frame[columns].to_numpy(dtype=np.float64)
    scaler = fit_scaler(train_values[np.isfinite(train_values).all(axis=1)])
    print(f"[data] held-out test split: {len(test_frame):,} rows, F={len(columns)}")

    # ---- The pieces (Phases 5 and 6) -----------------------------------------------------
    adapter = FeatureProvenanceAdapter.from_held_out_frame(
        test_frame, feature_columns=columns, scaler=scaler, seed=cfg.telemetry.seed
    )
    model = load_model(
        checkpoint_path,
        n_features=cfg.model.n_features,
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
    )
    buffer = DeviceWindowBuffer(window=cfg.sequence.window)

    cloud = MockCloudReceiver()
    alert = AlertSink()
    router = VerdictRouter(cloud, alert)

    messages = list(
        simulate_stream(
            cfg.telemetry.device_ids,
            n_messages=cfg.telemetry.messages_per_stream,
            seed=cfg.telemetry.seed,
            schema_version=cfg.telemetry.schema_version,
            temperature_range_c=tuple(cfg.telemetry.temperature_range_c),
            soil_moisture_range_pct=tuple(cfg.telemetry.soil_moisture_range_pct),
        )
    )
    n_devices = len(cfg.telemetry.device_ids)
    print(f"[telemetry] simulated {len(messages)} messages across {n_devices} devices")

    # ---- The real runtime loop: one message at a time, in order ----------------------------
    stage_latency_us: dict[str, list[float]] = {
        "provenance": [],
        "window": [],
        "inference": [],
        "routing": [],
    }
    n_buffering = n_classified = n_benign = n_malicious = 0
    agreement_with_true_label = 0  # informal sanity signal only -- see module docstring

    for message in messages:
        t0 = time.perf_counter()
        provenanced = adapter.network_features_for(message.stream_index)
        stage_latency_us["provenance"].append((time.perf_counter() - t0) * 1e6)

        t0 = time.perf_counter()
        window = buffer.push(message.device_id, provenanced.features)
        stage_latency_us["window"].append((time.perf_counter() - t0) * 1e6)

        if window is None:
            n_buffering += 1
            continue

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(torch.as_tensor(window[None, :, :], dtype=torch.float32))
            predicted_class = int(logits.argmax(dim=1).item())
        stage_latency_us["inference"].append((time.perf_counter() - t0) * 1e6)

        verdict_benign = predicted_class == BENIGN_CLASS_INDEX  # Eq. (5): y > 0 <=> malicious
        n_classified += 1
        n_benign += verdict_benign
        n_malicious += not verdict_benign
        if provenanced.true_label is not None:
            true_is_benign = provenanced.true_label == "BenignTraffic"
            agreement_with_true_label += verdict_benign == true_is_benign

        t0 = time.perf_counter()
        ad = AssociatedData(EDGE_ID, message.device_id, message.counter, message.schema_version)
        router.route(
            verdict_benign=verdict_benign,
            associated_data=ad,
            payload=json.dumps(message.payload).encode("utf-8"),
        )
        stage_latency_us["routing"].append((time.perf_counter() - t0) * 1e6)

    # ---- Report ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 7 END-TO-END RUN")
    print("=" * 78)
    print(f"messages simulated      : {len(messages)}")
    print(f"  still buffering (< W) : {n_buffering}")
    print(f"  classified            : {n_classified}  ({n_benign} benign, {n_malicious} malicious)")
    print(
        f"cloud received          : {cloud.received_count}  (expect == benign classified: "
        f"{cloud.received_count == n_benign})"
    )
    print(f"cloud rejected          : {cloud.rejected_count}")
    print(
        f"alert count             : {alert.alert_count}  (expect == malicious classified: "
        f"{alert.alert_count == n_malicious})"
    )
    print(
        f"G1 check -- malicious traffic reaching the cloud: "
        f"{cloud.received_count - n_benign} (expect 0)"
    )
    if n_classified:
        print(
            f"informal agreement with true_label (NOT a formal eval, see docstring): "
            f"{agreement_with_true_label}/{n_classified} "
            f"({100 * agreement_with_true_label / n_classified:.1f}%)"
        )
    print("\nper-stage latency (median us):")
    stage_summary: dict[str, dict[str, float]] = {}
    for stage, values in stage_latency_us.items():
        if not values:
            continue
        mean, std = mean_std(values)
        median = sorted(values)[len(values) // 2]
        stage_summary[stage] = {"median_us": median, "mean_us": mean, "std_us": std}
        print(f"  {stage:<12} median={median:.1f}us  mean={mean:.1f}+/-{std:.1f}us")

    versions, hardware = collect_environment()
    checkpoint_seed = int(checkpoint_meta.get("seed", "-1"))
    manifest = RunManifest(
        run_name=f"phase7_e2e_{cfg.run_name}",
        seeds=[checkpoint_seed],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts={},
        # Genuinely empty: this runtime demo's cloud path no longer uses Ascon (implementation
        # deviation, see the module docstring). Ascon now protects the federated weight
        # transport, exercised by train_federated_model.py, not by this script.
        ascon_backend={},
        results={
            "checkpoint_metadata": checkpoint_meta,
            "n_messages": len(messages),
            "n_buffering": n_buffering,
            "n_classified": n_classified,
            "n_benign": n_benign,
            "n_malicious": n_malicious,
            "cloud_received": cloud.received_count,
            "cloud_rejected": cloud.rejected_count,
            "alert_count": alert.alert_count,
            "g1_malicious_reaching_cloud": cloud.received_count - n_benign,
            "informal_true_label_agreement": (
                agreement_with_true_label / n_classified if n_classified else None
            ),
            "per_stage_latency_us": stage_summary,
            "limitation_note": (
                "Demonstrates architectural correctness only: the pipeline composes, planes "
                "stay separate, routing is correct. Does NOT demonstrate detection against a "
                "live deployment -- see telemetry/provenance.py. The cloud leg is plaintext "
                "(implementation deviation, see docs/design_paper.md); Ascon protects the "
                "federated weight transport instead (federated/crypto.py)."
            ),
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"\nmanifest -> {path}")
    print(f"total elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

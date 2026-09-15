"""Train and SAVE one federated global model checkpoint for Phase 7's runtime pipeline.

No training run before this one in this project ever persisted a model to disk -- Phase 3/4's
runs trained, reported metrics, and discarded the weights. This script reuses the identical
Phase 4 configuration and data path (same partition, same alpha=0.5, R=20, E=3, weighted
FedAvg) but trains ONE seed only, then saves the resulting model.

Scope decision (flagged, Golden Rule 1): Phase 4 already trained 3 seeds under this exact
config to establish the STATISTICAL claim (macro-F1 0.8334 +/- 0.0026, reported in
artifacts/phase4_results.json). Training a further 3 seeds here to pick one would be a second,
redundant statistical exercise; Phase 7 needs *a* real, honestly-trained deployment artifact of
the architecture the paper specifies, not a fresh validation. Seed 0 is used -- the first of
Phase 4's own seed list, no other reason for that particular choice. Phase 4's reported mean+/-
std remains the only statistically meaningful performance claim; this checkpoint's own held-out
accuracy is reported too (it will differ slightly from the 3-seed mean, by design) but is not a
new finding.

    PYTHONPATH=. ./.venv/bin/python -u scripts/train_federated_model.py [--cache path.npz]
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from configs.base import load_run_config

from ascon_smart_agri.data.dedup import deduplicate
from ascon_smart_agri.data.scaling import apply_scaler, fit_scaler
from ascon_smart_agri.data.split import make_blocks, stratified_block_split
from ascon_smart_agri.data.subsample import stratified_capped_subsample
from ascon_smart_agri.data.taxonomy import CLASS_NAMES, to_class_index
from ascon_smart_agri.eval.baselines import federated_global_gru
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.features.selection import FeatureSelector
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.model.checkpoint import save_model
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

print = functools.partial(builtins.print, flush=True)

CHECKPOINT_SEED = 0  # see the module docstring's scope decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default="", help="optional .npz of prepared train/test arrays")
    parser.add_argument("--out", default="artifacts/federated_global_model.safetensors")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()

    if args.cache and Path(args.cache).exists():
        blob = np.load(args.cache, allow_pickle=False)
        x_tr, y_tr, src_tr, idx_tr = blob["Xtr"], blob["ytr"], blob["str_"], blob["itr"]
        x_te, y_te, src_te, idx_te = blob["Xte"], blob["yte"], blob["ste"], blob["ite"]
        per_class_counts: dict[str, int] = {}
        print(f"[data] loaded cached arrays: train {x_tr.shape}, test {x_te.shape}")
    else:
        frame, per_class_counts = stratified_capped_subsample(
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
        test_frame = deduped.loc[block_ids.isin(split.test_blocks)]

        selector = FeatureSelector()
        selection = selector.fit(
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

        def prepare(part, scaler=None):  # type: ignore[no-untyped-def]
            values = part[columns].to_numpy(dtype=np.float64)
            finite = np.isfinite(values).all(axis=1)
            kept = part.loc[finite]
            values = values[finite]
            scaler = scaler or fit_scaler(values)
            scaled = apply_scaler(values, scaler).astype(np.float32)
            return (
                scaled,
                to_class_index(kept["label"]).to_numpy(),
                kept["source_file"].to_numpy().astype(str),
                kept.index.to_numpy(),
                scaler,
            )

        x_tr, y_tr, src_tr, idx_tr, fitted = prepare(train_frame)
        x_te, y_te, src_te, idx_te, _ = prepare(test_frame, fitted)
        print(f"[data] train {x_tr.shape} | test {x_te.shape}")

    seg_tr = contiguity_segments(src_tr, idx_tr)
    seg_te = contiguity_segments(src_te, idx_te)

    # ---- IDENTICAL partition to run_phase4.py: same seed, same alpha -> same 3-way split ---
    block_ids_tr = make_blocks(
        pd.DataFrame({"label": y_tr.astype(str)}), block_size=cfg.data.block_size
    )
    block_labels = pd.Series(y_tr.astype(str)).groupby(block_ids_tr).first().to_numpy()
    client_blocks = dirichlet_block_partition(
        block_labels,
        n_clients=cfg.federated.n_clients,
        alpha=cfg.federated.dirichlet_alpha,
        seed=cfg.data.seed,
    )
    histograms = per_client_class_histograms(client_blocks, block_labels)
    sizes = [len(c) for c in client_blocks]
    print(f"[partition] alpha={cfg.federated.dirichlet_alpha}, sizes={sizes}")

    client_seqs, client_y = [], []
    for blocks in client_blocks:
        idx = np.flatnonzero(block_ids_tr.isin(blocks).to_numpy())
        seqs, labs = build_windows(x_tr[idx], y_tr[idx], seg_tr[idx], cfg.sequence.window)
        client_seqs.append(seqs)
        client_y.append(labs)
        print(f"  -> client sequences: {len(seqs):,}")

    seq_te, lab_te = build_windows(x_te, y_te, seg_te, cfg.sequence.window)
    print(f"[sequences] test: {len(seq_te):,}")

    # ---- Train ONE seed, for real, and keep the model this time -----------------------------
    print(
        f"[train] federated global model, seed={CHECKPOINT_SEED}, "
        f"R={cfg.federated.rounds}, E={cfg.federated.local_epochs}"
    )
    t0 = time.time()
    metrics, convergence, bytes_per_round, model = federated_global_gru(
        client_seqs,
        client_y,
        seq_te,
        lab_te,
        seed=CHECKPOINT_SEED,
        class_names=list(CLASS_NAMES),
        n_classes=cfg.model.n_classes,
        rounds=cfg.federated.rounds,
        local_epochs=cfg.federated.local_epochs,
        hidden_size=cfg.model.hidden_size,
        aggregation=cfg.federated.aggregation,
        verbose=True,
    )
    print(f"[train] done in {time.time() - t0:.0f}s, macro-F1 {metrics.macro_f1:.4f}")

    out_path = Path(args.out)
    save_model(
        model,
        out_path,
        metadata={
            "seed": str(CHECKPOINT_SEED),
            "alpha": str(cfg.federated.dirichlet_alpha),
            "rounds": str(cfg.federated.rounds),
            "local_epochs": str(cfg.federated.local_epochs),
            "aggregation": cfg.federated.aggregation,
            "n_features": str(cfg.model.n_features),
            "hidden_size": str(cfg.model.hidden_size),
            "n_classes": str(cfg.model.n_classes),
            "window": str(cfg.sequence.window),
            "macro_f1": f"{metrics.macro_f1:.4f}",
        },
    )
    print(f"[checkpoint] saved -> {out_path}")

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase7_train_{cfg.run_name}",
        seeds=[CHECKPOINT_SEED],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts=per_class_counts,
        ascon_backend={},
        results={
            "checkpoint_path": str(out_path),
            "checkpoint_seed": CHECKPOINT_SEED,
            "macro_f1": metrics.macro_f1,
            "balanced_accuracy": metrics.balanced_accuracy,
            "mcc": metrics.mcc,
            "accuracy": metrics.accuracy,
            "per_class_f1": metrics.per_class_f1,
            "convergence": convergence,
            "measured_bytes_per_round": bytes_per_round,
            "per_client_class_histograms": histograms,
            "phase4_reference_macro_f1_mean": 0.8334,
            "phase4_reference_macro_f1_std": 0.0026,
            "scope_note": (
                "ONE seed trained and saved for deployment, not a fresh statistical claim -- "
                "see this script's module docstring. Phase 4's 3-seed mean +/- std remains the "
                "reported performance finding."
            ),
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"[manifest] -> {path}")
    print(f"[total] {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

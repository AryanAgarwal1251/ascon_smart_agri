"""Phase 4 minimal-gate experiment (Section III-F, baselines 4-5).

Runs baseline 4 (three local-only GRUs) and baseline 5 (the federated global GRU) at the
paper's default configuration (alpha=0.5, R=20, E=3, weighted FedAvg), over >= 3 seeds, and
writes a manifest that ``scripts/check_phase4_gate.py`` verifies against the minimal Phase 4
gate written before this script.

Scope decision (flagged, see CHANGELOG.md): this runs ONE configuration, not the Section
III-I3 ablation sweep (alpha in {0.1, 0.5, 100}, E in {1, 3, 5}, weighted vs unweighted, R up
to 20). That sweep is deferred as separate, larger follow-on work -- chosen with the user after
measuring that a single configuration alone costs several hours on this machine.

Partitioning decision (flagged per Golden Rule 1): the Dirichlet partition is drawn ONCE, seeded
by ``cfg.data.seed``, not redrawn per experimental seed. The paper does not say whether the
partition itself should vary across the >= 3 reported seeds; redrawing it would conflate
partition variance with training variance in a single number, unlike Phase 3, where the
train/test split was held fixed across the 3 training seeds. Holding the partition fixed keeps
Phase 4's seeds measuring the same thing Phase 3's did: variance from model training alone.

Baseline 4's epoch budget matches Phase 3's centralised-GRU baseline (10 epochs), so a
difference between the two isolates "how much data you have" rather than "how long you
trained".

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase4.py [--cache path.npz]
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
from ascon_smart_agri.eval.baselines import federated_global_gru, local_only_grus
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.report import format_seed_summary, mean_std
from ascon_smart_agri.features.selection import FeatureSelector
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.federated.server import theoretical_bytes_per_round
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

print = functools.partial(builtins.print, flush=True)

HEADLINE = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")
LOCAL_ONLY_EPOCHS = 10  # matches Phase 3's centralised-GRU baseline budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default="", help="optional .npz of prepared train/test arrays")
    parser.add_argument(
        "--centralized-manifest",
        default="artifacts/manifest_phase3_complete_default.json",
        help="Phase 3 manifest to pull the centralised-GRU reference macro-F1 from (G4 bracket)",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()

    # ---- Phases 1-2 pipeline (same as Phase 3's runners) --------------------------------
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

    x_tr.shape[1]
    seg_tr = contiguity_segments(src_tr, idx_tr)
    seg_te = contiguity_segments(src_te, idx_te)

    # ---- Partition: ONE draw, fixed across all experimental seeds (see module docstring) --
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
    for i, h in enumerate(histograms):
        print(f"  client {i}: {h}")

    client_seqs, client_y = [], []
    for blocks in client_blocks:
        mask = block_ids_tr.isin(blocks).to_numpy()
        idx = np.flatnonzero(mask)
        seqs, labs = build_windows(x_tr[idx], y_tr[idx], seg_tr[idx], cfg.sequence.window)
        client_seqs.append(seqs)
        client_y.append(labs)
        print(f"  -> client sequences: {len(seqs):,}")

    seq_te, lab_te = build_windows(x_te, y_te, seg_te, cfg.sequence.window)
    print(f"[sequences] test: {len(seq_te):,}")

    theoretical = theoretical_bytes_per_round(cfg.federated.n_clients, 33_800)

    centralized_ref = None
    ref_path = Path(args.centralized_manifest)
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        centralized_ref = ref["results"]["summary"]["16"]["gru"]["macro_f1_mean"]
        print(f"[reference] centralised GRU (Phase 3, W=16): macro-F1 {centralized_ref:.4f}")

    # ---- Baseline 4: three local-only GRUs, every seed -----------------------------------
    local_only_per_seed: dict[int, list[dict[str, object]]] = {
        i: [] for i in range(len(client_seqs))
    }
    for seed in cfg.evaluation.seeds:
        t0 = time.time()
        results = local_only_grus(
            client_seqs,
            client_y,
            seq_te,
            lab_te,
            seed=seed,
            class_names=list(CLASS_NAMES),
            n_classes=cfg.model.n_classes,
            epochs=LOCAL_ONLY_EPOCHS,
            hidden_size=cfg.model.hidden_size,
        )
        for i, m in enumerate(results):
            local_only_per_seed[i].append(
                {
                    "seed": seed,
                    "macro_f1": m.macro_f1,
                    "macro_precision": m.macro_precision,
                    "macro_recall": m.macro_recall,
                    "balanced_accuracy": m.balanced_accuracy,
                    "mcc": m.mcc,
                    "accuracy": m.accuracy,
                    "per_class_f1": m.per_class_f1,
                    "confusion": m.confusion.tolist(),
                }
            )
        print(
            f"[local-only seed {seed}] "
            + " | ".join(f"client{i} F1={r.macro_f1:.4f}" for i, r in enumerate(results))
            + f" ({time.time() - t0:.0f}s)"
        )

    # ---- Baseline 5: federated global GRU, every seed ------------------------------------
    federated_per_seed: list[dict[str, object]] = []
    convergence_by_seed: dict[int, list[float]] = {}
    measured_bytes = 0
    for seed in cfg.evaluation.seeds:
        t0 = time.time()
        metrics, convergence, bytes_per_round = federated_global_gru(
            client_seqs,
            client_y,
            seq_te,
            lab_te,
            seed=seed,
            class_names=list(CLASS_NAMES),
            n_classes=cfg.model.n_classes,
            rounds=cfg.federated.rounds,
            local_epochs=cfg.federated.local_epochs,
            hidden_size=cfg.model.hidden_size,
            aggregation=cfg.federated.aggregation,
            # Without this, nothing prints for an entire seed's R rounds (~70+ min at the
            # default config) -- found the hard way on the run this script was written for.
            verbose=True,
        )
        measured_bytes = bytes_per_round
        convergence_by_seed[seed] = convergence
        federated_per_seed.append(
            {
                "seed": seed,
                "macro_f1": metrics.macro_f1,
                "macro_precision": metrics.macro_precision,
                "macro_recall": metrics.macro_recall,
                "balanced_accuracy": metrics.balanced_accuracy,
                "mcc": metrics.mcc,
                "accuracy": metrics.accuracy,
                "per_class_f1": metrics.per_class_f1,
                "confusion": metrics.confusion.tolist(),
            }
        )
        print(
            f"[federated seed {seed}] final macro-F1 {metrics.macro_f1:.4f} "
            f"(round curve: {[round(c, 3) for c in convergence]}) ({time.time() - t0:.0f}s)"
        )

    # ---- Summaries (mean +/- std, never a single run) -------------------------------------
    local_only_summary: dict[str, dict[str, float]] = {}
    for client_id, runs in local_only_per_seed.items():
        local_only_summary[str(client_id)] = {}
        for metric in HEADLINE:
            values = [float(r[metric]) for r in runs]  # type: ignore[arg-type]
            mean, std = mean_std(values)
            local_only_summary[str(client_id)][f"{metric}_mean"] = mean
            local_only_summary[str(client_id)][f"{metric}_std"] = std

    federated_summary: dict[str, float] = {}
    for metric in HEADLINE:
        values = [float(r[metric]) for r in federated_per_seed]  # type: ignore[arg-type]
        mean, std = mean_std(values)
        federated_summary[f"{metric}_mean"] = mean
        federated_summary[f"{metric}_std"] = std

    print("\n" + "=" * 78)
    print(
        f"PHASE 4 RESULTS -- minimal gate, alpha={cfg.federated.dirichlet_alpha}, "
        f"R={cfg.federated.rounds}, E={cfg.federated.local_epochs}, "
        f"{cfg.federated.aggregation}, {len(cfg.evaluation.seeds)} seeds"
    )
    print("=" * 78)
    print("\nBaseline 3 (Phase 3, centralised, upper bound):")
    print(
        f"  macro-F1 {centralized_ref:.4f}" if centralized_ref is not None else "  (not available)"
    )
    print("\nBaseline 4 (local-only, lower bound):")
    for client_id, _summary in local_only_summary.items():
        print(
            f"  client {client_id}: "
            + format_seed_summary(
                "macro_f1",
                [
                    r["macro_f1"]
                    for r in local_only_per_seed[int(client_id)]  # type: ignore[arg-type]
                ],
            )
        )
    print("\nBaseline 5 (federated global, method under test):")
    print("  " + format_seed_summary("macro_f1", [r["macro_f1"] for r in federated_per_seed]))  # type: ignore[arg-type]

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase4_{cfg.run_name}",
        seeds=list(cfg.evaluation.seeds),
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts=per_class_counts,
        ascon_backend={},
        results={
            "local_only_summary": local_only_summary,
            "local_only_per_seed": {str(k): v for k, v in local_only_per_seed.items()},
            "federated_summary": federated_summary,
            "federated_per_seed": federated_per_seed,
            "federated_convergence_by_seed": {str(k): v for k, v in convergence_by_seed.items()},
            "centralized_reference_macro_f1": centralized_ref,
            "per_client_class_histograms": histograms,
            "measured_bytes_per_round": measured_bytes,
            "theoretical_bytes_per_round": theoretical,
            "alpha": cfg.federated.dirichlet_alpha,
            "rounds": cfg.federated.rounds,
            "local_epochs": cfg.federated.local_epochs,
            "aggregation": cfg.federated.aggregation,
            "local_only_epochs": LOCAL_ONLY_EPOCHS,
            "scope_note": (
                "Minimal gate only: one configuration (paper default), not the Section III-I3 "
                "ablation sweep. See CHANGELOG.md for the scope decision."
            ),
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"\n  manifest -> {path}")
    print(f"  total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

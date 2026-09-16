"""Phase 3 completion run: the Section III-I1 baselines and the window ablation.

Extends ``run_phase3.py`` from "the GRU's metrics" to the full Phase-3 deliverable:

    * baseline 1  random forest on single records      (S13's dissent; can contradict the GRU)
    * baseline 2  MLP on single records, parameter-matched (isolates recurrence)
    * baseline 3  centralised GRU                       (the method under test at this phase)
    * ablation    W = 1 vs W = 16 -- Section III-D: "if it matches W = 16, recurrence has not
                  earned its place in the architecture, and we will say so"

Baselines 4-5 (local-only GRUs, federated global GRU) are Phase 4 and are not run here.

Every figure is mean +/- std over >= 3 seeds (III-I4) and lands in one manifest, which
``scripts/check_phase3_gate.py`` then checks against the exit criterion.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase3_complete.py --epochs 10
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path

import numpy as np
from configs.base import load_run_config

from ascon_smart_agri.data.dedup import deduplicate
from ascon_smart_agri.data.scaling import apply_scaler, fit_scaler
from ascon_smart_agri.data.split import make_blocks, stratified_block_split
from ascon_smart_agri.data.subsample import stratified_capped_subsample
from ascon_smart_agri.data.taxonomy import CLASS_NAMES, to_class_index
from ascon_smart_agri.eval.baselines import (
    centralized_gru,
    mlp_single_record,
    random_forest_single_record,
)
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.report import mean_std
from ascon_smart_agri.features.selection import FeatureSelector
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

print = functools.partial(builtins.print, flush=True)

BASELINES = ("random_forest", "mlp", "gru")


def _log(window: int, seed: int, name: str, macro_f1: float, seconds: float) -> None:
    print(f"  [W={window} seed {seed}] {name:<4} macro-F1 {macro_f1:.4f} ({seconds:.0f}s)")


HEADLINE = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--rf-max-rows", type=int, default=300_000)
    parser.add_argument("--cache", default="", help="optional .npz of prepared arrays")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()

    # ---- Phases 1-2 pipeline (W-independent, so built once) ----------------------------
    if args.cache and Path(args.cache).exists():
        blob = np.load(args.cache, allow_pickle=False)
        x_tr, y_tr, src_tr, idx_tr = blob["Xtr"], blob["ytr"], blob["str_"], blob["itr"]
        x_te, y_te, src_te, idx_te = blob["Xte"], blob["yte"], blob["ste"], blob["ite"]
        columns = [str(c) for c in blob["cols"]]
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
            scaler = scaler or fit_scaler(values)  # TRAINING ONLY
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
        print(f"[data] train {x_tr.shape} | test {x_te.shape} | F={len(columns)}")

    seg_tr = contiguity_segments(src_tr, idx_tr)
    seg_te = contiguity_segments(src_te, idx_te)

    # ---- Per window: every baseline, every seed ----------------------------------------
    results: dict[str, dict[str, list[dict[str, object]]]] = {}
    for window in args.windows:
        seq_tr, lab_tr = build_windows(x_tr, y_tr, seg_tr, window)
        seq_te, lab_te = build_windows(x_te, y_te, seg_te, window)
        print(f"\n=== W={window}: train {len(seq_tr):,} | test {len(seq_te):,} sequences ===")
        results[str(window)] = {name: [] for name in BASELINES}

        for seed in cfg.evaluation.seeds:
            common = {
                "class_names": list(CLASS_NAMES),
                "seed": seed,
            }
            t0 = time.time()
            rf = random_forest_single_record(
                seq_tr,
                lab_tr,
                seq_te,
                lab_te,
                n_estimators=cfg.features.rf_n_estimators,
                max_train_rows=args.rf_max_rows,
                **common,  # type: ignore[arg-type]
            )
            _log(window, seed, "RF", rf.macro_f1, time.time() - t0)

            t0 = time.time()
            mlp = mlp_single_record(
                seq_tr,
                lab_tr,
                seq_te,
                lab_te,
                n_classes=cfg.model.n_classes,
                epochs=args.epochs,
                gru_hidden_size=cfg.model.hidden_size,
                **common,  # type: ignore[arg-type]
            )
            _log(window, seed, "MLP", mlp.macro_f1, time.time() - t0)

            t0 = time.time()
            gru = centralized_gru(
                seq_tr,
                lab_tr,
                seq_te,
                lab_te,
                n_classes=cfg.model.n_classes,
                epochs=args.epochs,
                hidden_size=cfg.model.hidden_size,
                **common,  # type: ignore[arg-type]
            )
            _log(window, seed, "GRU", gru.macro_f1, time.time() - t0)

            for name, m in (("random_forest", rf), ("mlp", mlp), ("gru", gru)):
                results[str(window)][name].append(
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

    # ---- Report ------------------------------------------------------------------------
    summary: dict[str, dict[str, dict[str, float]]] = {}
    print("\n" + "=" * 78)
    print(
        f"PHASE 3 COMPLETION -- baselines 1-3 of III-I1, W in {args.windows}, "
        f"{len(cfg.evaluation.seeds)} seeds, {args.epochs} epochs"
    )
    print("=" * 78)
    for window, by_baseline in results.items():
        print(f"\nW = {window}")
        summary[window] = {}
        for name, runs in by_baseline.items():
            summary[window][name] = {}
            line = []
            for metric in HEADLINE:
                values = [float(r[metric]) for r in runs]  # type: ignore[arg-type]
                mean, std = mean_std(values)
                summary[window][name][f"{metric}_mean"] = mean
                summary[window][name][f"{metric}_std"] = std
                line.append(f"{metric} {mean:.4f}+/-{std:.4f}")
            print(f"  {name:<14} " + " | ".join(line))

    # The Section III-D question this ablation exists to answer.
    verdict = None
    if "1" in summary and "16" in summary:
        w1, w16 = summary["1"]["gru"]["macro_f1_mean"], summary["16"]["gru"]["macro_f1_mean"]
        spread = max(summary["16"]["gru"]["macro_f1_std"], summary["1"]["gru"]["macro_f1_std"])
        gain = w16 - w1
        earned = gain > 2 * spread  # a gain worth more than the seed noise it sits in
        verdict = {
            "w1_macro_f1": w1,
            "w16_macro_f1": w16,
            "gain": gain,
            "seed_std": spread,
            "recurrence_earned_its_place": bool(earned),
        }
        print(
            f"\n  ABLATION (Section III-D): W=1 macro-F1 {w1:.4f} vs W=16 {w16:.4f} "
            f"-> gain {gain:+.4f}, seed std {spread:.4f}"
        )
        print(
            f"  => recurrence {'EARNED' if earned else 'DID NOT EARN'} its place "
            f"(gain {'>' if earned else '<='} 2x seed std)"
        )

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase3_complete_{cfg.run_name}",
        seeds=list(cfg.evaluation.seeds),
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts=per_class_counts,
        ascon_backend={},
        results={
            "summary": summary,
            "per_seed": results,
            "baselines_run": list(BASELINES),
            "baselines_deferred_to_phase4": ["local_only_grus", "federated_global_gru"],
            "windows": args.windows,
            "epochs": args.epochs,
            "selected_columns": columns,
            "ablation_verdict": verdict,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"\n  manifest -> {path}")
    print(f"  total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Phase 3 centralised-GRU experiment: the run that meets the Phase 3 exit criterion.

Chains the whole pipeline and reports the Section III-I2 protocol over >= 3 seeds
(III-I4), writing a manifest. Not a library module -- this is the experiment driver.

Run from the repo root with the root on the path, so the top-level ``configs`` package resolves
the same way ``pyproject.toml``'s ``pythonpath = [".", "src"]`` makes it resolve under pytest:

    PYTHONPATH=. ./.venv/bin/python scripts/run_phase3.py [--epochs N] [--max-train-sequences N]
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
from ascon_smart_agri.data.taxonomy import BENIGN_CLASS_INDEX, CLASS_NAMES, to_class_index
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.metrics import binary_metrics, multiclass_metrics
from ascon_smart_agri.eval.report import format_seed_summary, mean_std, near_ceiling_note
from ascon_smart_agri.features.selection import FeatureSelector
from ascon_smart_agri.model.train import predict, train_centralized
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments


def main() -> None:
    # Progress must stay visible when stdout is redirected to a log file: Python
    # block-buffers a non-TTY stdout, which hides a long run's progress entirely.
    print = functools.partial(builtins.print, flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-train-sequences",
        type=int,
        default=0,
        help="cap training sequences (0 = use all); the cap is a seeded, stratified draw",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()

    # ---- Phases 1-2: the data pipeline, exactly as the gates left it --------------------
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
    print(f"[data] train {len(train_frame):,} rows | test {len(test_frame):,} rows")

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
    print(f"[features] F={len(columns)} of F0={len(selection.fused_ranking)}")

    def prepare(frame_part, scaler=None):  # type: ignore[no-untyped-def]
        """Frame -> (sequences, labels, scaler). Non-finite rows are DROPPED, never imputed."""
        values = frame_part[columns].to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        kept = frame_part.loc[finite]
        values = values[finite]
        if scaler is None:
            scaler = fit_scaler(values)  # TRAINING ONLY
        scaled = apply_scaler(values, scaler).astype(np.float32)
        y = to_class_index(kept["label"]).to_numpy()
        # Dropping a non-finite row leaves a positional gap, which breaks the run here -- so no
        # window spans the hole, exactly as it must not span a held-out block.
        segments = contiguity_segments(kept["source_file"].to_numpy(), kept.index.to_numpy())
        sequences, seq_labels = build_windows(scaled, y, segments, cfg.sequence.window)
        return sequences, seq_labels, scaler, int((~finite).sum())

    x_train, y_train, scaler, dropped_train = prepare(train_frame)
    x_test, y_test, _, dropped_test = prepare(test_frame, scaler)
    print(
        f"[sequences] W={cfg.sequence.window} | train {len(x_train):,} | test {len(x_test):,} "
        f"| non-finite rows dropped: train {dropped_train}, test {dropped_test}"
    )

    if args.max_train_sequences and len(x_train) > args.max_train_sequences:
        rng = np.random.default_rng(cfg.data.seed)
        take = np.sort(rng.choice(len(x_train), size=args.max_train_sequences, replace=False))
        x_train, y_train = x_train[take], y_train[take]
        print(f"[sequences] training capped to {len(x_train):,} sequences")

    # ---- Phase 3: train and evaluate over >= 3 seeds (III-I4) ---------------------------
    per_seed: list[dict[str, object]] = []
    for seed in cfg.evaluation.seeds:
        t0 = time.time()
        model = train_centralized(
            x_train,
            y_train,
            hidden_size=cfg.model.hidden_size,
            n_classes=cfg.model.n_classes,
            epochs=args.epochs,
            seed=seed,
            verbose=True,
        )
        y_pred, probabilities = predict(model, x_test)

        multiclass = multiclass_metrics(y_test, y_pred, list(CLASS_NAMES))
        # Eq. (5): the binary decision is a deterministic projection of the 8-class output.
        binary = binary_metrics(
            (y_test > BENIGN_CLASS_INDEX).astype(int),
            1.0 - probabilities[:, BENIGN_CLASS_INDEX],
        )
        per_seed.append(
            {
                "seed": seed,
                "macro_f1": multiclass.macro_f1,
                "macro_precision": multiclass.macro_precision,
                "macro_recall": multiclass.macro_recall,
                "balanced_accuracy": multiclass.balanced_accuracy,
                "mcc": multiclass.mcc,
                "accuracy": multiclass.accuracy,
                "per_class_f1": multiclass.per_class_f1,
                "confusion": multiclass.confusion.tolist(),
                "binary_precision": binary.precision,
                "binary_recall": binary.recall,
                "binary_pr_auc": binary.pr_auc,
                "binary_fpr": binary.false_positive_rate,
            }
        )
        print(
            f"[seed {seed}] macro-F1 {multiclass.macro_f1:.4f} | bal-acc "
            f"{multiclass.balanced_accuracy:.4f} | MCC {multiclass.mcc:.4f} | FPR "
            f"{binary.false_positive_rate:.4f} | {time.time() - t0:.0f}s"
        )

    # ---- Report (mean +/- std over seeds, never a single run) ---------------------------
    print("\n" + "=" * 78)
    print(
        f"PHASE 3 RESULTS -- centralised GRU, {len(cfg.evaluation.seeds)} seeds, W="
        f"{cfg.sequence.window}, F={len(columns)}"
    )
    print("=" * 78)
    headline = [
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "mcc",
        "binary_pr_auc",
        "binary_fpr",
        "accuracy",
    ]
    summary: dict[str, object] = {}
    for name in headline:
        values = [float(r[name]) for r in per_seed]  # type: ignore[arg-type]
        mean, std = mean_std(values)
        summary[f"{name}_mean"], summary[f"{name}_std"] = mean, std
        print("  " + format_seed_summary(name, values))

    print("\n  per-class F1 (mean +/- std over seeds):")
    per_class_summary: dict[str, list[float]] = {}
    for i, name in enumerate(CLASS_NAMES):
        values = [float(r["per_class_f1"][name]) for r in per_seed]  # type: ignore[index]
        mean, std = mean_std(values)
        per_class_summary[name] = [mean, std]
        support = int(np.sum(np.asarray(y_test) == i))
        print(f"    {name:<16} {mean:.4f} +/- {std:.4f}   (test support {support:,})")
    summary["per_class_f1"] = per_class_summary

    note = near_ceiling_note(
        float(summary["accuracy_mean"]),  # type: ignore[arg-type]
        threshold=cfg.evaluation.near_ceiling_accuracy,
    )
    if note:
        print(f"\n  {note}")

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase3_{cfg.run_name}",
        seeds=list(cfg.evaluation.seeds),
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts=per_class_counts,
        ascon_backend={},  # crypto path not exercised by this run
        results={
            "summary": summary,
            "per_seed": per_seed,
            "selected_columns": columns,
            "n_train_sequences": len(x_train),
            "n_test_sequences": len(x_test),
            "epochs": args.epochs,
            "near_ceiling_note": note,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"\n  manifest -> {path}")


if __name__ == "__main__":
    main()

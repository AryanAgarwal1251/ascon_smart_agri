"""Empirical client-count (K) sweep for the federated GRU (Phase 4 extension).

NOT PART OF THE PAPER'S ABLATION SET, flagged per Golden Rule 1 -- see the module docstring of
``scripts/analyze_k_threshold.py`` for why (K is fixed at 3 by assumption A1 / objective O3).
That module predicts where macro-F1 should fall as K grows; this one measures it.

Only **baseline 5 (the federated global GRU)** is re-run per K. The other two are deliberately
skipped:

* **Baseline 3 (centralised)** is K-independent by construction -- it pools the whole training
  split and never sees the partition -- so re-training it once per K would burn hours to
  reproduce a constant. The Phase 4 figure (macro-F1 0.8543 +/- 0.0040) is the upper bound for
  every point on this sweep and is quoted from ``manifest_phase4_default.json``.
* **Baseline 4 (local-only)** would mean K models per seed; at K = 50 that is 150 GRUs for a
  lower bound that is not what the sweep is asking about.

Everything else matches Phase 4 exactly so the K = 3 point reproduces it: same config, same
seeds, same R and E, same compute budget per client, same shared global test set (III-B3), and
the same federated scaler built from sufficient statistics only (Eqs. 23-24).

Recorded per (K, seed) alongside the usual metrics:

* **omega_c from the true Eq. (21) weights.** ``analyze_k_threshold.py`` approximates omega with
  block counts; here it is computed from the sequence counts FedAvg actually weights by, so the
  prediction can be checked against the quantity it was predicting.
* **The convergence curve**, so "does federation converge more slowly at high K?" is answerable
  rather than assumed.

Phases 1-2 are expensive and run once for the whole sweep; ``--save-cache`` writes the prepared
arrays and ``--cache`` reads them back, sharing the format ``run_phase4.py`` uses.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_k_sweep.py \
        --k-values 3,5,10,20,30,50 --save-cache artifacts/phase4_cache.npz
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
from scripts.run_phase4 import binary_fpr, build_pipeline, federated_scaler

from ascon_smart_agri.data.scaling import fit_scaler
from ascon_smart_agri.data.taxonomy import CLASS_NAMES
from ascon_smart_agri.eval.baselines import run_federation
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.report import mean_std
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.model.gru import expected_param_count
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

print = functools.partial(builtins.print, flush=True)

HEADLINE = ("macro_f1", "balanced_accuracy", "mcc", "accuracy", "false_positive_rate")


def omega_from_weights(client_labels: list[np.ndarray], n_classes: int) -> dict[str, float]:
    """Share of the Eq. (21) aggregate held by clients carrying each class.

    ``omega_c`` is the fraction of the weighted-FedAvg sum contributed by clients with at least
    one training SEQUENCE of class c. A client holding none of it still contributes weight to
    the average, and its local model has been trained to push that class's logits down, so
    ``1 - omega_c`` is the share of the global model that has never seen the class.
    """
    counts = np.array([len(labels) for labels in client_labels], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return dict.fromkeys(CLASS_NAMES[:n_classes], 0.0)
    weights = counts / total

    omega = np.zeros(n_classes)
    for weight, labels in zip(weights, client_labels, strict=True):
        if len(labels) == 0:
            continue
        present = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes) > 0
        omega += weight * present
    return {CLASS_NAMES[i]: float(omega[i]) for i in range(n_classes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--k-values", default="3,5,10,20,30,50")
    parser.add_argument("--rounds", type=int, default=0, help="R; default: config value")
    parser.add_argument("--local-epochs", type=int, default=0, help="E; default: config value")
    parser.add_argument("--alpha", type=float, default=0.0, help="default: config value")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", default="")
    parser.add_argument("--save-cache", default="")
    parser.add_argument("--out", default="artifacts/k_sweep_results.json")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    rounds = args.rounds or cfg.federated.rounds
    local_epochs = args.local_epochs or cfg.federated.local_epochs
    alpha = args.alpha or cfg.federated.dirichlet_alpha
    aggregation = cfg.federated.aggregation
    k_values = [int(k) for k in args.k_values.split(",")]
    started = time.time()

    (x_tr, y_tr, src_tr, idx_tr, blk_tr, x_te, y_te, src_te, idx_te, columns, per_class_counts) = (
        build_pipeline(cfg, args.cache, args.save_cache)
    )

    seg_te = contiguity_segments(src_te, idx_te)
    pooled = fit_scaler(x_tr)  # reference only; the III-F4 claim is checked against it per run

    train_blocks = np.unique(blk_tr)
    first_row_of_block = (
        np.searchsorted(blk_tr, train_blocks)
        if np.all(np.diff(blk_tr) >= 0)
        else np.array([np.flatnonzero(blk_tr == b)[0] for b in train_blocks])
    )
    block_labels = y_tr[first_row_of_block]

    print(
        f"\n[setup] K sweep {k_values} | alpha={alpha} | R={rounds} | E={local_epochs}"
        f" | aggregation={aggregation} | W={cfg.sequence.window}"
    )
    print(
        f"[setup] {len(train_blocks):,} train blocks | {len(x_te):,} test rows"
        f" | seeds {list(cfg.evaluation.seeds)}"
    )

    names = list(CLASS_NAMES)
    benign = names.index("Benign")
    records: list[dict[str, object]] = []

    for K in k_values:
        for seed in cfg.evaluation.seeds:
            t0 = time.time()
            assignment = dirichlet_block_partition(
                block_labels, n_clients=K, alpha=alpha, seed=seed
            )
            histograms = per_client_class_histograms(assignment, block_labels)
            client_blocks = [train_blocks[np.asarray(ids, dtype=int)] for ids in assignment]
            row_masks = [np.isin(blk_tr, blocks) for blocks in client_blocks]

            mean, std = federated_scaler([x_tr[mask] for mask in row_masks])
            scaler_gap = float(
                max(
                    np.abs(mean - pooled.mean).max() / max(np.abs(pooled.mean).max(), 1e-12),
                    np.abs(std - pooled.std).max() / max(np.abs(pooled.std).max(), 1e-12),
                )
            )
            scaled_tr = ((x_tr - mean) / std).astype(np.float32)
            seq_te, lab_te = build_windows(
                ((x_te - mean) / std).astype(np.float32), y_te, seg_te, cfg.sequence.window
            )

            client_seqs: list[np.ndarray] = []
            client_labels: list[np.ndarray] = []
            for mask in row_masks:
                if not mask.any():
                    client_seqs.append(
                        np.empty((0, cfg.sequence.window, x_tr.shape[1]), np.float32)
                    )
                    client_labels.append(np.empty((0,), np.int64))
                    continue
                seg = contiguity_segments(src_tr[mask], idx_tr[mask])
                seqs, labels = build_windows(scaled_tr[mask], y_tr[mask], seg, cfg.sequence.window)
                client_seqs.append(seqs)
                client_labels.append(labels)

            empty = sum(1 for seqs in client_seqs if len(seqs) == 0)
            run = run_federation(
                client_seqs,
                client_labels,
                seq_te,
                lab_te,
                seed=seed,
                class_names=names,
                n_classes=cfg.model.n_classes,
                rounds=rounds,
                local_epochs=local_epochs,
                hidden_size=cfg.model.hidden_size,
                aggregation=aggregation,
                device=args.device,
                evaluate_each_round=True,
            )
            metrics = run.metrics
            omega = omega_from_weights(client_labels, cfg.model.n_classes)

            records.append(
                {
                    "K": K,
                    "seed": seed,
                    "macro_f1": metrics.macro_f1,
                    "balanced_accuracy": metrics.balanced_accuracy,
                    "mcc": metrics.mcc,
                    "accuracy": metrics.accuracy,
                    "false_positive_rate": binary_fpr(metrics.confusion, benign),
                    "per_class_f1": metrics.per_class_f1,
                    "omega": omega,
                    "omega_macro": float(np.mean(list(omega.values()))),
                    "sequence_counts": run.sequence_counts,
                    "clients_without_sequences": empty,
                    "per_round_macro_f1": run.per_round_macro_f1,
                    "bytes_per_round": run.bytes_per_round[0],
                    "block_histograms": histograms,
                    "federated_vs_pooled_scaler_gap": scaler_gap,
                    "elapsed_seconds": round(time.time() - t0, 1),
                }
            )
            print(
                f"  K={K:<3} seed={seed}  macro-F1 {metrics.macro_f1:.4f}"
                f" | w_macro {records[-1]['omega_macro']:.3f}"
                f" | w({min(omega, key=lambda c: omega[c])}) {min(omega.values()):.3f}"
                f" | {empty} empty clients | {time.time() - t0:.0f}s"
            )

            # A ~20 h sweep must not lose everything to one interruption: rewrite the partial
            # record set after every (K, seed) point, so a kill leaves usable results.
            progress = Path(args.out).with_name(Path(args.out).stem + "_progress.json")
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text(
                json.dumps({"k_values": k_values, "per_run": records}, indent=2, default=str)
                + "\n",
                encoding="utf-8",
            )

            del client_seqs, client_labels, scaled_tr, seq_te, lab_te

    print("\n" + "=" * 78)
    print(
        f"K SWEEP -- baseline 5 only, alpha={alpha}, R={rounds}, E={local_epochs}, "
        f"{len(cfg.evaluation.seeds)} seeds"
    )
    print("=" * 78)

    summary: dict[str, dict[str, float]] = {}
    print(f"{'K':>5} " + " ".join(f"{m:>22}" for m in HEADLINE) + f" {'omega_macro':>12}")
    for K in k_values:
        rows = [r for r in records if r["K"] == K]
        entry: dict[str, float] = {}
        cells = []
        for metric in HEADLINE:
            mean_value, std_value = mean_std([float(r[metric]) for r in rows])  # type: ignore[arg-type]
            entry[f"{metric}_mean"] = mean_value
            entry[f"{metric}_std"] = std_value
            cells.append(f"{mean_value:>13.4f}+/-{std_value:.4f}")
        omega_mean, _ = mean_std([float(r["omega_macro"]) for r in rows])
        entry["omega_macro_mean"] = omega_mean
        entry["clients_without_sequences_mean"] = float(
            np.mean([float(r["clients_without_sequences"]) for r in rows])
        )
        summary[str(K)] = entry
        print(f"{K:>5} " + " ".join(cells) + f" {omega_mean:>12.3f}")

    baseline = summary[str(k_values[0])]["macro_f1_mean"]
    print(f"\n  macro-F1 relative to K={k_values[0]} ({baseline:.4f}):")
    for K in k_values:
        delta = summary[str(K)]["macro_f1_mean"] - baseline
        print(f"    K={K:<4} {summary[str(K)]['macro_f1_mean']:.4f}  ({delta:+.4f})")

    versions, hardware = collect_environment()
    results = {
        "note": (
            "K = 3 is the HARDWARE configuration (two Pis plus one laptop-simulated "
            "client), chosen by budget and because K = 2 federation is degenerate -- "
            "not by measurement. This sweep is what tests the scalability claim, and "
            "the user asked for it on 2026-09-30 as a research-paper result."
        ),
        "baseline_5_only": True,
        "centralized_upper_bound_from_phase4": {"macro_f1_mean": 0.8543, "macro_f1_std": 0.0040},
        "k_values": k_values,
        "summary": summary,
        "per_run": records,
        "dirichlet_alpha": alpha,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "aggregation": aggregation,
        "window": cfg.sequence.window,
        "selected_columns": columns,
        "expected_param_count": expected_param_count(
            len(columns), cfg.model.hidden_size, cfg.model.n_classes
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    manifest = RunManifest(
        run_name=f"ksweep_{cfg.run_name}",
        seeds=list(cfg.evaluation.seeds),
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts=per_class_counts,
        ascon_backend={},
        results=results,
    )
    path = manifest.write(Path(cfg.output_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n  manifest -> {path}")
    print(f"  summary  -> {out}")
    print(f"  total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

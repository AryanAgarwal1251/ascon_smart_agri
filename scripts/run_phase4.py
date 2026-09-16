"""Phase 4 run: the three-client federated simulation (Section III-F).

Completes the Section III-I1 baseline set by adding the two the Phase 3 run deferred, and
reports them against the upper bound Phase 3 established:

    * baseline 3  centralised GRU        (upper bound attainable by pooling)      -- re-run here
    * baseline 4  three local-only GRUs  (lower bound attainable without federating)
    * baseline 5  federated global GRU   (the method under test)

Baselines 3 and 4 bracket baseline 5; the bracket is the answer to gap G4, and is the thing
this run exists to produce. Every figure is mean +/- std over >= 3 seeds (III-I4).

What the run demonstrates beyond the numbers:

    * **Per-client class histograms are published** (Section III-F1 / gap G3), including zero
      counts, so the Dirichlet partition can be inspected rather than trusted.
    * **The global scaler is built federatedly** (Eqs. 23-24): clients emit count/mean/M2 only,
      the server combines them, and the run checks the result against a pooled fit -- the
      Section III-F4 claim, verified on the real corpus rather than only in a unit test.
    * **Communication cost is measured, not assumed** (Eq. 22): the serialised bytes actually
      sent each round are recorded alongside the theoretical figure.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase4.py --rounds 20 --local-epochs 3

The Phases 1-2 pipeline is expensive, so ``--save-cache`` writes the prepared arrays (including
the per-row block ids this phase needs, which the Phase 3 cache does not carry) and ``--cache``
reads them back.
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
from ascon_smart_agri.data.scaling import fit_scaler
from ascon_smart_agri.data.split import make_blocks, stratified_block_split
from ascon_smart_agri.data.subsample import stratified_capped_subsample
from ascon_smart_agri.data.taxonomy import CLASS_NAMES, to_class_index
from ascon_smart_agri.eval.baselines import centralized_gru, local_only_grus, run_federation
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.report import client_to_global_gap, mean_std
from ascon_smart_agri.features.selection import FeatureSelector
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.federated.scaler_stats import combine_stats, local_sufficient_stats
from ascon_smart_agri.federated.server import theoretical_bytes_per_round
from ascon_smart_agri.model.gru import expected_param_count
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

print = functools.partial(builtins.print, flush=True)

# FPR is in the headline set deliberately (III-I2), not reported as an afterthought.
HEADLINE = ("macro_f1", "balanced_accuracy", "mcc", "accuracy", "false_positive_rate")
# local_only's per-seed dict only carries the mean-over-clients fields computed for it below;
# accuracy and false_positive_rate are not (yet) among them.
LOCAL_ONLY_HEADLINE = ("macro_f1", "balanced_accuracy", "mcc")


def build_pipeline(cfg, cache: str, save_cache: str):  # type: ignore[no-untyped-def]
    """Phases 1-2: subsample -> dedup -> block split -> feature selection.

    Returns UNSCALED train/test features plus the per-row block ids the Dirichlet partition
    needs. Standardisation is deliberately not done here -- it is the federated step of
    Section III-F4 and happens after the partition, from client sufficient statistics only.
    """
    if cache and Path(cache).exists():
        blob = np.load(cache, allow_pickle=False)
        print(f"[data] loaded cache {cache}")
        return (
            blob["Xtr"],
            blob["ytr"],
            blob["str_"],
            blob["itr"],
            blob["btr"],
            blob["Xte"],
            blob["yte"],
            blob["ste"],
            blob["ite"],
            [str(c) for c in blob["cols"]],
            {},
        )

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

    def prepare(part):  # type: ignore[no-untyped-def]
        """Select columns and drop non-finite rows. Deliberately does NOT scale.

        Standardisation is the federated step (Section III-F4): it happens after the partition,
        from client sufficient statistics only. Scaling here with a pooled fit would quietly
        pool the very data the phase exists to keep apart.
        """
        values = part[columns].to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        kept = part.loc[finite]
        return (
            values[finite],
            to_class_index(kept["label"]).to_numpy(),
            kept["source_file"].to_numpy().astype(str),
            kept.index.to_numpy(),
            block_ids.loc[kept.index].to_numpy(),
        )

    x_tr, y_tr, src_tr, idx_tr, blk_tr = prepare(train_frame)
    x_te, y_te, src_te, idx_te, _ = prepare(test_frame)
    print(f"[data] train {x_tr.shape} | test {x_te.shape} | F={len(columns)}")

    if save_cache:
        Path(save_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_cache,
            Xtr=x_tr,
            ytr=y_tr,
            str_=src_tr,
            itr=idx_tr,
            btr=blk_tr,
            Xte=x_te,
            yte=y_te,
            ste=src_te,
            ite=idx_te,
            cols=np.array(columns, dtype=object).astype(str),
        )
        print(f"[data] cache -> {save_cache}")

    return (
        x_tr,
        y_tr,
        src_tr,
        idx_tr,
        blk_tr,
        x_te,
        y_te,
        src_te,
        idx_te,
        columns,
        per_class_counts,
    )


def binary_fpr(confusion: np.ndarray, benign_index: int = 0) -> float:
    """False-positive rate FP/(FP+TN) under the Eq. (5) binary projection, Eq. (31).

    Section III-I2 makes FPR a first-class metric rather than a footnote, because in production
    a high false-alarm rate is what gets a detector switched off. It is a pure function of the
    confusion matrix (benign row: everything off the diagonal is a false alarm), so it needs no
    extra inference pass and stays consistent with the matrix reported beside it.
    """
    matrix = np.asarray(confusion, dtype=np.float64)
    true_negative = matrix[benign_index, benign_index]
    false_positive = matrix[benign_index].sum() - true_negative
    denominator = false_positive + true_negative
    return float(false_positive / denominator) if denominator > 0 else float("nan")


def federated_scaler(client_rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Global (mean, std) from client sufficient statistics only (Eqs. 23-24).

    No client's rows cross the boundary -- only count/mean/M2 do.
    """
    parts = [local_sufficient_stats(rows) for rows in client_rows if len(rows) > 0]
    return combine_stats(parts)


def block_strata(blk_tr: np.ndarray, y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_blocks, block_labels)``: each train block id and its single label.

    Shared with ``train_federated_model.py`` so Phase 7's checkpoint is partitioned from the
    SAME block ids as Phase 4's run, not from blocks re-derived over the train rows (which
    re-cuts them and changes the Dirichlet draw).
    """
    train_blocks = np.unique(blk_tr)
    first_row_of_block = (
        np.searchsorted(blk_tr, train_blocks)
        if np.all(np.diff(blk_tr) >= 0)
        else np.array([np.flatnonzero(blk_tr == b)[0] for b in train_blocks])
    )
    return train_blocks, y_tr[first_row_of_block]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rounds", type=int, default=0, help="R; default: config value")
    parser.add_argument("--local-epochs", type=int, default=0, help="E; default: config value")
    parser.add_argument("--alpha", type=float, default=0.0, help="default: config value")
    parser.add_argument(
        "--central-epochs",
        type=int,
        default=0,
        help="baseline 3 epochs; default 0 = R*E, matching the federated local-pass budget",
    )
    parser.add_argument("--aggregation", default="", help="weighted|unweighted")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", default="")
    parser.add_argument("--save-cache", default="")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    rounds = args.rounds or cfg.federated.rounds
    local_epochs = args.local_epochs or cfg.federated.local_epochs
    alpha = args.alpha or cfg.federated.dirichlet_alpha
    aggregation = args.aggregation or cfg.federated.aggregation
    n_clients = cfg.federated.n_clients

    # COMPUTE-MATCHED BRACKET. Federation makes R*E local passes over a client's data; the two
    # bounds must be given the same budget or the bracket measures the budget instead of the
    # method. Leaving baseline 3 at a fixed 10 epochs while federation ran 20*3 = 60 put the
    # federated model ABOVE its own upper bound on the first real seed -- which reads as
    # "federation beats pooling" and is nothing of the sort, just a 6x training advantage.
    # Baseline 4 was already matched this way; baseline 3 now is too.
    local_passes = rounds * local_epochs
    central_epochs = args.central_epochs or local_passes
    started = time.time()

    (x_tr, y_tr, src_tr, idx_tr, blk_tr, x_te, y_te, src_te, idx_te, columns, per_class_counts) = (
        build_pipeline(cfg, args.cache, args.save_cache)
    )

    # The shared global test set: one population for every baseline (Section III-B3). The
    # segments are seed-invariant; the scaled values are not, so windows are built per seed.
    seg_te = contiguity_segments(src_te, idx_te)
    seg_tr = contiguity_segments(src_tr, idx_tr)  # for the pooled (centralised) baseline
    pooled = fit_scaler(x_tr)  # reference only -- the III-F4 claim is checked against it

    # Blocks carry exactly one label (data/split.make_blocks), so a block's label is its
    # stratum for the Dirichlet draw of Eq. (20).
    train_blocks, block_labels = block_strata(blk_tr, y_tr)

    print(
        f"\n[setup] K={n_clients} clients | alpha={alpha} | R={rounds} rounds | E={local_epochs}"
        f" | aggregation={aggregation} | W={cfg.sequence.window}"
    )
    print(f"[setup] {len(train_blocks):,} train blocks | {len(x_te):,} test rows")

    results: dict[str, list[dict[str, object]]] = {"centralized": [], "federated": []}
    local_results: list[list[dict[str, object]]] = []
    partitions: list[dict[str, object]] = []

    for seed in cfg.evaluation.seeds:
        print(f"\n=== seed {seed} ===")

        # ---- Partition (Eq. 20), published as histograms (gap G3) ----------------------
        assignment = dirichlet_block_partition(
            block_labels, n_clients=n_clients, alpha=alpha, seed=seed
        )
        histograms = per_client_class_histograms(assignment, block_labels)
        client_blocks = [train_blocks[np.asarray(ids, dtype=int)] for ids in assignment]
        row_masks = [np.isin(blk_tr, blocks) for blocks in client_blocks]

        # ---- Global scaler from sufficient statistics only (Eqs. 23-24) ----------------
        # Only count/mean/M2 leave a client. The result is distributed once, before round one.
        mean, std = federated_scaler([x_tr[mask] for mask in row_masks])
        scaler_gap = float(
            max(
                np.abs(mean - pooled.mean).max() / max(np.abs(pooled.mean).max(), 1e-12),
                np.abs(std - pooled.std).max() / max(np.abs(pooled.std).max(), 1e-12),
            )
        )
        print(f"  [scaler] federated vs pooled max relative gap {scaler_gap:.2e}")

        scaled_tr = ((x_tr - mean) / std).astype(np.float32)
        scaled_te = ((x_te - mean) / std).astype(np.float32)
        seq_te, lab_te = build_windows(scaled_te, y_te, seg_te, cfg.sequence.window)

        names = list(CLASS_NAMES)
        common = {"class_names": names, "n_classes": cfg.model.n_classes}

        # ---- Baseline 3: centralised GRU, the upper bound ------------------------------
        # Windowed over the WHOLE training split, not over the concatenated client windows.
        # "Upper bound attainable by pooling" means exactly that: a pooled trainer never sees
        # the partition, so its windows are not fragmented at partition boundaries the way a
        # client's are. Concatenating client windows would hand the upper bound the federated
        # setting's handicap and understate the gap the bracket exists to measure. It also
        # matches how Phase 3 built this same baseline, so the two phases stay comparable.
        t0 = time.time()
        pooled_seqs, pooled_labels = build_windows(scaled_tr, y_tr, seg_tr, cfg.sequence.window)
        central = centralized_gru(
            pooled_seqs,
            pooled_labels,
            seq_te,
            lab_te,
            seed=seed,
            epochs=central_epochs,
            hidden_size=cfg.model.hidden_size,
            **common,  # type: ignore[arg-type]
        )
        print(
            f"  central  macro-F1 {central.macro_f1:.4f} ({time.time() - t0:.0f}s)"
            f" | {len(pooled_seqs):,} pooled sequences"
        )
        # Freed before the per-client tensors are built: holding both at once roughly doubles
        # peak memory for no reason, and the pooled copy is not needed again.
        del pooled_seqs, pooled_labels

        # ---- Per-client windows, built inside each client's own blocks -----------------
        client_seqs, client_labels = [], []
        for client_id, mask in enumerate(row_masks):
            if not mask.any():
                client_seqs.append(np.empty((0, cfg.sequence.window, x_tr.shape[1]), np.float32))
                client_labels.append(np.empty((0,), np.int64))
                print(f"  [client {client_id}] no blocks assigned (legitimate at low alpha)")
                continue
            seg = contiguity_segments(src_tr[mask], idx_tr[mask])
            seqs, labels = build_windows(scaled_tr[mask], y_tr[mask], seg, cfg.sequence.window)
            client_seqs.append(seqs)
            client_labels.append(labels)
            present = sum(1 for v in histograms[client_id].values() if v > 0)
            print(
                f"  [client {client_id}] {int(mask.sum()):,} rows | {len(seqs):,} sequences"
                f" | {present}/{len(CLASS_NAMES)} classes present"
            )

        partitions.append(
            {
                "seed": seed,
                "alpha": alpha,
                "block_histograms": histograms,
                "sequence_counts": [len(s) for s in client_seqs],
                "row_counts": [int(m.sum()) for m in row_masks],
                "federated_vs_pooled_scaler_gap": scaler_gap,
            }
        )

        # ---- Baseline 4: local-only GRUs, the lower bound ------------------------------
        t0 = time.time()
        locals_ = local_only_grus(
            client_seqs,
            client_labels,
            seq_te,
            lab_te,
            seed=seed,
            epochs=local_passes,  # matched compute: same local passes as federation
            hidden_size=cfg.model.hidden_size,
            device=args.device,
            **common,  # type: ignore[arg-type]
        )
        empty_clients = [i for i, seqs in enumerate(client_seqs) if len(seqs) == 0]
        for client_id, m in enumerate(locals_):
            tag = "  (no data: constant-benign)" if client_id in empty_clients else ""
            print(f"  local {client_id}  macro-F1 {m.macro_f1:.4f}{tag}")
        print(f"  local    ({time.time() - t0:.0f}s)")

        # ---- Baseline 5: the federated global GRU --------------------------------------
        t0 = time.time()
        run = run_federation(
            client_seqs,
            client_labels,
            seq_te,
            lab_te,
            seed=seed,
            rounds=rounds,
            local_epochs=local_epochs,
            hidden_size=cfg.model.hidden_size,
            aggregation=aggregation,
            device=args.device,
            evaluate_each_round=True,
            **common,  # type: ignore[arg-type]
        )
        print(
            f"  federated macro-F1 {run.metrics.macro_f1:.4f} ({time.time() - t0:.0f}s)"
            f" | {run.bytes_per_round[0] / 1024:.0f} KiB/round measured"
        )

        for name, m in (("centralized", central), ("federated", run.metrics)):
            results[name].append(
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
                    # Eq. (31), derived from the matrix above -- III-I2 treats it as primary.
                    "false_positive_rate": binary_fpr(m.confusion, names.index("Benign")),
                }
            )
        local_results.append(
            [
                {
                    "client": client_id,
                    "macro_f1": m.macro_f1,
                    "balanced_accuracy": m.balanced_accuracy,
                    "mcc": m.mcc,
                    "accuracy": m.accuracy,
                    "per_class_f1": m.per_class_f1,
                    "had_no_data": client_id in empty_clients,
                }
                for client_id, m in enumerate(locals_)
            ]
        )
        results.setdefault("federated_detail", []).append(
            {
                "seed": seed,
                "per_round_macro_f1": run.per_round_macro_f1,
                "sequence_counts": run.sequence_counts,
                "bytes_per_round": run.bytes_per_round,
            }
        )
        # The lower bound is the MEAN over clients that could train at all.
        results.setdefault("local_only", []).append(
            {
                "seed": seed,
                # Every client counts, including one that got nothing: excluding it would
                # make the lower bound look better than declining to federate actually is.
                "macro_f1": float(np.mean([m.macro_f1 for m in locals_])),
                "balanced_accuracy": float(np.mean([m.balanced_accuracy for m in locals_])),
                "mcc": float(np.mean([m.mcc for m in locals_])),
                "per_client_macro_f1": [m.macro_f1 for m in locals_],
                "clients_without_data": len(empty_clients),
                # Section III-I2's client-to-global gap: positive => federation helped that
                # client relative to going it alone.
                "client_to_global_gap": client_to_global_gap(
                    {str(i): m.macro_f1 for i, m in enumerate(locals_)}, run.metrics.macro_f1
                ),
            }
        )

    # ---- Report --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(
        f"PHASE 4 -- baselines 3-5 of III-I1, K={n_clients}, alpha={alpha}, R={rounds}, "
        f"E={local_epochs}, {len(cfg.evaluation.seeds)} seeds, "
        f"{local_passes} local passes each"
    )
    print("=" * 78)

    summary: dict[str, dict[str, float]] = {}
    for name in ("centralized", "local_only", "federated"):
        summary[name] = {}
        line = []
        for metric in HEADLINE if name != "local_only" else LOCAL_ONLY_HEADLINE:
            values = [float(r[metric]) for r in results[name]]  # type: ignore[arg-type]
            mean_value, std_value = mean_std(values)
            summary[name][f"{metric}_mean"] = mean_value
            summary[name][f"{metric}_std"] = std_value
            line.append(f"{metric} {mean_value:.4f}+/-{std_value:.4f}")
        print(f"  {name:<12} " + " | ".join(line))

    # The gap-G4 question this phase exists to answer.
    lower = summary["local_only"]["macro_f1_mean"]
    upper = summary["centralized"]["macro_f1_mean"]
    federated_mean = summary["federated"]["macro_f1_mean"]
    inside = lower <= federated_mean <= upper
    recovered = (federated_mean - lower) / (upper - lower) if upper > lower else float("nan")
    print(
        f"\n  BRACKET (gap G4): local-only {lower:.4f} <= federated {federated_mean:.4f}"
        f" <= centralised {upper:.4f} -> {'INSIDE' if inside else 'OUTSIDE'}"
    )
    print(f"  => federation recovers {recovered:.1%} of the gap local-only leaves on the table")

    measured = int(np.mean([d["bytes_per_round"][0] for d in results["federated_detail"]]))
    theoretical = theoretical_bytes_per_round(
        n_clients, expected_param_count(len(columns), cfg.model.hidden_size, cfg.model.n_classes)
    )
    print(f"  COST (Eq. 22): measured {measured:,} B/round vs theoretical {theoretical:,} B/round")

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
            "summary": summary,
            "per_seed": results,
            "local_only_per_client": local_results,
            "partitions": partitions,
            "baselines_run": ["centralized_gru", "local_only_grus", "federated_global_gru"],
            "bracket": {
                "local_only_macro_f1": lower,
                "federated_macro_f1": federated_mean,
                "centralized_macro_f1": upper,
                "federated_inside_bracket": bool(inside),
                "fraction_of_gap_recovered": recovered,
            },
            "communication": {
                "measured_bytes_per_round": measured,
                "theoretical_bytes_per_round": theoretical,
            },
            "rounds": rounds,
            "local_epochs": local_epochs,
            "epoch_budget": {
                "federated_local_passes": local_passes,
                "local_only_epochs": local_passes,
                "centralized_epochs": central_epochs,
            },
            "dirichlet_alpha": alpha,
            "aggregation": aggregation,
            "n_clients": n_clients,
            "window": cfg.sequence.window,
            "selected_columns": columns,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"\n  manifest -> {path}")
    print(f"  total elapsed: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Phase 9: the generalised detector -- corpora as federated farms, leave-one-dataset-out.

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase9.py [--config configs/generalised.yaml]
        [--cache artifacts/phase9_cache.npz] [--rounds R] [--local-epochs E]
        [--sequence-cap N] [--scaling per-corpus|global] [--skip-lodo] [--skip-pooled]
        [--corpora ciciot2023,ciciomt2024]

    # The run that decides whether CICIoMT2024 helps (Phase 4's budget; ~12 h on a laptop):
    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase9.py --rounds 20 --local-epochs 3 \\
        --sequence-cap 400000 --scaling per-corpus --rebuild-cache

What "works well generally" means here, measured (docs/plans/phase9-multi-dataset.md, §3):

1. **Each corpus through the Phase 2 discipline separately**: harmonised load
   (``data/corpus.py``) -> dedup -> block split -> the R3 leakage gate (no record hash in both
   train and test), reported per corpus. Nothing is pooled before the split.
2. **K clients allocated across the training corpora** (K = ``federated.n_clients``, the
   Phase 8 topology of two Pis and one simulated node): each corpus is one or more farms, the
   larger corpus takes the extra client, and a corpus with more than one client is split
   between them by the Phase 4 block-level Dirichlet draw (Eq. 20, ``federated.dirichlet_alpha``).
   Two training corpora at K = 3 give CICIoT2023 two farms and CICIoMT2024 one; a single
   training corpus (a leave-one-out fold) gives three Dirichlet farms, exactly as Phase 4.
3. **Standardisation** (``--scaling``): ``per-corpus`` (default) combines each corpus's
   clients' sufficient statistics (Eqs. 23-24) into that corpus's own scaler, so the
   per-testbed offsets found on 2026-09-21 (the same canonical column on a different scale in
   each corpus) never reach the GRU. A test corpus is scaled with the statistics of its own
   training split -- for a held-out corpus that is the label-free feature statistics a new
   farm computes over its own traffic before running the model; no test row and no label is
   used. ``global`` is III-F4's single federated scaler over every client, kept as the
   ablation. Feature selection (step 4) runs on the scaled rows in both modes, so it sees the
   data the model sees.
4. **Feature selection over the cross-corpus intersection** (``features.candidate_columns``),
   fitted on the *training* corpora of each experiment only; the held-out corpus of a
   leave-one-dataset-out fold never touches selection, scaling or training.
5. **Every parameter vector crosses as an Ascon-AEAD128 frame** (``SealedFederatedServer``),
   as on the Phase 8 hardware.
6. **Experiments, >= 3 seeds each, mean +/- std**, run seed-major so one seed of everything
   lands before the next seed starts (a progress file is rewritten after every run):
   * ``in_distribution`` -- federated over every training corpus, scored on each corpus's own
     test split;
   * ``pooled_centralised`` -- one GRU trained on the concatenation of the *same* capped
     client windows for R x E epochs, scored the same way. Same sequences, no federation: the
     difference to ``in_distribution`` is what federation costs, and its difference to the
     single-corpus ceilings below is what mixing corpora costs;
   * ``lodo/<held-out>`` -- federated over the other corpora, scored on the held-out corpus's
     test split (the generalisation claim) **and** on the training corpora's own test splits,
     which is the single-corpus in-distribution ceiling at the same budget, for free.
   Leave-one-dataset-out is the generalisation claim; in-distribution is the ceiling it is
   read against.

Metrics follow Section III-I2, with one addition that the multi-corpus setting forces:
``macro_f1`` is the C = 8 macro over every family (absent families score 0, as the metrics
module defines it), and ``macro_f1_present`` averages only over families present in that
test split -- CICIoMT2024 has no Mirai or BruteForce, and a detector cannot be marked down
for classes a corpus does not contain. Both are reported.

Compute: the default budget (R = 10, E = 2, 150k sequences per client) is sized for a laptop
run of the full protocol; the Phase 4 budget (R = 20, E = 3, 400k per client ~ Phase 4's
per-client load) is reproducible with the flags and takes hours. The cache holds the
*prepared* corpora (after step 1: all candidate columns, unscaled, with block ids), so a
second run with a different budget skips the loading.
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

from ascon_smart_agri.data.corpus import load_corpus
from ascon_smart_agri.data.datasets import REGISTRY, TRAINING_CORPORA, get_dataset
from ascon_smart_agri.data.dedup import deduplicate, record_hash
from ascon_smart_agri.data.split import make_blocks, stratified_block_split
from ascon_smart_agri.data.taxonomy import CLASS_NAMES, to_class_index
from ascon_smart_agri.eval.baselines import run_federation
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.eval.metrics import multiclass_metrics
from ascon_smart_agri.eval.report import mean_std
from ascon_smart_agri.features.selection import FeatureSelector, restrict_candidates
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.federated.scaler_stats import combine_stats, local_sufficient_stats
from ascon_smart_agri.federated.transport import generate_demo_keys
from ascon_smart_agri.model.train import predict, train_centralized
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

print = functools.partial(builtins.print, flush=True)

HEADLINE = ("macro_f1", "macro_f1_present", "balanced_accuracy", "mcc", "false_positive_rate")
CACHE_KEYS = ("x_tr", "y_tr", "s_tr", "i_tr", "b_tr", "x_te", "y_te", "s_te", "i_te")
POOLED = "pooled_centralised"


# ----------------------------------------------------------------------------- step 1: data


def prepare_corpus(name: str, cfg, seed: int):  # type: ignore[no-untyped-def]
    """Load -> dedup -> block split -> R3 gate for one corpus. Returns unscaled arrays."""
    spec = get_dataset(name)
    t0 = time.time()
    frame, per_leaf = load_corpus(
        spec,
        per_class_cap=cfg.data.per_class_cap,
        chunk_size=cfg.data.chunk_size,
        window_packets=cfg.sequence.window_packets,
        seed=seed,
    )
    n_raw = len(frame)
    deduped = deduplicate(frame)
    block_ids = make_blocks(deduped, block_size=cfg.data.block_size)
    split = stratified_block_split(
        deduped, block_ids, test_fraction=cfg.data.test_fraction, seed=seed
    )
    train = deduped.loc[block_ids.isin(split.train_blocks)]
    test = deduped.loc[block_ids.isin(split.test_blocks)]

    # R3, per corpus: no record hash in both partitions. The gate the whole phase rests on.
    overlap = len(set(record_hash(train)) & set(record_hash(test)))
    if overlap:
        raise RuntimeError(f"{name}: R3 leakage gate FAILED -- {overlap} hashes in train and test")

    feature_cols = [c for c in cfg.features.candidate_columns if c in train.columns]
    missing = [c for c in cfg.features.candidate_columns if c not in train.columns]

    def arrays(part):  # type: ignore[no-untyped-def]
        values = part[feature_cols].to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        kept = part.loc[finite]
        return (
            values[finite],
            to_class_index(kept["label"], spec.leaf_to_family).to_numpy(),
            kept["source_file"].to_numpy().astype(str),
            kept.index.to_numpy(),
            block_ids.loc[kept.index].to_numpy(),
        )

    x_tr, y_tr, s_tr, i_tr, b_tr = arrays(train)
    x_te, y_te, s_te, i_te, _ = arrays(test)
    print(
        f"[{name}] {n_raw:,} rows -> {len(deduped):,} after dedup | train {len(x_tr):,} "
        f"({len(np.unique(b_tr)):,} blocks) | test {len(x_te):,} | R3 overlap 0 | "
        f"{len(feature_cols)} candidate columns"
        + (f" (lacks {missing})" if missing else "")
        + f" | {time.time() - t0:.0f}s"
    )
    return {
        "columns": feature_cols,
        "per_leaf": per_leaf,
        "n_raw": n_raw,
        "n_dedup": len(deduped),
        "x_tr": x_tr,
        "y_tr": y_tr,
        "s_tr": s_tr,
        "i_tr": i_tr,
        "b_tr": b_tr,
        "x_te": x_te,
        "y_te": y_te,
        "s_te": s_te,
        "i_te": i_te,
    }


def save_cache(path: Path, corpora: dict[str, dict]) -> None:  # type: ignore[type-arg]
    blobs: dict[str, np.ndarray] = {}
    for name, c in corpora.items():
        for key in CACHE_KEYS:
            blobs[f"{name}__{key}"] = c[key]
        blobs[f"{name}__columns"] = np.array(c["columns"], dtype=str)
        blobs[f"{name}__meta"] = np.array(
            json.dumps({"per_leaf": c["per_leaf"], "n_raw": c["n_raw"], "n_dedup": c["n_dedup"]})
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **blobs)
    print(f"[cache] -> {path}")


def load_cache(path: Path, names: list[str], candidate_columns: list[str]) -> dict[str, dict]:  # type: ignore[type-arg]
    """Load the prepared corpora; refuse a cache built for another corpus set or column set."""
    blob = np.load(path, allow_pickle=False)
    corpora: dict[str, dict] = {}  # type: ignore[type-arg]
    for name in names:
        wanted = [f"{name}__{key}" for key in (*CACHE_KEYS, "columns", "meta")]
        absent = [k for k in wanted if k not in blob.files]
        if absent:
            raise SystemExit(
                f"{path} lacks {absent[:3]}{'...' if len(absent) > 3 else ''}: built for "
                "another corpus set or by an older driver -- rerun with --rebuild-cache"
            )
        meta = json.loads(str(blob[f"{name}__meta"]))
        corpora[name] = {
            "columns": [str(c) for c in blob[f"{name}__columns"]],
            **meta,
            **{key: blob[f"{name}__{key}"] for key in CACHE_KEYS},
        }
        c = corpora[name]
        expected = [col for col in candidate_columns if col in c["columns"]]
        stale = [col for col in candidate_columns if col not in c["columns"]]
        if c["columns"] != expected or (stale and len(c["columns"]) < len(candidate_columns)):
            # A corpus may legitimately lack a candidate; a cache built from a *narrower*
            # candidate list is the case to catch (e.g. the 17-column 2026-09-21 cache).
            raise SystemExit(
                f"{path}: {name} holds {len(c['columns'])} columns but the config lists "
                f"{len(candidate_columns)} candidates -- rerun with --rebuild-cache"
            )
        print(f"[cache] {name}: train {len(c['x_tr']):,} | test {len(c['x_te']):,}")
    return corpora


# ---------------------------------------------------------------------- step 2: farms (clients)


def allocate_clients(train_names: list[str], corpora: dict, n_clients: int) -> dict[str, int]:  # type: ignore[type-arg]
    """How many of the K clients each training corpus gets: at least one, extras to the larger."""
    if n_clients < len(train_names):
        raise SystemExit(
            f"federated.n_clients = {n_clients} is fewer than the {len(train_names)} training "
            "corpora; every corpus needs at least one client"
        )
    by_size = sorted(train_names, key=lambda n: (-len(corpora[n]["x_tr"]), n))
    counts = dict.fromkeys(train_names, 1)
    for i in range(n_clients - len(train_names)):
        counts[by_size[i % len(by_size)]] += 1
    return counts


def corpus_client_masks(c: dict, n_clients: int, alpha: float, seed: int):  # type: ignore[type-arg,no-untyped-def]
    """Split one corpus's training rows into ``n_clients`` farms: Phase 4's block Dirichlet.

    Returns ``(row_masks, histograms)``; a single client takes every row and no draw is made.
    """
    b_tr, y_tr = c["b_tr"], c["y_tr"]
    if n_clients == 1:
        return [np.ones(len(y_tr), dtype=bool)], None
    # Blocks are single-label by construction (data/split.make_blocks): the first row's label
    # is the block's label. One pass, not one scan per block.
    train_blocks, first = np.unique(b_tr, return_index=True)
    block_labels = y_tr[first]
    assignment = dirichlet_block_partition(
        block_labels, n_clients=n_clients, alpha=alpha, seed=seed
    )
    histograms = per_client_class_histograms(assignment, block_labels)
    masks = [np.isin(b_tr, train_blocks[np.asarray(ids, dtype=int)]) for ids in assignment]
    return masks, [{CLASS_NAMES[int(k)]: v for k, v in h.items()} for h in histograms]


# ------------------------------------------------------------------------ step 3-4: one run


def select_columns(  # type: ignore[no-untyped-def]
    train_names: list[str], scaled_train: dict, corpora: dict, cfg, seed: int
) -> list[str]:  # type: ignore[type-arg]
    """Four-stage selection over the intersection, fitted on the (scaled) training corpora."""
    import pandas as pd

    common = [
        c
        for c in cfg.features.candidate_columns
        if all(c in corpora[n]["columns"] for n in train_names)
    ]
    frames = []
    for name in train_names:
        cols = corpora[name]["columns"]
        idx = [cols.index(k) for k in common]
        frames.append(
            pd.DataFrame(scaled_train[name][:, idx], columns=common).assign(
                label=[CLASS_NAMES[i] for i in corpora[name]["y_tr"]]
            )
        )
    pooled = pd.concat(frames, ignore_index=True)
    del frames
    selection = FeatureSelector().fit(
        restrict_candidates(pooled.drop(columns=["label"]), common),
        pooled["label"],
        tau=cfg.features.correlation_tau,
        f=cfg.features.selected_f,
        f_sweep=cfg.features.f_sweep,
        rf_n_estimators=cfg.features.rf_n_estimators,
        rrf_k=cfg.features.rrf_k,
        sample_size=cfg.features.selection_sample_size,
        seed=seed,
    )
    return list(selection.selected_columns)


def cap_sequences(
    seqs: np.ndarray, labels: np.ndarray, cap: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded, stratified cap on a client's training sequences (compute budget only)."""
    if cap <= 0 or len(seqs) <= cap:
        return seqs, labels
    rng = np.random.default_rng(seed)
    take: list[np.ndarray] = []
    classes, counts = np.unique(labels, return_counts=True)
    share = counts / counts.sum()
    for cls, frac in zip(classes, share, strict=True):
        pos = np.flatnonzero(labels == cls)
        k = max(1, round(cap * frac))
        take.append(pos if len(pos) <= k else rng.choice(pos, size=k, replace=False))
    keep = np.sort(np.concatenate(take))
    return seqs[keep], labels[keep]


def score(model, seqs: np.ndarray, labels: np.ndarray) -> dict[str, object]:  # type: ignore[no-untyped-def]
    names = list(CLASS_NAMES)
    y_pred, _ = predict(model, seqs)
    m = multiclass_metrics(labels, y_pred, names)
    present = [names[i] for i in np.unique(labels)]
    benign = names.index("Benign")
    row = np.asarray(m.confusion, dtype=np.float64)[benign]
    fpr = float((row.sum() - row[benign]) / row.sum()) if row.sum() > 0 else float("nan")
    return {
        "macro_f1": m.macro_f1,
        "macro_f1_present": float(np.mean([m.per_class_f1[n] for n in present])),
        "families_present": present,
        "balanced_accuracy": m.balanced_accuracy,
        "mcc": m.mcc,
        "accuracy": m.accuracy,
        "per_class_f1": m.per_class_f1,
        "confusion": m.confusion.tolist(),
        "false_positive_rate": fpr,
        "n_sequences": len(labels),
    }


def print_scores(label: str, seed: int, scores: dict) -> None:  # type: ignore[type-arg]
    for name, s in scores.items():
        print(
            f"  [{label} | seed {seed}] test={name:<12} macro-F1 {s['macro_f1']:.4f} "
            f"(present-only {s['macro_f1_present']:.4f}) | bal-acc {s['balanced_accuracy']:.4f} "
            f"| FPR {s['false_positive_rate']:.4f}"
        )


def run_experiment(  # type: ignore[no-untyped-def]
    label: str,
    train_names: list[str],
    test_names: list[str],
    corpora: dict,  # type: ignore[type-arg]
    cfg,
    seed: int,
    *,
    rounds: int,
    local_epochs: int,
    sequence_cap: int,
    scaling: str,
    pooled: bool,
    verbose: bool,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """One federated run (and, if asked, the pooled-centralised twin on the same windows)."""
    t0 = time.time()
    W = cfg.sequence.window
    K = cfg.federated.n_clients
    alpha = cfg.federated.dirichlet_alpha

    # ---- Farms: K clients over the training corpora, Dirichlet within a corpus ------------
    n_per_corpus = allocate_clients(train_names, corpora, K)
    farms: list[tuple[str, int, np.ndarray]] = []  # (corpus, index within corpus, row mask)
    histograms: dict[str, object] = {}
    for name in train_names:
        masks, hist = corpus_client_masks(corpora[name], n_per_corpus[name], alpha, seed)
        farms.extend((name, i, m) for i, m in enumerate(masks))
        if hist is not None:
            histograms[name] = hist
    client_ids = [f"{name}:{i}" for name, i, _ in farms]

    # ---- Standardisation from client sufficient statistics only (Eqs. 23-24) --------------
    # Stats are over every candidate column (scaling is per column, so selecting afterwards
    # is the same as scaling the selected columns); only count/mean/M2 leave a farm.
    farm_stats = {
        cid: local_sufficient_stats(corpora[n]["x_tr"][m])
        for cid, (n, _, m) in zip(client_ids, farms, strict=True)
    }
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if scaling == "global":
        g = combine_stats(list(farm_stats.values()))
        stats = dict.fromkeys({*train_names, *test_names}, g)
    else:
        for name in {*train_names, *test_names}:
            own = [
                farm_stats[cid]
                for cid, (n, _, _) in zip(client_ids, farms, strict=True)
                if n == name
            ]
            if not own:  # a held-out corpus: its own training-split statistics, label-free
                own = [local_sufficient_stats(corpora[name]["x_tr"])]
            stats[name] = combine_stats(own)

    def scaled(name: str, part: str) -> np.ndarray:
        mean, std = stats[name]
        return ((corpora[name][f"x_{part}"] - mean) / std).astype(np.float32)

    scaled_train = {name: scaled(name, "tr") for name in train_names}
    columns = select_columns(train_names, scaled_train, corpora, cfg, seed)

    def take(name: str, x: np.ndarray) -> np.ndarray:
        cols = corpora[name]["columns"]
        return x[:, [cols.index(k) for k in columns]]

    # ---- Per-farm windows, built inside each farm's own rows -----------------------------
    client_seqs, client_labels, counts_raw = [], [], []
    for cid, (name, _, mask) in zip(client_ids, farms, strict=True):
        c = corpora[name]
        seg = contiguity_segments(c["s_tr"][mask], c["i_tr"][mask])
        seqs, labels = build_windows(take(name, scaled_train[name][mask]), c["y_tr"][mask], seg, W)
        counts_raw.append(len(seqs))
        seqs, labels = cap_sequences(seqs, labels, sequence_cap, seed)
        client_seqs.append(seqs)
        client_labels.append(labels)
        present = len(np.unique(labels))
        print(
            f"  [{label} | seed {seed}] farm {cid:<14} {int(mask.sum()):>9,} rows | "
            f"{counts_raw[-1]:>9,} sequences -> {len(seqs):>7,} kept | "
            f"{present}/{len(CLASS_NAMES)} classes"
        )
    del scaled_train

    test_sets = {}
    for name in test_names:
        c = corpora[name]
        seg = contiguity_segments(c["s_te"], c["i_te"])
        test_sets[name] = build_windows(take(name, scaled(name, "te")), c["y_te"], seg, W)

    # ---- Federated (Algorithm 1, sealed channel); curve tracked on the first test corpus --
    first_x, first_y = test_sets[test_names[0]]
    run = run_federation(
        client_seqs,
        client_labels,
        first_x,
        first_y,
        seed=seed,
        class_names=list(CLASS_NAMES),
        n_classes=cfg.model.n_classes,
        rounds=rounds,
        local_epochs=local_epochs,
        hidden_size=cfg.model.hidden_size,
        aggregation=cfg.federated.aggregation,
        evaluate_each_round=True,
        verbose=verbose,
        sealed=(client_ids, generate_demo_keys(client_ids)),
    )
    scores = {name: score(run.model, x, y) for name, (x, y) in test_sets.items()}
    print_scores(label, seed, scores)
    print(
        f"  [{label} | seed {seed}] F={len(columns)} {columns} | federated {time.time() - t0:.0f}s"
    )
    federated: dict[str, object] = {
        "seed": seed,
        "train_corpora": train_names,
        "clients": client_ids,
        "client_histograms_blocks": histograms,
        "scaler_stats_used": scaling,
        "selected_columns": columns,
        "sequence_counts_uncapped": counts_raw,
        "sequence_counts": run.sequence_counts,
        "per_round_macro_f1": run.per_round_macro_f1,
        "bytes_per_round": run.bytes_per_round,
        "scores": scores,
    }
    if not pooled:
        return federated, None

    # ---- Pooled-centralised twin: the SAME capped windows, one trainer, R x E epochs -------
    # Unlike Phase 4's upper bound (windowed over the whole split), this deliberately reuses
    # the farm windows so that the only difference to the federated run is the absence of
    # federation; the ~W/block_size windows lost at farm boundaries are lost to both.
    t1 = time.time()
    pooled_x = np.concatenate(client_seqs)
    pooled_y = np.concatenate(client_labels)
    del client_seqs, client_labels, run
    epochs = rounds * local_epochs
    model = train_centralized(
        pooled_x,
        pooled_y,
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
        epochs=epochs,
        seed=seed,
        verbose=verbose,
    )
    n_pooled = len(pooled_y)
    del pooled_x, pooled_y
    pooled_scores = {name: score(model, x, y) for name, (x, y) in test_sets.items()}
    print_scores(POOLED, seed, pooled_scores)
    print(
        f"  [{POOLED} | seed {seed}] {n_pooled:,} sequences x {epochs} epochs | "
        f"{time.time() - t1:.0f}s"
    )
    return federated, {
        "seed": seed,
        "train_corpora": train_names,
        "selected_columns": columns,
        "n_sequences": n_pooled,
        "epochs": epochs,
        "scores": pooled_scores,
    }


def summarise(per_seed: list[dict[str, object]], test_names: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in test_names:
        block: dict[str, object] = {}
        for metric in HEADLINE:
            values = [float(r["scores"][name][metric]) for r in per_seed]  # type: ignore[index]
            m, s = mean_std(values)
            block[f"{metric}_mean"], block[f"{metric}_std"] = m, s
        block["per_class_f1_mean"] = {
            cls: float(np.mean([r["scores"][name]["per_class_f1"][cls] for r in per_seed]))  # type: ignore[index]
            for cls in CLASS_NAMES
        }
        block["families_present"] = per_seed[0]["scores"][name]["families_present"]  # type: ignore[index]
        block["held_out"] = name not in per_seed[0]["train_corpora"]  # type: ignore[operator]
        block["n_seeds"] = len(per_seed)
        out[name] = block
    return out


# -------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generalised.yaml")
    parser.add_argument(
        "--corpora",
        default=",".join(TRAINING_CORPORA),
        help="registered corpora to train on; one marked not_for_training must be named "
        f"explicitly (registry: {', '.join(REGISTRY)})",
    )
    parser.add_argument("--cache", default="artifacts/phase9_cache.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rounds", type=int, default=10, help="R (Phase 4 used 20)")
    parser.add_argument("--local-epochs", type=int, default=2, help="E (Phase 4 used 3)")
    parser.add_argument(
        "--sequence-cap", type=int, default=150_000, help="per client, 0 = uncapped"
    )
    parser.add_argument("--skip-lodo", action="store_true", help="in-distribution only")
    parser.add_argument("--skip-pooled", action="store_true", help="no centralised twin")
    parser.add_argument(
        "--scaling",
        choices=("per-corpus", "global"),
        default="per-corpus",
        help="per-corpus = each corpus its own scaler from its farms' statistics (default); "
        "global = one federated scaler over every farm (III-F4, the ablation)",
    )
    parser.add_argument("--verbose", action="store_true", help="print every round's macro-F1")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if not cfg.features.candidate_columns:
        raise SystemExit("Phase 9 needs features.candidate_columns (configs/generalised.yaml)")
    names = [n.strip() for n in args.corpora.split(",") if n.strip()]
    for name in names:
        reason = get_dataset(name).not_for_training
        if reason:
            print(f"[warning] {name} is registered not_for_training; named explicitly: {reason}")
    started = time.time()

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild_cache:
        corpora = load_cache(cache, names, list(cfg.features.candidate_columns))
    else:
        corpora = {name: prepare_corpus(name, cfg, cfg.data.seed) for name in names}
        save_cache(cache, corpora)

    experiments: list[tuple[str, list[str], list[str]]] = [("in_distribution", names, names)]
    if not args.skip_lodo and len(names) > 1:
        for held_out in names:
            others = [n for n in names if n != held_out]
            experiments.append((f"lodo/{held_out}", others, [held_out, *others]))

    print(
        f"\n[setup] K={cfg.federated.n_clients} over {names} "
        f"(alpha={cfg.federated.dirichlet_alpha} within a corpus) | R={args.rounds} | "
        f"E={args.local_epochs} | W={cfg.sequence.window} | "
        f"cap {args.sequence_cap:,}/client | scaling {args.scaling} | seeds {cfg.evaluation.seeds}"
        + ("" if args.skip_pooled else f" | + {POOLED} at {args.rounds * args.local_epochs} epochs")
    )
    tag = f"phase9_{cfg.run_name}" + ("" if args.scaling == "global" else "_percorpus")
    progress = Path(cfg.output_dir) / f"{tag}_progress.json"
    results: dict[str, dict[str, object]] = {label: {"per_seed": []} for label, _, _ in experiments}
    if not args.skip_pooled:
        results[POOLED] = {"per_seed": []}
    tests_of = {label: t for label, _, t in experiments}
    tests_of[POOLED] = names

    for seed in cfg.evaluation.seeds:
        for label, train_names, test_names in experiments:
            print(f"\n=== seed {seed} | {label}: train {train_names} -> test {test_names} ===")
            federated, pooled = run_experiment(
                label,
                train_names,
                test_names,
                corpora,
                cfg,
                seed,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                sequence_cap=args.sequence_cap,
                scaling=args.scaling,
                pooled=(label == "in_distribution" and not args.skip_pooled),
                verbose=args.verbose,
            )
            results[label]["per_seed"].append(federated)  # type: ignore[attr-defined]
            if pooled is not None:
                results[POOLED]["per_seed"].append(pooled)  # type: ignore[attr-defined]
            for done, block in results.items():
                if block["per_seed"]:
                    block["summary"] = summarise(block["per_seed"], tests_of[done])  # type: ignore[arg-type]
            progress.write_text(
                json.dumps(
                    {"elapsed_seconds": round(time.time() - started, 1), "experiments": results},
                    indent=1,
                )
            )

    print("\n" + "=" * 92)
    print(
        f"PHASE 9 RESULTS -- {len(cfg.evaluation.seeds)} seeds, mean +/- std, "
        f"scaling {args.scaling}"
    )
    print("=" * 92)
    print(
        f"{'experiment':<24}{'test corpus':<14}{'':<10}{'macro-F1':>18}"
        f"{'present-only':>18}{'FPR':>16}"
    )
    for label, block in results.items():
        for name, s in block["summary"].items():  # type: ignore[attr-defined]
            print(
                f"{label:<24}{name:<14}{'HELD OUT' if s['held_out'] else '':<10}"
                f"{s['macro_f1_mean']:>10.4f} ± {s['macro_f1_std']:.4f}"
                f"{s['macro_f1_present_mean']:>10.4f} ± {s['macro_f1_present_std']:.4f}"
                f"{s['false_positive_rate_mean']:>9.4f} ± {s['false_positive_rate_std']:.4f}"
            )

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=tag,
        seeds=list(cfg.evaluation.seeds),
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts={
            f"{name}/{leaf}": n for name in names for leaf, n in corpora[name]["per_leaf"].items()
        },
        ascon_backend={"weight_channel": "SealedFederatedServer (Ascon-AEAD128, per-client keys)"},
        results={
            "corpora": {
                name: {
                    "n_raw": corpora[name]["n_raw"],
                    "n_dedup": corpora[name]["n_dedup"],
                    "n_train_rows": len(corpora[name]["x_tr"]),
                    "n_test_rows": len(corpora[name]["x_te"]),
                    "candidate_columns": corpora[name]["columns"],
                    "r3_overlap": 0,
                }
                for name in names
            },
            "budget": {
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "sequence_cap_per_client": args.sequence_cap,
                "scaling": args.scaling,
                "n_clients": cfg.federated.n_clients,
                "dirichlet_alpha": cfg.federated.dirichlet_alpha,
            },
            "experiments": results,
            "elapsed_seconds": round(time.time() - started, 1),
        },
    )
    path = manifest.write(Path(cfg.output_dir))
    print(f"\n  manifest -> {path}")


if __name__ == "__main__":
    main()

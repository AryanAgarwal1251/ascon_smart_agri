"""Train and SAVE one federated global model checkpoint for Phase 7's runtime pipeline.

No training run before this one in this project ever persisted a model to disk -- Phase 3/4's
runs trained, reported metrics, and discarded the weights. This script reuses the identical
Phase 4 configuration and data path (same partition, same alpha=0.5, R=20, E=3, weighted
FedAvg) but trains ONE seed only, then saves the resulting model.

Scope decision (flagged, Golden Rule 1): Phase 4 already trained 3 seeds under this exact
config to establish the STATISTICAL claim (mean +/- std in artifacts/manifest_phase4_default.json,
copied into this run's manifest as ``phase4_reference_*`` rather than typed in). Training a
further 3 seeds here to pick one would be a second,
redundant statistical exercise; Phase 7 needs *a* real, honestly-trained deployment artifact of
the architecture the paper specifies, not a fresh validation. Seed 0 is used -- the first of
Phase 4's own seed list, no other reason for that particular choice. Phase 4's reported mean+/-
std remains the only statistically meaningful performance claim; this checkpoint's own held-out
accuracy is reported too (it will differ slightly from the 3-seed mean, by design) but is not a
new finding.

    PYTHONPATH=. ./.venv/bin/python -u scripts/train_federated_model.py [--cache path.npz]

The data path is run_phase4.py's own (``build_pipeline``): the same block ids, the same
seed-0 Dirichlet draw, and the same federated scaler from client statistics (Eqs. 23-24),
so the checkpoint is trained exactly as Phase 4's seed-0 model was. ``--cache`` takes the
Phase 4 cache (``run_phase4.py --save-cache``), which stores UNSCALED features.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from configs.base import load_run_config

from ascon_smart_agri.crypto.ascon_aead import backend_provenance
from ascon_smart_agri.data.taxonomy import CLASS_NAMES
from ascon_smart_agri.eval.baselines import federated_global_gru
from ascon_smart_agri.eval.manifest import RunManifest, collect_environment
from ascon_smart_agri.federated.partition import (
    dirichlet_block_partition,
    per_client_class_histograms,
)
from ascon_smart_agri.model.checkpoint import save_model
from ascon_smart_agri.sequences.windowing import build_windows, contiguity_segments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase4 import block_strata, build_pipeline, federated_scaler

print = functools.partial(builtins.print, flush=True)

CHECKPOINT_SEED = 0  # see the module docstring's scope decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default="", help="optional .npz of prepared train/test arrays")
    parser.add_argument("--out", default="artifacts/federated_global_model.safetensors")
    parser.add_argument(
        "--phase4-manifest",
        default="artifacts/manifest_phase4_default.json",
        help="Phase 4 run whose 3-seed federated mean +/- std this checkpoint is reported against",
    )
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()

    # Same Phases 1-2 pipeline as run_phase4.py: UNSCALED features plus per-row block ids.
    (x_tr, y_tr, src_tr, idx_tr, blk_tr, x_te, y_te, src_te, idx_te, _columns, per_class_counts) = (
        build_pipeline(cfg, args.cache, "")
    )
    print(f"[data] train {x_tr.shape} | test {x_te.shape}")

    seg_te = contiguity_segments(src_te, idx_te)

    # ---- IDENTICAL partition to run_phase4.py's seed-0 run: same block ids, same draw ------
    train_blocks, block_labels = block_strata(blk_tr, y_tr)
    assignment = dirichlet_block_partition(
        block_labels,
        n_clients=cfg.federated.n_clients,
        alpha=cfg.federated.dirichlet_alpha,
        seed=CHECKPOINT_SEED,
    )
    histograms = per_client_class_histograms(assignment, block_labels)
    client_blocks = [train_blocks[np.asarray(ids, dtype=int)] for ids in assignment]
    row_masks = [np.isin(blk_tr, blocks) for blocks in client_blocks]
    sizes = [len(c) for c in client_blocks]
    print(f"[partition] alpha={cfg.federated.dirichlet_alpha}, sizes={sizes}")

    # ---- Global scaler from client sufficient statistics only (Eqs. 23-24), as in Phase 4 --
    mean, std = federated_scaler([x_tr[mask] for mask in row_masks])
    scaled_tr = ((x_tr - mean) / std).astype(np.float32)
    scaled_te = ((x_te - mean) / std).astype(np.float32)

    client_seqs, client_y = [], []
    for mask in row_masks:
        seg = contiguity_segments(src_tr[mask], idx_tr[mask])
        seqs, labs = build_windows(scaled_tr[mask], y_tr[mask], seg, cfg.sequence.window)
        client_seqs.append(seqs)
        client_y.append(labs)
        print(f"  -> client sequences: {len(seqs):,}")

    seq_te, lab_te = build_windows(scaled_te, y_te, seg_te, cfg.sequence.window)
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

    # The 3-seed claim this checkpoint is compared against comes from the Phase 4 manifest,
    # never from a literal in this file (a literal here once outlived the run it described).
    reference: dict[str, float] = {}
    if Path(args.phase4_manifest).exists():
        summary = json.loads(Path(args.phase4_manifest).read_text())["results"]["summary"]
        reference = summary.get("federated", {})
    print(
        f"[reference] Phase 4 federated macro-F1 {reference.get('macro_f1_mean')} "
        f"+/- {reference.get('macro_f1_std')} ({args.phase4_manifest})"
    )

    versions, hardware = collect_environment()
    manifest = RunManifest(
        run_name=f"phase7_train_{cfg.run_name}",
        seeds=[CHECKPOINT_SEED],
        library_versions=versions,
        hardware=hardware,
        config_snapshot=json.loads(cfg.model_dump_json()),
        subsample_per_class_counts=per_class_counts,
        # Implementation deviation (federated/crypto.py): this training run's weight transport
        # is genuinely Ascon-encrypted, unlike run_phase7.py's runtime demo.
        ascon_backend=backend_provenance(),
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
            "phase4_reference_manifest": args.phase4_manifest,
            "phase4_reference_macro_f1_mean": reference.get("macro_f1_mean"),
            "phase4_reference_macro_f1_std": reference.get("macro_f1_std"),
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

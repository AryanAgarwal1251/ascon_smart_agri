"""Distil the two Phase 7 manifests (training + end-to-end run) into one short results file.

Same distillation relationship ``phase4_results.json`` has to ``manifest_phase4_default.json``:
the manifests are the full provenance records, this pulls the headline facts into a compact
summary a reader can open without wading through per-round/per-message JSON.

    PYTHONPATH=. ./.venv/bin/python scripts/summarize_phase7.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def summarize(
    train_manifest: dict[str, Any],
    e2e_manifest: dict[str, Any],
    phase4_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train = train_manifest["results"]
    e2e = e2e_manifest["results"]

    # Compare against Phase 4's ACTUAL recorded seed-0 result, read from its own manifest --
    # not a hardcoded constant. An earlier version of this function compared against a
    # hand-typed "0.8338" (the value as printed, truncated to 4 decimals) and reported "False"
    # for a run that was in fact bit-for-bit identical (difference exactly 0.0): the truncated
    # constant differed from the true value by 1.45e-6, just over the 1e-6 tolerance. Reading
    # the real number from Phase 4's own manifest makes this comparison correct by
    # construction rather than by hoping a copied-in digit string stays in sync.
    phase4_seed0 = None
    if phase4_manifest is not None:
        runs = phase4_manifest["results"]["per_seed"]["federated"]
        for seed, entry in zip(phase4_manifest["seeds"], runs, strict=True):
            if seed == train["checkpoint_seed"]:
                phase4_seed0 = entry["macro_f1"]
                break

    return {
        "checkpoint": {
            "path": train["checkpoint_path"],
            "seed": train["checkpoint_seed"],
            "macro_f1": round(train["macro_f1"], 4),
            "balanced_accuracy": round(train["balanced_accuracy"], 4),
            "mcc": round(train["mcc"], 4),
            "phase4_reference_macro_f1_mean": train["phase4_reference_macro_f1_mean"],
            "phase4_reference_macro_f1_std": train["phase4_reference_macro_f1_std"],
            "phase4_seed0_macro_f1_exact": phase4_seed0,
            "reproduces_phase4_seed0_exactly": (
                phase4_seed0 is not None and train["macro_f1"] == phase4_seed0
            ),
            "training_seconds": train["elapsed_seconds"],
            "scope_note": train["scope_note"],
        },
        "end_to_end_run": {
            "n_messages": e2e["n_messages"],
            "n_buffering": e2e["n_buffering"],
            "n_classified": e2e["n_classified"],
            "n_benign": e2e["n_benign"],
            "n_malicious": e2e["n_malicious"],
            "cloud_received": e2e["cloud_received"],
            "cloud_rejected": e2e["cloud_rejected"],
            "alert_count": e2e["alert_count"],
            "g1_malicious_reaching_cloud": e2e["g1_malicious_reaching_cloud"],
            "g1_holds": e2e["g1_malicious_reaching_cloud"] == 0,
            "informal_true_label_agreement": e2e["informal_true_label_agreement"],
            "per_stage_latency_us": e2e["per_stage_latency_us"],
            "limitation_note": e2e["limitation_note"],
            "elapsed_seconds": e2e["elapsed_seconds"],
        },
        "training_git_commit": train_manifest["hardware"].get("git_commit"),
        "e2e_git_commit": e2e_manifest["hardware"].get("git_commit"),
        "source_manifests": [
            "manifest_phase7_train_default.json",
            "manifest_phase7_e2e_default.json",
        ],
    }


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")
    train_path = out_dir / "manifest_phase7_train_default.json"
    e2e_path = out_dir / "manifest_phase7_e2e_default.json"
    phase4_path = out_dir / "manifest_phase4_default.json"
    for p in (train_path, e2e_path):
        if not p.exists():
            print(f"Missing {p} -- run train_federated_model.py and run_phase7.py first.")
            return 2

    phase4_manifest = json.loads(phase4_path.read_text()) if phase4_path.exists() else None
    summary = summarize(
        json.loads(train_path.read_text()), json.loads(e2e_path.read_text()), phase4_manifest
    )
    result_path = out_dir / "phase7_results.json"
    result_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"wrote {result_path}")
    print(f"  checkpoint macro-F1        : {summary['checkpoint']['macro_f1']}")
    print(
        f"  reproduces Phase 4 seed 0  : {summary['checkpoint']['reproduces_phase4_seed0_exactly']}"
    )
    print(
        f"  e2e messages/classified    : {summary['end_to_end_run']['n_messages']}"
        f"/{summary['end_to_end_run']['n_classified']}"
    )
    print(f"  G1 holds                   : {summary['end_to_end_run']['g1_holds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Distil the Phase 4 manifest into a short, human-readable results file.

``artifacts/manifest_phase4_default.json`` (written by ``run_phase4.py``) is the full
provenance record -- every seed, every confusion matrix, the config snapshot, library versions.
This script pulls the headline numbers out of it into ``artifacts/phase4_results.json``, the
same distillation relationship ``phase2_feature_selection.json`` already has to its own run
(a compact summary a reader can open without wading through per-seed JSON), not a replacement
for the manifest.

    PYTHONPATH=. ./.venv/bin/python scripts/summarize_phase4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ascon_smart_agri.eval.report import client_to_global_gap


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    """Distil a Phase 4 manifest into the short results file.

    Reads the schema ``run_phase4.py`` writes (``summary``/``per_seed``/``partitions``/
    ``bracket``). An earlier revision read a flatter schema from a parallel Phase 4 driver; the
    two were reconciled during the merge onto this one, and the G4 questions it asks are kept
    verbatim because they are the ones Section III-I1 poses.
    """
    results = manifest["results"]
    summary = results["summary"]
    federated = summary["federated"]
    centralized = summary["centralized"]["macro_f1_mean"]

    # Per-client local-only figures, averaged across seeds.
    collected: dict[str, list[float]] = {}
    for entry in results["local_only_per_client"]:
        for client in entry:
            collected.setdefault(str(client["client"]), []).append(float(client["macro_f1"]))
    per_client = {k: sum(v) / len(v) for k, v in collected.items()}
    local_only_f1 = list(per_client.values())

    # Section III-I2's client-to-global gap: positive => federation helped that client.
    gap = client_to_global_gap(per_client, federated["macro_f1_mean"])

    g4_bracket = {
        "local_only_range": [round(min(local_only_f1), 4), round(max(local_only_f1), 4)],
        "federated_macro_f1": round(federated["macro_f1_mean"], 4),
        "centralized_macro_f1": round(centralized, 4),
        "federation_beats_best_local_only": federated["macro_f1_mean"] > max(local_only_f1),
        "federation_beats_worst_local_only": federated["macro_f1_mean"] > min(local_only_f1),
        "federation_reaches_centralized": (
            federated["macro_f1_mean"] >= centralized - 2 * federated["macro_f1_std"]
        ),
        # Only meaningful when the three baselines got equal training budgets.
        "compute_matched": results.get("epoch_budget"),
    }

    return {
        "run_name": manifest["run_name"],
        "seeds": manifest["seeds"],
        "config": {
            "alpha": results["dirichlet_alpha"],
            "rounds": results["rounds"],
            "local_epochs": results["local_epochs"],
            "aggregation": results["aggregation"],
            "n_clients": results["n_clients"],
            "window": results["window"],
            "epoch_budget": results.get("epoch_budget"),
        },
        "baseline_3_centralized": summary["centralized"],
        "baseline_4_local_only": {"mean": summary["local_only"], "per_client": per_client},
        "baseline_5_federated_global": federated,
        "g4_bracket": g4_bracket,
        "bracket_verdict": results["bracket"],
        "client_to_global_gap": {k: round(v, 4) for k, v in gap.items()},
        "communication_cost": results["communication"],
        "per_client_class_histograms": [p["block_histograms"] for p in results["partitions"]],
        "convergence_by_seed": [
            d["per_round_macro_f1"] for d in results["per_seed"]["federated_detail"]
        ],
        "elapsed_seconds": results["elapsed_seconds"],
        "git_commit": manifest["hardware"].get("git_commit"),
        "source_manifest": "manifest_phase4_default.json",
    }


def main() -> int:
    manifest_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "artifacts/manifest_phase4_default.json"
    )
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path} yet -- run scripts/run_phase4.py first.")
        return 2

    summary = summarize(json.loads(manifest_path.read_text()))
    out_path = manifest_path.parent / "phase4_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"wrote {out_path}")
    central = summary["baseline_3_centralized"]
    print(
        f"  baseline 3 (centralised)      : "
        f"{central['macro_f1_mean']:.4f} +/- {central['macro_f1_std']:.4f}"
    )
    for client, macro_f1 in summary["baseline_4_local_only"]["per_client"].items():
        print(f"  baseline 4 (local client {client})   : {macro_f1:.4f}")
    federated = summary["baseline_5_federated_global"]
    print(
        f"  baseline 5 (federated)        : "
        f"{federated['macro_f1_mean']:.4f} +/- {federated['macro_f1_std']:.4f}"
    )
    print(f"  G4 bracket                    : {summary['bracket_verdict']}")
    print(f"  client-to-global gap          : {summary['client_to_global_gap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

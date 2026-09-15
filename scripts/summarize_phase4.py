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
    results = manifest["results"]
    local_only = results["local_only_summary"]
    federated = results["federated_summary"]
    centralized = results["centralized_reference_macro_f1"]

    # G4: does federation sit between local-only (lower bound) and centralised (upper bound)?
    local_only_f1 = [v["macro_f1_mean"] for v in local_only.values()]
    # Section III-I2: "the client-to-global gap" -- computed retroactively (flagged in
    # CHANGELOG.md): not produced by the original run_phase4.py, added while scoping Phase 7
    # after grepping the codebase and finding this metric had never been computed anywhere.
    gap = client_to_global_gap(
        {client: v["macro_f1_mean"] for client, v in local_only.items()},
        federated["macro_f1_mean"],
    )
    g4_bracket = {
        "local_only_range": [round(min(local_only_f1), 4), round(max(local_only_f1), 4)],
        "federated_macro_f1": round(federated["macro_f1_mean"], 4),
        "centralized_macro_f1": round(centralized, 4) if centralized is not None else None,
        "federation_beats_best_local_only": federated["macro_f1_mean"] > max(local_only_f1),
        "federation_beats_worst_local_only": federated["macro_f1_mean"] > min(local_only_f1),
        "federation_reaches_centralized": (
            centralized is not None
            and federated["macro_f1_mean"] >= centralized - 2 * federated["macro_f1_std"]
        ),
    }

    return {
        "run_name": manifest["run_name"],
        "seeds": manifest["seeds"],
        "config": {
            "alpha": results["alpha"],
            "rounds": results["rounds"],
            "local_epochs": results["local_epochs"],
            "aggregation": results["aggregation"],
            "local_only_epochs": results["local_only_epochs"],
        },
        "scope_note": results["scope_note"],
        "baseline_3_centralized": {"macro_f1": centralized},
        "baseline_4_local_only": local_only,
        "baseline_5_federated_global": federated,
        "g4_bracket": g4_bracket,
        "client_to_global_gap": {k: round(v, 4) for k, v in gap.items()},
        "communication_cost": {
            "measured_bytes_per_round": results["measured_bytes_per_round"],
            "theoretical_bytes_per_round": results["theoretical_bytes_per_round"],
        },
        "per_client_class_histograms": results["per_client_class_histograms"],
        "convergence_by_seed": results["federated_convergence_by_seed"],
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
    print(f"  baseline 3 (centralised)   : {summary['baseline_3_centralized']['macro_f1']}")
    for client, stats in summary["baseline_4_local_only"].items():
        print(f"  baseline 4 (local client {client}): {stats['macro_f1_mean']:.4f}")
    federated_f1 = summary["baseline_5_federated_global"]["macro_f1_mean"]
    print(f"  baseline 5 (federated)     : {federated_f1:.4f}")
    print(f"  G4 bracket                 : {summary['g4_bracket']}")
    print(f"  client-to-global gap       : {summary['client_to_global_gap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4 exit-criterion checker, written BEFORE any Phase 4 experiment was run.

This is deliberate, after the Phase 3 lesson: Phase 3 was first declared complete by checking
its phase-table summary ("full evaluation protocol... reported") rather than the paper's actual
Section III-I1 requirement (all five baselines). The fix there was `check_phase3_gate.py`,
written *after* the gap was found. This time the criterion is written first, against the
paper text below, so it cannot be quietly fitted to whatever a run happens to produce.

What Phase 4 actually requires, read from the source rather than a summary:

  * Work plan (Section III-J4): Phase 4's stated gate is "three clients with weighted FedAvg" --
    narrower than Phase 3's. The six ablations of Section III-I3 (alpha, E, weighted/unweighted,
    W, F, R) are evaluation REPORTING, not listed among the seven phase gates. This script
    checks the MINIMAL gate: baselines 4-5 exist and are reported once, at a representative
    config, over >= 3 seeds -- not the full sweep. See CHANGELOG.md for the scope decision.
  * Section III-I1: baseline 4 is three local-only GRUs (the lower bound an operator gets by
    declining to federate); baseline 5 is the federated global GRU. Together with baseline 3
    (centralised GRU, already reported in Phase 3) they answer gap G4.
  * Section III-F1 (G3): per-client class histograms published, so the partition is inspectable.
  * Eq. (22): communication bytes per round measured, not merely asserted.
  * Section III-I4: manifest with seeds, config, versions, git commit.

    PYTHONPATH=. ./.venv/bin/python scripts/check_phase4_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_METRICS = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")
MIN_SEEDS = 3
N_CLIENTS = 3


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def check(manifest: dict[str, Any]) -> bool:
    results = manifest.get("results", {})
    ok = True

    print("\nPhase 4 exit criterion (minimal gate: 'three clients with weighted FedAvg')\n")

    seeds = manifest.get("seeds", [])
    ok &= _check(f">= {MIN_SEEDS} seeds (III-I4)", len(seeds) >= MIN_SEEDS, f"found {seeds}")

    # --- Baseline 4: three local-only GRUs -----------------------------------------------
    local_only = results.get("local_only_summary", {})
    ok &= _check(
        f"baseline 4: {N_CLIENTS} local-only clients reported (III-I1)",
        len(local_only) == N_CLIENTS,
        f"found {len(local_only)} clients: {list(local_only)}",
    )
    for client_id, client_summary in local_only.items():
        for metric in REQUIRED_METRICS:
            present = f"{metric}_mean" in client_summary and f"{metric}_std" in client_summary
            ok &= _check(f"  client {client_id}: {metric} as mean +/- std", present)

    # --- Baseline 5: federated global GRU -------------------------------------------------
    federated = results.get("federated_summary", {})
    ok &= _check("baseline 5: federated global GRU reported (III-I1)", bool(federated))
    for metric in REQUIRED_METRICS:
        present = f"{metric}_mean" in federated and f"{metric}_std" in federated
        ok &= _check(f"federated global: {metric} as mean +/- std", present)

    federated_runs = results.get("federated_per_seed", [])
    ok &= _check(
        f"federated global ran on >= {MIN_SEEDS} seeds",
        len(federated_runs) >= MIN_SEEDS,
        f"found {len(federated_runs)}",
    )

    # --- G4: the bracket (Phase 3's centralised GRU sits above, local-only below) --------
    centralized_ref = results.get("centralized_reference_macro_f1")
    ok &= _check(
        "centralised-GRU reference recorded, to bracket G4",
        centralized_ref is not None,
        f"macro-F1 {centralized_ref}" if centralized_ref is not None else "missing",
    )

    # --- G3: per-client histograms ---------------------------------------------------------
    histograms = results.get("per_client_class_histograms", [])
    ok &= _check(
        "per-client class histograms published (G3, III-F1)",
        len(histograms) == N_CLIENTS,
        f"found {len(histograms)}",
    )

    # --- Eq. (22): measured, not merely asserted -------------------------------------------
    measured_bytes = results.get("measured_bytes_per_round")
    theoretical_bytes = results.get("theoretical_bytes_per_round")
    ok &= _check(
        "communication cost measured (Eq. 22)",
        measured_bytes is not None and theoretical_bytes is not None,
        f"measured {measured_bytes} vs theoretical {theoretical_bytes}"
        if measured_bytes is not None
        else "missing",
    )

    # --- Manifest provenance (III-I4) -------------------------------------------------------
    for field in ("config_snapshot", "library_versions"):
        ok &= _check(f"manifest carries {field} (III-I4)", bool(manifest.get(field)))
    ok &= _check(
        "manifest carries the git commit (III-I4)",
        bool(manifest.get("hardware", {}).get("git_commit"))
        and manifest["hardware"]["git_commit"] != "unavailable",
        f"commit {manifest.get('hardware', {}).get('git_commit', '?')[:12]}",
    )

    print(
        "\n  (NOT checked here, by scope decision -- see CHANGELOG.md: the alpha/E/"
        "weighted-vs-unweighted/R ablation sweep of Section III-I3. This script certifies "
        "the minimal gate only.)"
    )
    return bool(ok)


def main() -> int:
    default = Path("artifacts/manifest_phase4_default.json")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not path.exists():
        print(f"No manifest at {path}. Run scripts/run_phase4.py first.")
        return 2

    passed = check(json.loads(path.read_text()))
    print("\n" + "=" * 70)
    print("PHASE 4: MINIMAL GATE CLOSED" if passed else "PHASE 4: GATE OPEN")
    print("=" * 70)
    if passed:
        print(
            "\nNote: this certifies the minimal gate (baselines 4-5 reported once, >= 3\n"
            "seeds) was met, NOT that the full Section III-I3 ablation sweep is done, and\n"
            "NOT that federation outperforms the alternatives -- G4 explicitly allows for\n"
            "the opposite finding. Read the reported numbers for that."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

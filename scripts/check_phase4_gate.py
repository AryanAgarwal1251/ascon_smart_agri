"""Phase 4 exit-criterion checker -- the machine-checkable answer to "is Phase 4 done?".

Mirrors ``check_phase3_gate.py``: it reads a Phase 4 run manifest and checks it against the
criterion item by item, exiting non-zero if any item fails, so closing the phase is not a
judgement call.

    PYTHONPATH=. ./.venv/bin/python scripts/check_phase4_gate.py

Phase 4's criterion (README work plan, Section III-F): *weighted FedAvg with n_k = sequences;
the global model loads into every client; per-client class histograms published.* Each item
below, and why it is in the criterion:

  * >= 3 seeds, every headline number as mean +/- std   Section III-I4: a single-run number is
                                                        not a finding.
  * baselines 3-5 of Section III-I1                     3 and 4 bracket 5; without the bracket
                                                        the federated number means nothing (G4).
  * the full metric suite, never accuracy alone         Section III-I2: at IR ~ 5751 accuracy is
                                                        not evidence.
  * aggregation weighted, n_k = SEQUENCE counts         Eq. (21). Weighting by rows instead is
                                                        the silent bug `test_fedavg_weighting`
                                                        exists for; the manifest must show the
                                                        weights actually applied.
  * per-client class histograms, incl. zero counts      Section III-F1 / gap G3: an absent class
                                                        is the most informative thing a skewed
                                                        partition can report.
  * federated scaler == pooled scaler                   Eqs. (23-24). Checked on the real run,
                                                        not only in a unit test.
  * communication cost measured, not assumed            Eq. (22).
  * a manifest with seeds, config, versions and commit  Section III-I4.

Like the Phase 3 checker, it deliberately does NOT check that federation *worked well*. A
federated model that lands at the bottom of its bracket, properly measured and honestly
reported, closes this gate. A better one that is not measured this way does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_METRICS = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")
BRACKET_BASELINES = ("centralized", "local_only", "federated")
MIN_SEEDS = 3
# Chan's parallel formula is algebraically exact, so the gap is pure floating-point error.
# Anything above this means the federated path is not computing the pooled statistic.
MAX_SCALER_GAP = 1e-9


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def check(manifest: dict[str, Any]) -> bool:
    results = manifest.get("results", {})
    summary = results.get("summary", {})
    per_seed = results.get("per_seed", {})
    ok = True

    print("\nPhase 4 exit criterion (Section III-F / III-J4)\n")

    seeds = manifest.get("seeds", [])
    ok &= _check(
        f">= {MIN_SEEDS} seeds (III-I4)", len(seeds) >= MIN_SEEDS, f"found {len(seeds)}: {seeds}"
    )

    # ---- Baselines 3-5 and their metrics ------------------------------------------------
    for baseline in BRACKET_BASELINES:
        present = isinstance(summary.get(baseline), dict)
        ok &= _check(f"baseline '{baseline}' reported (III-I1)", present)
        if not present:
            continue
        # local_only is summarised by macro-F1 alone (its per-client detail carries the rest);
        # the two whole-population models must report the full suite.
        needed = ("macro_f1",) if baseline == "local_only" else REQUIRED_METRICS
        for metric in needed:
            ok &= _check(
                f"{baseline}.{metric} as mean +/- std",
                f"{metric}_mean" in summary[baseline] and f"{metric}_std" in summary[baseline],
            )
        runs = per_seed.get(baseline, [])
        ok &= _check(
            f"{baseline} ran on >= {MIN_SEEDS} seeds", len(runs) >= MIN_SEEDS, f"found {len(runs)}"
        )

    # Section III-I2: accuracy may appear, never alone.
    for baseline in ("centralized", "federated"):
        block = summary.get(baseline, {})
        if "accuracy_mean" in block:
            ok &= _check(
                f"{baseline}: accuracy is not reported alone (III-I2)",
                "macro_f1_mean" in block and "balanced_accuracy_mean" in block,
            )

    for baseline in ("centralized", "federated"):
        runs = per_seed.get(baseline, [])
        ok &= _check(
            f"{baseline} reports per-class F1 and a confusion matrix",
            bool(runs) and "per_class_f1" in runs[0] and "confusion" in runs[0],
        )

    # ---- Eq. (21): weighted FedAvg on SEQUENCE counts ------------------------------------
    aggregation = results.get("aggregation")
    ok &= _check(
        "aggregation is weighted FedAvg (Eq. 21)", aggregation == "weighted", f"got {aggregation!r}"
    )

    detail = results.get("per_seed", {}).get("federated_detail", [])
    partitions = results.get("partitions", [])
    ok &= _check(
        f"federated run recorded for >= {MIN_SEEDS} seeds",
        len(detail) >= MIN_SEEDS,
        f"found {len(detail)}",
    )

    n_clients = results.get("n_clients")
    weights_match = bool(detail) and bool(partitions)
    if weights_match:
        for run, partition in zip(detail, partitions, strict=False):
            applied = run.get("sequence_counts")
            expected = partition.get("sequence_counts")
            # The weights the server actually used must be the per-client SEQUENCE counts --
            # not row counts, which the partition records separately.
            if applied != expected or applied == partition.get("row_counts") != expected:
                weights_match = False
                break
    ok &= _check(
        "FedAvg weights are the per-client SEQUENCE counts, not row counts (Eq. 21)",
        weights_match,
        "manifest lacks the applied weights" if not detail or not partitions else "",
    )
    ok &= _check(
        "the global model reached every client",
        bool(detail) and all(len(r.get("sequence_counts", [])) == n_clients for r in detail),
        f"K={n_clients}",
    )

    # ---- Section III-F1 / gap G3: published partition ------------------------------------
    histograms_ok = bool(partitions)
    zero_counts_kept = False
    for partition in partitions:
        histograms = partition.get("block_histograms")
        if not isinstance(histograms, list) or len(histograms) != n_clients:
            histograms_ok = False
            break
        vocabularies = {frozenset(h) for h in histograms}
        # Every client must report every class, so an absent class shows as 0 rather than
        # vanishing from that client's histogram.
        if len(vocabularies) != 1:
            histograms_ok = False
            break
        if any(0 in h.values() for h in histograms):
            zero_counts_kept = True
    ok &= _check("per-client class histograms published (III-F1 / G3)", histograms_ok)
    # Informational, not gating: a run at high alpha can legitimately give every client every
    # class, so the absence of zero counts is not a failure.
    _check(
        "  (histograms retain zero counts where a client lacks a class)",
        zero_counts_kept,
        "" if zero_counts_kept else "no zero counts here -- expected only at high alpha",
    )

    # ---- Eqs. (23-24): the federated scaler equals the pooled one ------------------------
    gaps = [p.get("federated_vs_pooled_scaler_gap") for p in partitions]
    measured_gaps = [g for g in gaps if isinstance(g, int | float)]
    ok &= _check(
        "federated scaler == pooled scaler (Eqs. 23-24)",
        bool(measured_gaps) and max(measured_gaps) <= MAX_SCALER_GAP,
        f"max gap {max(measured_gaps):.2e}" if measured_gaps else "not recorded",
    )

    # ---- Eq. (22): measured communication cost -------------------------------------------
    communication = results.get("communication", {})
    measured = communication.get("measured_bytes_per_round")
    theoretical = communication.get("theoretical_bytes_per_round")
    ok &= _check(
        "communication cost measured against Eq. (22)",
        isinstance(measured, int) and isinstance(theoretical, int) and measured > 0,
        f"measured {measured:,} B/round vs theoretical {theoretical:,} B/round"
        if isinstance(measured, int) and isinstance(theoretical, int)
        else "not recorded",
    )

    # ---- Compute-matched bracket -----------------------------------------------------------
    # The bracket compares three training regimes, so it only measures the METHOD if all three
    # get the same number of passes over their data. An under-trained baseline 3 can land below
    # the federated model and make federation look like it beats pooling; that happened on the
    # first real seed of this phase (centralised 10 epochs vs federation's 20*3 = 60) and is
    # what this item exists to catch.
    budget = results.get("epoch_budget", {})
    matched = (
        isinstance(budget, dict)
        and len(
            {
                budget.get("federated_local_passes"),
                budget.get("local_only_epochs"),
                budget.get("centralized_epochs"),
            }
        )
        == 1
        and budget.get("centralized_epochs") is not None
    )
    ok &= _check(
        "baselines 3-5 are compute-matched (equal local passes)",
        matched,
        f"centralised {budget.get('centralized_epochs')} epochs, local-only "
        f"{budget.get('local_only_epochs')} epochs, federated "
        f"{budget.get('federated_local_passes')} local passes"
        if isinstance(budget, dict) and budget
        else "epoch_budget not recorded",
    )

    # ---- Gap G4: the bracket --------------------------------------------------------------
    bracket = results.get("bracket")
    ok &= _check(
        "local-only / federated / centralised bracket reported (G4)",
        isinstance(bracket, dict) and "federated_inside_bracket" in bracket,
        (
            f"local-only {bracket['local_only_macro_f1']:.4f} <= federated "
            f"{bracket['federated_macro_f1']:.4f} <= centralised "
            f"{bracket['centralized_macro_f1']:.4f}: "
            f"{'inside' if bracket['federated_inside_bracket'] else 'OUTSIDE'}"
            if isinstance(bracket, dict) and "local_only_macro_f1" in bracket
            else "missing"
        ),
    )

    # ---- Section III-I4: the manifest ------------------------------------------------------
    for field in ("config_snapshot", "library_versions"):
        ok &= _check(f"manifest carries {field} (III-I4)", bool(manifest.get(field)))
    commit = manifest.get("hardware", {}).get("git_commit", "")
    ok &= _check(
        "manifest carries the git commit (III-I4)",
        bool(commit) and commit != "unavailable",
        f"commit {commit[:12]}, dirty={manifest.get('hardware', {}).get('git_dirty')}",
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
    print("PHASE 4: GATE CLOSED -- Phase 5 may begin" if passed else "PHASE 4: GATE OPEN")
    print("=" * 70)
    if passed:
        print(
            "\nNote: this certifies the federated protocol was measured and reported as\n"
            "Section III-F requires, NOT that federation performed well. Read the bracket\n"
            "for that -- where the federated model sits between the local-only lower bound\n"
            "and the centralised upper bound is the finding."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

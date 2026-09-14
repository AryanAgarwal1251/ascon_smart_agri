"""Phase 3 exit-criterion checker -- the machine-checkable answer to "is Phase 3 done?".

Phase 3 is the HARD GATE of Section III-J4: Phases 4-7 may not begin until it closes. Whether
it has closed should not be anybody's judgement call, so this script reads the run manifest and
checks it against the criterion, item by item, exiting non-zero if any item fails.

    PYTHONPATH=. ./.venv/bin/python scripts/check_phase3_gate.py

What it checks, and why each item is in the criterion:

  * >= 3 seeds, every headline number as mean +/- std        Section III-I4: a single-run
                                                             number is not a finding.
  * the full metric suite, never accuracy alone              Section III-I2 / CLAUDE.md: at
                                                             IR ~ 5751 accuracy is not evidence.
  * baselines 1-3 of Section III-I1                          The RF baseline is the one that can
                                                             contradict the GRU (S13); the MLP
                                                             is what isolates recurrence.
  * the W=1 vs W=16 ablation                                 Section III-D: "if it matches
                                                             W = 16, recurrence has not earned
                                                             its place ... and we will say so".
  * a manifest with seeds, config, versions and commit       Section III-I4: no headline number
                                                             without a manifest.

It deliberately does NOT check that the detector is *good*. A poor result that is properly
measured and honestly reported closes this gate; a good result that is not does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_METRICS = ("macro_f1", "balanced_accuracy", "mcc", "accuracy")
REQUIRED_BASELINES = ("random_forest", "mlp", "gru")
REQUIRED_WINDOWS = ("1", "16")
MIN_SEEDS = 3


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def check(manifest: dict[str, Any]) -> bool:
    results = manifest.get("results", {})
    summary = results.get("summary", {})
    per_seed = results.get("per_seed", {})
    ok = True

    print("\nPhase 3 exit criterion (Section III-J4 hard gate)\n")

    seeds = manifest.get("seeds", [])
    ok &= _check(
        f">= {MIN_SEEDS} seeds (III-I4)",
        len(seeds) >= MIN_SEEDS,
        f"found {len(seeds)}: {seeds}",
    )

    for window in REQUIRED_WINDOWS:
        ok &= _check(f"window W={window} evaluated", window in summary)

    # A manifest from an earlier run shape has a FLAT summary (metric -> float) rather than one
    # nested by window. Detect that rather than crashing on it: a checker that dies on the very
    # manifests it is meant to reject is worse than useless.
    # Window keys are stringified integers; anything else at this level (e.g. an old flat
    # summary's "per_class_f1" map) is not a window and must not be treated as one.
    windowed = {k: v for k, v in summary.items() if k.isdigit() and isinstance(v, dict)}
    if summary and not windowed:
        _check(
            "summary is nested by window",
            False,
            "flat summary -- this manifest predates the baseline/ablation run",
        )
        ok = False

    for window in windowed:
        for baseline in REQUIRED_BASELINES:
            present = isinstance(windowed[window].get(baseline), dict)
            ok &= _check(f"W={window}: baseline '{baseline}' reported (III-I1)", present)
            if not present:
                continue
            for metric in REQUIRED_METRICS:
                has_mean = f"{metric}_mean" in windowed[window][baseline]
                has_std = f"{metric}_std" in windowed[window][baseline]
                ok &= _check(
                    f"W={window}: {baseline}.{metric} as mean +/- std",
                    has_mean and has_std,
                )
            runs = per_seed.get(window, {}).get(baseline, [])
            ok &= _check(
                f"W={window}: {baseline} ran on >= {MIN_SEEDS} seeds",
                len(runs) >= MIN_SEEDS,
                f"found {len(runs)}",
            )
            ok &= _check(
                f"W={window}: {baseline} reports per-class F1 and a confusion matrix",
                bool(runs) and "per_class_f1" in runs[0] and "confusion" in runs[0],
            )

    verdict = results.get("ablation_verdict")
    ok &= _check(
        "W=1 vs W=16 ablation answered (Section III-D)",
        isinstance(verdict, dict) and "recurrence_earned_its_place" in verdict,
        (
            f"gain {verdict['gain']:+.4f}, recurrence earned its place: "
            f"{verdict['recurrence_earned_its_place']}"
            if isinstance(verdict, dict) and "gain" in verdict
            else "missing"
        ),
    )

    for field in ("config_snapshot", "library_versions"):
        ok &= _check(f"manifest carries {field} (III-I4)", bool(manifest.get(field)))
    ok &= _check(
        "manifest carries the git commit (III-I4)",
        bool(manifest.get("hardware", {}).get("git_commit"))
        and manifest["hardware"]["git_commit"] != "unavailable",
        f"commit {manifest.get('hardware', {}).get('git_commit', '?')[:12]}, "
        f"dirty={manifest.get('hardware', {}).get('git_dirty')}",
    )

    deferred = results.get("baselines_deferred_to_phase4", [])
    print(f"\n  (baselines 4-5 correctly deferred to Phase 4: {deferred or 'NOT RECORDED'})")
    return bool(ok)


def main() -> int:
    default = Path("artifacts/manifest_phase3_complete_default.json")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if not path.exists():
        print(f"No manifest at {path}. Run scripts/run_phase3_complete.py first.")
        return 2

    passed = check(json.loads(path.read_text()))
    print("\n" + "=" * 70)
    print("PHASE 3: GATE CLOSED -- Phase 4 may begin" if passed else "PHASE 3: GATE OPEN")
    print("=" * 70)
    if passed:
        print(
            "\nNote: this certifies the evaluation protocol was followed, NOT that the\n"
            "detector performs well. Read the reported numbers for that."
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

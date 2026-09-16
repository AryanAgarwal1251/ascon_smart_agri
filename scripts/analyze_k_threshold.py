"""Client-count (K) threshold analysis for the federated GRU (Phase 4 extension).

NOT PART OF THE PAPER'S ABLATION SET, flagged per Golden Rule 1. The design paper fixes K = 3
(assumption A1, "Three edge nodes are simulated"; objective O3) and ``configs/base.py`` defines
``alpha_sweep``, ``e_sweep``, ``w_sweep`` and ``f_sweep`` but no ``k_sweep``. This module adds a
scalability question the paper does not ask: how many edge gateways can join one federation
before the detector degrades, and what sets that limit.

It answers it *without training anything* -- every figure here is exact algebra or a Monte-Carlo
over the real :func:`dirichlet_block_partition`. The companion ``scripts/run_k_sweep.py`` runs
the matching empirical sweep; this module's job is to predict what that sweep should find.

The quantity the analysis turns on is **omega_c(K)**, defined here and not in the paper: the
share of the weighted-FedAvg aggregate (Eq. 21) contributed by clients that hold at least one
block of class c. A client with no class-c data trains a local model whose class-weighted loss
only knows the classes it *does* hold, so it pushes class-c logits down; omega_c is therefore
the fraction of the averaged model that has actually seen the class.

Three results, in decreasing order of how much they can be trusted:

1. **Pigeonhole (exact, any alpha).** Blocks are indivisible -- ``data/split.make_blocks``
   allocates contiguous 256-row blocks and the partition never splits one -- so a class with
   B_c blocks reaches at most ``min(K, B_c)`` clients. Past K = B_c, extra clients necessarily
   hold none of it.
2. **Dilution (exact as alpha -> infinity).** With blocks split evenly,
   ``omega_c(K) = min(1, B_c / K)``: flat while K <= B_c, then decaying as 1/K, with the knee
   exactly at K = B_c. ``--verify`` checks this against the Monte-Carlo.
3. **Heterogeneity (simulated; no closed form).** At finite alpha there is no clean expression.
   The obvious Beta-marginal approximation ``1 - I_{1/B_c}(alpha+1, alpha(K-1))`` was tried and
   overestimates omega by up to 0.80, because the partition hands each client a *contiguous
   slice* of the shuffled block list (``cuts = cumsum(p) * len(blocks)``), which couples
   consecutive clients rather than leaving their shares independent. So finite-alpha figures are
   measured by running the real partition function, not derived.

Block counts are read from the Phase 4 manifest rather than typed in, so the analysis cannot
drift from the run it describes.

    PYTHONPATH=. ./.venv/bin/python -u scripts/analyze_k_threshold.py --verify
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from ascon_smart_agri.data.taxonomy import CLASS_NAMES
from ascon_smart_agri.federated.partition import dirichlet_block_partition
from ascon_smart_agri.model.gru import expected_param_count

# Eq. (22): each round ships theta down to every client and back up again.
BYTES_PER_PARAM = 4
# Paper abstract: pooling the same corpus centrally costs about 276 MB.
POOLING_MB = 276.0

K_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200)


def blocks_per_class(manifest_path: Path) -> dict[str, int]:
    """Per-class TRAIN block counts, summed from the Phase 4 per-client histograms (gap G3).

    The partition is a partition, so the per-client counts sum to the split's totals and are
    identical across seeds; that invariant is asserted rather than assumed.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = manifest["results"]["partitions"]
    if not partitions:
        raise ValueError(f"{manifest_path} records no partitions")

    totals: list[Counter[str]] = []
    for partition in partitions:
        counter: Counter[str] = Counter()
        for histogram in partition["block_histograms"]:
            counter.update({str(k): int(v) for k, v in histogram.items()})
        totals.append(counter)
    if any(counter != totals[0] for counter in totals[1:]):
        raise ValueError("per-seed block totals disagree; the partitions are not of one split")

    return {CLASS_NAMES[int(index)]: count for index, count in sorted(totals[0].items())}


def monte_carlo(labels: np.ndarray, K: int, alpha: float, trials: int) -> dict[str, object]:
    """Expected coverage statistics at (K, alpha), from the real partition function.

    Weights here are block counts, which stand in for the Eq. (21) sequence weights: blocks are
    equal-sized by construction, so n_k is proportional to the block count up to the windowing
    remainder. ``run_k_sweep.py`` recomputes omega from the true sequence weights.
    """
    n_classes = len(CLASS_NAMES)
    counts = np.zeros((trials, K, n_classes), dtype=np.int64)
    for trial in range(trials):
        assignment = dirichlet_block_partition(
            labels, n_clients=K, alpha=alpha, seed=10_000 + trial
        )
        for client, block_ids in enumerate(assignment):
            if block_ids:
                np.add.at(counts[trial, client], labels[np.asarray(block_ids)], 1)

    totals = counts.sum(axis=2)
    holds = counts > 0
    grand = totals.sum(axis=1, keepdims=True).astype(np.float64)
    weights = np.divide(totals, grand, out=np.zeros_like(totals, dtype=float), where=grand > 0)
    omega = (weights[:, :, None] * holds).sum(axis=1)

    return {
        "K": K,
        "alpha": alpha,
        "classes_present_per_client_mean": float(holds.sum(axis=2).mean()),
        "empty_clients_mean": float((totals == 0).sum(axis=1).mean()),
        "omega_mean": {name: float(omega[:, i].mean()) for i, name in enumerate(CLASS_NAMES)},
        "omega_macro": float(omega.mean()),
    }


def crossing(sweep: list[dict[str, object]], alpha: float, klass: str, floor: float) -> float:
    """Largest K whose omega_c clears ``floor``, interpolated between grid points."""
    points = [(int(r["K"]), r["omega_mean"][klass]) for r in sweep if r["alpha"] == alpha]  # type: ignore[index]
    for (k0, y0), (k1, y1) in pairwise(points):
        if y1 < floor <= y0:
            return k0 + (y0 - floor) / (y0 - y1) * (k1 - k0)
    return float("inf") if points[-1][1] >= floor else float(points[0][0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/manifest_phase4_default.json")
    parser.add_argument("--out", default="artifacts/k_threshold_analysis.json")
    parser.add_argument("--trials", type=int, default=240, help="Monte-Carlo draws per (K, alpha)")
    parser.add_argument("--alphas", default="0.1,0.5,100.0")
    parser.add_argument("--rounds", type=int, default=20, help="R, for the Eq. (22) total")
    parser.add_argument("--floor", type=float, default=0.5, help="omega floor defining K*")
    parser.add_argument("--verify", action="store_true", help="check the alpha -> inf closed form")
    args = parser.parse_args()

    per_class = blocks_per_class(Path(args.manifest))
    labels = np.concatenate(
        [np.full(n, CLASS_NAMES.index(c), dtype=np.int64) for c, n in per_class.items()]
    )
    rarest = min(per_class, key=lambda c: per_class[c])
    b_min = per_class[rarest]
    alphas = [float(a) for a in args.alphas.split(",")]

    print(
        f"train blocks {len(labels):,} across {len(per_class)} classes; "
        f"rarest = {rarest} at {b_min} blocks (<- the ceiling)"
    )
    print("  " + "  ".join(f"{c} {n}" for c, n in sorted(per_class.items(), key=lambda kv: -kv[1])))

    sweep = [monte_carlo(labels, K, a, args.trials) for a in alphas for K in K_GRID]

    params = expected_param_count(16, 96, len(CLASS_NAMES))
    per_round = 2 * params * BYTES_PER_PARAM
    break_even = POOLING_MB * 1e6 / (per_round * args.rounds)

    for alpha in alphas:
        print(f"\n--- alpha = {alpha} " + "-" * 56)
        print(f"{'K':>5} {'cls/client':>11} {'empty':>7} {f'w({rarest})':>16} {'w macro':>9}")
        for row in sweep:
            if row["alpha"] != alpha:
                continue
            print(
                f"{row['K']:>5} {row['classes_present_per_client_mean']:>11.2f} "
                f"{row['empty_clients_mean']:>7.2f} {row['omega_mean'][rarest]:>16.3f} "  # type: ignore[index]
                f"{row['omega_macro']:>9.3f}"
            )

    print(f"\nK* (largest K holding omega >= {args.floor} for {rarest}):")
    for alpha in alphas:
        print(f"  alpha = {alpha:<6} K* = {crossing(sweep, alpha, rarest, args.floor):.0f}")
    print(f"  closed form (alpha -> inf): K* = B_c / floor = {b_min / args.floor:.0f}")

    print(f"\nPigeonhole ceiling (exact, any alpha): K_hard = {b_min}")
    print(
        f"Communication (Eq. 22): {per_round:,} * K bytes/round; "
        f"{args.rounds} rounds equals the {POOLING_MB:.0f} MB pooling cost at K = {break_even:.1f}"
    )

    if args.verify:
        print(
            f"\nVerifying omega_c(K) = min(1, B_c/K) against Monte-Carlo at alpha = {max(alphas)}"
        )
        worst = 0.0
        for row in sweep:
            if row["alpha"] != max(alphas):
                continue
            for name, count in per_class.items():
                closed = min(1.0, count / int(row["K"]))
                worst = max(worst, abs(row["omega_mean"][name] - closed))  # type: ignore[index]
        print(f"  worst absolute error across all K and classes: {worst:.4f}")

    payload = {
        "note": "Not a paper ablation; K is fixed at 3 by assumption A1 / objective O3.",
        "source_manifest": args.manifest,
        "blocks_per_class": per_class,
        "total_blocks": len(labels),
        "rarest_class": rarest,
        "pigeonhole_ceiling": b_min,
        "omega_floor": args.floor,
        "k_star": {str(a): crossing(sweep, a, rarest, args.floor) for a in alphas},
        "k_star_closed_form_near_iid": b_min / args.floor,
        "bytes_per_round_per_client_pair": per_round,
        "communication_break_even_K": break_even,
        "monte_carlo_trials": args.trials,
        "sweep": sweep,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

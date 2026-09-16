"""Single-command entry point, one subcommand per phase (Section III-J4).

``asa <phase> [args...]`` runs that phase's driver under ``scripts/`` with the remaining
arguments passed through unchanged, so ``asa federate --rounds 2`` is
``scripts/run_phase4.py --rounds 2``. It needs a checkout: the drivers live beside the
package, not inside it.

Phases 2, 5 and 6 have no driver of their own and say so instead of pretending: Phase 2 is the
data pipeline every training driver rebuilds (or loads with ``--cache``), and Phases 5-6 are
library modules exercised by ``run-e2e`` and the test suite.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

# phase -> (help, driver script or None)
PHASES: dict[str, tuple[str, str | None]] = {
    "characterize": ("Phase 1: dataset characterisation report (III-B)", "run_phase1.py"),
    "preprocess": ("Phase 2: leakage-controlled preprocessing + feature selection", None),
    "train-centralized": (
        "Phase 3: centralised GRU with full evaluation (hard gate)",
        "run_phase3_complete.py",
    ),
    "federate": (
        "Phase 4: three-client federated simulation with weighted FedAvg",
        "run_phase4.py",
    ),
    "telemetry": ("Phase 5: telemetry simulation + feature-provenance adapter", None),
    "secure": ("Phase 6: Ascon integration + alerting path", None),
    "run-e2e": ("Phase 7: end-to-end integration", "run_phase7.py"),
}

LIBRARY_ONLY = {
    "preprocess": "run inside every training driver; `asa federate --save-cache x.npz` keeps it",
    "telemetry": "library module (telemetry/); exercised by `asa run-e2e`, tests/test_telemetry_*",
    "secure": "library module (crypto/, routing/); exercised by `asa run-e2e`, tests/test_ascon_*",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with one subcommand per phase."""
    parser = argparse.ArgumentParser(prog="asa", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="path to run config")
    sub = parser.add_subparsers(dest="phase", required=True)
    for name, (help_text, _) in PHASES.items():
        sub.add_parser(name, help=help_text, add_help=False)  # -h reaches the driver
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: dispatch to the phase's driver, passing unknown arguments through."""
    args, passthrough = build_parser().parse_known_args(argv)
    _, script = PHASES[args.phase]
    if script is None:
        print(
            f"asa {args.phase}: no standalone driver -- {LIBRARY_ONLY[args.phase]}", file=sys.stderr
        )
        return 2
    path = SCRIPTS / script
    if not path.exists():
        print(f"asa {args.phase}: driver {path} not found (run from a checkout)", file=sys.stderr)
        return 2
    # The drivers import `configs.base` from the repo root, exactly as `PYTHONPATH=.` provides.
    sys.path.insert(0, str(REPO_ROOT))
    sys.argv = [str(path), "--config", args.config, *passthrough]
    runpy.run_path(str(path), run_name="__main__")
    return 0

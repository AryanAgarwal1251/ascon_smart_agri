"""Single-command entry points, one subcommand per phase (Section III-J4).

Each phase is gated on the previous one's exit criterion. The subcommands are wired here so
the pipeline is reproducible from one binary (``asa <phase> --config configs/default.yaml``),
but the phase logic lives in the respective packages and is added phase by phase.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with one subcommand per phase."""
    parser = argparse.ArgumentParser(prog="asa", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="path to run config")
    sub = parser.add_subparsers(dest="phase", required=True)
    for name, help_text in [
        ("characterize", "Phase 1: dataset characterisation report (III-B)"),
        ("preprocess", "Phase 2: leakage-controlled preprocessing + feature selection"),
        ("train-centralized", "Phase 3: centralised GRU with full evaluation (hard gate)"),
        ("federate", "Phase 4: three-client federated simulation with weighted FedAvg"),
        ("telemetry", "Phase 5: telemetry simulation + feature-provenance adapter"),
        ("secure", "Phase 6: Ascon integration + alerting path"),
        ("run-e2e", "Phase 7: end-to-end integration"),
    ]:
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Dispatches to the selected phase (added phase by phase)."""
    args = build_parser().parse_args(argv)
    raise NotImplementedError(
        f"Phase '{args.phase}' is not implemented yet (scaffolding only; see README work plan)."
    )

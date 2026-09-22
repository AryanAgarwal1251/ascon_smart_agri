"""Phase 1: characterise the corpus and write the report (Section III-B).

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase1.py [--config configs/default.yaml]

Streams every CSV under ``data.dataset_root`` and writes
``<output_dir>/phase1_characterization_report_raw.json``: record count, per-column nulls and
infinities, zero-variance columns, exact duplicates, label vocabulary (empty for the official
raw distribution, whose label is implicit in the per-capture filename) and the imbalance ratio.
This is the script that produced the committed report; re-run it to check that file.

Multi-dataset (Phase 9 plan): ``--dataset ciciomt2024|edge_iiotset [--root PATH]`` runs the
same scan over a registered corpus (``data/datasets.py``), deriving labels the way that corpus
does, and writes ``<output_dir>/phase1_characterization_report_<name>.json``. The report's
``placeholder_collisions`` and ``label_counts`` are what the registry's column aliases and
taxonomy map must be reconciled against before that corpus is trained on.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from configs.base import load_run_config

from ascon_smart_agri.data.characterize import characterize_dataset
from ascon_smart_agri.data.datasets import discover_parts, get_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="", help="default: <output_dir>/phase1_..._raw.json")
    parser.add_argument("--dataset", default="", help="registered corpus name (data/datasets.py)")
    parser.add_argument("--root", default="", help="corpus root; default: the registry's")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()
    if args.dataset:
        spec = get_dataset(args.dataset)
        root = Path(args.root) if args.root else spec.default_root
        report = characterize_dataset(
            root,
            chunk_size=cfg.data.chunk_size,
            parts=discover_parts(spec, root),
            label_of=spec.label_of,
        )
        default_out = Path(cfg.output_dir) / f"phase1_characterization_report_{spec.name}.json"
    else:
        report = characterize_dataset(Path(cfg.data.dataset_root), chunk_size=cfg.data.chunk_size)
        default_out = Path(cfg.output_dir) / "phase1_characterization_report_raw.json"
    out = Path(args.out or default_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True) + "\n")
    print(
        f"[phase1] {report.n_records:,} records | {report.exact_duplicate_count:,} exact "
        f"duplicates | {len(report.label_counts)} labels | {time.time() - started:.0f}s -> {out}"
    )
    if report.placeholder_collisions:
        print(
            "[phase1] WARNING placeholder spellings collide in "
            f"{sorted(report.placeholder_collisions)} -- provenance leak; see data/datasets.py"
        )


if __name__ == "__main__":
    main()

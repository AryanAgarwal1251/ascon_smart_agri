"""Phase 1: characterise the corpus and write the report (Section III-B).

    PYTHONPATH=. ./.venv/bin/python -u scripts/run_phase1.py [--config configs/default.yaml]

Streams every CSV under ``data.dataset_root`` and writes
``<output_dir>/phase1_characterization_report_raw.json``: record count, per-column nulls and
infinities, zero-variance columns, exact duplicates, label vocabulary (empty for the official
raw distribution, whose label is implicit in the per-capture filename) and the imbalance ratio.
This is the script that produced the committed report; re-run it to check that file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from configs.base import load_run_config

from ascon_smart_agri.data.characterize import characterize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="", help="default: <output_dir>/phase1_..._raw.json")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    started = time.time()
    report = characterize_dataset(Path(cfg.data.dataset_root), chunk_size=cfg.data.chunk_size)
    out = Path(args.out or Path(cfg.output_dir) / "phase1_characterization_report_raw.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True) + "\n")
    print(
        f"[phase1] {report.n_records:,} records | {report.exact_duplicate_count:,} exact "
        f"duplicates | {len(report.label_counts)} labels | {time.time() - started:.0f}s -> {out}"
    )


if __name__ == "__main__":
    main()

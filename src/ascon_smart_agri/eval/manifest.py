"""Per-run manifest logger (Phase 1+, Section III-I4).

Every run writes a manifest capturing what is needed to reproduce and audit it: seeds, library
versions, hardware, the full hyperparameter/config snapshot, the subsample seed and resulting
per-class counts (III-B1), and --- critically --- the Ascon package NAME, VERSION, and VARIANT
string with its KAT-conformance result (III-G1, R5). No headline number is reported without a
manifest, and no single-run number is reported as a finding (III-I4).

TODO(Phase 1+): implement manifest assembly + write; capture git commit, torch/lib versions,
CPU/GPU info, and the Ascon variant record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    """The reproducibility record written once per run."""

    run_name: str
    seeds: list[int]
    library_versions: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, str] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    subsample_per_class_counts: dict[str, int] = field(default_factory=dict)
    # Ascon conformance record (III-G1): package, version, variant string, KAT pass/fail.
    ascon_backend: dict[str, str] = field(default_factory=dict)

    def write(self, output_dir: Path) -> Path:
        """Serialise the manifest into ``output_dir`` and return the written path."""
        del output_dir
        raise NotImplementedError("Phase 1+: manifest writing not implemented yet.")

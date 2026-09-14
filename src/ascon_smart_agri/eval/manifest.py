"""Per-run manifest logger (Phase 1+, Section III-I4).

Every run writes a manifest capturing what is needed to reproduce and audit it: seeds, library
versions, hardware, the full hyperparameter/config snapshot, the subsample seed and resulting
per-class counts (III-B1), and --- critically --- the Ascon package NAME, VERSION, and VARIANT
string with its KAT-conformance result (III-G1, R5). No headline number is reported without a
manifest, and no single-run number is reported as a finding (III-I4).

``collect_environment`` captures the git commit, library versions and CPU/platform facts so a
reported number can be traced to the tree and machine that produced it. The Ascon backend record
is passed in by the caller rather than imported here, so this module stays usable in phases where
the crypto path is not exercised (the field simply stays empty and the manifest says so).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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

    # Metrics reported as mean +/- std over the seeds above (III-I4); never a single run.
    results: dict[str, Any] = field(default_factory=dict)

    def write(self, output_dir: Path) -> Path:
        """Serialise the manifest into ``output_dir`` and return the written path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"manifest_{self.run_name}.json"
        # Trailing newline: manifests are committed (they are III-I4 provenance), so they must
        # satisfy the end-of-file-fixer pre-commit hook or every run dirties the tree.
        payload = json.dumps(asdict(self), indent=2, default=str) + "\n"
        path.write_text(payload, encoding="utf-8")
        return path


def _git_commit() -> str:
    """Return the current commit, or a marker when the tree is not a usable git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def _git_is_dirty() -> bool:
    """True when tracked files differ from the commit -- a reported number is then unpinned."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(result.stdout.strip())


def collect_environment() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(library_versions, hardware)`` for the manifest."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "sklearn", "scipy", "torch"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "unavailable"

    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "git_commit": _git_commit(),
        # A dirty tree means the commit does not fully identify the code that ran.
        "git_dirty": str(_git_is_dirty()),
    }
    return versions, hardware

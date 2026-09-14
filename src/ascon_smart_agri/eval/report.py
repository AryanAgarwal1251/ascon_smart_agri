"""Reporting, incl. the stated-expectation guard (Phase 3+, Section III-I5, risk R7).

Published CICIoT2023 results routinely exceed 98-99% accuracy; that is a known artifact of
the dataset's imbalance and the separability of the flooding classes, NOT evidence of a good
model. Reporting is built so that near-ceiling accuracy automatically emits a note directing
the reader to macro-F1 on the rare families and to FPR, where methods actually differ. The
expectation is committed in code/config comments BEFORE experiments run, to guard against
post-hoc rationalisation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_NEAR_CEILING_NOTE = (
    "NOTE (Section III-I5): accuracy {accuracy:.4f} is at or above the {threshold:.4f} "
    "near-ceiling threshold. On CICIoT2023 this is the expected artifact of class imbalance "
    "and the separability of the flooding classes, NOT evidence of a good detector. Read "
    "macro-F1 on the rare families and the false-positive rate instead; that is where methods "
    "actually differ."
)


def near_ceiling_note(accuracy: float, *, threshold: float) -> str | None:
    """Return the III-I5 caution note when accuracy is at/above ``threshold``, else ``None``."""
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be a fraction in [0, 1], got {accuracy}")
    if accuracy < threshold:
        return None
    return _NEAR_CEILING_NOTE.format(accuracy=accuracy, threshold=threshold)


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return ``(mean, population std)`` over seeds.

    Section III-I4 requires >= 3 seeds reported as mean +/- std; a single run is not a finding,
    so one value yields a std of 0.0 and must not be presented as if it were a spread.
    """
    if not values:
        raise ValueError("cannot summarise an empty sequence of seed results")
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    variance = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, math.sqrt(variance)


def format_seed_summary(name: str, values: Sequence[float]) -> str:
    """Format one metric across seeds as ``name: mean +/- std (n seeds)``."""
    mean, std = mean_std(values)
    return f"{name}: {mean:.4f} +/- {std:.4f} (n={len(values)})"

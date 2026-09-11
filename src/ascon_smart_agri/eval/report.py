"""Reporting, incl. the stated-expectation guard (Phase 3+, Section III-I5, risk R7).

Published CICIoT2023 results routinely exceed 98-99% accuracy; that is a known artifact of
the dataset's imbalance and the separability of the flooding classes, NOT evidence of a good
model. Reporting is built so that near-ceiling accuracy automatically emits a note directing
the reader to macro-F1 on the rare families and to FPR, where methods actually differ. The
expectation is committed in code/config comments BEFORE experiments run, to guard against
post-hoc rationalisation.

TODO(Phase 3+): implement formatted reports (mean +/- std over seeds) + the note emission.
"""

from __future__ import annotations


def near_ceiling_note(accuracy: float, *, threshold: float) -> str | None:
    """Return the III-I5 caution note when accuracy is at/above ``threshold``, else ``None``."""
    del accuracy, threshold
    raise NotImplementedError("Phase 3+: near-ceiling note not implemented yet.")

"""FedAvg weighting correctness (Section III-F2, Eq. 21).

The aggregation weight n_k must be the number of TRAINING SEQUENCES per client, not raw rows.
Using row counts over-weights clients whose data fragments into short runs (fewer sequences
per row, Eq. 12). This is a common, silent bug. Activates in Phase 4.
"""

from __future__ import annotations

import pytest

from ascon_smart_agri.federated.aggregation import unweighted_average, weighted_fedavg


@pytest.mark.skip(reason="pending Phase 4: aggregation not implemented yet")
def test_weight_uses_sequence_count_not_row_count() -> None:
    assert callable(weighted_fedavg)
    assert callable(unweighted_average)
    raise AssertionError("implement in Phase 4: assert weighting matches sequence counts")

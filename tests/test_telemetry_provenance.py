"""Unit tests for the feature-provenance adapter (Section III-H, gap G6).

The property that matters most: every feature the adapter hands out is traceable to a real,
held-out CICIoT2023 row and NEVER derived from the application-layer payload or invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ascon_smart_agri.data.scaling import fit_scaler
from ascon_smart_agri.telemetry.provenance import FeatureProvenanceAdapter, ProvenancedFeatures
from ascon_smart_agri.telemetry.simulate import simulate_stream


def _pool(n: int = 5, f: int = 3) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, f)), [f"file{i}.csv:{i}" for i in range(n)]


def test_returns_provenanced_features_traceable_to_a_real_row() -> None:
    features, refs = _pool()
    adapter = FeatureProvenanceAdapter(features, refs, seed=0)

    result = adapter.network_features_for(0)

    assert isinstance(result, ProvenancedFeatures)
    assert result.source_record_ref in refs
    assert result.origin == "held_out_ciciot2023_record"
    # The returned vector is EXACTLY one of the pool's rows, never a blend or invention.
    assert any(np.array_equal(result.features, row) for row in features)


def test_never_fabricates_a_feature_vector() -> None:
    """Every possible output must be byte-identical to one of the pool's rows."""
    features, refs = _pool(n=4)
    adapter = FeatureProvenanceAdapter(features, refs, seed=0)

    outputs = [adapter.network_features_for(i).features for i in range(20)]

    for out in outputs:
        assert any(np.array_equal(out, row) for row in features)


def test_assignment_is_a_deterministic_pure_function_of_the_counter() -> None:
    features, refs = _pool()
    adapter = FeatureProvenanceAdapter(features, refs, seed=7)

    first_call = adapter.network_features_for(3)
    second_call = adapter.network_features_for(3)  # same counter, called again

    assert first_call.source_record_ref == second_call.source_record_ref
    np.testing.assert_array_equal(first_call.features, second_call.features)


def test_different_seeds_give_different_assignment_orders() -> None:
    features, refs = _pool(n=10)

    a = FeatureProvenanceAdapter(features, refs, seed=1)
    b = FeatureProvenanceAdapter(features, refs, seed=2)

    refs_a = [a.network_features_for(i).source_record_ref for i in range(10)]
    refs_b = [b.network_features_for(i).source_record_ref for i in range(10)]
    assert refs_a != refs_b


def test_counter_cycles_when_it_exceeds_the_pool_size() -> None:
    features, refs = _pool(n=3)
    adapter = FeatureProvenanceAdapter(features, refs, seed=0)

    assert (
        adapter.network_features_for(0).source_record_ref
        == adapter.network_features_for(3).source_record_ref
    )
    assert (
        adapter.network_features_for(1).source_record_ref
        == adapter.network_features_for(4).source_record_ref
    )


def test_true_label_is_carried_but_never_shapes_the_feature_vector() -> None:
    features, refs = _pool(n=3)
    labels = ["Benign", "DDoS-TCP_Flood", "XSS"]
    adapter = FeatureProvenanceAdapter(features, refs, labels=labels, seed=0)

    result = adapter.network_features_for(0)

    assert result.true_label in labels
    assert result.features.shape == (3,)  # unchanged shape: label never appended to features


def test_true_label_defaults_to_none_when_not_supplied() -> None:
    features, refs = _pool()
    adapter = FeatureProvenanceAdapter(features, refs, seed=0)

    assert adapter.network_features_for(0).true_label is None


def test_rejects_mismatched_lengths_and_empty_pools() -> None:
    features, refs = _pool(n=5)

    with pytest.raises(ValueError, match="source_refs"):
        FeatureProvenanceAdapter(features, refs[:3], seed=0)
    with pytest.raises(ValueError, match="labels"):
        FeatureProvenanceAdapter(features, refs, labels=["only one"], seed=0)
    with pytest.raises(ValueError, match="zero held-out records"):
        FeatureProvenanceAdapter(np.empty((0, 3)), [], seed=0)


def test_rejects_a_negative_message_counter() -> None:
    features, refs = _pool()
    adapter = FeatureProvenanceAdapter(features, refs, seed=0)

    with pytest.raises(ValueError, match="non-negative"):
        adapter.network_features_for(-1)


def test_pairing_with_real_simulated_devices_gives_each_device_distinct_records() -> None:
    """Regression test for the exact bug found integrating this with simulate.py: pairing on
    TelemetryMessage.counter (which restarts at 0 per device) gave every device's message 0 the
    identical held-out record. Pairing on stream_index must not."""
    features, refs = _pool(n=20)
    adapter = FeatureProvenanceAdapter(features, refs, seed=0)
    messages = list(simulate_stream(["soil01", "soil02", "soil03"], n_messages=9, seed=0))

    # The wrong way (kept here as a demonstration of the bug, not a recommendation).
    wrong_refs = [adapter.network_features_for(m.counter).source_record_ref for m in messages]
    firsts_wrong = [wrong_refs[i] for i, m in enumerate(messages) if m.counter == 0]
    assert len(set(firsts_wrong)) == 1  # the bug: all three devices' message 0 collide

    # The correct way.
    right_refs = [adapter.network_features_for(m.stream_index).source_record_ref for m in messages]
    firsts_right = [right_refs[i] for i, m in enumerate(messages) if m.counter == 0]
    assert len(set(firsts_right)) == 3  # fixed: each device's message 0 gets its own record


# ---------------------------------------------------------------- from_held_out_frame


def _held_out_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0],
            "feat_b": [10.0, 20.0, 30.0, 40.0],
            "source_file": [
                "DDoS/part1.csv",
                "DDoS/part1.csv",
                "Benign/part1.csv",
                "XSS/part1.csv",
            ],
            "label": ["DDoS-TCP_Flood", "DDoS-TCP_Flood", "BenignTraffic", "XSS"],
        },
        index=[100, 101, 205, 310],  # simulates the pooled-corpus index dedup.py leaves behind
    )


def test_from_held_out_frame_builds_correct_provenance_refs() -> None:
    frame = _held_out_frame()
    scaler = fit_scaler(frame[["feat_a", "feat_b"]].to_numpy())

    adapter = FeatureProvenanceAdapter.from_held_out_frame(
        frame, feature_columns=["feat_a", "feat_b"], scaler=scaler, seed=0
    )

    all_refs = {adapter.network_features_for(i).source_record_ref for i in range(4)}
    assert all_refs == {
        "DDoS/part1.csv:100",
        "DDoS/part1.csv:101",
        "Benign/part1.csv:205",
        "XSS/part1.csv:310",
    }


def test_from_held_out_frame_scales_features_matching_training() -> None:
    frame = _held_out_frame()
    scaler = fit_scaler(frame[["feat_a", "feat_b"]].to_numpy())

    adapter = FeatureProvenanceAdapter.from_held_out_frame(
        frame, feature_columns=["feat_a", "feat_b"], scaler=scaler, seed=0
    )

    seen = np.stack([adapter.network_features_for(i).features for i in range(4)])
    # Standardised: mean ~0 across the pool, since the scaler was fit on this same frame.
    np.testing.assert_allclose(seen.mean(axis=0), 0.0, atol=1e-6)


def test_from_held_out_frame_drops_non_finite_rows_rather_than_imputing() -> None:
    frame = _held_out_frame()
    frame.loc[101, "feat_a"] = np.inf
    scaler = fit_scaler(frame[["feat_a", "feat_b"]].to_numpy()[np.isfinite(frame["feat_a"])])

    adapter = FeatureProvenanceAdapter.from_held_out_frame(
        frame, feature_columns=["feat_a", "feat_b"], scaler=scaler, seed=0
    )

    refs = {adapter.network_features_for(i).source_record_ref for i in range(3)}
    assert "DDoS/part1.csv:101" not in refs


def test_from_held_out_frame_requires_source_file_and_feature_columns() -> None:
    frame = _held_out_frame().drop(columns=["source_file"])
    scaler = fit_scaler(frame[["feat_a", "feat_b"]].to_numpy())

    with pytest.raises(ValueError, match="source_file"):
        FeatureProvenanceAdapter.from_held_out_frame(
            frame, feature_columns=["feat_a", "feat_b"], scaler=scaler, seed=0
        )


def test_from_held_out_frame_true_label_comes_from_the_label_column() -> None:
    frame = _held_out_frame()
    scaler = fit_scaler(frame[["feat_a", "feat_b"]].to_numpy())

    adapter = FeatureProvenanceAdapter.from_held_out_frame(
        frame, feature_columns=["feat_a", "feat_b"], scaler=scaler, seed=0
    )

    labels_seen = {adapter.network_features_for(i).true_label for i in range(4)}
    assert labels_seen == {"DDoS-TCP_Flood", "BenignTraffic", "XSS"}

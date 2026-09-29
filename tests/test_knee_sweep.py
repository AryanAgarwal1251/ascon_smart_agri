"""Stage 4's injected evaluator: the real validation macro-F1 curve (Section III-C).

``tests/test_feature_selection.py`` already pins the *seam* -- that ``FeatureSelector.fit``
consults an injected ``evaluator`` and takes the knee of whatever curve it returns, using a stub
that returns fixed numbers. These tests cover the thing that was missing for real: an evaluator
that actually trains a detector and scores it, so Stage 4 can stop falling back to the
configured F.

The leakage tests here are the important ones. Choosing F is a modelling decision, so scoring a
candidate against the held-out test set would be leakage of exactly the kind gate R3 exists to
prevent. ``make_knee_evaluator`` cannot reach the test set -- it is not a parameter -- and these
tests pin that structurally rather than trusting the docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ascon_smart_agri.data.split import make_blocks
from ascon_smart_agri.data.taxonomy import CLASS_NAMES
from ascon_smart_agri.eval.knee_sweep import make_knee_evaluator
from ascon_smart_agri.features.selection import FeatureSelector

WINDOW = 4


def sweep_frame(runs: int = 24, run_length: int = 24, seed: int = 0) -> pd.DataFrame:
    """Alternating benign/attack runs long enough to window, with a learnable signal.

    Runs alternate label so ``make_blocks`` produces many single-label blocks for the inner
    split to draw from, and each run is comfortably longer than ``WINDOW`` so windowing over
    contiguous same-label runs yields sequences on both sides of that split.
    """
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    for index in range(runs):
        leaf = "BenignTraffic" if index % 2 == 0 else "DDoS-ACK_Fragmentation"
        labels.extend([leaf] * run_length)
    n = len(labels)
    is_attack = np.array([label != "BenignTraffic" for label in labels], dtype=np.float64)
    return pd.DataFrame(
        {
            # Two genuinely informative columns, then pure noise -- so a curve over candidate F
            # has somewhere to rise and somewhere to flatten.
            "signal_a": is_attack + rng.normal(0, 0.05, n),
            "signal_b": is_attack * 2.0 + rng.normal(0, 0.05, n),
            "noise_a": rng.normal(0, 1.0, n),
            "noise_b": rng.normal(0, 1.0, n),
            "label": labels,
            "source_file": "synthetic.pcap.csv",
        }
    )


def build_evaluator(frame: pd.DataFrame, **overrides: object):  # type: ignore[no-untyped-def]
    """``make_knee_evaluator`` at test scale: one epoch, small net."""
    kwargs: dict[str, object] = {
        "window": WINDOW,
        "class_names": list(CLASS_NAMES),
        "n_classes": len(CLASS_NAMES),
        "epochs": 1,
        "hidden_size": 8,
        "batch_size": 64,
        "validation_fraction": 0.25,
        "seed": 0,
    }
    kwargs.update(overrides)
    return make_knee_evaluator(frame, make_blocks(frame, block_size=8), **kwargs)  # type: ignore[arg-type]


class TestEvaluatorProducesARealCurve:
    """The gap this closed: `f_sweep_scores` was empty on every run ever committed."""

    def test_scores_a_candidate_as_a_macro_f1_in_range(self) -> None:
        frame = sweep_frame()
        evaluator, _ = build_evaluator(frame)

        score = evaluator(["signal_a", "signal_b"])

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fit_records_a_non_empty_curve_and_selects_from_it(self) -> None:
        """End to end: the curve Stage 4 fell back from is now populated and used."""
        frame = sweep_frame()
        evaluator, _ = build_evaluator(frame)

        result = FeatureSelector().fit(
            frame.drop(columns=["label"]),
            frame["label"],
            tau=0.99,
            f=2,
            f_sweep=[1, 2, 3],
            evaluator=evaluator,
            sample_size=None,
        )

        assert result.f_sweep_scores, "Stage 4 recorded no curve; it fell back to configured F"
        assert result.selected_f in result.f_sweep_scores
        assert len(result.selected_columns) == result.selected_f

    def test_provenance_describes_how_the_curve_was_measured(self) -> None:
        """A curve nobody can interpret later is not a result; the manifest needs the how."""
        frame = sweep_frame()
        evaluator, provenance = build_evaluator(frame, epochs=2)

        evaluator(["signal_a", "signal_b"])

        assert provenance["epochs_per_candidate"] == 2
        assert provenance["window"] == WINDOW
        assert provenance["inner_train_rows"] > 0
        assert provenance["validation_rows"] > 0
        # Scores accumulate as candidates are scored, keyed by the F they were measured at.
        assert provenance["scores"] == {2: pytest.approx(evaluator(["signal_a", "signal_b"]))}


class TestLeakageContract:
    """Gate R3 applies to feature selection too: the test set must be unreachable."""

    def test_inner_split_partitions_the_training_rows_with_no_overlap(self) -> None:
        frame = sweep_frame()
        _, provenance = build_evaluator(frame)

        inner_train = provenance["inner_train_rows"]
        validation = provenance["validation_rows"]
        assert isinstance(inner_train, int) and isinstance(validation, int)
        # Every training row lands on exactly one side: a partition, not a resample.
        assert inner_train + validation == len(frame)
        assert inner_train > 0 and validation > 0

    def test_validation_rows_never_train_the_model_scoring_them(self) -> None:
        """Pinned by construction: the two frames the evaluator builds share no row index."""
        frame = sweep_frame()
        block_ids = make_blocks(frame, block_size=8)
        _, provenance = build_evaluator(frame)

        # Reconstruct the same inner split the evaluator drew and check disjointness.
        from ascon_smart_agri.data.split import stratified_block_split

        inner = stratified_block_split(
            frame,
            block_ids,
            test_fraction=0.25,
            seed=int(provenance["inner_split_seed"]),  # type: ignore[call-overload]
        )
        train_index = set(frame.loc[block_ids.isin(inner.train_blocks)].index)
        validation_index = set(frame.loc[block_ids.isin(inner.test_blocks)].index)
        assert not (train_index & validation_index)
        assert set(inner.train_blocks).isdisjoint(inner.test_blocks)

    def test_the_evaluator_has_no_parameter_that_could_carry_the_test_set(self) -> None:
        """Structural, not conventional: there is nowhere to pass test data in.

        ``make_knee_evaluator`` takes the training frame and its block ids only. If a future
        change adds a test-set parameter, this test fails and the reviewer has to justify it.
        """
        import inspect

        parameters = set(inspect.signature(make_knee_evaluator).parameters)
        forbidden = {name for name in parameters if "test" in name.lower()}
        assert not forbidden, f"make_knee_evaluator gained test-set parameters: {forbidden}"


class TestGuardsAgainstUnmeasurableCurves:
    """A candidate that cannot be scored must raise, never return a number that reads as bad."""

    def test_rejects_a_validation_fraction_outside_the_open_unit_interval(self) -> None:
        frame = sweep_frame()
        with pytest.raises(ValueError, match="validation_fraction"):
            build_evaluator(frame, validation_fraction=0.0)

    def test_rejects_a_non_positive_epoch_budget(self) -> None:
        frame = sweep_frame()
        with pytest.raises(ValueError, match="epochs"):
            build_evaluator(frame, epochs=0)

    def test_rejects_a_frame_whose_block_ids_do_not_line_up(self) -> None:
        frame = sweep_frame()
        with pytest.raises(ValueError, match="block_ids"):
            make_knee_evaluator(
                frame,
                make_blocks(frame, block_size=8).iloc[:-1],
                window=WINDOW,
                class_names=list(CLASS_NAMES),
                n_classes=len(CLASS_NAMES),
                epochs=1,
            )

    def test_raises_when_windowing_leaves_a_candidate_unscorable(self) -> None:
        """W longer than every run yields no sequences; 0.0 would read as 'measured and awful'."""
        frame = sweep_frame(runs=6, run_length=6)
        evaluator, _ = build_evaluator(frame, window=64)
        with pytest.raises(ValueError, match="cannot score this candidate"):
            evaluator(["signal_a", "signal_b"])


class TestDriverWiring:
    """The Phase 4 driver must actually inject the evaluator, not merely be able to.

    The original defect was not a missing evaluator -- ``make_knee_evaluator``'s seam existed
    from the start -- but that **no driver ever passed one**, so every committed manifest carried
    ``f_sweep_scores: {}`` and an F that came from the config rather than from a measured knee.
    These tests pin the wiring itself.
    """

    @staticmethod
    def run_pipeline(monkeypatch: pytest.MonkeyPatch, *, knee_sweep: bool):  # type: ignore[no-untyped-def]
        """Drive ``build_pipeline`` over a synthetic corpus, standing in for CICIoT2023."""
        from configs.base import load_run_config
        from scripts import run_phase4

        frame = sweep_frame(runs=40, run_length=30)
        monkeypatch.setattr(
            run_phase4,
            "stratified_capped_subsample",
            lambda *args, **kwargs: (frame, {"BenignTraffic": len(frame) // 2}),
        )
        cfg = load_run_config("configs/default.yaml")
        cfg.sequence.window = WINDOW
        cfg.model.hidden_size = 8
        cfg.features.f_sweep = [1, 2, 3]
        cfg.features.selected_f = 2
        cfg.features.selection_sample_size = None

        result = run_phase4.build_pipeline(cfg, "", "", knee_sweep=knee_sweep, knee_epochs=1)
        return result[11]  # the selection report

    def test_sweep_off_by_default_records_that_f_was_not_measured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default must stay bit-compatible with every committed manifest."""
        report = self.run_pipeline(monkeypatch, knee_sweep=False)

        assert report["knee_sweep"] == {"ran": False}
        assert report["f_sweep_scores"] == {}
        # F came from the config, and the manifest says so rather than implying a measurement.
        assert report["selected_f"] == report["configured_f"]

    def test_sweep_on_records_a_real_curve_in_the_manifest_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = self.run_pipeline(monkeypatch, knee_sweep=True)

        knee = report["knee_sweep"]
        assert isinstance(knee, dict)
        assert knee["ran"] is True
        assert report["f_sweep_scores"], "the driver did not inject the evaluator"
        # Keys are JSON-safe strings, since this goes straight into the run manifest.
        assert all(isinstance(key, str) for key in report["f_sweep_scores"])  # type: ignore[union-attr]
        assert knee["inner_train_rows"] > 0 and knee["validation_rows"] > 0
        assert report["selected_f"] in {int(k) for k in report["f_sweep_scores"]}  # type: ignore[union-attr]

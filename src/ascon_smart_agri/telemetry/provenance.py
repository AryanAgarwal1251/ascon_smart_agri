"""Feature-provenance adapter (Phase 5, Section III-H, gap G6).

DECLARED LIMITATION --- READ THIS BEFORE USING THE ADAPTER.

A model trained on CICIoT2023 flow-derived features cannot consume an application-layer JSON
telemetry payload: they are different objects with different origins. This adapter keeps the
two planes apart and records the provenance of every feature value the model consumes:

    * Application plane: the JSON payload (e.g. deviceId/temperature/soilMoisture). It is
      NEVER parsed into network features. IMPLEMENTATION DEVIATION FROM THE DESIGN PAPER
      (flagged per CLAUDE.md golden rule 1; see ``docs/design_paper.md``'s "Implementation
      deviation" section): this plane is no longer Ascon-protected on its way to the cloud --
      that confidentiality/integrity is now assumed to be handled by mechanisms outside this
      codebase. Ascon-AEAD128 protects Channel 3 (the federated weight transport,
      ``federated/crypto.py``) instead.
    * Network plane: a flow-feature vector drawn from HELD-OUT CICIoT2023 records never seen
      during training, standing in for the network conditions of that message. This is what
      the GRU classifies.

NO feature value is ever invented or synthesised to bridge the two planes. Two alternatives
were considered and rejected (III-H): synthesising features with a generative model (cannot
be verified; evaluates the detector against its own generator's artefacts) and instrumenting
a live MQTT broker (recovers only part of the feature set without the original extractor).

What a runtime demonstration built on this adapter DOES show: architectural correctness ---
the pipeline separates the two planes, routes on the verdict, and never places a
malicious-verdict payload on the cloud path. What it does NOT show: that a CICIoT2023-trained
model would detect attacks against a live agricultural MQTT deployment. That claim would
require capture and feature re-extraction on the target network, and is not made. This
limitation is also stated in the README.

Implementation notes:

* **LEAKAGE CONTRACT, same shape as ``data/scaling.py``'s: this class trusts the caller to
  pass genuinely held-out rows and cannot verify that from inside.** There is no way for the
  adapter to know, from a feature vector alone, whether a row was used in training. The
  constructor accepts any ``(features, source_refs)`` pair; correctness is the caller's
  responsibility. :meth:`from_held_out_frame` exists specifically to make the CORRECT
  construction the easy one: pass it the already-known Phase 2 test split (never trained on,
  by that phase's own R3 leakage gate) and it derives everything else -- selected columns,
  scaling, provenance refs -- from data the pipeline already produced, rather than requiring a
  fresh held-out pool to be carved out and re-verified by hand.
* **Provenance refs reuse `data/subsample.py`'s `source_file` column and the frame's own
  index**, which `data/dedup.py` resets to a clean contiguous range over the pooled corpus and
  `data/split.py` preserves through block-level filtering. A ref is therefore
  ``"<source_file>:<pooled_row_index>"`` -- exact and traceable back to one physical CSV row,
  with no new bookkeeping invented for this module.
* **Features are returned already selected and scaled**, matching training exactly. The
  adapter's whole purpose is to hand the classifier something it can consume without further
  processing; if scaling were deferred to the caller, a mismatch between training-time and
  runtime preprocessing (classic train/serve skew) becomes possible. Centralising it here
  removes that failure mode structurally rather than by convention.
* **`true_label` is demo/test-only and is never read by anything that classifies.** It lets a
  runtime demonstration deliberately construct a "this message pairs with a known-malicious
  record" scenario to show the routing behaves correctly (Phase 7), and lets tests assert the
  adapter is really drawing from held-out data with the label distribution expected. Nothing
  about the F=16 selected feature vector encodes it.
* **Assignment is a fixed, seeded permutation, cycling if the key exceeds the pool.**
  `network_features_for` is a pure function of its argument given the constructor's seed: the
  same key always returns the same held-out record, which is what makes a demo run
  reproducible. The permutation is built once at construction, not reshuffled per call.
* **Call `network_features_for` with a `TelemetryMessage`'s `stream_index`, never its
  `counter`.** `counter` restarts at 0 for every device (it is the crypto layer's per-device
  replay counter, Eq. 27); pairing on it would give every device's first message the identical
  held-out record, second message the identical next one, and so on. `stream_index` is
  monotonic across the whole emitted stream and is what actually differentiates messages here.
  See `telemetry/simulate.py`'s `TelemetryMessage` docstring for the full reasoning -- this was
  found and fixed by pairing a real simulated stream with a real held-out split and noticing
  every device's message 0 landed on the same network record.
* **This module returns ONE record's feature vector per call, not a (W, F) window.** Section
  III-H's own dataclass shape (`features: Array`, singular; `source_record_ref: str`, singular)
  matches this. Assembling a rolling window from successive calls for a given device is a
  runtime-pipeline concern for Phase 7's end-to-end integration
  (`sequences/windowing.py` already exists for the offline case); building that here would
  blur a phase boundary this project treats as load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._types import Array
from ..data.scaling import ScalerStats, apply_scaler


@dataclass(frozen=True)
class ProvenancedFeatures:
    """A network-feature vector plus the provenance of every value it contains."""

    features: Array  # from held-out CICIoT2023 records only; already selected + scaled
    source_record_ref: str  # e.g. "<file>:<row_index>" of the held-out record
    origin: str = "held_out_ciciot2023_record"  # never "synthesised" / "parsed_from_payload"
    # Demo/test-only: the record's true label, NEVER consumed by the classifier. See the
    # module docstring's note on true_label.
    true_label: str | None = None


class FeatureProvenanceAdapter:
    """Pairs a simulated application payload with a held-out network-feature vector.

    The adapter enforces the III-H boundary in code: it draws the network features only from
    held-out records and refuses to derive any feature from the JSON payload.
    """

    def __init__(
        self,
        features: Array,
        source_refs: list[str],
        *,
        labels: list[str] | None = None,
        seed: int = 0,
    ) -> None:
        if len(features) != len(source_refs):
            raise ValueError(
                f"features has {len(features)} rows but source_refs has {len(source_refs)}"
            )
        if labels is not None and len(labels) != len(features):
            raise ValueError(f"labels has {len(labels)} entries, expected {len(features)}")
        if len(features) == 0:
            raise ValueError("cannot build an adapter over zero held-out records")

        self._features = np.asarray(features)
        self._source_refs = source_refs
        self._labels = labels
        # Fixed once, not reshuffled per call -- see the module docstring.
        self._order = np.random.default_rng(seed).permutation(len(features))

    @classmethod
    def from_held_out_frame(
        cls,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        scaler: ScalerStats,
        seed: int = 0,
    ) -> FeatureProvenanceAdapter:
        """Build from a held-out frame carrying the columns ``data/subsample.py`` adds.

        Intended input: the Phase 2 TEST split (never trained on, by that phase's own R3
        leakage gate) -- passing it here needs no separate held-out pool to be carved out and
        re-verified. ``frame`` must carry a ``source_file`` column and its pooled-corpus index
        (both already present after ``data/dedup.py``/``data/split.py``). ``scaler`` must be
        the SAME fitted scaler used for training, so runtime features match training exactly.
        """
        missing = [c for c in (*feature_columns, "source_file") if c not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing required columns: {missing}")

        raw = frame[feature_columns].to_numpy(dtype=np.float64)
        finite = np.isfinite(raw).all(axis=1)
        if not finite.all():
            # Same policy as the training pipeline (data/scaling.py never imputes): drop
            # non-finite rows rather than inventing a value for them.
            raw = raw[finite]
        kept = frame.loc[finite] if not finite.all() else frame
        scaled = apply_scaler(raw, scaler).astype(np.float32)

        source_refs = [f"{row.source_file}:{idx}" for idx, row in kept.iterrows()]
        labels = list(kept["label"].astype(str)) if "label" in kept.columns else None
        return cls(scaled, source_refs, labels=labels, seed=seed)

    def network_features_for(self, stream_index: int) -> ProvenancedFeatures:
        """Return the held-out network-feature vector standing in for this message.

        Renamed from the scaffold's ``message_counter`` (flagged, Golden Rule 1): that name
        collides with ``TelemetryMessage.counter``, the crypto layer's PER-DEVICE replay
        counter, which restarts at 0 for every device and would silently pair every device's
        Nth message with the identical held-out record. Pass ``TelemetryMessage.stream_index``
        (monotonic across the whole stream) here instead -- see this module's docstring.
        """
        if stream_index < 0:
            raise ValueError(f"stream_index must be non-negative, got {stream_index}")

        index = int(self._order[stream_index % len(self._order)])
        return ProvenancedFeatures(
            features=self._features[index],
            source_record_ref=self._source_refs[index],
            true_label=self._labels[index] if self._labels is not None else None,
        )

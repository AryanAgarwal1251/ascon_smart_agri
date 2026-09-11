"""Feature-provenance adapter (Phase 5, Section III-H, gap G6).

DECLARED LIMITATION --- READ THIS BEFORE USING THE ADAPTER.

A model trained on CICIoT2023 flow-derived features cannot consume an application-layer JSON
telemetry payload: they are different objects with different origins. This adapter keeps the
two planes apart and records the provenance of every feature value the model consumes:

    * Application plane: the JSON payload (e.g. deviceId/temperature/soilMoisture). This is
      what Ascon protects. It is NEVER parsed into network features.
    * Network plane: a flow-feature vector drawn from HELD-OUT CICIoT2023 records never seen
      during training, standing in for the network conditions of that message. This is what
      the GRU classifies.

NO feature value is ever invented or synthesised to bridge the two planes. Two alternatives
were considered and rejected (III-H): synthesising features with a generative model (cannot
be verified; evaluates the detector against its own generator's artefacts) and instrumenting
a live MQTT broker (recovers only part of the feature set without the original extractor).

What a runtime demonstration built on this adapter DOES show: architectural correctness ---
the pipeline separates the two planes, routes on the verdict, encrypts/verifies correctly,
and never places a malicious-verdict payload on the cloud path. What it does NOT show: that a
CICIoT2023-trained model would detect attacks against a live agricultural MQTT deployment.
That claim would require capture and feature re-extraction on the target network, and is not
made. This limitation is also stated in the README.

TODO(Phase 5): implement the adapter pairing held-out network records to simulated messages,
with a provenance record per feature (source file, record index).
"""

from __future__ import annotations

from dataclasses import dataclass

from .._types import Array


@dataclass(frozen=True)
class ProvenancedFeatures:
    """A network-feature vector plus the provenance of every value it contains."""

    features: Array  # from held-out CICIoT2023 records only
    source_record_ref: str  # e.g. "<file>:<row_index>" of the held-out record
    origin: str = "held_out_ciciot2023_record"  # never "synthesised" / "parsed_from_payload"


class FeatureProvenanceAdapter:
    """Pairs a simulated application payload with a held-out network-feature vector.

    The adapter enforces the III-H boundary in code: it draws the network features only from
    held-out records and refuses to derive any feature from the JSON payload.
    """

    def network_features_for(self, message_counter: int) -> ProvenancedFeatures:
        """Return the held-out network-feature vector standing in for this message."""
        del message_counter
        raise NotImplementedError("Phase 5: provenance adapter not implemented yet.")

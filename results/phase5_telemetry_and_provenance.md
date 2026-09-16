# Phase 5 — Telemetry Simulation + Feature-Provenance Adapter

**Status: ✅ Done.** Code: [`telemetry/simulate.py`](../src/ascon_smart_agri/telemetry/simulate.py),
[`telemetry/provenance.py`](../src/ascon_smart_agri/telemetry/provenance.py). Tests:
[`tests/test_telemetry_simulate.py`](../tests/test_telemetry_simulate.py) (11),
[`tests/test_telemetry_provenance.py`](../tests/test_telemetry_provenance.py) (15).

## What the paper says

> A payload such as `{"deviceId":"soil01","temperature":24.8,"soilMoisture":42.5}` contains no
> flow-derived features at all... Each simulated message therefore carries two planes. The
> application plane is the JSON payload, and it is what Ascon protects. The network plane is a
> flow-feature vector drawn from **held-out CICIoT2023 records never seen during training**...
> No value is invented. — Section III-H (gap G6)

> "The runtime demonstration establishes architectural correctness... It does not show that a
> CICIoT2023-trained model would detect attacks against a live agricultural MQTT deployment."
> — Section III-H, the declared limitation

## What we achieved

**Two planes, kept structurally apart** — not just documented apart. `telemetry/simulate.py`
has no import of, or reference to, anything network-feature-related; `telemetry/provenance.py`
draws only from held-out rows and never reads a payload field. Neither module can accidentally
bridge the planes because neither has the other's data in scope.

**G6 verified on real data**, not only asserted: built the adapter from the actual 311,573-row
Phase 2 test split, paired it with a real simulated stream, and checked **200+ distinct
provenance references against the 1,237,958-row training index — zero leaked**.

**`FeatureProvenanceAdapter.from_held_out_frame`** makes the correct construction the easy one:
point it at the Phase 2 test split (already proven never-trained-on by the R3 gate) and it
derives provenance refs from `data/subsample.py`'s existing `source_file` column and the
frame's own pooled-corpus index — no new held-out pool invented, no new bookkeeping.

## A real bug found by running two modules together, not by either one's own tests

`TelemetryMessage` originally carried only `counter`, monotonic **per device** (correct for its
actual purpose — Eq. 27's replay-protection AD tuple scopes counters per device). But the
adapter's pairing function took a bare int with no documented distinction from that field —
pairing on `counter` gave **every device's message 0 the identical held-out network record**,
message 1 the identical next one, and so on. Only surfaced by running a real simulated stream
against a real adapter and inspecting the output; neither module's own unit tests, run in
isolation, could have caught it. Fixed by adding a second field, `stream_index` (monotonic
across the whole stream, never repeating), and renaming the adapter's parameter from the
scaffold's `message_counter` to `stream_index` so the correct call is the only obviously-named
one. A regression test demonstrates the bug directly, not only the fix.

## Decisions flagged, not silently made

- **`true_label` added to `ProvenancedFeatures`** — the held-out record's ground truth, for demo
  narration and test assertions only, never read by anything that classifies. The returned
  feature vector's shape is unchanged by its presence.
- **Runtime window assembly deliberately deferred**, not built here: the adapter returns one
  record's feature vector per call (matching the paper's own dataclass shape), and turning a
  stream of single vectors into `(W,F)` windows is Phase 7's job — see that file.

## What this phase does not claim

Exactly what Section III-H itself states: this demonstrates that the two-plane architecture
composes correctly, not that a CICIoT2023-trained model would detect attacks on a live
deployment. That claim would need capture and feature re-extraction on a target network, and is
not made anywhere in this project.

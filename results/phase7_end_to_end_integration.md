# Phase 7 — End-to-End Integration

**Status: ✅ Done.** Source: [`artifacts/phase7_results.json`](../artifacts/phase7_results.json),
[`manifest_phase7_train_default.json`](../artifacts/manifest_phase7_train_default.json),
[`manifest_phase7_e2e_default.json`](../artifacts/manifest_phase7_e2e_default.json). Scripts:
[`train_federated_model.py`](../scripts/train_federated_model.py), [`run_phase7.py`](../scripts/run_phase7.py).

## What the paper says

> "The training plane runs offline and periodically... The runtime plane runs continuously:
> telemetry arrives, is classified using the current global model, and is routed according to
> Equation (5)." — Section III-A

This is the first script in the project where that sentence is executed for real, one message
at a time, rather than each piece being proven correct in isolation.

## What we achieved

**A model was trained and saved for the first time in this project.** Every prior training run
(Phases 3 and 4) trained, reported metrics, and discarded the weights —
`model/checkpoint.py` (new this phase) closes that gap via safetensors, never pickle.

**Checkpoint result: macro-F1 0.8338, bit-for-bit identical to Phase 4's original seed-0 run**
(`0.8338014523294024` in both, difference exactly `0.0`) — confirming the training path is
genuinely deterministic end to end, verified against Phase 4's own recorded value rather than a
hand-typed constant (see the bug note below).

**The full runtime loop, assembled and run for real** (100 simulated messages, 3 devices,
W=16): `simulate_stream` → `FeatureProvenanceAdapter` (G6) → `DeviceWindowBuffer` (new this
phase) → the real saved model → Eq. (5)'s `y > 0` projection → `VerdictRouter` → real
`AsconAEAD128`/`MockCloudReceiver` or `AlertSink`.

| | Count |
| --- | --- |
| Messages simulated | 100 |
| Still buffering (< W messages for that device) | 45 — correct, no window was padded to fake completion |
| Classified | 55 (0 benign, 55 malicious) |
| Cloud received | 0 (exactly matches benign-classified) |
| Alert count | 55 (exactly matches malicious-classified) |
| **Malicious-verdict messages reaching the cloud** | **0 — G1 holds in the fully assembled loop** |

**Per-stage median latency:** provenance lookup 3.6 µs, window buffering 10.5 µs, model
inference 273.7 µs, routing (encrypt+send or alert) 4.1 µs — inference dominates, as expected
for a GRU forward pass against dict lookups and byte serialisation.

## A real, verified finding: why this run classified zero benign messages

Not a bug. The held-out pool's true benign rate is **4.52%**, and the seeded draw for this run's
classified range (stream positions 45–99) landed **exactly 2** true-benign records — almost
exactly the 4.52% × 55 ≈ 2.5 expected by chance. Both of those 2 were (informally) misclassified
as malicious — 53/55 = 96.4% informal agreement with ground truth overall, **not a formal
evaluation** (see the module's own disclaimer), and n=2 is far too small a sample to be a new
finding about the model. It is a small-sample echo of the already-documented FPR limitation
(0.2975, Phase 4), not new evidence. Phase 6's own integration test already proved benign
routing works correctly with real crypto (8 of 200 messages there were genuinely benign and all
8 round-tripped correctly) — this run simply didn't draw one in its classified range.

## A real bug in the verification script itself, caught and fixed

`scripts/summarize_phase7.py` first compared the checkpoint's macro-F1 against a hand-typed
`0.8338` (the value as printed, truncated to 4 decimals) with a `1e-6` tolerance, and reported
**"reproduces Phase 4 seed 0 exactly: False"** for a run that was in fact bit-for-bit identical
— the true value differs from the truncated constant by `1.45e-6`, just over the tolerance.
Fixed to read Phase 4's actual recorded seed-0 value from its own manifest and compare for exact
equality, instead of trusting a copied-in digit string to stay in sync with the real number.

## An honest caveat about this checkpoint's vintage

The deployed checkpoint (`git commit 1cdfa1a`) was trained **before** the client-seeding fix
described in [phase4_federated_learning.md](phase4_federated_learning.md) landed — it used the
original per-round seeding (client_id only), not the corrected `(seed, client_id, round)` mix.
Its macro-F1 (0.8338) is real and was independently reproduced, but it is not from the exact
same code path as Phase 4's now-corrected federated numbers. The difference is expected to be
small (the fix changes shuffle order within an otherwise-correct Algorithm 1, not its
correctness), but re-training the deployed checkpoint against the reconciled codebase is
recorded here as follow-on work rather than silently assumed equivalent.

## Decisions flagged, not silently made

- **Deployment checkpoint trains one seed, not a fresh 3-seed statistical run.** Phase 4 already
  established the statistical claim; Phase 7 needs a real artifact to deploy, not a second
  validation.
- **Window assembly lives here, not in Phase 5's adapter** — a device's buffer slides by one
  once full; nothing is ever padded to manufacture a window before real data fills it.

## What this phase does not claim

Exactly Section III-H's own limitation, inherited: this demonstrates the pipeline composes —
planes stay separate, routing is correct, crypto verifies — not that this model would detect
attacks against a live agricultural deployment.

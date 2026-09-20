# Phase 7 — End-to-End Integration

**Status: ✅ Done.** Source: [`artifacts/phase7_results.json`](../artifacts/phase7_results.json),
[`manifest_phase7_train_default.json`](../artifacts/manifest_phase7_train_default.json),
[`manifest_phase7_e2e_default.json`](../artifacts/manifest_phase7_e2e_default.json). Scripts:
[`train_federated_model.py`](../scripts/train_federated_model.py), [`run_phase7.py`](../scripts/run_phase7.py).

> **Implementation deviation from the design paper (user-approved, 2026-09-20).** The runtime
> cloud leg is now plaintext — the benign-verdict payload is sent to the mock cloud receiver
> without Ascon (cloud-payload protection is assumed handled outside this codebase). Ascon now
> protects the *training-plane* weight transport instead ([`federated/crypto.py`](../src/ascon_smart_agri/federated/crypto.py)),
> which is exercised by `train_federated_model.py`, not by this runtime demo. See the design
> paper's [Implementation Deviation section](../docs/design_paper.md). The routing counts and
> macro-F1 below are unaffected: classification and path selection are unchanged, and the
> weight-transport encryption is lossless.

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

**Checkpoint result: macro-F1 0.8507, bit-for-bit identical to seed 0 of Phase 4's
compute-matched run** (`0.8507461612861715` in both, difference exactly `0.0`; balanced accuracy
0.8528, MCC 0.9021). The training script reuses Phase 4's own pipeline, block ids, Dirichlet
draw and federated scaler, so the checkpoint is that run's seed-0 model, not a re-creation of
it — verified against Phase 4's recorded value rather than a hand-typed constant (see the bug
note below). Training took 18.6 minutes from the Phase 4 cache. The 3-seed claim it is reported
against, 0.8308 ± 0.0150, is read from `manifest_phase4_default.json` at run time.

**The full runtime loop, assembled and run for real** (100 simulated messages, 3 devices,
W=16): `simulate_stream` → `FeatureProvenanceAdapter` (G6) → `DeviceWindowBuffer` (new this
phase) → the real saved model → Eq. (5)'s `y > 0` projection → `VerdictRouter` → plaintext
`MockCloudReceiver` or `AlertSink` (per the deviation, the cloud leg no longer runs `AsconAEAD128`).

| | Count |
| --- | --- |
| Messages simulated | 100 |
| Still buffering (< W messages for that device) | 45 — correct, no window was padded to fake completion |
| Classified | 55 (0 benign, 55 malicious) |
| Cloud received | 0 (exactly matches benign-classified) |
| Alert count | 55 (exactly matches malicious-classified) |
| **Malicious-verdict messages reaching the cloud** | **0 — G1 holds in the fully assembled loop** |

**Per-stage median latency** (from the pre-deviation run; the routing stage then still included
the benign-path encrypt): provenance lookup 1.2 µs, window buffering 3.9 µs, model inference
112.0 µs, routing 2.4 µs — inference dominates. With the deviation the routing stage now does a
plaintext send on the benign branch (no encrypt), so its latency can only fall; the figure will
refresh on the next real run (see the note at the foot of this file).

## A real, verified finding: why this run classified zero benign messages

Not a bug. The held-out pool's true benign rate is **4.52%**, and the seeded draw for this run's
classified range (stream positions 45–99) landed **exactly 2** true-benign records — almost
exactly the 4.52% × 55 ≈ 2.5 expected by chance. Both of those 2 were (informally) misclassified
as malicious — 53/55 = 96.4% informal agreement with ground truth overall, **not a formal
evaluation** (see the module's own disclaimer), and n=2 is far too small a sample to be a new
finding about the model. It is a small-sample echo of the already-documented FPR limitation
(0.2975, Phase 4), not new evidence. Phase 6's own integration test already proved benign
routing reaches the cloud correctly (8 of 200 messages there were genuinely benign and all 8
were delivered) — this run simply didn't draw one in its classified range. (Under the deviation
the benign path is plaintext, so "delivered" no longer means "decrypted"; the routing behaviour
is otherwise identical.)

## A real bug in the verification script itself, caught and fixed

`scripts/summarize_phase7.py` first compared the checkpoint's macro-F1 against a hand-typed
`0.8338` (the value as printed, truncated to 4 decimals) with a `1e-6` tolerance, and reported
**"reproduces Phase 4 seed 0 exactly: False"** for a run that was in fact bit-for-bit identical
— the true value differs from the truncated constant by `1.45e-6`, just over the tolerance.
Fixed to read Phase 4's actual recorded seed-0 value from its own manifest and compare for exact
equality, instead of trusting a copied-in digit string to stay in sync with the real number.

## The checkpoint's vintage, and the two bugs behind the first one

The first deployed checkpoint (macro-F1 0.8338, `git commit 1cdfa1a`) was trained before the
client-seeding fix and before the two Phase 4 drivers were reconciled, and was recorded here as
follow-on work. Re-training it against the reconciled codebase exposed two problems in
`train_federated_model.py` that the earlier caveat had not seen:

- **It did not train on Phase 4's partition.** Despite an "IDENTICAL partition" comment, it
  re-cut blocks from the train labels (4,842 blocks, client sizes 1164/1849/1829) instead of
  reading the per-row block ids Phase 4 uses (4,854 blocks, 1166/1855/1833), so the Dirichlet
  draw differed.
- **Given the Phase 4 cache, it fed the GRU unscaled features.** The cache stores raw values
  (scaling is the federated step, Eqs. 23-24); the script assumed pre-scaled ones. At a smoke
  budget of R=2, E=1 the same seed scored 0.50 before the fix and 0.71 after.

Both are fixed by reusing `run_phase4.py`'s `build_pipeline`, the shared `block_strata`, and the
federated scaler. The regenerated checkpoint's sequence counts (284,500 / 458,447 / 452,390)
and macro-F1 match Phase 4's seed 0 exactly. The end-to-end run on it drew the same seeded
message stream and produced identical routing counts (0 benign, 55 malicious, 53/55 informal
agreement); only the per-stage latencies moved, and those are wall-clock noise.

## Decisions flagged, not silently made

- **Deployment checkpoint trains one seed, not a fresh 3-seed statistical run.** Phase 4 already
  established the statistical claim; Phase 7 needs a real artifact to deploy, not a second
  validation.
- **Window assembly lives here, not in Phase 5's adapter** — a device's buffer slides by one
  once full; nothing is ever padded to manufacture a window before real data fills it.

## What this phase does not claim

Exactly Section III-H's own limitation, inherited: this demonstrates the pipeline composes —
planes stay separate, routing is correct — not that this model would detect attacks against a
live agricultural deployment. (Under the deviation the runtime cloud leg is plaintext, so this
run no longer demonstrates AEAD verification; that guarantee now lives on the training-plane
weight transport, exercised by `train_federated_model.py` and covered by
[Phase 6](phase6_ascon_and_alerting.md).)

---

*Re-run note (2026-09-20 deviation): the checkpoint and both Phase 7 manifests here were produced
by the pre-deviation code. Re-running `train_federated_model.py` (now with the Ascon-encrypted
weight transport) and `run_phase7.py` (now with a plaintext cloud leg) requires the raw
CICIoT2023 corpus, which is not present in this environment, so the committed artifacts have not
been regenerated. The checkpoint's macro-F1 (0.8507) is provably unchanged — the weight
encryption is lossless — and the routing counts are unchanged; only the routing-stage latency and
the manifests' `ascon_backend` field will refresh on the next real run. Commands are in the
[results README](README.md).*

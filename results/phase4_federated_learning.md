# Phase 4 — Three-Client Federated Simulation

**Status: ✅ Done — gate closed.** Verified by [`scripts/check_phase4_gate.py`](../scripts/check_phase4_gate.py)
(exit 0). Source: [`artifacts/phase4_results.json`](../artifacts/phase4_results.json),
[`artifacts/manifest_phase4_default.json`](../artifacts/manifest_phase4_default.json). Code:
[`federated/`](../src/ascon_smart_agri/federated/), [`eval/baselines.py`](../src/ascon_smart_agri/eval/baselines.py).

> **Implementation deviation from the design paper (user-approved, 2026-09-20).** The federated
> weight transport is now Ascon-AEAD128 encrypted on both legs of every round (see
> [`federated/crypto.py`](../src/ascon_smart_agri/federated/crypto.py) and the design paper's
> [Implementation Deviation section](../docs/design_paper.md)). The encryption is **lossless**, so
> every detection-quality figure in this report is unchanged; only the communication cost gains a
> constant per-round overhead, detailed below.

## What the paper says

> Clients are formed by a **Dirichlet partition** applied at block level, α treated as an
> experimental variable; per-client class histograms are published. — Section III-F1, Eq. (20)

> Aggregation weight `n_k` is the number of training **sequences** held by client k, not raw
> rows — "using row counts would systematically over-weight clients whose data happens to be
> fragmented into short runs." — Section III-F2, Eq. (21)

> "R4 admits the possibility that federation underperforms the local-only baseline under strong
> heterogeneity. That would still be a legitimate finding... we will report it as such rather
> than tuning until it disappears." — Section III-J

> Communication: `B_round = 2K|θ|b`. For K=3, |θ|=33,800, b=4: **811 kB ≈ 0.77 MiB/round**,
> ~15.5 MiB over 20 rounds, against ~276 MB to pool the same data centrally. — Eq. (22)

## What we achieved — the corrected, compute-matched result

**This section reports the reconciled figures, not this project's first attempt.** See
"A real correction" below for why.

At α=0.5, R=20, E=3, weighted FedAvg, **every baseline trained on the same 60-epoch-equivalent
budget**, 3 seeds:

| Baseline | macro-F1 |
| --- | --- |
| Local-only (mean across 3 clients) | 0.7205 ± 0.0750 (range 0.6743–0.7586) |
| **Federated global** | **0.8308 ± 0.0150** |
| Centralised (upper bound) | 0.8543 ± 0.0040 |

**Federation sits properly between the two bounds, recovering 82.4% of the achievable gap**
(`fraction_of_gap_recovered: 0.8239`) — the textbook G4 relationship the paper describes, not
the bracket-inverted result this project reported at first.

**Client-to-global gap, positive for every client** (federation helps all three relative to
going it alone): client 0 **+0.1023**, client 1 **+0.1565**, client 2 **+0.0721**.

**Communication cost matches Eq. (22) almost exactly**: the parameter traffic is 815,136
bytes/round against a theoretical 811,200 (the small excess is the safetensors container's own
header) — **0.78 MiB/round**, matching the paper's stated 0.77. The 815,136 figure is the
parameter-only cost from the pre-deviation run (`manifest_phase4_default.json`), unchanged by the
encryption because the tensor bytes are identical.

**Weight-encryption overhead (implementation deviation).** Each round now carries a 16-byte nonce
plus a 16-byte tag on each of the two legs per client — `2K·32 = 192` bytes for K=3
(`aead_overhead_bytes_per_round(3)`, deterministic, not measured). Total: **815,328 bytes/round**,
a **0.024%** increase — negligible against the ~0.77 MiB the parameters themselves cost, exactly
as expected for a fixed AEAD overhead on a ~132 KiB payload (see
[Phase 6](phase6_ascon_and_alerting.md)'s expansion table). *The end-to-end re-run that would
regenerate `manifest_phase4_default.json` with the crypto path wired in has not been executed in
this environment (the raw CICIoT2023 corpus is not present here); the +192 B delta is exact and
the detection numbers are provably unchanged by lossless encryption, but the manifest's own
`measured_bytes_per_round` and populated `ascon_backend` fields will only refresh on the next real
run — command in the [results README](README.md).*

**Binary FPR: 0.2975** (derivable from the stored per-seed confusion matrices;
population-std convention, matching every other σ figure in this project, gives ±0.0418) — too
high for deployment, and read first per Section III-I2's own instruction to weight FPR over
accuracy.

## A real correction, on the record

This project's first Phase 4 run compared the federated model (trained for R×E = 60 total local
passes) against a centralised reference **borrowed from Phase 3 at only 10 epochs**, and a
local-only baseline also capped at 10. That is not a fair comparison — of course a model that
saw 6× more total training does better, independent of whether that training was federated or
centralised. The result was a bracket that looked *too* good (federation appearing to beat
centralised on every seed), and the unequal budget is why.

A teammate (`AdvaitMalviya`) built an independent, parallel Phase 4 implementation from the same
starting commit, caught this, and reran all three baselines at a matched 60-epoch budget. The
two implementations were **reconciled by merge**, not by discarding either side — the original
441-minute run is kept at
[`artifacts/manifest_phase4_minimal_gate.json`](../artifacts/manifest_phase4_minimal_gate.json)
for the record, its bracket explicitly marked inverted and why.

## A second real bug the reconciliation fixed

`federated/client.py`'s per-round local-training seed was originally just `client_id` — a
constant across all 20 rounds. That meant every round after the first replayed the *identical*
batch shuffle, quietly narrowing the variance the ≥3-seed protocol (Section III-I4) is supposed
to report. Fixed to mix `(run seed, client id, round index)` via `numpy.random.SeedSequence`.

## Decisions flagged, not silently made

- **Replay/scaler/aggregation invariants were checked on real data, not only unit tests.** The
  federated scaler path (`local_sufficient_stats` + Chan's combination, Eqs. 23–24) was verified
  against the actual α=0.5 partition over raw, unscaled features: per-client means genuinely
  differ (e.g. `Tot sum`: 25,184 / 30,640 / 20,991 across clients), yet the combination
  reproduces the pooled fit to **2×10⁻¹² relative difference**.
- **FedProx exists, wired in, never exercised for real.** `fedprox_proximal_term` is correctly
  integrated into the training loop (verified: constrains drift to 16% of unconstrained), but
  α=0.5 never diverged, so the R4 fallback has not needed to fire in practice.

## What this phase does not claim

The full Section III-I3 ablation sweep (α ∈ {0.1, 0.5, 100}, E ∈ {1,3,5}, weighted vs.
unweighted, R up to 20) is not run — one configuration alone costs multiple hours, and it is
deferred as separate, larger follow-on work rather than attempted piecemeal.

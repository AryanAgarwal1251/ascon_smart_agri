# Phase 6 — Ascon Integration + Alerting Path

**Status: ✅ Done.** Gating tests: [`tests/test_ascon_kat.py`](../tests/test_ascon_kat.py),
[`test_ascon_tamper.py`](../tests/test_ascon_tamper.py), [`test_nonce_collision.py`](../tests/test_nonce_collision.py),
[`test_path_disjointness.py`](../tests/test_path_disjointness.py) — all green, and the suite
reached **zero skips for the first time in this project** on this phase's completion. Code:
[`crypto/ascon_aead.py`](../src/ascon_smart_agri/crypto/ascon_aead.py),
[`routing/`](../src/ascon_smart_agri/routing/), [`eval/crypto_benchmark.py`](../src/ascon_smart_agri/eval/crypto_benchmark.py).

## What the paper says

> "The verdict selects the data path: the two paths are kept disjoint in the code rather than by
> convention... the malicious handler holds no reference to the cloud client at all, and an
> automated test asserts that no encrypted payload is emitted during a malicious-only run."
> — Section III-A, gap G1, Eq. (5)

> Decryption returns the payload only if the tag verifies: `Dec(ν,a,c,t) ∈ {p, ⊥}`. "A scheme
> offering confidentiality alone would return some plaintext for a modified ciphertext, and the
> farm would act on it." — Section III-G, Eq. (26)

> Wire overhead: `|c|_wire = |p| + 32 bytes`. "For a typical 96-byte sensor reading this is a
> 33% expansion, falling to 6.3% at 512 bytes." — Section III-G4, Eq. (29)

> "Authenticating a monotonic counter also gives replay detection, which encryption alone would
> not provide." — Section III-G2, near Eq. (27)

## What we achieved

**G1's structural half**: `AlertSink.__init__` takes no transport dependency (checked by
introspecting its constructor signature), and no instance attribute of any `AlertSink` is ever a
`CloudTransport` (checked by walking its `vars()`).

**G1's behavioural half**: routes real malicious-verdict messages through the real
`VerdictRouter` and asserts the cloud receiver's `received_count` **and** `rejected_count` both
stay 0 — not "nothing was accepted", but "nothing was even attempted", because the alert sink
cannot reach the cloud transport to try.

**Ciphertext expansion matches Eq. (29) exactly** (a structural property, checked against the
real vendored backend rather than assumed):

| Payload | Expansion | Encrypt (median / p95) | Decrypt (median / p95) |
| --- | --- | --- | --- |
| 96 B | 32 B (**33.33%**, paper: 33%) | 390.8 µs / 400.9 µs | 391.9 µs / 403.3 µs |
| 512 B | 32 B (**6.25%**, paper: 6.3%) | 1300.5 µs / 1327.8 µs | 1302.1 µs / 1330.3 µs |

*(Latency is on the vendored pure-Python reference backend, not an optimised implementation —
see the KAT-conformance note below.)*

**Replay enforcement, confirmed with the user before any code was written** (per the project's
own planning doc, which explicitly withheld authorization until the policy was chosen):
per-`(edge_id, device_id)` scope, strict-monotonic acceptance, in-memory state. Verification
order is fixed and tested as load-bearing: a tampered message at counter=5 is rejected *without*
consuming that counter, so the genuine message at counter=5 is still accepted afterwards —
proven directly, not just asserted.

## A real bug found by benchmarking, not assumed correct

`AsconAEAD128.encrypt()` returns only `ciphertext||tag` (payload + 16 bytes); Eq. (29)'s 32-byte
figure also counts the nonce, which travels as a separate field in this design and isn't part of
that return value. The benchmark's first run disagreed with the paper's own worked 33%/6.3%
examples — that disagreement is what caught the bug, not a code review.

## Decisions flagged, not silently made

- **Ascon backend, gated by KAT.** The PyPI `ascon` package implements only pre-standard
  Ascon-128/128a, not SP 800-232 Ascon-AEAD128. Per the paper's own "no primitive from scratch"
  fallback, the official reference implementation is vendored and gated behind all 1089 official
  known-answer vectors before any integration.
- **Associated-data byte serialisation** — Eq. (27) defines the tuple, not its wire encoding.
  Length-prefixed (TLV-style), injective by construction, confirmed with the user.
- **`route()`'s scaffold signature (`edge_id` alone) couldn't build a valid Eq. (27) AD tuple**
  — `device_id`, `counter`, `schema_version` were simply missing. Widened to take a full
  `AssociatedData`.

## What this phase does not claim

Replay-state persistence across a receiver restart is out of scope, the same class of exclusion
as production key management (named out of scope by the paper itself). The demo key store is
exactly that — demo keys, explicitly labelled, never committed.

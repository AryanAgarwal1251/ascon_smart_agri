# Phase 6 — Ascon Integration + Alerting Path

**Status: ✅ Done.** Gating tests: [`tests/test_ascon_kat.py`](../tests/test_ascon_kat.py),
[`test_ascon_tamper.py`](../tests/test_ascon_tamper.py), [`test_nonce_collision.py`](../tests/test_nonce_collision.py),
[`test_path_disjointness.py`](../tests/test_path_disjointness.py),
[`test_federated_weight_crypto.py`](../tests/test_federated_weight_crypto.py) — all green. Code:
[`crypto/ascon_aead.py`](../src/ascon_smart_agri/crypto/ascon_aead.py),
[`federated/crypto.py`](../src/ascon_smart_agri/federated/crypto.py),
[`routing/`](../src/ascon_smart_agri/routing/), [`eval/crypto_benchmark.py`](../src/ascon_smart_agri/eval/crypto_benchmark.py).

> **Implementation deviation from the design paper (user-approved, 2026-09-20).** The paper
> applies Ascon-AEAD128 to the gateway→cloud **telemetry** channel (Channel 2). The
> implementation instead applies it to the client↔aggregator **model-weight** channel
> (Channel 3), and sends telemetry in the clear. See the design paper's
> [Implementation Deviation section](../docs/design_paper.md) for the full rationale. This report
> describes the code as it stands after that deviation.

## What the paper says

> Decryption returns the payload only if the tag verifies: `Dec(ν,a,c,t) ∈ {p, ⊥}`. "A scheme
> offering confidentiality alone would return some plaintext for a modified ciphertext, and the
> farm would act on it." — Section III-G, Eq. (26)

> Wire overhead: `|c|_wire = |p| + 32 bytes` (a 16-byte tag plus a 16-byte nonce), independent of
> payload size. — Section III-G4, Eq. (29)

> "The verdict selects the data path: the two paths are kept disjoint in the code rather than by
> convention... the malicious handler holds no reference to the cloud client at all." — Section
> III-A, gap G1, Eq. (5)

> Channel 3 (node-to-aggregator model updates): "Federated learning addresses [reconstruction]
> only partially and [poisoning] not at all." — Section I-B. The deviation adds wire-level AEAD to
> this channel; it does **not** claim to close the reconstruction/poisoning gaps the paper leaves
> open there.

## What we achieved

**Ascon-AEAD128 now protects every model-weight blob on the federated transport**, both legs of
every round (server→client broadcast and client→server upload), via
`federated/crypto.py`'s `protect_state`/`unprotect_state` wired into `federated/server.py`. A
tampered blob makes decryption return ⊥ (Eq. 26), surfaced as a `WeightIntegrityError` that
aborts the round rather than aggregating unverified parameters — proven by
`test_federated_weight_crypto.py` for a bit-flipped ciphertext and for a swapped `client_id` or
`direction` in the associated data.

**G1's structural half (retained on the telemetry path)**: `AlertSink.__init__` takes no
transport dependency (checked by introspecting its constructor signature), and no instance
attribute of any `AlertSink` is ever a `CloudTransport` (checked by walking its `vars()`).

**G1's behavioural half (retained)**: routes real malicious-verdict messages through the real
`VerdictRouter` and asserts the cloud receiver's `received_count` **and** `rejected_count` both
stay 0 — not "nothing was accepted", but "nothing was even attempted", because the alert sink
cannot reach the cloud transport to try. The cloud path is now plaintext, so the guarantee is
"zero payloads reach the cloud", not "zero *encrypted* payloads".

**Ciphertext expansion matches Eq. (29) exactly at weight-blob sizes** (a structural property,
checked against the real vendored backend rather than assumed). Latency is measured on the
vendored pure-Python reference backend — not an optimised implementation — so absolute times are
a correctness/overhead-shape figure, not a deployment SLA. Sizes span a single GRU parameter
tensor up to a full 33,800-parameter state-dict blob (135,200 raw bytes), via
`eval/crypto_benchmark.py` (100 samples each):

| Blob | Expansion | Encrypt (median / p95) | Decrypt (median / p95) |
| --- | --- | --- | --- |
| 1,024 B | 32 B (**3.13%**) | 1,769 µs / 2,029 µs | 1,784 µs / 2,114 µs |
| 37,056 B | 32 B (**0.086%**) | 58,267 µs / 62,455 µs | 57,960 µs / 61,792 µs |
| 135,200 B (full θ) | 32 B (**0.024%**) | 230,441 µs / 241,783 µs | 230,086 µs / 240,015 µs |

The relative overhead is tiny on the weight channel — a flat 32 bytes against a ~132 KiB blob is
0.024%, versus the 33% Eq. (29) reports for a 96-byte telemetry payload. The channel Ascon now
protects is exactly the one where the fixed AEAD overhead matters least. (Per-round crypto *time*
on the reference backend is the larger cost: two ~230 ms legs × 3 clients ≈ 1.4 s/round, which an
optimised backend would cut by orders of magnitude; the primitive is a drop-in behind the
KAT-gated facade.)

**Replay/ordering protection carries into the weight channel via the AD.** `round_index` and
`direction` are authenticated in every weight blob's associated data, so a blob cannot be
replayed across rounds or across the broadcast/upload legs.

## Associated-data design for the weight channel

Eq. (27) defines the telemetry AD tuple; the weight channel mirrors its shape as
`⟨client_id, round_index, direction, schema_version⟩` (`federated/crypto.py`'s
`WeightAssociatedData`). Both use one shared length-prefixed (TLV) wire encoding
(`crypto/_ad_wire.py`), injective by construction, so neither can authenticate a blob against the
wrong fields by string-boundary collision.

## Decisions flagged, not silently made

- **Ascon channel moved (the deviation itself).** Recorded in the design paper's addendum and
  `CLAUDE.md`; telemetry confidentiality is assumed handled outside this codebase.
- **Corrupted weight blob is a hard failure, not a re-route.** Unlike the telemetry path (which
  diverts a bad message to the alert sink), a federated round has no alternate path — it cannot
  proceed on unverified parameters — so `unprotect_state` raises rather than returning ⊥ silently.
- **One pre-shared key per client↔server pair, reused across rounds with fresh nonces** — an
  extension of assumption A5. Production key management stays out of scope; demo keys are
  generated inline by the driver scripts and never written to a manifest (III-J3).
- **Ascon backend, gated by KAT.** The PyPI `ascon` package implements only pre-standard
  Ascon-128/128a, not SP 800-232 Ascon-AEAD128. Per the paper's own "no primitive from scratch"
  fallback, the official reference implementation is vendored and gated behind all official
  known-answer vectors before any integration — unchanged by the deviation.

## What this phase does not claim

The weight-channel AEAD adds confidentiality and tamper-evidence to the *wire*. It does **not**
provide secure aggregation, does **not** defend against a curious-but-honest aggregator
inspecting the plaintext it legitimately decrypts (assumption A4 stands), and does **not** defend
against gradient inversion or a poisoned update — all out of scope in the paper, all still out of
scope. Replay-state persistence across a receiver restart remains out of scope, as does
production key management.

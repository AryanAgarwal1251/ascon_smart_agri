# Phase 6 implementation plan — replay-window enforcement (receiver path)

**Status: research/planning only. Nothing in this document is authorization to write code, and
the core policy choices below are NOT yet decided.** Per [CLAUDE.md](../../CLAUDE.md) golden
rule 1, the parameters the design paper leaves open (window scope, size, persistence) must be
flagged to the user and decided *before* implementation, not silently resolved here. This file
exists so a future agent inherits the analysis instead of re-deriving it; treat every claim as a
snapshot to re-verify against [docs/design_paper.md](../../docs/design_paper.md) and the code.

Covers the receiver-side replay check that consumes the authenticated `counter` field carried in
the associated data. It is the counterpart to
[phase6-ad-serialization.md](phase6-ad-serialization.md), which deliberately scoped replay
enforcement *out* of the crypto core (§4 of that doc).

## 1. What the paper requires, and what it leaves open

- Design paper Section III-G2 (near Eq. 27, line ~574): "Authenticating a monotonic **counter**
  also gives replay detection, which encryption alone would not provide." Eq. (27) puts
  `counter` inside the authenticated AD tuple `⟨edge_id, device_id, counter, schema_version⟩`.
- Threat model, Section III (line ~94): on Channel 2 (gateway→cloud) "the adversary is a network
  observer who can read, modify, **replay**, or drop messages in transit." Replay defence is
  therefore in scope for the benign path, not optional.
- **What the paper does NOT specify (open, must be surfaced per golden rule 1):**
  - **Scope of the "last seen" counter** — per `⟨edge_id, device_id⟩` pair (most natural: the
    counter is a per-device sequence number), per `edge_id`, or per key. This is the primary
    open decision.
  - **Acceptance policy** — strict monotonic (accept only `counter > last_seen`, reject equal or
    lower) vs. a sliding window that tolerates limited out-of-order delivery (a bitmap of recently
    seen counters, like IPsec AH/ESP RFC 6479). Strict-increasing is simplest and matches "a
    monotonic counter"; a window only matters if the transport can reorder.
  - **Persistence** — whether last-seen state must survive a receiver restart. In-memory is fine
    for the mock/demo; a note that production would need durable state (out of scope, A6).
  - **Counter source & wrap** — who increments `counter` on the sender side, and behaviour near
    the `u64` ceiling (the AD encoding already bounds it to `[0, 2**64)`; at realistic message
    rates wrap is unreachable, but state the assumption).

## 2. Where it lives (and where it must NOT)

- **Belongs on the receiver / cloud-ingest path:** `routing/cloud_sink.py`
  (`MockCloudReceiver.send_encrypted`, currently a `NotImplementedError` stub). After the AEAD tag
  verifies (Eq. 26), the receiver parses the AD with `AssociatedData.from_bytes`, reads
  `edge_id`/`device_id`/`counter`, and applies the replay check.
- **Ordering constraint:** the replay check runs **after** successful AEAD verification, never
  before. The counter is only trustworthy because it is authenticated — a message whose tag fails
  is discarded (`⊥`) and must not touch replay state, or an attacker could poison the window with
  forged counters.
- **Must NOT live in the crypto core.** `crypto/ascon_aead.py` only *carries* the counter in the
  authenticated AD and exposes it via `from_bytes`; it holds no replay state. This boundary is
  called out in the `AssociatedData` docstring — do not move replay logic into that module.
- **Path disjointness (G1) is unaffected.** Replay enforcement is entirely on the benign
  (cloud) path; the malicious path (`routing/alert_sink.py`) still holds no `CloudTransport` and
  emits no encrypted payloads. Keep `tests/test_path_disjointness.py` green — do not introduce
  any coupling from the alert sink to this state.

## 3. Proposed shape (subject to the §1 decisions)

Assuming the most likely resolution — per-`⟨edge_id, device_id⟩` scope, strict-increasing
acceptance, in-memory state (mock/demo, A6) — a sketch to confirm with the user before writing:

- A small `ReplayGuard` (likely its own object under `routing/`, testable in isolation) holding
  `dict[tuple[str, str], int]` of last-seen counter per device.
- `check(edge_id, device_id, counter) -> bool`: returns `True` and updates state iff
  `counter > last_seen` (or first-seen); returns `False` (replay/stale) otherwise. A rejected
  message is dropped, not decrypted-and-acted-on.
- Wire it into `MockCloudReceiver.send_encrypted` **after** Ascon verification succeeds.
- Record the chosen policy in `backend_provenance()` / the run manifest (e.g.
  `"replay_policy": "per-device-strict-monotonic-v1"`) for auditability, mirroring how the crypto
  backend and `ad_encoding` are recorded.

## 4. Tests to add (`tests/test_replay_window.py`)

- **Accept in order:** increasing counters for one device all accepted.
- **Reject replay:** a repeated counter, and any counter ≤ last-seen, rejected.
- **Independence across devices:** a low counter for device B is not blocked by device A's high
  counter (validates the scope decision once made).
- **Post-verification only:** a message with a tampered tag (`decrypt → ⊥`) never updates replay
  state — a subsequent legitimate message at that counter is still accepted.
- **Integration:** end-to-end through `MockCloudReceiver` — encrypt with a real
  `AssociatedData.to_bytes()`, deliver twice, assert the second delivery is rejected as a replay
  while the first is accepted.
- If a sliding window is chosen instead of strict-monotonic, add reordering-tolerance and
  window-eviction cases.

## 5. Bookkeeping (do these in the same change, per golden rule 6)

- Update [CHANGELOG.md](../../CHANGELOG.md): the replay policy chosen, that it completes the
  receiver side of Phase 6's benign path, and any golden-rule-1 flag raised for the policy
  decision.
- Update the `AssociatedData` docstring in `crypto/ascon_aead.py` to point at the concrete
  implementation once it lands (it currently points here).
- Add the replay-policy identity to the run manifest (see §3).

## 6. Open items to resolve with the user, not silently assume

- The §1 decisions (scope, acceptance policy, persistence) — all genuinely unspecified by the
  paper; surface them per golden rule 1.
- Whether replay enforcement is even required for the *demo* deliverable or is future work like
  the live-broker option (Section III, line ~614). The paper lists it as a real defence, so the
  default assumption is "in scope for Phase 6", but confirm against the current phase goal.
- Interaction with the counter's sender-side origin (Phase 5 telemetry simulation): who assigns
  `counter`, and whether the simulator guarantees monotonicity per device.

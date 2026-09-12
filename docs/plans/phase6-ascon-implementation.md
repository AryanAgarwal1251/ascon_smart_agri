# Phase 6 implementation plan — Ascon-AEAD128 integration

**Status: research/planning only. Nothing in this document is authorization to write code.**
Per [CLAUDE.md](../../CLAUDE.md) golden rule 2, Phase 6 substantive logic does not start until
Phase 3 (centralised GRU, hard gate) has met its exit criterion **and** the user has explicitly
signed off on starting Phase 6. This file exists so that whichever agent picks up Phase 6 does
not have to re-derive the library research from scratch. Treat every factual claim below as a
snapshot to re-verify (library versions, KAT vector locations, and licenses can all change) —
do not treat this file as ground truth without checking.

Covers: `src/ascon_smart_agri/crypto/ascon_aead.py` (`AsconAEAD128`, `NonceRegistry`,
`verify_kat_conformance`), `tests/test_ascon_kat.py`, `tests/test_ascon_tamper.py`,
`tests/test_nonce_collision.py`, and the manifest fields in `eval/manifest.py` that record
crypto backend provenance.

## 1. What the paper and CLAUDE.md require

- Design paper Section III-G1 (line ~563): implementation must be verified against the
  **official known-answer test vectors** as a gating unit test before any integration; package
  name, version, and variant string go in the run manifest; **no primitive from scratch, no
  placeholder encryption at any stage**.
- Design paper Section II-E: the standardised **Ascon-AEAD128** (NIST SP 800-232) is *not* the
  same parameterisation as pre-standard **Ascon-128** from Ascon v1.2 — it derives from
  Ascon-128a with revised initial values and little-endian formatting. A library that only
  implements v1.2 will silently produce non-conformant ciphertext if this isn't checked.
- CLAUDE.md golden rule 3: never hand-roll cryptography. Backend must be a maintained library
  or the official reference implementation, gated behind the KAT.
- CLAUDE.md "Open decision": if the PyPI `ascon` candidate fails the KAT gate, fall back to
  **binding the official reference implementation**, not hand-rolling. This was confirmed
  paper-consistent in a prior session (nothing in III-G1 forbids vendoring a pinned external
  reference implementation; it only forbids writing one from scratch).

## 2. Library research (done; re-verify before use)

### Candidate 1 — PyPI `ascon` package (already listed as the `crypto` optional dependency in `pyproject.toml`, pinned `>=1.3`)

- Implements pre-standard **Ascon v1.2**: `Ascon-128`, `Ascon-128a`, `Ascon-80pq`. It is *not*
  maintained by the official Ascon design team.
- Expectation, not yet verified experimentally: this will **fail** an SP 800-232 KAT, because
  Ascon-AEAD128 uses revised initialization vectors and little-endian formatting relative to
  Ascon-128a. Run the KAT test against it anyway and record the actual failure — don't skip
  straight to the fallback on the basis of this document alone.
- Source: [pypi.org/project/ascon](https://pypi.org/project/ascon/)

### Candidate 2 (fallback) — `meichlseder/pyascon` (official reference implementation)

- GitHub: [github.com/meichlseder/pyascon](https://github.com/meichlseder/pyascon)
- Authored by Maria Eichlseder, one of the four original Ascon designers (with Dobraunig,
  Mendel, Schläffer). Linked from the Ascon team's own implementations page
  ([ascon.isec.tugraz.at](https://ascon.isec.tugraz.at/specification.html)).
- Already implements NIST SP 800-232 directly:
  `ascon_encrypt(key, nonce, associateddata, plaintext, variant="Ascon-AEAD128")` /
  `ascon_decrypt(...)`, with `"Ascon-AEAD128"` as the default variant — this is the finalized
  standard parameterisation, not v1.2.
- **License: CC0 1.0 Universal (public domain dedication)** — no attribution or redistribution
  restriction, so vendoring it into this repo is legally unencumbered. Re-verify the `LICENSE`
  file at the pinned commit before vendoring, since this was checked via web search, not by
  reading the file directly.
- Also ships `genkat.py`, useful for cross-generating/checking KAT vectors locally.

### KAT vector sourcing

Two independent sources, ideally cross-check both:

1. **`ascon/ascon-c`** (official C reference repo):
   [github.com/ascon/ascon-c](https://github.com/ascon/ascon-c) — KAT files live at
   `crypto_aead/asconaead128/.../LWC_AEAD_KAT_*.txt` in CAVP-style format (key, nonce, AD,
   plaintext, ciphertext tuples).
2. **NIST's own ACVP server**: [github.com/usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server)
   — canonical NIST-side vectors, useful as a second, independently-sourced check.

Record in the run manifest *which* vector source(s) `tests/test_ascon_kat.py` validated against.

## 3. Proposed implementation sequence

1. **Land the KAT test first, backend-agnostic.** Parse KAT vectors (from `ascon-c`, cross-check
   against ACVP-Server) into `tests/test_ascon_kat.py` fixtures. It should be able to gate
   *any* candidate `AsconAEAD128`-shaped backend, not just one library.
2. **Wire up Candidate 1 (`ascon` PyPI package) and run the KAT test against it.** Document the
   actual pass/fail result — this is the "verified, not assumed" step. If it passes, stop here:
   use it, record its package name/version/`"Ascon-AEAD128"` (or whatever variant string it
   reports) in the manifest, and skip step 3.
3. **If step 2 fails:** vendor `meichlseder/pyascon`, pinned to a specific commit SHA (not
   `master`/latest). Re-verify its license at that commit. Store it under something like
   `src/ascon_smart_agri/crypto/_vendor/pyascon/` with the upstream commit hash and a link to
   the source recorded in a header comment (CC0 doesn't require this, but it keeps provenance
   auditable). Re-run the same KAT test against the vendored copy.
4. **Implement `AsconAEAD128.encrypt`/`.decrypt`** as a thin facade over whichever backend
   passed the gate — `decrypt` must return `None` (not raise, not return garbage plaintext) on
   any tag-verification failure, per Eq. (26) and `tests/test_ascon_tamper.py`.
5. **Implement `NonceRegistry.register`**: draw nonces from `secrets.token_bytes(16)` (CSPRNG,
   not `random`), track issued `(key, nonce)` pairs, raise on a collision. Cheap to check, and
   nonce reuse is unrecoverable (III-G3) — this is exactly what `test_nonce_collision.py` gates.
6. **Resolve the associated-data byte encoding before implementing it, not while implementing
   it.** Eq. (27) defines the *tuple* `⟨edge_id, device_id, counter, schema_version⟩` but the
   paper does not specify a byte-serialisation for it. Naive concatenation of variable-length
   strings is ambiguous (e.g. `edge_id="ab", device_id="c"` vs. `edge_id="a", device_id="bc"`
   collide under simple concatenation) — flag this as a discrepancy to the user per CLAUDE.md
   golden rule 1 rather than silently picking an encoding. A length-prefixed or fixed-width
   encoding is the standard fix; note the choice made and why in the docstring and manifest.
7. **Implement `verify_kat_conformance()`** as the single gate function everything else must
   pass before integration proceeds (routing/cloud_sink.py, alerting path).
8. **Record provenance in the manifest** (`eval/manifest.py`): backend identity (PyPI
   name+version, or vendored commit SHA), variant string, and which KAT vector source(s) were
   used.
9. **Update `pyproject.toml`** to reflect the actual outcome — either keep the `ascon` extra as
   the real dependency, or replace/annotate it once a vendored fallback is in place.
10. **Update [CHANGELOG.md](../../CHANGELOG.md)** per CLAUDE.md golden rule 6 as each of the
    above steps lands, not just at the end.

## 4. Open items for the implementing agent to resolve, not assume

- AD byte encoding (step 6 above) — genuinely unresolved by the paper; surface it.
- Whether the `ascon` PyPI package's `>=1.3` pin (already in `pyproject.toml`) has changed
  behaviour since this research was done — re-check its docs/changelog before assuming it's
  still v1.2-only.
- Confirm the exact upstream commit SHA and license text of `pyascon` at vendoring time, rather
  than trusting the summary in Section 2 above.
- Cross-check KAT vectors from both sources listed in Section 2 rather than relying on one.

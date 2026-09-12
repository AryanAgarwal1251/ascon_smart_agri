# Phase 6 implementation plan — associated-data (AD) byte serialization

**Status: a decision has been made and is authorized for implementation.** This resolves the
spec gap flagged in [phase6-ascon-implementation.md](phase6-ascon-implementation.md) (step 6 /
§4) and in the module docstring of
[`crypto/ascon_aead.py`](../../src/ascon_smart_agri/crypto/ascon_aead.py). Normally, per
[CLAUDE.md](../../CLAUDE.md) golden rule 1, an ambiguity the design paper leaves open is flagged
to the user rather than silently resolved; **the user explicitly waived that rule for this
specific decision** and chose the encoding below. Record the choice as an intentional
extension of the paper (docstring, manifest, CHANGELOG) so it stays auditable, and reconcile it
with [docs/design_paper.md](../../docs/design_paper.md) if the paper is later revised.

Treat every concrete value here (prefix widths, version byte) as the decided default, not as
sacred — but do not change them without recording why, because the AD bytes are **authenticated**
and any encoding change silently breaks interoperability between sender and receiver.

## 1. The problem being solved

Design paper Eq. (27) defines associated data as the *tuple*
`a = ⟨edge_id, device_id, counter, schema_version⟩` but specifies **no byte serialization**.
The AD is authenticated-but-not-encrypted (Ascon-AEAD128): it is transmitted in the clear
alongside the ciphertext, the receiver reads `edge_id` from it to select a decryption key, and
any mismatch between the sender's and receiver's AD bytes makes decryption return `⊥` (Eq. 26).

Naive concatenation of the variable-length string fields is **ambiguous**: `edge_id="ab",
device_id="c"` and `edge_id="a", device_id="bc"` concatenate to identical bytes, so a reading
could be authenticated against the wrong device's metadata. The serialization must therefore be
an **injective, unambiguously parseable** function of the tuple.

## 2. Decision: length-prefixed (TLV-style) encoding

**Chosen: length-prefixed.** Rejected: fixed-width padding.

### Why not fixed-width

Fixed-width fields do **not** actually close the ambiguity and introduce two silent failure
modes:

1. **Truncation aliasing.** If an id exceeds its field width it is truncated, so
   `edge_id="sensor-north-01"` and `edge_id="sensor-north-02"` truncated to 12 bytes both become
   `"sensor-north"` → identical AD. This re-creates the exact cross-device confusion the encoding
   exists to prevent, reachable through legitimate ids.
2. **Padding ambiguity.** Null-padding makes `edge_id="ab"` and `edge_id="ab\x00"` encode
   identically. Closing this requires forbidding byte values in ids; closing truncation requires
   storing a length — at which point the design has become length-prefixing anyway, worse.

Fixed-width also requires choosing field widths up front, which depends on the id-length
distribution — a **Phase 1 characterisation** output that does not yet exist. It forces a
security-relevant parameter to be guessed blind.

### Why length-prefixed

- Injective and unambiguously parseable by construction (canonical TLV framing — the standard
  way to build AAD from multiple structured fields, preventing canonicalization ambiguity).
- No length caps in practice, no truncation, no forbidden byte values; ids/schema strings may be
  any length and any bytes.
- Self-describing: the receiver can parse `edge_id` (needed in the clear for key selection)
  field-by-field with no out-of-band length agreement.

## 3. Byte layout (authoritative)

All multi-byte integers are **big-endian**. Strings are **UTF-8**. Field order follows Eq. (27).

```
AD := fmt_version : u8  = 0x01
   || L(edge_id)        : u16  || edge_id        : UTF-8 bytes
   || L(device_id)      : u16  || device_id      : UTF-8 bytes
   || counter           : u64
   || L(schema_version) : u16  || schema_version : UTF-8 bytes
```

where `L(s)` is the byte length of the UTF-8 encoding of `s`.

Rationale for each choice:

- **`fmt_version` (1 byte, `0x01`)** — a leading wire-format version so the encoding can evolve
  without ambiguity; a receiver rejects unknown versions.
- **`u16` string length prefixes** — up to 65 535 bytes per field, far beyond any realistic id
  or schema string. **Never truncate:** raise if a field's UTF-8 length exceeds 65 535.
- **`counter` as fixed `u64`** — monotonic replay counter (III-G3); fixed width, no prefix
  needed. Validate `0 ≤ counter < 2**64`.
- **Big-endian** — arbitrary for authentication (any fixed order works if both sides agree);
  chosen by network-byte-order convention.

Endianness/width note: this AD framing is **independent** of Ascon's internal little-endian
formatting (SP 800-232). The AEAD treats the AD as an opaque byte string; this layout only
governs how the tuple becomes those bytes.

## 4. API to implement

In [`crypto/ascon_aead.py`](../../src/ascon_smart_agri/crypto/ascon_aead.py), on the existing
`AssociatedData` dataclass (currently a typed container with no serialization — replace the
"deferred" note in the module docstring with the decided encoding):

```python
def to_bytes(self) -> bytes:
    """Serialize the Eq. (27) tuple to authenticated-data bytes (see phase6-ad-serialization.md).

    Length-prefixed, big-endian, UTF-8. Raises ValueError if any string field exceeds 65535
    UTF-8 bytes or if counter is out of u64 range. Never truncates.
    """


@classmethod
def from_bytes(cls, data: bytes) -> "AssociatedData":
    """Parse AD bytes back to the tuple; the receiver uses this to read edge_id in the clear.

    Raises ValueError on unknown fmt_version, truncated/over-long input, or trailing bytes
    (a strict, single-valid-encoding parse — no tolerated garbage).
    """
```

Requirements:

- `from_bytes(to_bytes(x)) == x` for all valid `x` (round-trip).
- `from_bytes` is **strict**: reject unknown `fmt_version`, inputs shorter than the declared
  lengths, and any trailing bytes after the last field. There must be exactly one valid byte
  encoding per tuple (canonical).
- Callers pass `associated_data.to_bytes()` to `AsconAEAD128.encrypt`/`.decrypt`. The AEAD facade
  signature stays `(nonce: bytes, associated_data: bytes, ...)` — do not couple the cipher to the
  dataclass.
- Replay-window enforcement (tracking the last-seen `counter` per `⟨edge_id, device_id⟩`) is
  **receiver/routing logic, out of scope here** — this module only carries the counter in the AD.

## 5. Tests to add (`tests/test_ascon_ad_encoding.py`)

- **Round-trip:** `from_bytes(to_bytes(x)) == x` across ids/schema of varied lengths, including
  empty strings, non-ASCII UTF-8, and `counter` at `0` and `2**64 - 1`.
- **Anti-ambiguity (the motivating cases):** `AssociatedData("ab","c",n,v).to_bytes() !=
  AssociatedData("a","bc",n,v).to_bytes()`, and the fixed-width truncation pair
  (`"sensor-north-01"` vs `"sensor-north-02"`) produce distinct AD.
- **Tamper integration:** encrypt with `ad1 = x.to_bytes()`, attempt decrypt with `ad2 =
  y.to_bytes()` for a neighbouring tuple `y` → returns `None` (⊥). Complements the single-bit AD
  mutation already in [`tests/test_ascon_tamper.py`](../../tests/test_ascon_tamper.py).
- **Strict parse:** `from_bytes` raises on unknown `fmt_version`, truncated input, and trailing
  bytes.
- **Bounds:** `to_bytes` raises on a >65535-byte field and on out-of-range `counter`.

## 6. Provenance / bookkeeping (do these in the same change)

- Add the AD encoding identity to `backend_provenance()` (e.g. `"ad_encoding":
  "length-prefixed-v1"`) so the run manifest records it (III-G1/III-I4 auditability), mirroring
  how the crypto backend is recorded.
- Update the `crypto/ascon_aead.py` module docstring: replace the "UNRESOLVED SPEC GAP … DEFERRED"
  block with the decided encoding, citing this file.
- Update [CHANGELOG.md](../../CHANGELOG.md) per golden rule 6: note the AD encoding decision, that
  golden rule 1 was waived by the user for it, and that it is an intentional extension of the
  paper.

## 7. Open items to reconcile, not silently assume

- **Paper reconciliation:** this encoding is an addition the paper does not specify. If
  [docs/design_paper.md](../../docs/design_paper.md) is later revised to define an AD
  serialization, reconcile against it and flag any conflict (golden rule 1).
- **`u16` vs `u32` prefixes:** `u16` is the decided default (ample headroom). Only widen to `u32`
  if a real field can plausibly exceed 65 535 bytes — record the reason if changed.
- **`counter` semantics:** width and endianness are fixed here, but the monotonicity/replay
  policy (per-key? per-device? window size?) belongs to the receiver path and is defined there.

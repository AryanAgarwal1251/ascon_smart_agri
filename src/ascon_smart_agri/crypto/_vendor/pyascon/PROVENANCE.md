# Vendored: pyascon (official Ascon reference implementation)

`ascon.py` in this directory is vendored **verbatim** from the official Ascon reference
implementation. It is not modified; provenance is recorded here rather than by editing the file,
so the copy can be re-verified by diffing against the pinned upstream commit.

| field | value |
| --- | --- |
| upstream repo | https://github.com/meichlseder/pyascon |
| author | Maria Eichlseder (one of the four original Ascon designers: Dobraunig, Eichlseder, Mendel, Schläffer) |
| pinned commit | `ed24e54abf9507d26fa49b46a56091570c7e743e` |
| vendored file | `ascon.py` |
| `ascon.py` sha256 | `b799653fc09e56fc4eb72c90b3574b967b3ead4c337b2402123148d5b1873eda` |
| standard | NIST SP 800-232 (`Ascon-AEAD128`, `Ascon-Hash256`, `Ascon-XOF128`, …) |
| license | CC0 1.0 Universal (public domain dedication) — see `LICENSE` |
| vendored on | 2026-09-12 |

## Why this backend, not the PyPI `ascon` package

Design paper III-G1 requires KAT conformance to SP 800-232 before any crypto integration, and
forbids implementing the primitive from scratch. The PyPI `ascon` package (only version `0.0.9`
is published — the `>=1.3` pin previously in `pyproject.toml` was unsatisfiable) implements only
the pre-standard Ascon v1.2 variants `Ascon-128`, `Ascon-128a`, `Ascon-80pq`; it does not
expose `Ascon-AEAD128` and cannot pass the standard's KAT. Vendoring the official reference
implementation is the paper-consistent fallback recorded in CLAUDE.md's "Open decision".

## KAT verification

Gated by `tests/test_ascon_kat.py` via `ascon_smart_agri.crypto.ascon_aead.verify_kat_conformance`,
which checks this backend against all vectors in `tests/kat/LWC_AEAD_KAT_128_128.txt`
(see that directory's `PROVENANCE.md`). At vendoring time: **1089 / 1089 vectors pass** for both
encryption and decryption.

## Re-verifying this copy

```bash
curl -s "https://raw.githubusercontent.com/meichlseder/pyascon/ed24e54abf9507d26fa49b46a56091570c7e743e/ascon.py" \
  | sha256sum   # must equal b799653fc09e56fc4eb72c90b3574b967b3ead4c337b2402123148d5b1873eda
```

Do not edit `ascon.py`. If it must be updated, re-pin to a new commit, re-record the sha256
here, and re-run the KAT gate.

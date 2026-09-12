# KAT vectors: Ascon-AEAD128 (NIST SP 800-232)

`LWC_AEAD_KAT_128_128.txt` holds the official known-answer test vectors for `Ascon-AEAD128`,
used by `tests/test_ascon_kat.py` (via `verify_kat_conformance`) as the R5 gating check.

| field | value |
| --- | --- |
| source repo | https://github.com/ascon/ascon-c (official C reference) |
| path in repo | `crypto_aead/asconaead128/LWC_AEAD_KAT_128_128.txt` |
| repo commit at download | `446347f21b209f3921c65ece70027c366cbe1693` |
| sha256 | `bbbc34692fe05e5fda0a3b025585622ab3e3747495e5e3655b29aae8c2a4bd33` |
| vectors | 1089 (CAVP-style `Count`/`Key`/`Nonce`/`PT`/`AD`/`CT` tuples) |
| downloaded on | 2026-09-12 |

Format: each record is `Count`, `Key`, `Nonce`, `PT` (plaintext), `AD` (associated data), `CT`
(ciphertext **with** the 16-byte tag appended), all hex, blank-line separated. `PT`/`AD` may be
empty. `verify_kat_conformance` asserts `encrypt(key, nonce, ad, pt) == ct` and
`decrypt(key, nonce, ad, ct) == pt` for every record.

The Phase 6 plan lists a second, independent source (NIST's ACVP-Server) for cross-checking;
this file is the ascon-c source. The manifest records which source(s) a given run validated
against (`ascon_smart_agri.crypto.ascon_aead.backend_provenance`).

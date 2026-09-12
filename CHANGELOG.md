# Changelog

All notable changes to this repository are recorded here, newest first. This project follows
the seven-phase gating discipline of [docs/design_paper.md](docs/design_paper.md) (Section
III-J4) rather than semantic-versioned releases, so entries are grouped by date and tagged with
the phase they belong to. See [CLAUDE.md](CLAUDE.md) for the rule requiring this file to be
kept current.

## Phase status

| # | Phase | Status | Exit criterion |
| - | --- | --- | --- |
| 1 | Characterisation report | Not started | Real columns/types, nulls, zero-variance, exact duplicate count, label vocab + counts, correlation matrix produced |
| 2 | Leakage-controlled preprocessing + four-stage feature selection | Not started | `tests/test_leakage.py` green on real data |
| 3 | Centralised GRU + full evaluation | Not started (**hard gate**) | Full evaluation protocol (macro-F1, per-class F1, balanced accuracy, MCC, confusion matrix, FPR; ≥3 seeds) reported |
| 4 | Three-client federated simulation, weighted FedAvg | Not started | `test_fedavg_weighting.py`, `test_scaler_equivalence.py` green |
| 5 | Telemetry simulation + feature-provenance adapter | Not started | Provenance adapter enforces G6 boundary |
| 6 | Ascon integration + alerting path | In progress (**gate deliberately overridden**) | `test_ascon_kat.py`, `test_ascon_tamper.py`, `test_nonce_collision.py`, `test_path_disjointness.py` green |
| 7 | End-to-end integration | Not started | Full pipeline run producing a manifest |

Most modules under `src/ascon_smart_agri/` are still typed stubs: they `del` their unused
parameters and raise `NotImplementedError("Phase N: ... not implemented yet.")`. The exception
is the Ascon-AEAD128 crypto core (`crypto/ascon_aead.py`), implemented ahead of its phase gate
(see the 2026-09-12 Phase 6 entry below).

## Unreleased

### Added

- **Phase 6 (Ascon-AEAD128 crypto core), gate deliberately overridden.** Implemented
  `crypto/ascon_aead.py` (`AsconAEAD128` encrypt/decrypt facade, `NonceRegistry`,
  `NonceReuseError`, `verify_kat_conformance`, `backend_provenance`) over a vendored official
  reference backend. **This started Phase 6 substantive logic while Phase 3 (the hard gate) and
  Phases 1, 2, 4, 5 are still unmet** — CLAUDE.md golden rule 2 was consciously waived by the
  user for this work, and is recorded here so the out-of-order start is auditable. The alerting
  path / path-disjointness half of Phase 6 (`routing/alert_sink.py`, `routing/cloud_sink.py`,
  `test_path_disjointness.py`) is **not** done, so the phase is *in progress*, not complete.
  - **Backend selection (KAT-gated, III-G1/R5).** The PyPI `ascon` candidate was rejected: only
    version `0.0.9` is published (the previous `>=1.3` pin was unsatisfiable) and it implements
    only the pre-standard Ascon v1.2 variants (`Ascon-128`/`Ascon-128a`), never SP 800-232
    `Ascon-AEAD128`. Per the paper-consistent fallback, the official reference implementation
    (`meichlseder/pyascon`, commit `ed24e54`, CC0) is vendored verbatim under
    `src/ascon_smart_agri/crypto/_vendor/pyascon/` (with `LICENSE` and `PROVENANCE.md`).
  - **KAT gate passes:** all 1089 official SP 800-232 vectors (from `ascon/ascon-c`, committed
    at `tests/kat/LWC_AEAD_KAT_128_128.txt` with `PROVENANCE.md`) verify for both encryption and
    decryption. `test_ascon_kat.py`, `test_ascon_tamper.py`, and `test_nonce_collision.py` are
    activated (skips removed) and green — pytest now 41 passed / 5 skipped.
  - **Associated-data byte encoding — decided AND implemented (golden rule 1 waived by the
    user).** Eq. (27) defines the AD tuple but no byte serialisation. The user waived golden
    rule 1 for this specific decision and chose a **length-prefixed (TLV-style)** encoding over
    fixed-width (fixed-width can silently re-introduce the cross-device collision via
    truncation/padding, and needs id-length figures Phase 1 hasn't produced). The decision, exact
    byte layout, API, and test plan are recorded in `docs/plans/phase6-ad-serialization.md`.
    `AssociatedData.to_bytes()`/`.from_bytes()` now implement it — `fmt_version:u8=0x01 ||
    L(edge_id):u16 || edge_id || L(device_id):u16 || device_id || counter:u64 ||
    L(schema_version):u16 || schema_version`, big-endian, UTF-8, injective and canonical.
    `to_bytes` raises on a >65535-byte field or out-of-u64 `counter` (never truncates);
    `from_bytes` is a strict single-valid-encoding parse (rejects unknown `fmt_version`,
    truncated input, and trailing bytes). The AEAD facade stays decoupled — it still operates on
    `bytes` and callers pass `associated_data.to_bytes()`. `backend_provenance()` now records
    `"ad_encoding": "length-prefixed-v1"` for the run manifest, and the module docstring's
    "UNRESOLVED SPEC GAP … DEFERRED" block is replaced with the decided encoding, noted as an
    intentional extension of the paper. New `tests/test_ascon_ad_encoding.py` (round-trip incl.
    empty/non-ASCII/u64-max, the anti-ambiguity and truncation-aliasing cases, neighbouring-AD
    decryption → `⊥`, strict-parse rejections, bounds, and exact wire layout) — pytest now
    52 passed / 5 skipped.
  - **Tooling:** removed the unsatisfiable `crypto = ["ascon>=1.3"]` extra and the `ascon.*`
    mypy override; excluded the vendored code from ruff, mypy (strict), and vulture so it stays
    byte-identical and diff-verifiable against upstream.
- `docs/plans/phase6-replay-window.md`: a planning doc for the receiver-side replay-window
  enforcement that consumes the authenticated `counter` (the counterpart to
  `phase6-ad-serialization.md`, which scoped replay *out* of the crypto core). Records where the
  check belongs (`routing/cloud_sink.py`, after AEAD verification, never in the crypto core),
  the path-disjointness constraint, a proposed `ReplayGuard` shape, and a test plan.
  Planning/research only — the core policy choices (per-device vs. per-key scope, strict-monotonic
  vs. sliding window, persistence) are **unspecified by the paper and must be surfaced to the
  user per golden rule 1 before implementation**; nothing is implemented. The `AssociatedData`
  docstring in `crypto/ascon_aead.py` now states replay enforcement is out of scope for that
  module and points here.
- `CHANGELOG.md` (this file), tracking repository status per phase.
- `CLAUDE.md` golden rule 6, requiring `CHANGELOG.md` to be kept current after every change.
- `docs/plans/phase6-ascon-implementation.md`: a Phase 6 planning doc for the Ascon-AEAD128
  backend decision — library research (PyPI `ascon` package vs. vendoring the official
  `meichlseder/pyascon` reference implementation), KAT vector sourcing (`ascon/ascon-c`,
  NIST's ACVP-Server), and a proposed step-by-step implementation sequence. Linked from
  `CLAUDE.md`'s "Open decision" section. Planning/research only — no crypto logic implemented;
  Phase 3's hard gate is still unmet and Phase 6 has not started.

### Repository state as of 2026-09-12

- **Mostly scaffolding.** All source files under `src/ascon_smart_agri/` are typed stubs
  (function/class signatures + docstrings citing the relevant design-paper section or equation)
  **except** the Ascon-AEAD128 crypto core `crypto/ascon_aead.py` and the vendored backend under
  `crypto/_vendor/pyascon/`, which are now implemented. Otherwise no working logic beyond
  `__init__.py` files and the `Array` type alias in `_types.py`.
- **Tests:** `pytest` reports 52 passed, 5 skipped, 0 failed. The 5 remaining skips are explicit
  `pytest.skip("pending Phase N: ...")` markers on tests that require unstubbed logic
  (`test_fedavg_weighting.py`, `test_leakage.py`, `test_model_param_count.py`,
  `test_path_disjointness.py`, `test_scaler_equivalence.py`). The Ascon KAT/tamper/nonce and
  AD-encoding tests are active and green.
- **Quality gates:** `ruff check`, `ruff format --check`, `mypy src`, and `vulture` all pass
  clean on the current tree.
- **Config scaffold:** typed pydantic config (`configs/base.py`, `configs/default.yaml`) exists
  and validates; sweep values for the six ablations are not yet exercised by any run.
- **Open decision resolved:** the Ascon backend fallback is settled — the PyPI `ascon` candidate
  is non-conformant (v1.2-only) and unsatisfiable at the pinned version, so the official
  `pyascon` reference implementation is vendored and gated by the KAT (see the Phase 6 entry
  above).

## 2026-09-12 — `1c446ee`

- Added `data/` to `.gitignore` (dataset artifacts are not checked into version control).

## 2026-09-11 — `8944683`

- Initial commit: project scaffolding — typed stub modules for all seven phases, pydantic
  config, pre-commit gate configuration (ruff, mypy, vulture, pytest), test suite with
  phase-gated skips, `README.md`, `AGENTS.md`, `CLAUDE.md`, and `docs/design_paper.md`.

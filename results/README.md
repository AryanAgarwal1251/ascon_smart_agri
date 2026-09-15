# Results — paper claims vs. what was achieved

This folder is the project's results record: for each of the seven phases in
[docs/design_paper.md](../docs/design_paper.md) (Section III-J4), what the paper specifies
alongside what this codebase actually produced, with numbers pulled from the committed
manifests in [artifacts/](../artifacts/) — never typed in from memory. Every figure below is
traceable to a source file; where a claim couldn't be verified against real data, that is
stated rather than smoothed over.

**Status: all seven phases complete**, verified by the gate scripts committed alongside the
code (`scripts/check_phase3_gate.py`, `scripts/check_phase4_gate.py`), not by narrative alone.
Current test suite: **344 passed, 0 skipped**, all lint/type/dead-code gates green.

| # | Phase | Status | Headline result |
| - | --- | --- | --- |
| 1 | [Characterisation](phase1_characterisation.md) | ✅ Done | 46,776,700 records (raw), matches paper's ~46.7M |
| 2 | [Leakage control + feature selection](phase2_leakage_and_feature_selection.md) | ✅ Done | R3 gate passes on real data; F0=31 → F=16 |
| 3 | [Centralised GRU](phase3_centralised_gru.md) | ✅ Done — hard gate closed | macro-F1 0.8297 ± 0.0013 at W=16; recurrence earns its place |
| 4 | [Federated learning](phase4_federated_learning.md) | ✅ Done — gate closed | Federated 0.8308 recovers 82.4% of the local-only→centralised gap |
| 5 | [Telemetry + provenance](phase5_telemetry_and_provenance.md) | ✅ Done | G6 verified: 0 leaked into training index across 200+ checks |
| 6 | [Ascon + alerting](phase6_ascon_and_alerting.md) | ✅ Done — zero skips | G1 holds: 0 malicious-verdict payloads ever reached the cloud |
| 7 | [End-to-end integration](phase7_end_to_end_integration.md) | ✅ Done | Real model, real pipeline, real routing — G1 holds in the assembled loop |

## How to read each phase file

Every file follows the same shape:

- **What the paper says** — direct citations to `docs/design_paper.md`, by section and
  equation number, so a claim can be checked against the source rather than against this
  summary.
- **What we achieved** — the real, measured result, with the exact source file (manifest or
  report) named so the number can be re-derived.
- **Decisions flagged, not silently made** — every place the paper left something open (a
  serialisation format, a taxonomy ambiguity, a replay policy) and how it was resolved, per
  this project's own operating rule (`CLAUDE.md` Golden Rule 1).
- **Honest gaps** — what is *not* done or not as strong as it could be, stated plainly rather
  than omitted.

## A correction, on the record

Phase 4's first internal report (superseded, kept at
[`artifacts/manifest_phase4_minimal_gate.json`](../artifacts/manifest_phase4_minimal_gate.json)
for the record rather than deleted) compared a federated model trained for 60 total local
passes against a centralised baseline trained for only 10 epochs — an unequal budget that made
"federation beats centralised" a statement about training time, not about the method. A
teammate's independent, parallel Phase 4 implementation caught this and reran all three
baselines on the same 60-epoch budget; [phase4_federated_learning.md](phase4_federated_learning.md)
reports that corrected, compute-matched result. The two implementations were reconciled by
merge, not by discarding either — see that file for the full account.

## Regenerating these numbers

Every figure below can be reproduced from the committed manifests without rerunning anything:

```bash
cat artifacts/phase4_results.json      # Phase 4's distilled summary
cat artifacts/phase7_results.json      # Phase 7's distilled summary
PYTHONPATH=. ./.venv/bin/python scripts/check_phase3_gate.py
PYTHONPATH=. ./.venv/bin/python scripts/check_phase4_gate.py
```

Reproducing the underlying experiments from raw data requires the CICIoT2023 raw distribution
at `data/ciciot2023_raw/` (not committed — see `.gitignore`) and the run scripts in `scripts/`,
each named `run_phaseN*.py` or `train_*.py`.

# Changelog

All notable changes to this repository are recorded here, newest first. This project follows
the seven-phase gating discipline of [docs/design_paper.md](docs/design_paper.md) (Section
III-J4) rather than semantic-versioned releases, so entries are grouped by date and tagged with
the phase they belong to. See [CLAUDE.md](CLAUDE.md) for the rule requiring this file to be
kept current.

## Results folder

[`results/`](results/) holds a paper-claim-vs-achieved writeup for each of the seven phases,
each figure traceable to the manifest in `artifacts/` it was pulled from rather than typed from
memory. Start at [`results/README.md`](results/README.md).

## Phase status

| # | Phase | Status | Exit criterion |
| - | --- | --- | --- |
| 1 | Characterisation report | Done, on both a subsampled/pre-split Kaggle mirror and the official UNB raw corpus (see 2026-09-13 entries) | Real columns/types, nulls, zero-variance, exact duplicate count, label vocab + counts, correlation matrix produced |
| 2 | Leakage-controlled preprocessing + four-stage feature selection | **Done** (Stage 4's knee sweep is wired but inert until Phase 3 supplies a detector — see the 2026-09-14 entry): subsample -> dedup -> split -> four-stage selection all run end-to-end on the real corpus; R3 gate passes on real data | `tests/test_leakage.py` green on real data ✅ |
| 3 | Centralised GRU + full evaluation | **DONE — gate CLOSED** (**hard gate**), verified by `scripts/check_phase3_gate.py` (exit 0). Baselines 1-3 of III-I1 reported over 3 seeds at W ∈ {1,16}; GRU macro-F1 **0.8297 ± 0.0013** at W=16 vs random forest 0.6855 and MLP 0.6070. Ablation answered: recurrence **earned its place** (+0.2322, 51× seed std) | Full evaluation protocol (macro-F1, per-class F1, balanced accuracy, MCC, confusion matrix, FPR; ≥3 seeds) reported **for baselines 1-3 of Section III-I1** ✅ |
| 4 | Three-client federated simulation, weighted FedAvg | **DONE — gate CLOSED**, verified by `scripts/check_phase4_gate.py` (exit 0). Two independent runs were merged; the reported figures are the **compute-matched** run (α=0.5, R=20, E=3, W=16, 60 local passes for every baseline): federated global macro-F1 **0.8308 ± 0.0150**, bracketed by local-only **0.7205 ± 0.0750** and centralised **0.8543 ± 0.0040**, recovering **82.4 %** of the gap; client-to-global gap positive for all three clients (+0.10 / +0.16 / +0.07). FPR **0.2975 ± 0.0512** — read it first. The earlier minimal-gate run is kept at `artifacts/manifest_phase4_minimal_gate.json`; its bracket is inverted because its baselines were not compute-matched. **Not done:** the α/E/aggregation ablation sweep of Section III-I3 | `test_fedavg_weighting.py`, `test_scaler_equivalence.py` green ✅; gate checker exit 0 ✅ |
| 5 | Telemetry simulation + feature-provenance adapter | **Done.** `telemetry/simulate.py` and `telemetry/provenance.py` implemented; G6 boundary verified on real data (200+ provenance refs checked, zero leaked into the training index) | Provenance adapter enforces G6 boundary ✅ |
| 6 | Ascon integration + alerting path | **Done.** `test_ascon_kat.py`, `test_ascon_tamper.py`, `test_nonce_collision.py`, `test_path_disjointness.py` all green -- the suite has **zero skips** for the first time in this project | `test_ascon_kat.py`, `test_ascon_tamper.py`, `test_nonce_collision.py`, `test_path_disjointness.py` green ✅ |
| 7 | End-to-end integration | **Done.** A federated global model was trained and saved for the first time in this project (macro-F1 0.8338, bit-for-bit identical to Phase 4's seed-0 result), and the full runtime pipeline ran for real: telemetry → held-out network features (G6) → streaming windows → the real model → Eq. (5) → routing → Ascon/alert. G1 held throughout (0 malicious-verdict messages reached the cloud) | Full pipeline run producing a manifest ✅ |

Most modules under `src/ascon_smart_agri/` are still typed stubs: they `del` their unused
parameters and raise `NotImplementedError("Phase N: ... not implemented yet.")`. The exceptions
are the Ascon-AEAD128 crypto core (`crypto/ascon_aead.py`), implemented ahead of its phase gate
(see the 2026-09-12 Phase 6 entry below), `data/characterize.py` (Phase 1), and
`data/subsample.py`/`data/dedup.py`/`data/split.py`/`features/selection.py` (Phase 2, complete)
-- see the 2026-09-14 and 2026-09-13 entries below.

## 2026-09-16

### Phase 4 local-only baseline now reports balanced accuracy and MCC, not macro-F1 alone

`scripts/run_phase4.py` computed `balanced_accuracy` and `mcc` per local-only client all along
(`local_only_per_client` in the manifest already had them), but the per-seed `local_only` entry
and its seed-level summary tracked only `macro_f1` -- so `baseline_4_local_only.mean` in
`phase4_results.json` reported macro-F1 alone for the lower bound of the G4 bracket while
`baseline_3_centralized` and `baseline_5_federated_global` reported macro-F1, balanced accuracy,
MCC, and accuracy. Added the mean-over-clients `balanced_accuracy` and `mcc` to the per-seed
`results["local_only"]` dict and a new `LOCAL_ONLY_HEADLINE` tuple so the seed-level
mean +/- std summary picks them up the same way the other two baselines' `HEADLINE` does;
`scripts/summarize_phase4.py` needed no change since it copies `summary["local_only"]` wholesale
into `baseline_4_local_only.mean`. Re-running `run_phase4.py` (and `summarize_phase4.py`) will
regenerate the manifest and results file with the new fields; the committed artifacts have not
been regenerated in this change. `false_positive_rate` for local-only is a separate, not yet
implemented gap (the confusion matrix is not currently retained per client) -- flagged, not
fixed here.

## 2026-09-15

### Phase 7 checkpoint regenerated on the fixed training path

`artifacts/federated_global_model.safetensors` (gitignored) and both Phase 7 manifests were
re-created with the fixed `train_federated_model.py` (previous entry). **Macro-F1 0.8507,
bit-for-bit identical to seed 0 of the compute-matched Phase 4 run** (`0.8507461612861715`;
`summarize_phase7.py` reports `reproduces Phase 4 seed 0: True`), which closes the "checkpoint
vintage" caveat in `results/phase7_end_to_end_integration.md`: the deployed model is now the
Phase 4 run's own seed-0 model, trained on its partition and its federated scaler. 18.6 min
from the Phase 4 cache. The end-to-end run on the new checkpoint gives the same routing counts
as before (100 messages, 55 classified, 0 benign / 55 malicious, 0 malicious-verdict payloads
at the cloud, 53/55 informal agreement); per-stage medians 1.2 / 3.9 / 112.0 / 2.4 µs.

### End-to-end verification of all seven phases; four reproducibility breaks fixed

Every phase was executed in order on this machine from the raw corpus (Phase 1 report
re-derived and byte-identical; Phases 2-4 from raw at a reduced budget, both gates closed;
Phase 6 benchmark at 96 B and 512 B; a checkpoint trained and run through Phase 7). The
committed full-budget manifests still close both gates. What broke on the way, and the fix:

- **`scripts/train_federated_model.py` did not train on Phase 4's partition or scaling**,
  despite its "IDENTICAL partition" comment. It re-cut blocks from the train labels (4,842
  blocks against the cache's 4,854, a different Dirichlet draw: client sizes 1164/1849/1829
  against Phase 4's 1166/1855/1833) and, when given the Phase 4 cache, fed the GRU **unscaled**
  features (the cache stores raw values; the script assumed pre-scaled ones -- a smoke run at
  R=2/E=1 scored 0.50 before the fix and 0.71 after, same seed, same data). It now calls
  `run_phase4.py`'s own `build_pipeline`, the new shared `block_strata`, and the federated
  scaler (Eqs. 23-24), and reproduces Phase 4's seed-0 sequence counts exactly
  (284,500 / 458,447 / 452,390). The deployed checkpoint and both Phase 7 manifests are
  regenerated from the fixed script in the entry that follows.
- **The same script hard-coded the superseded 0.8334 +/- 0.0026** from the inverted
  minimal-gate run as the Phase 4 reference and wrote it into every Phase 7 training manifest.
  It now reads the 3-seed mean +/- std from `--phase4-manifest`
  (default `artifacts/manifest_phase4_default.json`) and records which file it read.
- **`scripts/summarize_phase7.py` crashed** (`KeyError: 'federated_per_seed'`) against the
  current Phase 4 manifest: it still read the minimal-gate schema. Ported to `per_seed.federated`.
- **The `asa` CLI raised `NotImplementedError` for every phase** while its docstring promised
  one-binary reproducibility. It now dispatches to the driver under `scripts/` with arguments
  passed through (`asa federate --rounds 2`), and says plainly that Phases 2, 5 and 6 have no
  standalone driver (exit 2). `scripts/run_phase1.py` added so Phase 1's committed report has
  a script that produces it. `tests/test_cli.py` (3 tests) covers the dispatch.

### Merged two parallel Phase 4 implementations

Two Phase 4 drivers were written independently and collided on merge (7 conflicted paths).
Resolved by union rather than by picking a side; what each contributed:

- **`federated_global_gru` keeps the 4-tuple contract** `(metrics, macro_f1_per_round,
  measured_bytes_per_round, model)`. Returning the trained model is not optional —
  `scripts/train_federated_model.py` and Phase 7's runtime need an actual classifier to drive
  routing, and the single-metrics form could not supply one. It now delegates to
  `run_federation`, which additionally carries the Eq. (21) weights and the per-round byte
  series the Phase 4 manifest records, so both callers are served by one implementation.
- **A local-only client with zero sequences is reported, not filtered.** The competing revision
  returned `None` so such a client could be excluded from the mean. That was wrong in the
  direction the bracket exists to expose: Section III-I1 asks for "three local-only GRUs" as a
  report, and dropping the client that received nothing makes the lower bound look better than
  declining to federate actually is. It is now scored against a constant-benign prediction.
- **An explicit all-empty guard** in `run_federation`, raising "every client holds zero
  sequences" at the top rather than letting the failure surface from `weighted_fedavg` as
  "every client reported n_k = 0" — the symptom, not the cause.
- **FedProx (`fedprox_mu`) and the (run seed, client id, round) training seed coexist** in
  `FederatedClient`; only their docstrings conflicted, and both paragraphs are kept.
- **Section III-I2's client-to-global gap** is now computed by `run_phase4.py` and recorded per
  seed. It had existed only in the other driver's post-hoc summariser; on the merged run it is
  positive for every client (+0.1023 / +0.1565 / +0.0721), i.e. federation helped all three,
  which is a stronger statement than the mean alone.
- **`scripts/summarize_phase4.py` was ported** to the surviving manifest schema, keeping the G4
  questions it asks verbatim.
- **Both experiments are kept.** `artifacts/manifest_phase4_default.json` is the compute-matched
  run (161 min); `artifacts/manifest_phase4_minimal_gate.json` is the earlier 441-min run.

Two things the merge exposed that are worth stating plainly:

- **The minimal-gate run has the inverted bracket this repository has already fixed once.** It
  reports federated 0.8334 against a centralised *reference* of 0.8297 borrowed from Phase 3 at
  10 epochs, and a local-only baseline capped at 10 epochs, while federation ran R*E = 60 local
  passes. "Beating the centralised reference on every seed" is a statement about the training
  budget, not the method. Its manifest is retained as a record; its conclusion is not.
- **Git's auto-merge silently dropped test coverage.** `tests/test_federated.py` merged without
  conflict markers to 23 tests, from sides holding 26 and 25 — every test either side had added
  was gone, including both FedProx tests and all three training-seed tests. Rebuilt as the
  union (28). A clean merge is not evidence of a correct one.

Suite after the merge: **344 passed**, all gates green, `check_phase4_gate.py` exit 0.


### Phase 4 gate CLOSED — the federated simulation, run on the real corpus

`scripts/run_phase4.py` ran end-to-end on the official UNB CICIoT2023 distribution and
`scripts/check_phase4_gate.py` exits 0 on the resulting manifest
(`artifacts/manifest_phase4_default.json`). Every number below is mean ± std over seeds
{0, 1, 2} (III-I4); nothing here comes from synthetic data.

Corpus as prepared by Phases 1-2: 46,776,700 raw records -> capped subsample -> dedup -> block
split -> four-stage selection at F=16, giving **1,237,911 train / 311,565 test rows** across
4,854 train blocks, 1,222,009 pooled training sequences at W=16. Wall clock 160.8 min.

| baseline (III-I1) | macro-F1 | balanced acc. | MCC | accuracy |
| --- | --- | --- | --- | --- |
| 4. local-only GRUs (lower bound) | 0.7205 ± 0.0750 | — | — | — |
| 5. **federated global GRU** | **0.8308 ± 0.0150** | 0.8404 ± 0.0088 | 0.8845 ± 0.0174 | 0.9090 ± 0.0151 |
| 3. centralised GRU (upper bound) | 0.8543 ± 0.0040 | 0.8612 ± 0.0035 | 0.9021 ± 0.0020 | 0.9235 ± 0.0020 |

**The G4 answer:** the federated model sits inside its bracket and recovers **82.4 %** of the
macro-F1 that local-only training leaves on the table, for **815,136 measured bytes per round**
(Eq. 22 predicts 811,200; the 0.5 % excess is safetensors framing). Local-only is also by far the
least stable baseline (± 0.0750 against the federated ± 0.0150), because its score depends
entirely on which classes its client happened to be dealt: at seed 1, where two clients were
missing classes, local-only fell to 0.6145 while the federated model still reached 0.8145.

**Read the FPR before reading anything else.** Under the Eq. (5) binary projection the federated
model has a false-positive rate of **0.2975 ± 0.0512** (centralised 0.2767 ± 0.0283) at ~98.8 %
attack recall — roughly **three in ten benign flows raise an alarm**. Section III-I2 makes FPR
first-class precisely because this is the number that gets a detector switched off in
production, and no amount of 0.92 accuracy compensates for it. Accuracy (0.9090) sits below the
0.98 near-ceiling threshold, so the III-I5 note correctly does not fire — this run is not in
the dataset-artifact regime, and the weak classes are real: BruteForce 0.6339, WebBased 0.6522
and Benign 0.7148 per-class F1, against Mirai 0.9849 and Spoofing 0.9673.
  - These FPR figures are **derived from the confusion matrices in the committed manifest**, not
    separately measured: FPR is a pure function of the benign row. `run_phase4.py` now records
    `false_positive_rate` per seed and carries it in the headline set, so the next run stores it
    directly; this manifest predates that and was not re-run for a derived quantity.

**Eqs. (23-24) verified on real data:** the federated scaler, built only from per-client
count/mean/M2, matched a pooled fit to a maximum relative gap of **5.42e-12** across all three
seeds -- the III-F4 claim holding on 1.2M real rows, not only in a unit test.

### Fixed: the bracket was measuring the training budget, not the method

The first real seed put the federated model (0.8507) **above** its own centralised upper bound
(0.8293), which reads as "federation beats pooling" and is nothing of the sort. Federation makes
R*E = 20*3 = 60 local passes; baseline 4 was already matched to that, but baseline 3 was left at
the Phase 3 default of 10 epochs — a 6x training advantage handed to the method under test. With
the budget equalised the centralised baseline rose to 0.8584 on that seed and the ordering came
right. `--central-epochs` now defaults to `R*E`, the manifest records an `epoch_budget` block,
and `check_phase4_gate.py` gained an item that **fails** if the three budgets ever diverge again.
The run was stopped and restarted rather than allowed to finish and publish the inverted result.

### Changed: the centralised baseline is windowed over the whole training split

It previously trained on the concatenated client windows. "Upper bound attainable by pooling"
means a trainer that never sees the partition, so its windows must not be fragmented at
partition boundaries the way a client's are; concatenating handed the upper bound the federated
setting's handicap. It also matches how Phase 3 built this baseline, which is what makes the two
phases comparable — and that comparability is now evidence: at Phase 3's own 10-epoch budget
this pipeline reproduced Phase 3's centralised macro-F1 to **0.8293 against 0.8297 ± 0.0013**,
an independent end-to-end check that Phases 1-3 replay faithfully on this machine. Building the
pooled tensor before the per-client ones and freeing it also removed a ~1.2 GB duplicate from
peak memory.

## 2026-09-14

### Added: Phase 4 gate checker, and a real fix to the local training seed

- **`scripts/check_phase4_gate.py`** -- the machine-checkable answer to "is Phase 4 done?",
  mirroring `check_phase3_gate.py` so closing the phase is not a judgement call. It reads a
  Phase 4 manifest and checks: >= 3 seeds; baselines 3-5 reported with mean +/- std; accuracy
  never reported alone (III-I2); `aggregation == "weighted"` and the applied weights equal the
  per-client **sequence** counts rather than the row counts (Eq. 21); the global model reached
  all K clients; per-client class histograms published with a shared vocabulary so an absent
  class shows as 0 (III-F1 / G3); the federated-vs-pooled scaler gap within floating-point
  tolerance (Eqs. 23-24); communication cost measured against Eq. (22); the G4 bracket; and a
  manifest carrying config, versions and a real git commit (III-I4).
  - Like the Phase 3 checker it deliberately does **not** check that federation performed well.
    A federated model at the bottom of its bracket, properly measured, closes the gate.
  - **Verified to fail, not just to pass.** A gate that cannot reject is worse than none, so it
    was run against eight deliberately broken manifests -- unweighted aggregation, weights
    swapped to row counts, two seeds, a corrupted scaler gap, dropped histograms, a missing
    bracket, an absent git commit, and a client the global model never reached. All eight exit
    non-zero with the relevant item marked FAIL.

- **`FederatedClient` now seeds local training from (run seed, client id, round)** via a
  `SeedSequence`, replacing the bare `client_id`. The old seeding had two consequences that
  never surface as a crash, only as distorted numbers: within a run every round re-seeded
  identically, so a client replayed the *same* batch permutation in round 20 as in round 1 and
  the shuffle stopped being a shuffle after the first round; and across runs local training was
  independent of the experiment seed, so the >= 3-seed spread of Section III-I4 sampled only the
  initial parameters and the Dirichlet draw and reported a tighter std than the method actually
  has. This was raised twice as an open question before the real run rather than discovered
  after it. `training_seed()` is exposed so a test can pin the three inputs apart, and three
  tests in `test_federated.py` assert rounds differ, the run seed matters, clients differ from
  each other, the derivation stays reproducible, and the round counter actually advances.
  `run_federation` passes the run seed down to each client. Suite: **250 passed, 1 skipped**.

### Still blocking Phase 4 (the phase is NOT closed)

The federated experiment has still not been run on CICIoT2023, because the corpus is not on this
machine: `data/ciciot2023_raw/` does not exist, and there is no prepared cache. Everything above
was verified on synthetic arrays, which establishes that the runner and the gate behave
correctly and **nothing else** -- no macro-F1, bracket position, or convergence curve from a
synthetic run is a finding, and the synthetic manifest produced while testing the gate was
deleted rather than committed. Phase 4 closes when `scripts/run_phase4.py` has been run against
the real corpus and `scripts/check_phase4_gate.py` exits 0 on the resulting manifest.

### Changed: toolchain pinned to one set of versions; pre-commit hook actually installed

`pre-commit install` had never been run in this checkout, and the config had drifted from the
`dev` extra badly enough that installing it would have baked a contradiction into every commit.

- **Hook revs realigned with `pyproject.toml`'s `dev` extra:** ruff `v0.6.9` -> `v0.16.7`,
  vulture `v2.13` -> `v2.16`, mypy `v1.11.2` -> `v2.3.1`. pre-commit builds each hook its own
  isolated environment at the pinned rev, so a drifted pin means the hook and a local
  `ruff check .` are *different programs* -- one can pass while the other fails. vulture is the
  sharpest case: `pyproject.toml` pins it with an exact `==2.16` while the hook asked for 2.13,
  so the config contradicted itself. A comment at the top of the config now says to keep them
  in step.
- **`ruff` hook id -> `ruff-check`**; the bare `ruff` id is a deprecated alias in ruff-pre-commit
  >= 0.12 and emitted a warning on every run.
- **Python floor raised to 3.12** (`requires-python`, ruff `target-version`, mypy
  `python_version`). This is forced by the dependencies rather than chosen: numpy (>= 2.4) and
  scipy (>= 1.16) both declare `requires-python >= 3.12`, so `>= 3.11` was already a false
  claim -- the pinned stack cannot install on 3.11. It also un-blocked mypy, which under a 3.11
  target refuses to parse numpy's bundled stubs (they use PEP 695 `type` statements) and so
  failed before reaching any of our code.
- **`features/selection.py` Stage 2 now indexes the Spearman matrix as a float array**
  (`.to_numpy(dtype=np.float64)` plus a column->position map) instead of `spearman.at[row, col]`.
  Under the current pandas-stubs, `.at` is declared as a union spanning `str`/`bytes`/`datetime`,
  which cannot be compared against a float threshold; `float(...)` does not fix it either, since
  the union includes members `float()` rejects. A correlation is a float, so the array is the
  honest type. **Behaviour is unchanged** -- all 18 feature-selection tests still pass. This was
  a pre-existing failure surfaced, not caused, by the pin alignment.
- **`pre-commit install` run**; `pre-commit run --all-files` is green on every hook.

### Changed: setup docs now cover both platforms, and describe a venv that exists

`CLAUDE.md`'s Commands section documented `./.venv/Scripts/ruff.exe` -- Windows paths -- while
also claiming the scientific stack was "already installed in the system Python 3.11" and that
`.venv` was created with `--system-site-packages`. On a POSIX checkout none of those commands
run, and the system-Python claim is no longer true anywhere.

- `CLAUDE.md` and `README.md` now give **both** macOS/Linux (`.venv/bin/`) and Windows
  (`.venv\Scripts\*.exe`) invocations for setup and for every gate, state the 3.12 floor and
  why it exists, and drop the stale `--system-site-packages` / system-Python framing: everything
  installs into a project-local `.venv`.
- Documented the two things that actually bite: the pytest hook is `language: system` so the
  venv must be **activated** (not just addressed by path) when committing, and hook revs must
  track the `dev` extra.
- Dropped README's "add `,crypto` once the Ascon backend is chosen" -- there is no `crypto`
  extra; the backend is vendored.

### Added: Phase 4 baselines 4-5 and the federated experiment runner

- **Baselines 4 and 5 of Section III-I1 implemented** in `eval/baselines.py`, the last two
  `NotImplementedError` stubs Phase 4 owed. `local_only_grus` trains one GRU per client on that
  client's partition alone; `federated_global_gru` runs R rounds of Algorithm 1 and scores the
  resulting global model. Together with baseline 3 they form the bracket that answers gap G4.
  - **Signatures widened to take an explicit test set**, exactly as baselines 1-3 were: the
    scaffold's `(client_seqs, client_y, seed)` cannot express a train/test split and so could
    not return test metrics at all. Flagged here as a scaffold correction, not worked around.
  - **Every baseline is scored on the shared global test set** (Section III-B3), local-only
    clients included. Scoring a local model on its own partition's held-out slice would ask it
    an easier, different question -- its partition is class-skewed by the Eq. (20) draw -- and
    would make baselines 3, 4 and 5 three unrelated numbers instead of a bracket.
  - **A client with zero sequences yields `None`, not a zero-filled metric bundle.** Under
    alpha = 0.1 a client can legitimately receive no blocks (see `federated/partition.py`) and
    has no detector to report. A bundle of zeros would silently drag a reported mean down as
    though the client had trained and failed. This widens baseline 4's return type to
    `list[MulticlassMetrics | None]`, deliberately.
  - **Per-client seeds come from a `SeedSequence` spawned off the run seed**, so client 0 at
    seed 1 is not the same run as client 1 at seed 0.
  - `run_federation` returns a `FederatedRun` carrying the pieces the manifest needs beyond the
    headline metric: the Eq. (21) weights actually applied, the Eq. (22) bytes actually sent per
    round, and (optionally) the per-round convergence curve. `federated_global_gru` is a thin
    wrapper over it so the Section III-I1 baseline contract stays a single metric bundle.

- **`scripts/run_phase4.py`** -- the federated experiment, mirroring `run_phase3_complete.py`.
  Runs baselines 3-5 over >= 3 seeds at a given alpha, publishes the per-client class histograms
  (Section III-F1 / gap G3) including zero counts, reports the bracket, and writes a manifest.
  - **The global scaler is built federatedly and is now actually exercised** (Eqs. 23-24).
    `build_pipeline` deliberately does **not** standardise: clients emit count/mean/M2 only, the
    server combines them, and the scaled arrays are derived from that result. An earlier draft
    scaled with a pooled fit in the pipeline and computed the federated statistics beside it,
    which made the III-F4 path dead code; that was wrong and is fixed. The run prints the
    federated-vs-pooled gap as a live check (4e-15 on the synthetic smoke run).
  - Carries its **own cache format** (`--save-cache`/`--cache`) because Phase 4 needs the
    per-row block ids that the Phase 3 cache does not store.

- **Tests:** `test_phase4_baselines_remain_gated` -- which asserted the two stubs still raised --
  is removed, being obsolete the moment they were implemented, and replaced by seven tests
  covering the shared-test-set property, the empty-client `None`, seed independence, learning,
  the published Eq. (21)/Eq. (22) figures, and malformed-partition rejection. Suite: **247
  passed, 1 skipped** (the remaining skip is Phase 6 routing).

### Not done, and not to be reported as done

- **The federated experiment has not been run on CICIoT2023.** The runner was verified end-to-end
  on synthetic arrays only; every macro-F1 it printed there is noise from generated data. No
  Phase 4 number is a finding until the run happens on the real corpus, and the synthetic-data
  manifest that smoke run produced was deleted rather than committed.
- **Observation for the team, not silently changed:** `FederatedClient.local_train` seeds
  `train_module` with `self.client_id`, so a client's batch shuffling is identical across
  experiment seeds. Run-to-run variance therefore comes only from the initial global parameters
  and the Dirichlet partition, which under-samples the spread that III-I4's mean +/- std is
  meant to report. Left as-is because it is Phase 4 infrastructure someone else wrote and its
  tests pass; worth a decision before the real run.

### Added

- **Phase 3 completed: scaling, training loop, metrics, reporting guard and manifest.**
  New `data/scaling.py`; implemented `model/train.py`, `eval/metrics.py`, `eval/report.py` and
  `eval/manifest.py`; new experiment driver `scripts/run_phase3.py`. New `tests/test_scaling.py`
  (8), `tests/test_metrics.py` (9), `tests/test_training.py` (15), `tests/test_manifest.py` (4).
  - **Pooled scaler kept out of `federated/`.** The Phase 3 scaler lives in `data/scaling.py`
    rather than in Phase 4's `federated/scaler_stats.py`, so Phase 4's module stays closed until
    its gate opens. Statistics are stored as `(count, mean, M2)` — the form Chan's parallel
    formula (Eqs. 23-24) combines — specifically so `tests/test_scaler_equivalence.py` has a
    real pooled fit to compare the federated combination against when Phase 4 starts.
    Zero-variance features are divided by 1.0, not 0.0.
  - **Non-finite policy for training data, decided and documented.** The selected F=16 columns
    carry **47 NaN cells in `Variance`** (0.004% of training rows, genuine nulls in the raw
    CSVs). `fit_scaler` *refuses* non-finite input rather than imputing, and the driver drops
    the affected rows. Dropping leaves a positional gap, which `contiguity_segments` turns into
    a segment break — so no window spans the hole, exactly as none may span a held-out block.
    Measured: 47 rows dropped from train, 8 from test.
  - **Class-weighted CE (Eq. 18) with a guard the paper does not specify:** a class absent from
    the training split gets weight **0**, not infinity. Eq. (18) divides by n_c, undefined at
    n_c = 0; an infinite weight would let one stray prediction dominate the loss. The absence
    stays visible in the per-class F1 instead.
  - **Metrics honour the "never accuracy alone" invariant, and the macro average includes
    classes absent from the split** (F1 = 0 rather than dropping them from the denominator,
    which would flatter a detector precisely on the rare families). PR-AUC returns NaN — not a
    fabricated 0.0 or 1.0 — when only one class is present. A regression test asserts a
    majority-class detector scoring 0.96 accuracy is correctly punished to macro-F1 < 0.35.
  - **The Section III-I5 near-ceiling guard is live**: accuracy at or above the configured
    threshold automatically emits the note directing the reader to macro-F1 and FPR.
  - **Manifest writing implemented**, capturing seeds, full config snapshot, library versions,
    platform, git commit **and whether the tree was dirty** — a dirty tree means the commit does
    not fully identify the code that produced the number.
  - **PHASE 3 EXIT CRITERION MET — the centralised GRU trained and evaluated on the real
    corpus**, 3 seeds (0/1/2), W=16, F=16, 3 epochs, 1,222,009 training and 296,750 test
    sequences, ~11 min/seed on CPU. Manifest at `artifacts/manifest_phase3_default.json`
    (gitignored). Reported as mean ± std over seeds, never a single run:

    | Metric | Result |
    | --- | --- |
    | macro-F1 | **0.7866 ± 0.0199** |
    | macro precision / recall | 0.7575 ± 0.0129 / 0.8660 ± 0.0084 |
    | balanced accuracy | 0.8660 ± 0.0084 |
    | MCC | 0.8532 ± 0.0097 |
    | binary PR-AUC | 0.9993 ± 0.0000 |
    | **binary FPR** | **0.2573 ± 0.0612** |
    | accuracy (alongside only) | 0.8818 ± 0.0080 |

    Per-class F1: Mirai 0.999, Spoofing 0.945, DDoS 0.927, DoS 0.847, Reconnaissance 0.832,
    Benign 0.703, **WebBased 0.529 ± 0.063, BruteForce 0.512 ± 0.076**.
  - **Three findings that the evaluation protocol exists to surface, recorded rather than
    smoothed over:**
    1. **Accuracy is 0.88, not the 98-99% the CICIoT2023 literature routinely reports** — so the
       Section III-I5 near-ceiling note did *not* fire. This is the expected consequence of the
       leakage controls, not underperformance: R3's dedup-before-split removed 25.6M duplicate
       rows corpus-wide (54.8%) and block-level splitting keeps neighbouring records on one side
       of the split. Published near-ceiling figures on this benchmark are the number you get
       *without* those controls. The gap is evidence the controls bind.
    2. **The binary false-positive rate, 0.257, is far too high to deploy.** Roughly a quarter of
       benign windows are flagged as attack, which is precisely the failure mode Section III-I2
       makes FPR a first-class metric to expose ("a high false-alarm rate is what gets a detector
       switched off"). Binary PR-AUC of 0.9993 looks excellent and hides this completely — the
       threshold, not the ranking, is what is wrong. Tuning the operating point is future work.
    3. **The model is under-trained at 3 epochs.** Weighted CE was still falling when training
       stopped (0.575 → 0.377 → 0.311 per epoch, consistently across all three seeds). The
       reported numbers are therefore a floor, not the architecture's ceiling, and the two rare
       families (BruteForce, WebBased) also carry the largest seed-to-seed spread (±0.076,
       ±0.063). A longer run is expected to improve all three; the exit criterion is that the
       protocol is *reported*, which it is, not that the detector is good.
  - **Runner buffering fixed.** The first full run redirected stdout to a log file, where Python
    block-buffers a non-TTY stream — the log stayed empty for the entire 21-minute run, making
    progress unobservable. `scripts/run_phase3.py` now line-buffers its own output.

- **Phase 3 opened (user-authorised): taxonomy, windowing and the GRU detector implemented.**
  New `data/taxonomy.py`, implemented `sequences/windowing.py` and `model/gru.py`. New
  `tests/test_taxonomy.py` (10), `tests/test_windowing.py` (21), `tests/test_gru_detector.py`
  (9); `tests/test_model_param_count.py`'s Phase-3 skip is removed and three cases added.
  Pytest now 151 passed / 3 skipped (up from 102 passed / 4 skipped). Phase 3 is **not**
  complete: the training loop, feature scaling and the Section III-I evaluation protocol remain,
  so its exit criterion is unmet and Phase 4 stays closed.
  - **Taxonomy discrepancy reconciled (flagged per Golden Rule 1, confirmed with the user).**
    The paper states the class taxonomy twice and the statements disagree: Section III-B3's prose
    says "benign plus seven attack families" (C = 8), `ModelConfig` hard-validates
    `n_classes == 8` and Eq. (19)'s 33,800 parameters assume C = 8 — but **Table I lists only
    six family rows**, merging DDoS and DoS into one `DDoS / DoS` row. Splitting that merged row
    is the only reading satisfying prose, validator and Eq. (19) at once; the alternative (follow
    Table I literally at C = 7) was put to the user and rejected. `data/taxonomy.py` is now the
    single place the mapping lives. Two placements inside it are judgement calls following the
    canonical CICIoT2023 grouping, both recorded: `VulnerabilityScan` → Reconnaissance and
    `Backdoor_Malware` → WebBased. **Verified against the real dataset**: all 34 on-disk labels
    map, none unmapped, no mapped label absent from disk.
  - **Benign is fixed at class index 0**, so Eq. (5)'s binary projection is exactly `y > 0` and
    never needs a lookup. Unknown labels raise rather than falling back to a default class,
    which would otherwise corrupt the per-class metrics of Section III-I.
  - **Windowing implemented** with runs defined as maximal stretches sharing *both* label and
    source id, sequences emitted in input order (never shuffled — Section III-D reserves
    shuffling for complete sequences at batch time), labels taken from the final record
    (y_i = y_{i+W-1}), and Eq. (12) checked directly against `count_sequences`.
  - **Correctness bug found and fixed while composing the pipeline: windows could span a
    held-out block.** Splitting operates on whole blocks, so filtering the deduplicated frame
    down to the train blocks leaves rows that were never neighbours sitting adjacent — block 5
    and block 7 become adjacent when block 6 goes to test. Windowing over that filtered frame
    fabricates an adjacency the capture never contained, which the positional-ordering
    assumption of Section III-D does not license. Measured on the real training split: **977
    such discontinuities, which would have produced 14,508 bogus sequences (1.2% of the training
    set)** built on adjacency that never existed. Fixed by the new
    `windowing.contiguity_segments(source_ids, row_positions)`, which breaks a segment wherever
    the source file changes *or* the original row positions stop being consecutive; callers pass
    its output as `build_windows`' `source_ids`. Regression tests cover both break causes.
  - **`build_latency_eval_windows` gained a keyword-only `benign_label`** (a scaffold signature
    extension): the module needs to know which class counts as benign to find benign→attack
    transitions, and taking it as a parameter keeps `sequences/` independent of
    `data/taxonomy.py`'s indices, working on leaf labels or family indices alike.
  - **GRU detector implemented** as GRU → LayerNorm(h_W) → Linear, reading the final hidden
    state (consistent with y_i = y_{i+W-1}). LayerNorm never BatchNorm, asserted by a test that
    walks the module tree. The default sizing reproduces **exactly 33,800 parameters**, matching
    Eq. (19) term for term (GRU 32,832 + LayerNorm 192 + head 776), and the formula/built-model
    equality is now checked at four further sizings rather than only the reference point.
    `torch` moved from a `TYPE_CHECKING` guard to a real module-scope import.
  - **Tooling:** a narrowly-scoped `disallow_subclassing_any = false` override for
    `ascon_smart_agri.model.*` in `pyproject.toml`. `torch` is deliberately not a pre-commit
    mypy `additional_dependency` (it would add a multi-hundred-megabyte download to the hook
    environment for a stubs-only benefit), so `nn.Module` resolves to `Any` there and strict
    mode's `disallow_subclassing_any` fired on the detector, while the local `mypy src` run —
    which does see torch — type-checks the module properly. Both environments are now green.
  - **Full chain verified end-to-end on real data**: raw CSVs → subsample → dedup → split →
    four-stage selection → taxonomy → windowing → GRU forward pass. 1,237,958 train rows → 34
    leaves mapped to 8 classes → F=16 of F0=31 → **1,222,745 sequences** of shape (16, 16) over
    1,015 runs → (64, 8) logits from a 33,800-parameter model. At family level the imbalance
    ratio is **44.7**, not the leaf-level IR ≈ 5751 the paper quotes — which is exactly the
    motivation Section III-B3 gives for reporting at family granularity. Benign is 4.5% of
    training rows.

### Phase 7 complete: real model trained, saved, and run through the real pipeline

Closes the last item from the gap audit: `scripts/train_federated_model.py` trains and saves
(via `model/checkpoint.py`, new this entry) one federated global model checkpoint under the
IDENTICAL Phase 4 configuration and data partition -- same seed, same α=0.5, R=20, E=3,
weighted FedAvg. Scope decision (flagged): seed 0 only, not a fresh 3-seed run, since Phase 4
already established the statistical claim and Phase 7 needs a real deployment artifact, not a
second validation. 54.8 minutes (faster than Phase 4's contended 78.6-minute seed-0 run; no
contention this time). **Result: macro-F1 0.8338 -- bit-for-bit identical to Phase 4's seed-0
run** (verified against Phase 4's own manifest, not a hand-typed constant -- see the bug note
below), confirming the training path is genuinely deterministic end to end.

`scripts/run_phase7.py` is the first script in this project to execute the paper's own
sentence for real: *"telemetry arrives, is classified using the current global model, and is
routed according to Equation (5)."* One message at a time, in order:
`simulate_stream` → `FeatureProvenanceAdapter` (G6, held-out only) → `DeviceWindowBuffer` (new
this entry, `sequences/streaming.py`) → the real saved model → Eq. (5)'s `y > 0` projection →
`VerdictRouter` → real `AsconAEAD128`/`MockCloudReceiver` or `AlertSink`. Validated with a real
smoke run (a tiny 2-round checkpoint on sliced data, 80 messages) before the expensive training
finished, then run for real against the trained checkpoint.

**Real end-to-end results** (100 simulated messages, 3 devices, W=16): 45 still buffering
(correct -- no device had accumulated a full window yet, no padding was fabricated), 55
classified (0 benign, 55 malicious). Cloud received exactly 0, matching 0 benign verdicts
exactly; alert count exactly 55, matching every malicious verdict; **zero malicious-verdict
messages reached the cloud** -- G1 holds through the fully assembled runtime loop, not merely
in the isolated unit tests. Per-stage latency (median): provenance 3.6us, window 10.5us,
inference 273.7us, routing 4.1us -- inference dominates, as expected for a GRU forward pass
versus dict lookups and byte serialisation.

- **A real, verified finding, not a bug: this run's sample drew 0 true-benign records among
  its 55 classified windows' final records that the model called benign** (53/55 = 96.4%
  informal agreement with true_label -- NOT a formal evaluation, see the module's own
  disclaimer). Checked rather than assumed: the held-out pool's true benign rate is 4.52%, and
  the seeded draw for stream positions 45-99 (the ones that got classified, given 45 messages
  went to filling buffers first) landed exactly 2 true-benign records -- almost exactly the
  4.52% x 55 ≈ 2.5 expected by chance. Both of those 2 were (informally) misclassified as
  malicious, a small-sample (n=2) echo of the already-documented binary FPR limitation
  (0.257, measured formally in Phase 3) -- not new evidence, and far too small a sample to be
  one. Phase 6's own integration test already proved benign routing works correctly with real
  crypto (8 of 200 messages there were genuinely benign and all 8 round-tripped correctly); this
  run simply didn't draw one in its classified range.
- **A real bug in the verification script itself, caught and fixed before it shipped a wrong
  answer.** `scripts/summarize_phase7.py` first compared the checkpoint's macro-F1 against a
  hand-typed `0.8338` (the value as printed, truncated to 4 decimals) with a `1e-6` tolerance,
  and reported **"reproduces Phase 4 seed 0 exactly: False"** for a run that was in fact
  bit-for-bit identical -- the true value is `0.8338014523294024`, which differs from the
  truncated constant by `1.45e-6`, just over the tolerance. Fixed to read Phase 4's actual
  recorded seed-0 value from its own manifest and compare for exact equality, rather than
  trusting a copied-in digit string to stay in sync with the real number. Found by checking the
  comparison's own inputs rather than accepting "False" at face value.
- New `scripts/summarize_phase7.py` (distilling both `manifest_phase7_train_default.json` and
  `manifest_phase7_e2e_default.json` into `artifacts/phase7_results.json`, same relationship as
  Phase 4's summarizer). The model checkpoint itself
  (`artifacts/federated_global_model.safetensors`, 136 KB) stays gitignored as bulk output,
  consistent with the existing `*.safetensors` policy; the manifests and results file, being
  JSON provenance, are tracked.

### Phase 7 opened: three gaps closed that don't require retraining

Audited Section III-I2 against the codebase before writing any Phase 7 code (the Phase 3
lesson, applied proactively this time rather than after declaring something complete): grepped
for `client_to_global_gap`, encrypt/decrypt latency measurement, and any persisted model
weights. Found the first two never implemented and the third genuinely absent -- no training
run in this project has ever saved a model to disk. Closed the two that don't need retraining
now; the third (a real classifier driving the runtime pipeline) needs a fresh federated run and
is scoped separately below.

- **`sequences/streaming.py`, closing the gap `telemetry/provenance.py` explicitly deferred
  here.** `DeviceWindowBuffer` assembles `(W, F)` windows per device from single feature
  vectors arriving one call at a time -- the streaming counterpart to
  `sequences/windowing.py`'s offline, whole-batch construction. Rolling per DEVICE (not a
  single shared buffer, which would splice unrelated devices' readings together); a window is
  emitted the instant a device's buffer first reaches `W` and then slides by one on every
  subsequent push, consistent with `y_i = y_{i+W-1}`; nothing is ever padded to fake a window
  before real data fills it, the same objection that already rules out synthetic oversampling
  and fabricated network features. New `tests/test_streaming_windows.py` (12).
- **`eval/report.py`'s `client_to_global_gap`**, defined as `federated_macro_f1 -
  local_only_macro_f1` per client, both scored on the SAME shared test set (flagged, Golden
  Rule 1: the paper names this metric in Section III-I2 but never defines its sign or exact
  form). Positive means federation helps that client; negative is the G4/R4 finding the paper
  explicitly permits and asks to be reported honestly. **Retroactively applied to Phase 4's own
  results** -- `scripts/summarize_phase4.py` now computes it from the ALREADY-SAVED
  `manifest_phase4_default.json` (no retraining needed) and `artifacts/phase4_results.json` was
  regenerated: every client gains from federation, +0.063 to +0.100 macro-F1. New tests in
  `tests/test_training.py` (4).
- **`eval/crypto_benchmark.py`**, measuring the already-implemented, KAT-verified
  `AsconAEAD128` facade -- no new cryptography. A real bug in the benchmark itself, not the
  crypto, surfaced on its first run: `AsconAEAD128.encrypt()` returns only `ciphertext||tag`
  (payload + 16 bytes); Eq. (29)'s 32-byte wire-expansion figure additionally counts the nonce,
  which travels as a separate field in this design and isn't part of that return value. Fixed
  to add `len(nonce)` explicitly; the fix was found because the benchmark's own first run
  disagreed with the paper's worked 33%/6.3% examples, not assumed correct in advance. New
  `tests/test_crypto_benchmark.py` (7) checks the expansion exactly against those two worked
  examples. **Real measurements on this machine** (the vendored pure-Python reference backend,
  not an optimised implementation -- see the Phase 6 backend-selection entry):

  | Payload | Expansion | Encrypt (median / p95) | Decrypt (median / p95) |
  | --- | --- | --- | --- |
  | 96 B | 32 B (33.33%) | 390.8 us / 400.9 us | 391.9 us / 403.3 us |
  | 512 B | 32 B (6.25%) | 1300.5 us / 1327.8 us | 1302.1 us / 1330.3 us |

  Expansion matches the paper's stated 33%/6.3% exactly, since it is a structural property of
  the scheme (Eq. 29), not a measurement with noise -- latency is genuinely empirical and
  machine-dependent, reported as a distribution rather than a single number for that reason.

Pytest now **329 passed / 0 skipped** (up from 306/0).

### Phase 6 complete: alerting path, replay window, G1's behavioural half -- zero skips left

Implements `routing/alert_sink.py`, `routing/cloud_sink.py`, `routing/router.py`, and a new
`routing/replay_guard.py`, closing out Phase 6 (the crypto core was implemented ahead of its
gate back in Phase 6's first entry; this completes the routing half). New
`tests/test_replay_window.py` (9), `tests/test_cloud_sink.py` (8), `tests/test_router.py` (7),
`tests/test_alert_sink.py` (4); `test_path_disjointness.py`'s remaining skip activated with a
real behavioural test. **Pytest now 306 passed / 0 skipped** (up from 277/1) -- every test in
the suite is live for the first time in this project.

- **Replay-window policy confirmed with the user before any code was written**, per
  `docs/plans/phase6-replay-window.md`'s explicit instruction that this was not yet
  authorised. Chosen: per-`(edge_id, device_id)` scope, strict-monotonic acceptance
  (`counter > last_seen`), in-memory state -- the plan doc's own recommended sketch. Implemented
  as `routing/replay_guard.py`'s `ReplayGuard`, deliberately its own small object rather than
  folded into the crypto core, matching `AssociatedData`'s docstring which already stated
  replay state must not live there.
- **Verification order is fixed and tested as load-bearing**: `MockCloudReceiver.send_encrypted`
  decrypts/verifies (Eq. 26) FIRST, checks replay SECOND, never the reverse. A message with a
  tampered tag must never touch replay state, or a forged counter could poison the window for a
  legitimate later message -- `test_cloud_sink.py`'s
  `test_a_failed_verification_never_advances_replay_state` sends a corrupted message at
  counter=5 (rejected), then the genuine message at counter=5 (accepted), proving the counter
  was never consumed by the failed attempt.
- **`route()`'s scaffold signature (`edge_id: str` alone) could not build a valid Eq. (27) AD
  tuple** -- `device_id`, `counter`, `schema_version` were simply absent, so the benign path
  could not have encrypted correctly. Widened (flagged) to take a full `AssociatedData`, which
  already carries everything Eq. (27) needs, rather than adding three more scattered
  parameters. Same class of correction as the baseline-function widenings in Phases 3-4.
  `VerdictRouter` also gained an owned, cross-call-persistent `NonceRegistry` (injectable for
  testing) -- the scaffold's constructor had nowhere to draw nonces from at all.
  `MockCloudReceiver.__init__` gained a `keys: dict[str, bytes]` DEMO key store (explicitly
  labelled as such; production key management is named out of scope by the paper itself), since
  the receiver's own docstring already said it must "select the decryption key" and the
  scaffold gave it no way to.
- **G1's structural half is unchanged and still gating** (`AlertSink.__init__` takes no
  transport dependency; introspection confirms no instance attribute is ever a
  `CloudTransport`). The newly-activated behavioural half routes 5 real malicious-verdict
  messages through the real `VerdictRouter` and asserts the cloud receiver's
  `received_count` AND `rejected_count` both stay 0 -- not merely "nothing was accepted" but
  "nothing was even attempted", since the alert sink cannot reach the cloud transport to try.
- **Verified end-to-end on real data**, chaining Phase 5's telemetry/provenance layer into
  Phase 6's crypto/routing layer for the first time: 200 real simulated messages, each paired
  with a real held-out CICIoT2023 record via `FeatureProvenanceAdapter`, routed through the
  real `AsconAEAD128`/`VerdictRouter`/`MockCloudReceiver` stack using the record's true label as
  a stand-in verdict (never fed to a model -- a smoke-test substitute, since Phase 7 is where a
  real classifier gets wired in). Of 200 messages, 8 were benign and 192 malicious; the cloud
  received exactly 8, rejected 0, and every accepted payload decrypted to byte-identical
  original content; the alert sink recorded exactly 192, and the cloud saw zero of them.
- **Scope note:** replay persistence across a receiver restart, and the full III-G4 overhead
  measurement (wire expansion at multiple payload sizes) are not implemented -- the former is
  explicitly out of scope alongside production key management, the latter is a reporting task
  for Phase 7's end-to-end run rather than a Phase 6 gate item.

### Phase 5 complete: telemetry simulation + feature-provenance adapter (G6)

Implemented `telemetry/simulate.py` (`simulate_stream`) and `telemetry/provenance.py`
(`FeatureProvenanceAdapter`), plus a new `TelemetryConfig` in `configs/base.py`/`default.yaml`.
New `tests/test_telemetry_simulate.py` (11) and `tests/test_telemetry_provenance.py` (15) --
pytest now **277 passed / 1 skipped** (up from 251/1).

- **Two planes, kept structurally apart.** `simulate.py` generates only application-layer JSON
  payloads (`deviceId`/`temperature`/`soilMoisture`, matching Section III-H's own example) and
  has no import of or reference to anything network-feature-related; `provenance.py` draws only
  from held-out CICIoT2023 rows and never reads a payload field. Neither module can accidentally
  bridge the two planes, because neither has the other's data in scope.
- **`FeatureProvenanceAdapter.from_held_out_frame`, a convenience constructor that makes the
  correct construction the easy one.** Point it at the Phase 2 TEST split directly (already
  proven never-trained-on by the R3 leakage gate) and it derives provenance refs from
  `data/subsample.py`'s existing `source_file` column and the frame's own pooled-corpus index --
  no new held-out pool or bookkeeping invented. Features are returned already selected and
  scaled with the SAME fitted scaler training used, closing off train/serve skew structurally.
- **G6 verified on real data, not only synthetic fixtures.** Built the adapter from the actual
  311,573-row Phase 2 test split and paired it with a real simulated stream: sampled 200+
  distinct provenance refs and checked every one against the 1,237,958-row training index --
  **zero leaked**. This is the same "prove it on the real corpus, not just a unit test" standard
  applied earlier to the federated scaler path.
- **A real pairing bug found integrating the two modules, not caught by either module's own
  unit tests in isolation.** `TelemetryMessage` originally carried only `counter`, monotonic
  PER DEVICE (correct for its actual purpose -- Eq. 27's replay-protection AD tuple scopes
  counters per device). But the adapter's `network_features_for` took a bare int with no
  documented distinction, so pairing on `counter` gave every device's message 0 the *identical*
  held-out network record, message 1 the identical next one, and so on -- only surfaced by
  running a real simulated stream against a real adapter and inspecting the output, not by
  either module's tests alone. Fixed by adding a second field, `stream_index` (monotonic across
  the WHOLE stream, never repeating), and renaming the adapter's parameter from the scaffold's
  `message_counter` to `stream_index` so the correct call is the only obviously-named one. A
  regression test demonstrates the bug directly (pairing on `counter` collides; pairing on
  `stream_index` does not) rather than only testing the fix in isolation.
- **`true_label` added to `ProvenancedFeatures`** (flagged scaffold extension): carries the
  held-out record's ground-truth label for demo narration and test assertions only -- never
  read by anything that classifies, and the returned feature vector's shape is unchanged by it.
- **New `TelemetryConfig`** (`device_ids`, `messages_per_stream`, `schema_version`, sensor
  ranges, `seed`), following the project's typed-pydantic-config convention rather than
  scattering these as function defaults.

### Phase 4: two gaps closed after the minimal gate (FedProx wiring, scaler verification)

Audited Phase 4 after the gate closed rather than treating "gate closed" as "nothing left" --
found two real gaps neither the gate checker nor the experiment run had reason to catch, since
neither affects the minimal gate's own criterion.

- **FedProx was computed but never wired into training, until now.** `fedprox_proximal_term()`
  existed, was unit-tested, and was correctly documented as the Section III-F2 R4 fallback --
  but nothing in `federated/client.py` or `model/train.py` ever called it, so selecting it had
  no effect. Fixed: `model/train.py`'s `train_module` gained `fedprox_mu`/`fedprox_reference`
  parameters and adds the penalty to every step's loss, computed from the model's **live**
  parameters (`model.named_parameters()`, not a `state_dict()` snapshot) so the gradient
  actually reaches training rather than only being logged. `federated/client.py`'s
  `FederatedClient` gained a `fedprox_mu` constructor argument (default `None`, matching
  `configs/base.py`'s existing default -- the fallback stays opt-in, not the default regime),
  and passes the round's broadcast state as the fixed reference point.
  - **Verified behaviourally, not just "doesn't crash":** starting two identically-initialised
    models from the same point, plain training drifted 0.0207 (squared L2) from the start point
    over 8 epochs; with `fedprox_mu=10.0` it drifted only 0.0032 -- **16% of the unconstrained
    drift**, a real, substantial constraint, not a rounding-level effect. A second test confirms
    `fedprox_mu=0.0` is indistinguishable from plain training (the penalty is a true no-op at
    mu=0), and a third exercises the wiring through `FederatedClient.local_train` itself, not
    only the lower-level `train_module`. New tests in `test_training.py` (4) and
    `test_federated.py` (2) -- pytest now 251 passed / 1 skipped.
  - Not yet exercised against real data: this closes the implementation gap, not the deferred
    alpha=0.1 sweep that would be the actual test of whether R4 is needed.
- **The federated scaler-statistics path (`local_sufficient_stats`/`combine_stats`) had only
  ever been proven on synthetic data.** The real Phase 4 run reused Phase 3's already-pooled
  scaler rather than deriving it through the actual per-client-statistics mechanism, so the "no
  raw data crosses the boundary" property for scaling was demonstrated in a unit test but not on
  production data. Verified now on the real corpus: reconstructing the client partition
  actually used (alpha=0.5, same seed) over the RAW, unscaled selected features (not the
  already-standardised cache, which would have made the check vacuous -- mean already ~0), each
  client's local mean differs meaningfully from the others (e.g. `Tot sum`: 25,184 / 30,640 /
  20,991 across the three clients) and yet Chan's combination reproduces the pooled fit to
  **relative difference 2e-12** -- as exact on real, skewed production data as the synthetic
  tests already proved in principle. This was a verification exercise (manual, ~1 minute), not
  a code change; no new artifact was produced since nothing about the already-reported Phase 4
  results changes.

### Phase 4 minimal gate CLOSED: baselines 4-5, real federated experiment run

Runs `scripts/run_phase4.py` for real: 3 clients (Dirichlet α=0.5), R=20 rounds, E=3 local
epochs, weighted FedAvg, 3 seeds, against `scripts/check_phase4_gate.py`'s criterion (written
*before* this run, per the Phase 3 lesson). **25/25 checks PASS, exit 0 — `PHASE 4: MINIMAL GATE
CLOSED`.** Results written to `artifacts/manifest_phase4_default.json` (full provenance) and
`artifacts/phase4_results.json` (distilled summary, in the same relationship
`phase2_feature_selection.json` has to its own run). Total wall-clock: 440.5 minutes.

| macro-F1 | seed 0 | seed 1 | seed 2 | mean |
| --- | --- | --- | --- | --- |
| Baseline 4, client 0 (local-only) | -- | -- | -- | 0.7702 ± 0.0127 |
| Baseline 4, client 1 (local-only) | -- | -- | -- | 0.7332 ± 0.0045 |
| Baseline 4, client 2 (local-only) | -- | -- | -- | 0.7635 ± 0.0066 |
| Baseline 5 (federated global) | 0.8338 | 0.8364 | 0.8301 | **0.8334 ± 0.0026** |
| Baseline 3 (Phase 3 centralised, reference) | | | | 0.8297 |

- **G4 answered, computed rather than eyeballed** (`summarize_phase4.py`'s bracket check):
  federation beats every local-only client on every seed, **and slightly exceeds the
  centralised reference on all three** (0.8334 vs 0.8297). The paper explicitly allows the
  opposite finding (R4: federation underperforming local-only under heterogeneity is "a
  legitimate finding... the quantity an operator most needs to know"); this run did not need
  that allowance, but it was checked rather than assumed.
- **Convergence.** All three seeds' round curves climb steeply to ~round 8-10 then plateau in a
  tight 0.83-0.84 band with minor round-to-round noise (e.g. seed 0 dips to 0.811 at round 12,
  recovers next round) -- visibly converged well before R=20, a data point for judging R's
  necessity if the deferred R sweep is picked up later.
- **Partition sanity-checked against real data, not just synthetic tests.** At α=0.5 the
  per-client histograms are visibly skewed (client 1 holds 1,359 of class 1 against 12-24 of
  several others) without any client being starved to zero -- the heterogeneity knob is doing
  real work on the actual corpus, not only in `test_federated.py`'s synthetic fixtures.
- **A real logging bug found mid-run, fixed for future runs, left uncorrected in this one.**
  `federated_global_gru`'s `verbose` flag was never passed as `True` from `run_phase4.py`, so an
  entire seed's 20 rounds (order of an hour) produced no output. Fixed in `983828f`; the fix
  could not apply to the already-running process (Python had already loaded the old code), so
  seed 0's per-round progress went unobserved -- the final metrics are unaffected, only their
  visibility while running.
- **A severe, misleading timing anomaly, root-caused rather than left unexplained.** Seed 1 took
  14,807s (4.1h) against seed 0's 4,714s (1.3h) and seed 2's 4,008s (1.1h) -- a 3.1-3.7x outlier
  with no code-level explanation, since the process's own memory footprint stayed flat (~430-590
  MB RSS) throughout. Diagnosed as **system-wide memory pressure**: swap usage measured at 8.9 of
  10 GB (87%) during the slow stretch, most plausibly from other applications competing for RAM
  on the host, not from this process or its code. Recorded here so a future run's per-seed
  timing spread is not mistaken for a regression in the federated code.
- **Scope decision (flagged per Golden Rule 1, discussed with the user before this run started).**
  Section III-J4's "work plan" states Phase 4's gate as "three clients with weighted FedAvg" --
  narrower than Phase 3's, and the six ablations of Section III-I3 (α, E, weighted/unweighted,
  W, F, R) are evaluation *reporting*, not listed among the seven phase-gate criteria. This is
  the same summary-vs-source distinction that caused Phase 3 to be declared complete
  prematurely (see the correction entry below); this time `check_phase4_gate.py` was written
  first, against the paper text, specifically to not repeat it. The α/E/weighted-vs-unweighted/R
  sweep is real remaining work, deferred as a separate, much larger task (a single configuration
  alone cost 7.3 hours) -- not silently dropped.

### Phase 4 opened (user-authorised): federated infrastructure, both invariants green

Implemented `federated/scaler_stats.py`, `serialization.py`, `aggregation.py`, `partition.py`,
`client.py` and `server.py`. New `tests/test_federated.py` (23) and real bodies for the two
previously-skipped invariant tests. Pytest now **241 passed / 1 skipped** (up from 197/3) — the
only remaining skip is Phase 6's path-disjointness. **Phase 4 is not complete**: baselines 4-5
(local-only GRUs, federated global GRU) are still stubs and the federated experiment has not
been run.

- **FedAvg weight is the SEQUENCE count (Eq. 21), with a test that would fail on row counts.**
  `test_fedavg_weighting.py` constructs two clients holding equal rows but unequal run
  structure, so sequence-weighting gives 1.0 and the row-count bug gives 5.0 — the invariant is
  checked by the two answers differing, not merely by the right one appearing.
- **Chan's combination is exact against the pooled fit (Eqs. 23-24).** `test_scaler_equivalence`
  compares deliberately *unequal* client shares (an equal split would pass even with an
  unweighted mean-of-means, hiding the n_k weighting), repeats over ten random splits, and
  carries a negative control proving the weighting does real work. A further test uses features
  with mean ~1e8 and spread ~1, where the `E[x^2] - E[x]^2` route cancels catastrophically —
  which is why the transmitted triple is `(count, mean, M2)` rather than the prose's
  "count, sum, and sum of squares".
- **Eq. (22)'s communication figures reproduce exactly**, and are checked against the bytes this
  implementation actually serialises rather than trusted: |θ| = 33,800 → **811,200 B = 0.7736
  MiB per round** (paper: ~0.77 MiB), **15.47 MiB over 20 rounds** (paper: ~15.5 MiB), against
  276 MB to pool the records — a factor of **17.0** (paper: "roughly 18").
- **Per-class Dirichlet draws, not one draw reused (flagged per Golden Rule 1).** Eq. (20) is
  written per class; drawing once would give every client the same share of every class — a
  uniform partition wearing a Dirichlet's clothes, and exactly the L2 under-specification
  Section III-F1 exists to avoid. A test asserts client 0's share differs across classes, and
  another asserts large α is measurably more uniform than small α, so the sweep knob is known
  to do something.
- **Client independence is enforced, not asserted (A1).** Local data is private with no public
  attribute (a test checks the public surface), the broadcast global state is cloned before use
  so a client cannot mutate the server's tensors, and a **fresh optimiser is built every round**
  — carrying or aggregating Adam moments would make the method not FedAvg while still appearing
  to converge.
- **Every update crosses the boundary through safetensors even in-process**, so the no-pickle
  rule is structurally true rather than a claim about code that would transmit differently if
  wired to a socket. `n_k` travels in the safetensors *header metadata* rather than as a tensor
  smuggled into the state dict, so a received state loads into a module without first stripping
  a bookkeeping key; a missing count is an error, never a silent zero, since it is the
  aggregation weight.
- **A client holding zero sequences is legitimate, not an error.** Under α = 0.1 a client can
  receive too few blocks to form a single window at W = 16; it contributes weight 0 and appears
  in the published histogram. Raising there would make a declared experimental configuration
  unrunnable.
- **Scaffold inconsistency reconciled:** `aggregation.StateDict` was a `Mapping` while
  `serialization.StateDict` was a `dict`, so the two modules could not compose without a cast.
  Aggregation now accepts the wider `Mapping` and returns a concrete `dict`.

### Changed: run manifests are now version-controlled

`artifacts/` was gitignored wholesale, so every manifest and phase report existed only on the
machine that produced it. That contradicts Section III-I4 ("no headline number is reported
without a manifest"): the figures quoted throughout this file were claims with their evidence
sitting outside the repository, and losing one laptop would have meant re-deriving them (114
minutes for the Phase 3 run alone). Manifests and phase reports are provenance, not bulk output
— all five total ~200 KB, and each records the git commit it belongs to, so the repo is exactly
where they belong. Model checkpoints, figures and scratch output stay ignored.

Note for anyone editing the rule: the pattern is `artifacts/*`, **not** `artifacts/`. Git never
descends into an excluded *directory*, so a `!` re-include beneath one is silently ignored —
the first attempt at this change looked correct and tracked nothing.

### Phase 3 gate CLOSED — baselines 1-3 and the window ablation

Answers the correction below. `eval/baselines.py` implemented (baselines 1-3; 4-5 stay Phase 4
stubs, with a test asserting they still raise), plus `scripts/run_phase3_complete.py` and
`scripts/check_phase3_gate.py`. New `tests/test_baselines.py` (10). Run: 3 baselines × W ∈ {1,16}
× 3 seeds × 10 epochs, 113.8 min, manifest at
`artifacts/manifest_phase3_complete_default.json` (gitignored).

| macro-F1 | W=1 | W=16 |
| --- | --- | --- |
| random forest | 0.6884 ± 0.0016 | 0.6855 ± 0.0006 |
| MLP (parameter-matched) | 0.6097 ± 0.0033 | 0.6070 ± 0.0020 |
| **centralised GRU** | 0.5974 ± 0.0045 | **0.8297 ± 0.0013** |

At W=16 the GRU also reports balanced accuracy 0.8805 ± 0.0047, MCC 0.8785 ± 0.0024 and
accuracy 0.9030 ± 0.0022.

- **Ablation answered (Section III-D).** The paper set the test: "if it matches W = 16,
  recurrence has not earned its place in the architecture, and we will say so." It does not
  match — W=16 beats W=1 by **+0.2322 macro-F1, about 51× the seed standard deviation**. The
  verdict is computed by the runner (gain vs. 2× seed std) and recorded in the manifest, so it
  could not be rationalised after the fact. **Recurrence earned its place.**
- **S13's dissent was a real threat and was answered.** The random forest genuinely beats both
  neural models at W=1 (0.6884 vs GRU 0.5974) — so the baseline was doing work, not
  rubber-stamping the architecture — but the GRU clears it by **+0.144** once it has a full
  window. Notably the RF is nearly window-invariant (0.6884 → 0.6855), as it must be, since it
  only ever sees one record.
- **The MLP comparison isolates recurrence cleanly.** Parameter-matched to 33,808 against the
  GRU's 33,800 (0.02% apart) and trained through the *same* `train_module` loop — same
  optimiser, class-weighted CE, seeding, batching, epochs — so the **+0.2227** gap at W=16 is
  attributable to recurrence and not to capacity or training setup. Its near-identical scores at
  W=1 and W=16 (0.6097 / 0.6070) confirm it really is blind to history, as designed.
- **Recurrence helps most exactly where it is needed most.** Per-class F1 at W=16, GRU vs RF:
  BruteForce **0.662 vs 0.264**, WebBased **0.601 vs 0.345**, Benign **0.736 vs 0.486**. The
  rare families and the benign class — the ones that carry the operational cost — are where the
  window pays off; on the easy flooding classes the two are close (Mirai 0.9985 for both).
- **Longer training mattered, as flagged.** 10 epochs lifted the W=16 GRU from the earlier
  3-epoch 0.7866 to 0.8297 (+0.043), confirming that run was under-trained rather than at the
  architecture's ceiling.
- **Timing mis-estimated twice, recorded so the method is not repeated.** The first estimate
  extrapolated a random-forest benchmark run on *random labels*, which is pathological for trees
  (they grow to full depth memorising noise) and overstated RF cost ~10×. The second
  "correction" was worse: the MLP's seed-0 fit took 1455s against 62s for every other seed,
  because the test suite, ruff, mypy and the gate checker were run **on the same machine during
  that fit**. A timing taken under self-inflicted load was reported as the model's cost. Results
  were unaffected (contention changes wall-clock, not arithmetic); the estimate was not.
- **`scripts/check_phase3_gate.py` makes the gate machine-checkable** rather than a judgement
  call: it reads the manifest and checks seeds, mean ± std for every headline metric, all three
  baselines, per-class F1 and confusion matrices, the ablation verdict, and manifest provenance,
  exiting non-zero on any failure. It deliberately does **not** check that the detector is good —
  a poor result properly measured closes the gate; a good result improperly measured does not.
  Validating it against the *older* manifest (which it should reject) exposed two bugs in the
  checker itself — a `TypeError` crash on a flat-summary manifest, and mistaking the
  `per_class_f1` map for a window key — both fixed before it was trusted.

### Still open after Phase 3

- **Binary FPR remains too high to deploy** (0.257 ± 0.061 at the 3-epoch run; not re-measured
  here, since the completion run reports the multiclass bundle). Tuning the operating point is
  outstanding work, not a Phase 3 blocker.
- **The window sweep is partial.** Section III-I3 specifies W ∈ {1, 8, 16, 32}; only the
  decision-relevant endpoints W=1 and W=16 were run. W=8 and W=32 remain, as does the F sweep.
- The detection-latency set (below) is still unresolved.

### Correction (same day): Phase 3's gate was declared clear too early

An earlier version of this entry and of the phase table above recorded Phase 3's exit criterion
as **met** on the strength of the metric protocol alone. That was wrong, and is corrected here
rather than quietly edited away. **Section III-I1 requires five baselines "reported for every
configuration"**, three of which the scaffold itself tags Phase 3 in
`eval/baselines.py`'s `NotImplementedError` messages:

1. random forest on single records — **still a stub** (and the most pointed omission: S13 found
   a random forest strongest on tabular flow features, which is precisely why the paper carries
   this baseline — it is the one that can contradict the GRU choice);
2. MLP on single records — **still a stub** (it is what isolates the contribution of recurrence);
3. centralised GRU — done, though produced via `scripts/run_phase3.py` rather than through
   `eval/baselines.centralized_gru`'s entry point, which remains a stub.

Baselines 4 and 5 (local-only GRUs, federated global GRU) are correctly Phase 4.

Additionally, Section III-D states the W sweep includes **W = 1 as an ablation**: "if it matches
W = 16, recurrence has not earned its place in the architecture, and we will say so." That
comparison has not been run, so the architecture's central claim is so far unexamined on this
data. Phase 3 therefore remains **open**, and Phase 4 stays gated behind it.

### Open question raised, not resolved (blocks part of Section III-D)

- **The detection-latency evaluation set cannot be built as specified.** Section III-D requires
  "a separate held-out set of mixed windows spanning a benign-to-attack boundary … used to
  measure detection latency", *and* that windows be constructed "only over contiguous same-label
  runs within a single source file". On the raw UNB distribution these cannot both hold:
  **every capture file contains exactly one label** (verified — max labels per `source_file` is
  1), because the benchmark names each capture after its attack type. A benign→attack transition
  inside one source file therefore does not exist anywhere in the corpus, and
  `build_latency_eval_windows` correctly returns **zero** windows on the real data. The three
  constraints (mixed benign→attack windows, single-source-file windows, one-label-per-file data)
  are mutually unsatisfiable. Resolving it requires a user decision — synthesise transitions by
  concatenating a benign run to an attack run from a different file (fabricates a transition
  that never occurred, the same objection that rules out synthetic oversampling), relax the
  single-source-file rule for the latency set only, or drop the latency metric and report why.
  Nothing is implemented for it pending that decision.

- **Phase 2 (four-stage feature selection) implemented — Phase 2 is now complete.**
  `features/selection.py`'s `FeatureSelector.fit`/`.transform` implement all four stages of
  Section III-C, fitted on training blocks only. New `tests/test_feature_selection.py`
  (18 tests). Pytest now 102 passed / 4 skipped (up from 84 passed / 4 skipped).
  - **Stage 4 needs Phase 3, which is gated behind Phase 2 — a circular dependency in the
    paper's own phase ordering, flagged rather than resolved by breaking the gate.** Choosing
    "the knee of the validation macro-F1 curve" (Stage 4) requires training and evaluating a
    detector at each candidate F, but the detector is Phase 3 and Phase 3 is the hard gate that
    opens only once Phase 2 closes. Resolved by dependency injection instead of starting Phase 3
    early: `fit()` takes an optional `evaluator` callable mapping a candidate column list to a
    validation macro-F1. Supplied (Phase 3 onward), Stage 4 runs for real and records the curve
    in `SelectionResult.f_sweep_scores`; `None` (Phase 2, now), it falls back to the configured
    `f` and records an **empty** curve rather than fabricating one — so no GRU is trained before
    its gate. The knee itself is a standard maximum-distance-to-chord construction and is unit
    tested against a deliberately placed knee.
  - **The reciprocal-rank-fusion equation is not recoverable from the paper source (flagged,
    Golden Rule 1).** `docs/design_paper.md`'s PDF extraction dropped *every* display equation
    before Eq. (12) — which includes Stage 3's RRF formula and the mutual-information definition.
    The surrounding prose survives, the equations do not. Implemented as the canonical
    `RRF(j) = sum_m 1/(k + r_m(j))` with 1-based ranks and `k = 60` (Cormack et al., 2009),
    exposed as the new `FeatureConfig.rrf_k`. **This needs re-checking against the authoritative
    PDF before any Stage-3 number is quoted in a write-up** — the missing equations are a
    repo-wide source-fidelity gap, not just a Stage-3 one.
  - **Two further undefined terms, resolved and documented:** "univariate relevance" (Stage 2's
    tie-break for which member of a correlated pair survives) is read as mutual information with
    the label — the same statistic Stage 3 ranks by, so the two stages share one notion of
    relevance; and "disagreement" (Stage 3) is reported as the selection-relevant one, a feature
    in the top-F of exactly one of the two rankings, which is precisely what the fusion step
    would otherwise bury.
  - **Statistics computed on a seeded 200,000-row sample** of the training split (new
    `FeatureConfig.selection_sample_size`; `None` uses every row), shared by Spearman, mutual
    information and the random forest so all three see identical rows. Measured on this machine:
    MI and RF each cost ~16s per 100k rows and scale linearly, so a full-split fit would run to
    minutes for estimates already stable at this size; the sampled fit takes ~63s.
  - **Non-finite rows excluded from the statistics sample only, never imputed or zero-filled**,
    and the count reported in `SelectionResult.n_nonfinite_rows_excluded` — same policy
    `data/characterize.py` already applies to its correlation accumulation, and the reason is
    the genuine `+inf` in `Rate` that Phase 1 found in the raw corpus.
  - **Scaffold extension (flagged):** `SelectionResult` gained `fused_ranking`,
    `f_sweep_scores`, `selected_f` and `n_nonfinite_rows_excluded`. Section III-C requires the
    fused ranking and the Stage-4 knee curve to be reportable, and the stub dataclass had no
    field for either; the non-finite count is added on the same principle as Phase 1's
    `infinite_counts` (visible, not silent).
  - **Real results on the real training split** (1,237,958 rows; saved to
    `artifacts/phase2_feature_selection.json`, gitignored): Stage 1 dropped `source_file` and no
    zero-variance column (consistent with Phase 1's finding that the raw corpus, unlike the
    Kaggle mirror, has none). Stage 2 pruned **8** columns at tau=0.95 — `Rate`, `syn_count`,
    `fin_count`, `rst_count`, `ARP`, `IPv`, `AVG`, `Std` — leaving **F0 = 31**. 54 non-finite
    rows excluded. The selected F=16 set, in fused-rank order: `Tot sum`, `Header_Length`,
    `Tot size`, `Max`, `Variance`, `IAT`, `TCP`, `ack_count`, `Time_To_Live`, `syn_flag_number`,
    `ack_flag_number`, `UDP`, `Protocol Type`, `psh_flag_number`, `fin_flag_number`, `Min`.
    **The two rankings genuinely disagree on 4 features** (`fin_flag_number`, `ICMP`, `Min`,
    `Number`) — reported, per Section III-C, rather than averaged away; note mutual information
    ranks `IAT` outside its top 8 while the random forest ranks it *first*, the clearest instance
    of the non-monotonic-vs-high-cardinality split the paper predicts.

- **Phase 2 (leakage control), subsampling + label derivation implemented; the full
  SUBSAMPLE -> DEDUP -> SPLIT pipeline now runs end-to-end on real data.**
  `data/subsample.py`'s `stratified_capped_subsample` is implemented (Section III-B1, risk R1).
  New `tests/test_subsample.py` (10 tests, synthetic fixtures). Pytest now 84 passed / 4 skipped
  (up from 74 passed / 4 skipped).
  - **Dataset root decision (flagged per Golden Rule 1, confirmed with the user via an explicit
    choice, not silently resolved).** `configs/base.py`'s `DataConfig.dataset_root` and
    `configs/default.yaml` now point at the official UNB raw distribution
    (`data/ciciot2023_raw/`), not the pre-merged Kaggle mirror (`data/ciciot2023/`) used
    previously. Three things in the paper text make this the only choice consistent with
    Sections III-B/D, laid out for the user before the choice was made: (1) Section III-B1 says
    subsampling reads "whole CSV parts in fixed order" -- a description of many per-capture
    files, which only the raw distribution has; (2) Section III-D requires windows to stay
    "within a single source file", recoverable only when each CSV *is* one capture, as in the
    raw distribution -- the Kaggle mirror had already destroyed that identity by merging; (3) the
    raw distribution's record count (46,776,700) matches the paper's stated ~46.7M almost
    exactly, confirming it -- not the Kaggle mirror's ~17% subsample -- is the benchmark the
    paper describes. The Kaggle mirror is no longer consumed anywhere in the pipeline; it remains
    on disk for reference only.
  - **Label derivation (flagged per Golden Rule 1).** The raw distribution has no `label` column;
    the label lives only in each CSV part's *filename*. Directory name and filename disagree for
    benign traffic (`Benign_Final/` contains `BenignTraffic*.pcap.csv`), so labels are derived
    from the filename (strip trailing `.pcap.csv`, then a trailing digit run, then one leftover
    `-`/`_`), not the directory. This reproduces all 34 of the raw distribution's implied
    classes and matches the Kaggle mirror's own 34-label vocabulary almost exactly (e.g.
    `DDoS-PSHACK_Flood`, `DDoS-RSTFINFlood`), a useful cross-check even though the Kaggle mirror
    itself is no longer used as data.
  - **Per-class cap mechanic (flagged per Golden Rule 1).** Parts are read in natural
    part-number order (`Foo.pcap.csv`, `Foo1.pcap.csv`, `Foo2.pcap.csv`, ... -- lexicographic
    order would wrongly visit `Foo10` before `Foo2`) and a part, once started, is always read to
    completion (`chunk_size` bounds memory *within* one part's read; it never stops a read
    mid-part). Reading stops once a class's accumulated rows reach or exceed `per_class_cap`.
    Because some CICIoT2023 attack-type directories hold parts far larger than a typical cap (a
    single `DDoS-ICMP_Flood` part alone is ~268k rows against the old 200k cap), whole-part
    reading alone would let the cap overshoot by up to one whole part per class. To keep "cap" a
    real ceiling, a *seeded, order-preserving* random trim (no shuffling -- kept rows retain
    their original relative order) drops any excess down to exactly `per_class_cap` once a
    class's accumulated rows exceed it; this is the only place `seed` is used, since read order
    and per-part inclusion are otherwise fully deterministic. `target` is not used to derive
    `per_class_cap` -- the paper does not specify that relationship, and the two are independent
    config knobs -- it is only a sanity check (`warnings.warn` if the realised total deviates
    from it by more than 50%).
  - **Source-file identity added, resolving the gap flagged in the 2026-09-13 `dedup.py` entry.**
    Every row now carries a `source_file` column (the part's path relative to `dataset_root`), so
    Section III-D's "never window across a source file" rule is enforceable once
    `sequences/windowing.py` is implemented (Phase 2/3 boundary).
  - **Config recalibration, a real finding surfaced rather than silently patched over.** Running
    the implemented subsampler against the real raw distribution with the inherited
    `per_class_cap: 200_000` (tuned, it turns out, for the Kaggle mirror's per-class
    distribution, never re-validated against the raw one) produced 4,458,051 records -- more
    than double the paper's stated M ~ 1.5-2e6 (Section III-B1), because 19 of 34 classes hit
    that cap on the raw distribution, not just the "dominant DDoS classes" the paper names (also
    `BenignTraffic`, `MITM-ArpSpoofing`, `VulnerabilityScan`, ...). Swept `per_class_cap` against
    the real data and reset the default to **70,000**, which caps 24 of 34 classes and lands the
    realised total at **1,772,371** records, inside the paper's target range.
    `configs/base.py`/`configs/default.yaml` updated with the new default and this rationale
    recorded inline.
  - **Full pipeline validated end-to-end on real data (not just synthetic fixtures).** Chaining
    the new `subsample.py` with the existing `dedup.py`/`split.py` against
    `data/ciciot2023_raw/` under the recalibrated config: 1,772,371 records subsampled ->
    1,549,531 after dedup (222,840 exact duplicates removed, a 12.6% duplicate rate within this
    subsample) -> 6,072 label-respecting blocks -> stratified split (4,854 train / 1,218 test
    blocks, 1,237,958 / 311,573 rows) -> the R3 leakage assertion (no record hash in both
    partitions) **passes on real data**, not only on the synthetic fixture `test_leakage.py`
    already covered. This was a manual verification run, not a new automated test (the real
    dataset is gitignored and not available in CI-equivalent local runs), but it is the first
    time the R3 gate's *intent* -- not just its synthetic proxy -- has been confirmed.

## 2026-09-13

### Added

- **Phase 2 (leakage control), dedup + block-split half implemented; the R3 gate is now
  active.** `data/dedup.py`'s `record_hash`/`deduplicate` and `data/split.py`'s
  `make_blocks`/`stratified_block_split` are implemented, and `tests/test_leakage.py` has been
  rewritten from its skipped placeholder into a real gating test: it deduplicates synthetic data
  containing planted triple-duplicate rows, block-splits the result, and asserts the train/test
  hash sets are disjoint -- plus a negative-control test
  (`test_leakage_would_occur_without_dedup`) proving the same data *does* leak when split
  without deduplicating first, so the gate is not vacuously true. New `tests/test_dedup.py`
  (5 tests) and `tests/test_split.py` (8 tests). Pytest now 74 passed / 4 skipped (down from
  5 skipped: `test_leakage.py` no longer skips).
  - **`record_hash`**: a single 64-bit `pandas.util.hash_pandas_object` hash per row rendered as
    hex, matching the collision-probability reasoning already documented for Phase 1's
    `characterize.py`. Deliberately scoped to the capped-subsample regime (~1.5-2e6 rows, see
    below), where the birthday-bound collision risk is ~1e-10, not the full 46.7M-row corpus.
  - **`deduplicate`**: keeps first occurrence by that hash, preserves row order (never shuffles),
    resets the index to a clean contiguous range.
  - **`make_blocks`**: assigns contiguous block ids, breaking whenever `block_size` rows have
    accumulated OR the `label` column changes -- a resolution of something the paper leaves
    implicit (see below).
  - **`stratified_block_split`**: draws `test_fraction` of each label's blocks into a global test
    set via a seeded RNG, raises if any block is found to span more than one label (defensive;
    `make_blocks` itself never produces one).
  - **Ordering decision, flagged rather than left implicit (Golden Rule 1):** the paper's R3
    invariant fixes dedup *before split*, but says nothing about dedup's position relative to
    subsampling. `deduplicate`'s signature (`frame: pd.DataFrame` in memory) only fits the
    ~1.5-2e6-row capped subsample of Section III-B1, not the 46.7M-row raw corpus (which does not
    fit as one in-memory frame — Phase 1 handles that scale by streaming instead). This fixes the
    pipeline order as **SUBSAMPLE -> DEDUP -> SPLIT**, documented in `dedup.py`'s module
    docstring. Not yet run on real data for this reason: `data/subsample.py` (needed to produce
    a memory-feasible, real, labeled frame from `data/ciciot2023_raw/`) is still a stub, as is
    label derivation from the raw distribution's per-attack-type directory names.
  - **Block-boundary decision, flagged (Golden Rule 1):** `make_blocks` never lets a block span a
    label change, so blocks can be stratified by a single label in
    `stratified_block_split`. The paper doesn't specify block-construction mechanics explicitly;
    this anticipates Section III-D's later, explicit rule that *windows* may never cross a label
    boundary either, so blocks (the coarser unit windows are drawn from) are made to agree with
    that constraint rather than risk being finer-grained now and re-litigated at windowing time.
  - **Gap surfaced, not yet resolved:** neither this module nor `characterize.py` tracks which
    original source *file* each pooled row came from. Section III-D requires windows to stay
    within "a single source file" even when labels happen to match across files, so whatever
    ingests the raw distribution into one ordered frame (likely `data/subsample.py`, not yet
    implemented) will need to carry a source-file identity column through dedup, split, and
    windowing. Flagged here so it isn't lost before that module is written.
- **Phase 1 (dataset characterisation), implemented and run on real data.**
  `data/characterize.py`'s `characterize_dataset` now streams every `*.csv` file found under
  `dataset_root` in fixed-size chunks (`pandas.read_csv(chunksize=...)`) and reports: exact
  columns/dtypes, per-column null counts, zero-variance columns (min == max among observed
  values), an exact duplicate-row count (order-sensitive 64-bit row hash via
  `pandas.util.hash_pandas_object`; birthday-bound collision probability ~1e-6 at this corpus
  size, documented in the module docstring), label vocabulary + counts, the class imbalance
  ratio, and a Pearson correlation matrix over numeric columns (streamed via
  running sum/sum-of-squares/sum-of-products, one pass, O(k^2) memory). New
  `tests/test_characterize.py` (5 tests, synthetic fixtures via `tmp_path`) — pytest now
  57 passed / 5 skipped.
  - **Scaffold gap found and fixed (flagged per golden rule 1, confirmed with the user):** the
    stub `CharacterizationReport` dataclass had no field for the correlation matrix, even though
    Section III-B requires one in the Phase 1 report. Added `correlation_matrix: dict[str,
    dict[str, float]]`. Pearson was chosen over Spearman so the full pooled corpus can be
    covered exactly in one streaming pass; this is a separate statistic from Phase 2's Spearman
    correlation *pruning* (Section III-C, Stage 2, tau=0.95), which is training-data-only and
    feeds feature selection, not this report.
  - **Dataset provenance conflict found and resolved (flagged per golden rule 1, confirmed with
    the user):** the CICIoT2023 data supplied for this project (`data/ciciot2023/{train,test,
    validation}/*.csv`) arrived already split by an external mirror builder, and is a ~7.85M-row
    subsample (34 labels, 47 columns, schema matches the paper) rather than the paper's stated
    ~46.7M-record raw corpus. Consuming that external split as-is would make Section III-B2/R3
    (dedup must precede *our* split) unverifiable, since we have no visibility into how or
    whether the mirror deduplicated before assigning rows to train/test/validation. Resolution:
    `characterize_dataset` pools every CSV under `dataset_root` (recursively) into one corpus
    regardless of which subdirectory it came from, and Phase 2 will dedup + split that pooled
    corpus ourselves, ignoring the external split assignment. This also generalises correctly to
    the real CICIoT2023 distribution format (many per-capture CSV parts, no pre-existing split).
  - **Real numbers produced on the pooled corpus** (saved to
    `artifacts/phase1_characterization_report_kaggle.json`, gitignored): 7,845,673 records, 47
    columns, 0 nulls anywhere, zero-variance columns `Telnet` and `IRC`, 275,259 exact duplicate
    rows (confirms the paper's claimed leakage channel is real in this data), 34 labels,
    imbalance ratio ≈ 5,820 (paper states IR ≈ 5,751 on the full corpus; the difference is
    expected since this is a subsample, not the raw benchmark).
- Development environment stood up on this machine (`.venv`, `pip install -e ".[dev]"`,
  `pre-commit install`) — CLAUDE.md had assumed one already existed. All four gates (`ruff
  check`, `ruff format --check`, `mypy src`, `vulture`) confirmed clean before and after this
  work.
- **Second dataset added: the official UNB raw CICIoT2023 distribution** (`data/ciciot2023_raw/`,
  309 per-capture CSV files, 8.3 GB, one directory per attack type — the format the paper
  actually describes, as opposed to the pre-merged Kaggle mirror above), and `characterize_dataset`
  run on it too, at the user's request, specifically to diff it against the Kaggle mirror.
  - **`.gitignore` gap fixed:** added a blanket `/data/` rule. The prior `*.csv`/`*.parquet`-only
    rules would have let non-CSV dataset files (e.g. this distribution's `README_CSV.pdf`) get
    committed, contradicting the section's own "must not be committed" comment.
  - **Correctness bug found and fixed while running Phase 1 on this data.** The correlation
    accumulator filtered `NaN` but not `+/-inf` out of the numeric block before accumulating
    sums; the raw distribution's `Rate` column (a division by flow duration) has genuine `+inf`
    values for near-zero-duration flows, which silently corrupted the running covariance
    (surfaced as a `RuntimeWarning: invalid value encountered in subtract`, not a hard failure —
    it would have shipped a wrong correlation matrix unnoticed). Fixed the mask to
    `np.isfinite(...).all(axis=1)`, and added a new `infinite_counts: dict[str, int]` field
    to `CharacterizationReport` (parallel to `null_counts`) so infinite values are reported
    rather than silently dropped. New regression test
    `test_characterize_infinite_values_counted_and_excluded_from_correlation` — pytest now
    58 passed / 5 skipped.
  - **Real numbers on the raw corpus** (saved to
    `artifacts/phase1_characterization_report_raw.json`, gitignored): **46,776,700 records** —
    matches the paper's stated ~46.7M almost exactly, confirming the Kaggle mirror is indeed a
    ~17% subsample rather than the full benchmark. 39 columns, **no `label` column at all**
    (labels are implicit in the per-attack-type directory/file names in this distribution, unlike
    the Kaggle mirror which already merged in a `label` column). 0 zero-variance columns (vs. the
    Kaggle mirror's `Telnet`/`IRC` — the fuller corpus has at least one nonzero value for both).
    **25,641,443 exact duplicate rows — 54.8% of the corpus**, dramatically higher than the
    Kaggle mirror's 3.5%; this is the leakage channel Section III-B2/R3 exists to close, and its
    scale on the real corpus underscores why dedup-before-split is a hard invariant here, not a
    nicety. Small genuine null counts in several columns (the Kaggle mirror had zero), and 1,037
    `+inf` values in `Rate`.
  - **Schema diff (columns only in one dataset):** Kaggle-only —
    `flow_duration, Duration, Srate, Drate, urg_count, Magnitue, Radius, Covariance, Weight,
    label` (10 columns, all absent from the raw distribution); raw-only — `Time_To_Live, IGMP`
    (2 columns, absent from the Kaggle mirror). 37 columns are common to both. This means the
    Kaggle mirror is not merely a subsample: it was built with extra derived columns and an
    injected `label` column that the official raw distribution does not carry, so Phase 2's
    feature selection and label derivation will need to target whichever dataset is used as the
    project's source of truth — **not yet decided, flagged to the user**.

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

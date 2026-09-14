# Changelog

All notable changes to this repository are recorded here, newest first. This project follows
the seven-phase gating discipline of [docs/design_paper.md](docs/design_paper.md) (Section
III-J4) rather than semantic-versioned releases, so entries are grouped by date and tagged with
the phase they belong to. See [CLAUDE.md](CLAUDE.md) for the rule requiring this file to be
kept current.

## Phase status

| # | Phase | Status | Exit criterion |
| - | --- | --- | --- |
| 1 | Characterisation report | Done, on both a subsampled/pre-split Kaggle mirror and the official UNB raw corpus (see 2026-09-13 entries) | Real columns/types, nulls, zero-variance, exact duplicate count, label vocab + counts, correlation matrix produced |
| 2 | Leakage-controlled preprocessing + four-stage feature selection | **Done** (Stage 4's knee sweep is wired but inert until Phase 3 supplies a detector — see the 2026-09-14 entry): subsample -> dedup -> split -> four-stage selection all run end-to-end on the real corpus; R3 gate passes on real data | `tests/test_leakage.py` green on real data ✅ |
| 3 | Centralised GRU + full evaluation | **DONE — gate CLOSED** (**hard gate**), verified by `scripts/check_phase3_gate.py` (exit 0). Baselines 1-3 of III-I1 reported over 3 seeds at W ∈ {1,16}; GRU macro-F1 **0.8297 ± 0.0013** at W=16 vs random forest 0.6855 and MLP 0.6070. Ablation answered: recurrence **earned its place** (+0.2322, 51× seed std) | Full evaluation protocol (macro-F1, per-class F1, balanced accuracy, MCC, confusion matrix, FPR; ≥3 seeds) reported **for baselines 1-3 of Section III-I1** ✅ |
| 4 | Three-client federated simulation, weighted FedAvg | In progress: **both blocking invariants green** (`test_fedavg_weighting.py`, `test_scaler_equivalence.py` — the last two skips in the suite are gone); partition/aggregation/serialization/client/server implemented and Eq. (22)'s cost figures reproduced. **Not** done: baselines 4-5 are still stubs and the federated experiment has not been run | `test_fedavg_weighting.py`, `test_scaler_equivalence.py` green ✅ |
| 5 | Telemetry simulation + feature-provenance adapter | Not started | Provenance adapter enforces G6 boundary |
| 6 | Ascon integration + alerting path | In progress (**gate deliberately overridden**) | `test_ascon_kat.py`, `test_ascon_tamper.py`, `test_nonce_collision.py`, `test_path_disjointness.py` green |
| 7 | End-to-end integration | Not started | Full pipeline run producing a manifest |

Most modules under `src/ascon_smart_agri/` are still typed stubs: they `del` their unused
parameters and raise `NotImplementedError("Phase N: ... not implemented yet.")`. The exceptions
are the Ascon-AEAD128 crypto core (`crypto/ascon_aead.py`), implemented ahead of its phase gate
(see the 2026-09-12 Phase 6 entry below), `data/characterize.py` (Phase 1), and
`data/subsample.py`/`data/dedup.py`/`data/split.py`/`features/selection.py` (Phase 2, complete)
-- see the 2026-09-14 and 2026-09-13 entries below.

## 2026-09-14

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

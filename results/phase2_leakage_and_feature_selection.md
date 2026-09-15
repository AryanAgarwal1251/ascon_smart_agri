# Phase 2 — Leakage-Controlled Preprocessing + Feature Selection

**Status: ✅ Done.** Source: [`artifacts/phase2_feature_selection.json`](../artifacts/phase2_feature_selection.json).
Code: [`data/subsample.py`](../src/ascon_smart_agri/data/subsample.py),
[`data/dedup.py`](../src/ascon_smart_agri/data/dedup.py), [`data/split.py`](../src/ascon_smart_agri/data/split.py),
[`features/selection.py`](../src/ascon_smart_agri/features/selection.py). Gating test:
[`tests/test_leakage.py`](../tests/test_leakage.py) (R3).

## What the paper says

> "The full corpus does not fit in workstation memory. We draw a stratified, capped subsample
> of M ≈ 1.5–2 × 10⁶ records by reading whole CSV parts in fixed order and applying a per-class
> cap." — Section III-B1

> "If deduplication is applied after the train–test split, identical records appear on both
> sides, and the test set measures memorisation rather than generalisation... We therefore fix
> the order as [dedup, then split] and assert it with a test that fails if any record hash
> appears in both partitions." — Section III-B2 (risk R3)

> Feature selection runs in **four stages**: rule-based elimination; Spearman correlation
> pruning at τ = 0.95; mutual-information/impurity-importance reciprocal rank fusion; and a
> cardinality sweep over F ∈ {8, 12, 16, 24, F₀}. — Section III-C

## What we achieved

**Pipeline run end-to-end on the real raw corpus:**

```
46,776,700 records (309 per-capture CSVs)
  → SUBSAMPLE   1,772,371 rows   (34 classes, per-class cap calibrated to land in the paper's
                                   M ≈ 1.5–2×10⁶ target — the inherited default overshot 2.5×)
  → DEDUP       1,549,531 rows   (222,840 exact duplicates removed, 12.6%)
  → SPLIT       6,072 blocks → 4,854 train / 1,218 test (1,237,958 / 311,573 rows)
  → R3 GATE     PASS — zero record hashes shared between train and test
  → SELECT      39 columns → Stage 1 (F=38, drops source_file) → Stage 2 (F0=31)
                → Stage 3 fusion → F=16
```

**Stage 2 (Spearman, τ=0.95) pruned 8 columns:** `Rate`, `syn_count`, `fin_count`, `rst_count`,
`ARP`, `IPv`, `AVG`, `Std` — leaving **F0 = 31**.

**Selected F=16, in fused-rank order:** `Tot sum`, `Header_Length`, `Tot size`, `Max`,
`Variance`, `IAT`, `TCP`, `ack_count`, `Time_To_Live`, `syn_flag_number`, `ack_flag_number`,
`UDP`, `Protocol Type`, `psh_flag_number`, `fin_flag_number`, `Min`.

**Stage 3's two rankings genuinely disagree on 4 features** — `fin_flag_number`, `ICMP`,
`Min`, `Number` — reported rather than averaged away, per the paper's own instruction ("where
they disagree we report the disagreement instead of hiding it in an average"). The sharpest
case: mutual information ranks `IAT` outside its top 8, while random-forest impurity importance
ranks it **first** — exactly the non-monotonic-vs-high-cardinality split the paper predicts the
two methods will expose.

**54 non-finite rows** (genuine `+inf` in `Variance`, propagated from `Rate`'s division by
near-zero flow duration) excluded rather than imputed.

## Decisions flagged, not silently made

- **Per-class cap recalibrated.** The inherited default (`200,000`) produced 4.46M rows on the
  raw distribution — 19 of 34 classes hit that cap, not just the "dominant DDoS classes" the
  paper names. Recalibrated to `70,000`, landing at 1,772,371 rows inside the paper's stated
  target.
- **Label source: filename, not directory.** The raw distribution's directory `Benign_Final/`
  holds files named `BenignTraffic*.pcap.csv` — labels are derived from the filename, matching
  the Kaggle mirror's own 34-label vocabulary almost exactly (a useful cross-check).
- **Reciprocal rank fusion constant.** `docs/design_paper.md`'s PDF extraction dropped every
  display equation before Eq. (12), including Stage 3's RRF formula. Implemented as the
  canonical `RRF(j) = Σₘ 1/(k + rₘ(j))`, k=60 (Cormack et al., 2009) — flagged as needing
  re-verification against the authoritative PDF before being quoted as the paper's own formula.

## A correctness bug this phase caught

Splitting operates on whole blocks (Section III-B3), so filtering a deduplicated frame down to
just the train blocks leaves rows that were never physically adjacent sitting next to each
other — block 5 and block 7 become neighbours when block 6 is held out for test. Windowing
across that join would fabricate adjacency the capture never contained. Measured on the real
split: **977 such discontinuities**, which would have produced **14,508 sequences (1.2% of the
training set) built on adjacency that never existed** — caught before Phase 3 trained on it, via
`sequences/windowing.py`'s `contiguity_segments()`.

## What this phase does not claim

Stage 4 (the cardinality sweep) needs a trained detector to score candidate F values against —
a circular dependency, since Phase 3 (the detector) is gated behind Phase 2. Resolved by
dependency injection: `FeatureSelector.fit()` accepts an optional `evaluator` callable; without
one (true throughout Phase 2), Stage 4 falls back to the configured F=16 and records an *empty*
sweep curve rather than fabricating one.

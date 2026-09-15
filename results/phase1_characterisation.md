# Phase 1 — Dataset Characterisation

**Status: ✅ Done.** Source: [`artifacts/phase1_characterization_report_raw.json`](../artifacts/phase1_characterization_report_raw.json),
[`_kaggle.json`](../artifacts/phase1_characterization_report_kaggle.json). Code:
[`src/ascon_smart_agri/data/characterize.py`](../src/ascon_smart_agri/data/characterize.py).

## What the paper says

> "Before any modelling, a characterisation script reports the actual column names and types,
> per-column null and zero-variance checks, the exact duplicate count, the label vocabulary
> with counts, and the numeric correlation matrix. No column or class name enters the final
> report unless this script produced it. Figures quoted from the literature are treated as
> hypotheses to confirm." — Section III-B

The paper states the corpus is roughly **46.7 million records** and quotes a class imbalance
ratio of **IR ≈ 5751**.

## What we achieved

A streaming characteriser (bounded memory via `pandas.read_csv(chunksize=...)`, risk R1) run
on **both** candidate data sources, so the choice between them could be made on evidence:

| | Official UNB raw distribution | Kaggle mirror |
| --- | --- | --- |
| Records | **46,776,700** | 7,845,673 |
| Columns | 39 | 47 |
| Exact duplicate rows | **25,641,443 (54.8%)** | 275,259 (3.5%) |
| Zero-variance columns | none | `Telnet`, `IRC` |
| Null values | 1,449 | 0 |
| `+inf` values (in `Rate`) | 1,037 | 0 |
| Label column | none (implicit in 309 per-capture filenames) | explicit `label` column, 34 values |
| Imbalance ratio | — (no label column to compute it from directly) | 5,819.9 |

**The raw distribution's record count matches the paper's stated ~46.7M almost exactly**,
confirming it — not the Kaggle mirror, a third-party ~17% subsample — is the benchmark the
paper describes. This became the basis for adopting the raw distribution as canonical in
Phase 2 (see that file).

## A correctness bug this phase caught before it could propagate

The streaming correlation accumulator filtered `NaN` but not `±inf` out of the numeric block
before accumulating sums. The raw distribution's `Rate` column (a division by flow duration) has
genuine `+inf` values for near-zero-duration flows — a real property of the data, not a parsing
artefact. Left unfixed, this would have silently corrupted the running covariance (it surfaced
only as a `RuntimeWarning`, not a hard failure) and shipped a wrong correlation matrix
unnoticed. Fixed by masking to `np.isfinite(...).all(axis=1)`, with a new `infinite_counts`
field added to the report so infinite values are visible rather than silently dropped.

## Decisions flagged, not silently made

- **Dataset choice (raw vs. Kaggle mirror), confirmed with the user.** Three things in the
  paper's own text make the raw distribution the only defensible choice: Section III-B1 says
  subsampling reads "whole CSV parts in fixed order" (only the raw distribution has multiple
  per-capture parts); Section III-D requires windows to stay within "a single source file"
  (only recoverable when each CSV *is* one capture); and the record count matches almost
  exactly. See [phase2](phase2_leakage_and_feature_selection.md) for the full comparison table.

## What this phase does not claim

Characterisation reports facts about the corpus; it makes no modelling decisions. Feature
selection, leakage control, and the taxonomy mapping are Phase 2/3 concerns, deliberately kept
separate here per the paper's own framing ("no column or class name enters the final report
unless this script produced it").

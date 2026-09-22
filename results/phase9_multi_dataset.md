# Phase 9 — Multi-Dataset Generalisation

**Status: both runs complete (2026-09-22).** The first run (2026-09-21) is a recorded negative
result; the second run finished in 8.82 h over 3 seeds and every figure below is read from its
manifest. Sources: [`artifacts/manifest_phase9_generalised.json`](../artifacts/manifest_phase9_generalised.json)
(first run), [`artifacts/manifest_phase9_generalised_percorpus.json`](../artifacts/manifest_phase9_generalised_percorpus.json)
(second run). Code:
[`scripts/run_phase9.py`](../scripts/run_phase9.py), [`data/datasets.py`](../src/ascon_smart_agri/data/datasets.py),
[`data/corpus.py`](../src/ascon_smart_agri/data/corpus.py). Plan:
[`docs/plans/phase9-multi-dataset.md`](../docs/plans/phase9-multi-dataset.md).

## What the paper says

The design paper is CICIoT2023-only. Phase 9 exists because the user lifted that rule on
2026-09-19: a detector trained on one testbed's attack set should not fail on attacks that are
in scope but absent from it. What carries over from the paper unchanged is the discipline —
characterise first (III-J4 step 1), dedup before split (R3), windows only inside contiguous
same-label runs (III-D), the full metric set with ≥ 3 seeds (III-I2, III-I4) — applied *per
corpus*. What the paper does not cover, and this phase had to decide, is how corpora relate:
which columns are shared, whether their values mean the same thing, how corpora map onto
federated clients, and what "generalises" is measured by. The plan fixed leave-one-dataset-out
as the claim and pooled accuracy as explicitly *not* the claim.

## The first run (2026-09-21): a negative result, and why it is not the generalisation number

Three corpora, one per client, one global scaler (III-F4), the 13-column intersection
(Edge-IIoTset cannot supply `Header_Length`, `IAT`, `Time_To_Live`), R = 10, E = 2, 150k
sequences per client, 3 seeds. Macro-F1 over families present in the test split:

| experiment | test corpus | macro-F1 (present) | FPR |
| --- | --- | --- | --- |
| in_distribution (K = 3) | CICIoT2023 | 0.603 ± 0.001 | 0.19 |
| in_distribution (K = 3) | CICIoMT2024 | 0.562 ± 0.032 | 0.37 |
| in_distribution (K = 3) | Edge-IIoTset | 0.118 ± 0.022 | 0.93 |
| lodo (train MT + Edge) | CICIoT2023 | 0.064 ± 0.011 | 0.58 |
| lodo (train IoT + Edge) | CICIoMT2024 | 0.265 ± 0.056 | 0.92 |
| lodo (train IoT + MT) | Edge-IIoTset | 0.092 ± 0.022 | 0.99 |

Two things were true at once. (i) The 0.83 → 0.60 drop on CICIoT2023 stacks four changes —
Phase 3 complete 0.830 → Phase 3 default driver 0.787 → 13-feature intersection 0.763
(`manifest_phase3_generalised.json`) → three-corpus federation at a third of Phase 4's budget
0.603 — so the feature cut cost 0.024 and the rest was corpus mixing plus an unmatched budget.
(ii) Leave-one-dataset-out at 0.06 with FPR 0.6–0.99 is *below* majority-class chance: the
model was reading a scale offset, not traffic. The cache showed it column by column — the same
canonical name on a different scale per corpus (`ack_count` max 100 / 6 / 10, `Variance`
1e8 / **[0, 1]** / 1e6, `Tot sum` 2.6e4 / 1.2e3 / 3.7e3 across CICIoT2023 / CICIoMT2024 /
Edge-IIoTset), and Edge-IIoTset's reconstructed windows near-constant (99 % TCP, 93 %
ACK-flagged, 81 % exact duplicates). A single global scaler leaves each corpus its own offset
and the GRU learns the testbed. That is a real finding about *this harmonisation*, recorded as
such; it is not a measurement of cross-testbed generalisation.

## The second design (2026-09-22)

Decided by the user on the reading above; the full rationale is in the plan's §5.

- **Edge-IIoTset out of training** (`not_for_training` in the registry, with the reasons
  above as its recorded string). It stays characterisable. The one alternative third corpus,
  CIC IoT-DIAD 2024, was checked the same day: its packet-based set is the IoTDevID per-packet
  schema and its flow set is CICFlowMeter — none of our 39 canonical columns — so the training
  set is CICIoT2023 + CICIoMT2024 and there is no fourth schema.
- **All 16 features back.** Both corpora supply the full 39-column vocabulary; selection
  re-runs per experiment on the scaled training rows.
- **Per-corpus standardisation.** Each corpus's farms combine their sufficient statistics
  (Eqs. 23–24) into that corpus's scaler; a held-out corpus uses the label-free statistics of
  its own training split, which is what a new farm computes over its own traffic before the
  model runs. A departure from III-F4's single global scaler (kept as `--scaling global`).
- **K = 3 across corpora.** CICIoT2023 → two farms (Phase 4's block Dirichlet, α = 0.5),
  CICIoMT2024 → one; a single-corpus fold → three Dirichlet farms, which reproduces Phase 4's
  partition to within the 7 rows dropped for non-finite `Rate`/`IAT`.
- **Pooled-centralised twin** on the *same* capped farm windows for R × E epochs: same
  sequences, no federation.
- **Budget: Phase 4's** — R = 20, E = 3, 400k sequences per farm, 3 seeds, seed-major.

### What each comparison answers

| comparison | question |
| --- | --- |
| `lodo/ciciomt2024` scored on CICIoT2023 (its own training corpus) vs Phase 4's 0.831 | does the new driver reproduce Phase 4 on one corpus? (sanity) |
| `in_distribution` on CICIoT2023 vs that single-corpus ceiling | does adding CICIoMT2024 to the federation cost CICIoT2023 anything? |
| `pooled_centralised` vs `in_distribution` | what does federation cost, with mixing held fixed? |
| `lodo/<held-out>` on the held-out corpus | **the generalisation claim**: a model that never saw this testbed |

### Results (second run)

R = 20, E = 3, W = 16, 400k sequences per farm, 3 seeds, per-corpus scaling; 8.82 h wall clock.
Dedup-before-split held on both corpora independently (`r3_overlap = 0`). Mean ± std over seeds,
read from the manifest:

| experiment | test corpus | held out? | macro-F1 | present-only | FPR |
| --- | --- | --- | --- | --- | --- |
| in_distribution | CICIoT2023 | no | 0.7742 ± 0.0119 | 0.7742 ± 0.0119 | 0.3183 ± 0.0182 |
| in_distribution | CICIoMT2024 | no | 0.6025 ± 0.0155 | 0.8034 ± 0.0207 | 0.1188 ± 0.1018 |
| pooled_centralised | CICIoT2023 | no | 0.8390 ± 0.0035 | 0.8390 ± 0.0035 | 0.2929 ± 0.0330 |
| pooled_centralised | CICIoMT2024 | no | 0.6757 ± 0.0039 | 0.9009 ± 0.0052 | 0.0242 ± 0.0011 |
| lodo/ciciomt2024 | CICIoT2023 | no (ceiling) | 0.8407 ± 0.0150 | 0.8407 ± 0.0150 | 0.2320 ± 0.0458 |
| lodo/ciciomt2024 | CICIoMT2024 | **yes** | 0.1035 ± 0.0364 | 0.1380 ± 0.0485 | 0.8174 ± 0.0964 |
| lodo/ciciot2023 | CICIoMT2024 | no (ceiling) | 0.6778 ± 0.0058 | 0.9038 ± 0.0077 | 0.0095 ± 0.0060 |
| lodo/ciciot2023 | CICIoT2023 | **yes** | 0.0858 ± 0.0347 | 0.0858 ± 0.0347 | 0.9520 ± 0.0433 |

Answering the four comparisons in the order they were posed:

**The driver reproduces Phase 4 (sanity).** Trained on CICIoT2023 alone at Phase 4's budget,
the `lodo/ciciomt2024` fold scores **0.8407 ± 0.0150** on CICIoT2023 against Phase 4's
0.8308 ± 0.0150 — the same number to within a seed's spread, on a rewritten driver, a wider
candidate vocabulary and a per-corpus scaler. Nothing in the Phase 9 rework broke Phase 4.

**Per-corpus scaling fixed the in-distribution collapse.** CICIoT2023 in-distribution went
**0.603 → 0.774** and CICIoMT2024 present-only **0.562 → 0.803** against the first run. That
was the change the second design was built to test, and it worked: the first run's 0.60 was a
scale artefact plus an unmatched budget, as the cache had suggested.

**Mixing costs CICIoT2023 about 0.066, federation about another 0.065.** The single-corpus
ceiling is 0.8407; adding CICIoMT2024 to the federation drops it to 0.7742. Holding mixing
fixed and removing federation recovers it to 0.8390 (pooled). So roughly half the loss is
having a second testbed in the federation and half is FedAvg over heterogeneous farms — with
the notable result that the **pooled two-corpus model matches Phase 3's single-corpus 0.830**:
a model that can see both corpora at once pays nothing for the second one.

**Leave-one-dataset-out fails, and this is the headline.** A model that has never seen a corpus
scores **0.086 ± 0.035** on held-out CICIoT2023 (FPR 0.95) and **0.138 ± 0.049** present-only on
held-out CICIoMT2024 (FPR 0.82), while the same models score 0.84 and 0.90 on the corpus they
trained on. FPR 0.82–0.95 means the detector flags nearly every benign window of an unseen
testbed: it has learned what *that* testbed's benign traffic looks like, not what benign traffic
looks like. Per-corpus standardisation removed the scale offset that explained the first run's
LODO ≈ 0, and the number did not move — so the earlier scale explanation was incomplete, and the
residual gap is distribution shift in the features themselves, not in their units.

The expectation recorded in the plan before this run was that LODO "should be well below it but
no longer near zero". **That expectation was not met**, and per R4 (III-J) it is reported rather
than tuned away. What can honestly be claimed from Phase 9 is: the detector generalises across
*farms* (federated, within a testbed) and across *two testbeds seen jointly* (pooled 0.839, in-
distribution 0.774) — it does **not** transfer to an unseen testbed. Cross-testbed transfer would
need domain adaptation or corpus-invariant features, neither of which is in this project's scope.

## Decisions flagged, not silently made

- **Per-corpus scaler vs III-F4's single scaler.** The paper's federated scaler exists so that
  standardisation equals the pooled scaler without pooling data. That equivalence still holds
  *within* a corpus (Chan's formula over that corpus's farms); across corpora the pooled scaler
  is exactly what encodes the testbed. The paper is amended if the second run's result is kept.
- **`macro_f1_present`.** CICIoMT2024 has no Mirai or BruteForce. The C = 8 macro-F1 scores an
  absent family 0 by the metrics module's definition; the present-only macro is reported
  alongside so a corpus is not marked down for classes it does not contain. Both numbers are
  in every manifest.
- **The pooled twin reuses the farm windows** rather than windowing the whole split as
  Phase 4's upper bound did, so that its only difference to the federated run is the absence
  of federation; the ~W/block_size windows lost at farm boundaries are lost to both.
- **Taxonomy judgement calls** for CICIoMT2024's 19 leaves into the eight families are
  documented in `data/taxonomy.py` (MQTT-Malformed_Data → WebBased is the least obvious).

## What this phase does not claim

- No agriculture-specific attack data is in the training set: the most agriculture-relevant
  corpus (Edge-IIoTset) could not be made to speak CICIoT2023's feature language. The
  agriculture framing rests on the Phase 8 sensor topology, not on the training corpora.
- Two corpora is the smallest set on which leave-one-dataset-out is defined. It is a
  two-testbed generalisation number, not a survey.
- Nothing about the runtime Pi/ESP32 path changed: the provenance adapter still draws
  held-out CICIoT2023 records (G6); CICIoMT2024 only enters through training.

# Generalising across IoT testbeds — and the bug that must be fixed first

The goal is a detector that works on a testbed it was not trained on. Leave-one-dataset-out
currently reports **0.0853 ± 0.0288** and **0.0763 ± 0.0251** with FPR up to 0.95, which looks
like a hard generalisation failure.

**It is not. It is a feature-semantics bug.** This document shows the measurement, says exactly
which results it invalidates, and then sets out the real generalisation plan — because domain
adaptation applied on top of mismatched features would only be correcting a data defect.

---

## 1. The finding: three of the sixteen features do not mean the same thing in the two corpora

Measured directly from `artifacts/phase9_cache.npz` (the deduplicated training splits, before any
scaling):

| Feature | Phase 2 rank | CICIoT2023 range | CICIoMT2024 range | Verdict |
| --- | --- | --- | --- | --- |
| **`Variance`** | 5 | 0 … 1.13×10⁸ | **0 … 1** | **Not the same quantity.** 100 % of CICIoMT2024 rows are ≤ 1; only 34.9 % of CICIoT2023's are. A [0,1]-bounded statistic is not a variance of packet sizes. |
| **`Header_Length`** | 2 | 0 … 60 | **0 … 9.89×10⁶** | **Not the same quantity.** 0–60 bytes is a correct IP header length (median 20). 9.9 million is not a per-packet header length. |
| **`IAT`** | 6 | 2.4×10⁻⁷ … 4.7×10⁴ | **4.7×10⁻⁶ … 1.69×10⁸** | **Not the same quantity.** 79.6 % of CICIoMT2024 rows fall within ±1 % of 8.47×10⁷, across 586,036 distinct values — a tight cluster around a large constant, which is the signature of an absolute timestamp-derived value, not an inter-arrival delta. |

Medians for scale: `IAT` 3.6×10⁻⁴ vs 8.47×10⁷ — a ratio of **2.4×10¹¹**.

**Three of Phase 2's top six features are semantically mismatched.** A model trained on one
corpus and evaluated on the other is being fed numbers that do not denote what it learned.

### This was a known risk, recorded and not yet discharged

`data/datasets.py` says so in its own docstring:

> **Every column alias and leaf label below for the two new corpora was written from the
> published documentation, not from the files.**

And the 2026-09-22 changelog lists "re-verifying CICIoMT2024's feature semantics against its
README" as an open follow-up. The `Duration -> Time_To_Live` alias *was* verified against real
values (64/128, clearly a TTL). The other 38 columns were matched by name only.

Per-corpus standardisation hid the problem in-distribution — each corpus is scaled by its own
statistics, so a model trained and tested within one corpus never notices — which is exactly why
this survived until a cross-corpus evaluation.

---

## 2. What this invalidates, and what it does not

### Contaminated — must be withdrawn or re-measured

| Result | Figure | Status |
| --- | --- | --- |
| Leave-one-dataset-out | 0.0853 / 0.0763 | **Measures a broken mapping, not generalisation.** Cannot be reported as a generalisation finding. |
| Cost of adding CICIoMT2024, mixed farms | −0.023 | Contaminated |
| Cost of adding CICIoMT2024, corpus-per-farm | −0.070 | Contaminated |
| Two-corpus pooled-centralised twin | 0.8476 | Contaminated |

**The claims inventory's A9 entry calling the LODO failure "a contribution" must be struck.**
We came close to publishing a wrong negative result.

### Unaffected

Everything that never crosses corpora:

- **Phase 3** centralised GRU 0.8297, recurrence ablation +0.2322
- **Phase 4** federated 0.8308, the bracket, the 82.4 % gap recovery
- **The K sweep** — single corpus throughout
- **Per-client federation gain on CICIoT2023** (+0.1338 / +0.0789 / +0.0516)
- **Phase 7, Phase 8, G1, the Ascon work** — no corpus mapping involved
- **Edge-IIoTset's 0.9822** — its own native schema, never mapped onto the canonical one
- **CICIoMT2024's label coverage** (6 of 8 families, no Mirai, no BruteForce) — derived from
  filenames, not from feature columns

### What it means for the dataset decision

The decision to train on CICIoT2023 alone rested on three legs. Two stand, one must be withdrawn:

| Reason | Status |
| --- | --- |
| Medical testbed against an agriculture deployment domain | ✅ stands |
| Supplies no attack family CICIoT2023 lacks (6 of 8) | ✅ stands |
| Measurably degrades the model by 0.023 | ❌ **withdrawn — contaminated** |

The decision survives, but `docs/dataset-selection.md` overstates its evidence and must be
corrected.

---

## 3. Step 0 — fix the semantics (before any generalisation work)

Not optional, and not research: this is verification work.

1. **Read CICIoMT2024's README feature table against real column values**, per column, for all 39.
   The registry's aliases were written from documentation; they must be confirmed from data.
2. **For each of `Variance`, `Header_Length`, `IAT`**, decide whether the column is (a) the same
   quantity in different units — convertible; (b) a different statistic — must be dropped from the
   shared vocabulary; or (c) derivable from other columns present.
3. **Add a cross-corpus sanity gate to the test suite:** for every feature in the shared
   vocabulary, assert that the two corpora's ranges overlap within a stated tolerance. A feature
   whose ranges are disjoint by 10¹¹ must fail loudly, not be standardised into silence.
4. **Re-run the LODO folds** on the corrected mapping. Only then is the resulting number a
   generalisation result.

Expected outcome: the shared vocabulary shrinks (probably by `Variance` and `Header_Length` at
least), and LODO rises from ~0.08 to *something interpretable*. It may still be poor — but then it
will be poor for a real reason.

---

## 4. The generalisation toolkit, after step 0

Ranked by cost against expected value.

### 4.1 Rank / quantile features — cheapest, highest expected value

Replace each feature's value with its **within-corpus rank percentile**. "This window's packet
size is in the 90th percentile *for this network*" is comparable across testbeds in a way that
"1500 bytes" is not; link speeds, MTUs and device mixes differ, relative position does not.

- Removes location and scale shift **by construction** — no adaptation needed
- Each farm computes its own quantiles from its own traffic, which fits the federated setting:
  quantile sketches are sufficient statistics, like the existing mean/M2 scaler exchange
- Cost: a transform plus a re-run. Already listed as a project follow-up.
- Risk: discards absolute magnitude, which may carry real signal for volumetric floods

### 4.2 Shape features instead of magnitudes — cheap

Several of the 16 are absolute (`Tot sum`, `Tot size`, `Max`, `Min`, `Header_Length`). Their
**ratios** are more testbed-invariant: coefficient of variation (`Std/AVG`) rather than
`Variance`; `Max/AVG`; `Min/Max`; size entropy within the window. The flood signature is a
*shape* — many near-identical packets at near-constant spacing — and shape survives a change of
testbed better than magnitude does.

### 4.3 Domain-adversarial training (DANN) — real work, principled

Add a domain-classifier head on the GRU's hidden state with a gradient-reversal layer, so the
representation becomes one from which the corpus of origin cannot be predicted. This directly
attacks "the model learned which testbed it is on".

- Well-established, implementable in the existing training loop
- **Needs ≥ 3 training domains to be meaningful** (see 4.5)
- Interacts with federation: the domain label is the farm, so this is also a defence against
  clients being distinguishable — worth stating carefully, as it is adjacent to the
  privacy work this project declares out of scope

### 4.4 Per-deployment calibration — the honest fallback, and already the deployment story

Accept that zero-shot transfer to an unseen testbed may be unattainable, and make the first hours
on a new farm part of the design: capture local traffic, run controlled attacks via the ESP32
`scenario` switch, recompute the scaler, and fine-tune. The federated rounds then carry that
farm's knowledge to the others.

This is not a retreat. It is what the measured per-client gain already demonstrates, and it is
the claim the paper can make without reservation.

### 4.5 More training domains — and it reopens the corpus question honestly

**With two corpora, leave-one-dataset-out trains on exactly one domain.** That is domain
*transfer*, the hardest possible setting, not domain *generalisation*. The domain-generalisation
literature uses four or more source domains, because invariance can only be learned from observed
variation — with one source domain there is no variation to learn from.

So if **cross-testbed generalisation is a headline claim**, more corpora are needed — but for a
different reason than the one they were rejected on. They were rejected as *extra attack
coverage*, which they do not provide. As *additional domains* they would provide exactly what is
missing.

This is a genuine tension with the 2026-10-01 decision and should be decided deliberately:

| If the headline claim is… | then… |
| --- | --- |
| A deployable agriculture detector with measured federation benefit | **CICIoT2023 alone is right.** Generalisation is handled by per-farm calibration (4.4). |
| A detector that generalises to unseen IoT testbeds | **More source domains are required.** Corpora return as domains, with the semantics gate of §3 enforced first. |

---

## 5. Recommended sequence

| # | Step | Cost | Gate |
| --- | --- | --- | --- |
| 1 | Verify all 39 CICIoMT2024 aliases against real values | hours | — |
| 2 | Add the cross-corpus range-overlap test | hours | must fail on today's mapping |
| 3 | Correct `dataset-selection.md` and the claims inventory | hours | — |
| 4 | Re-run LODO on the corrected mapping | ~hours compute | the first honest generalisation number |
| 5 | Rank/quantile features, re-run LODO | ~hours compute | compare against step 4 |
| 6 | Shape features, re-run LODO | ~hours compute | compare |
| 7 | **Decide** headline claim (§4.5) | user | determines whether 8 happens |
| 8 | DANN with ≥ 3 source domains | days | only if generalisation is the claim |

Steps 1–3 are corrections and must happen regardless of which direction is chosen. Step 4 is the
first point at which the word "generalisation" can honestly be used about any number in this
project.

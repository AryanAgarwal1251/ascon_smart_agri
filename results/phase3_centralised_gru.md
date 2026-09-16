# Phase 3 — Centralised GRU (Hard Gate)

**Status: ✅ Done — gate closed.** Verified by [`scripts/check_phase3_gate.py`](../scripts/check_phase3_gate.py)
(49/49 checks pass, exit 0), not by narrative. Source:
[`artifacts/manifest_phase3_complete_default.json`](../artifacts/manifest_phase3_complete_default.json).
Code: [`model/gru.py`](../src/ascon_smart_agri/model/gru.py), [`model/train.py`](../src/ascon_smart_agri/model/train.py),
[`eval/baselines.py`](../src/ascon_smart_agri/eval/baselines.py).

## What the paper says

> Phase 3 is a **hard gate**: "federation is not begun until single-client detection is proven,
> because debugging an aggregation fault and a modelling fault simultaneously is substantially
> harder than debugging either alone." — Section III-J4

> Architecture: single GRU layer → LayerNorm(h_W) → linear head. Default sizing F=16, H=96, C=8
> gives **3(FH+H²+2H) + 2H + (HC+C) = 32,832 + 192 + 776 = 33,800 parameters** — Eq. (19)

> Five baselines "reported for every configuration": random forest on single records (motivated
> by S13's dissent that tree ensembles can beat recurrent models on tabular flow features);
> an MLP on single records ("isolates the contribution of recurrence"); centralised GRU; three
> local-only GRUs; the federated global GRU. — Section III-I1

> Window-length ablation: "we sweep W ∈ {1, 8, 16, 32}, and include W = 1 as an ablation. If it
> matches W = 16, recurrence has not earned its place in the architecture, and we will say so."
> — Section III-D

## What we achieved

**The model reproduces Eq. (19) exactly**: `count_parameters(build_detector(16, 96, 8)) ==
33,800`, term for term (GRU 32,832 + LayerNorm 192 + head 776) — checked by a unit test, not
assumed.

**Baselines 1–3 of Section III-I1, 3 seeds, W ∈ {1, 16}:**

| macro-F1 | W=1 | W=16 |
| --- | --- | --- |
| Random forest | 0.6884 ± 0.0016 | 0.6855 ± 0.0006 |
| MLP (parameter-matched, 33,808 params) | 0.6097 ± 0.0033 | 0.6070 ± 0.0020 |
| **Centralised GRU** | 0.5974 ± 0.0045 | **0.8297 ± 0.0013** |

At W=16 the GRU also reaches balanced accuracy 0.8805, MCC 0.8785, accuracy 0.9030.

**The W=1 vs. W=16 ablation is answered, computed by the runner itself, not eyeballed**: gain
**+0.2322**, about **51× the seed standard deviation** — `recurrence_earned_its_place: True`.
Section III-D's own trap ("if it matches W=16, recurrence has not earned its place") did not
catch this architecture.

**S13's dissent was tested, not assumed away.** The random forest genuinely beats both neural
models at W=1 (0.688 vs. GRU 0.597) — the baseline was doing real work. The GRU clears it by
+0.144 once given a full window.

## Decisions flagged, not silently made

- **34-leaf-label → 8-class taxonomy, confirmed with the user.** Table I lists only 6 family
  rows (merging DDoS/DoS into one), but Section III-B3's prose says "seven attack families" and
  the `n_classes==8` validator plus Eq. (19) both assume C=8. Splitting the merged row is the
  only reading consistent with all three; verified against the real dataset — all 34 on-disk
  labels map, none unmapped.

## An honest limitation

**Only baselines 1–3 are Phase 3's.** Baselines 4–5 (local-only GRUs, the federated global GRU)
are correctly Phase 4's — see that file. The W=8 and W=32 points of the full {1,8,16,32} sweep,
and the full feature-count sweep, were not run; W=1 vs. W=16 (the two decision-relevant
endpoints, and the one the paper explicitly frames as the test) were.

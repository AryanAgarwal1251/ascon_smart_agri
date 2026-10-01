# Paper rewrite audit — what the existing draft says, what we measured, and what must change

The draft in [`design_paper.md`](design_paper.md) was written **before** the system was built. It
is a design document: it specifies intentions and claims "we claim no new algorithm; the
contribution is the composition and the rigour of its specification." Eighteen months of
measurement later, several of its statements are contradicted by our own results, and most of the
paper's actual contribution did not exist when it was written.

This audit is the worklist for the rewrite. It is organised as: what is wrong, what is missing,
what must be renamed, and what must be verified.

---

## 1. Statements the draft makes that measurement contradicts

| Draft says | Measured reality | Action |
| --- | --- | --- |
| "We claim no new algorithm; the contribution is the composition and the rigour of its specification" (§I-G) | The contributions are now **empirical**: a sealed bidirectional weight channel validated over 120 frames, per-client federation gain, a 21-point client-count sweep, and three dataset rejections decided by measurement | **Rewrite §I-G entirely.** The paper is no longer a design paper. |
| Single global scaler (§III-F4) | We use **per-corpus standardisation**; it moved CICIoT2023 from 0.603 to 0.7742 | Amend to per-corpus, state the departure and why |
| Dataset is CICIoT2023 (implicit throughout) | Three corpora were characterised and **two were rejected by measurement**; the rejection is a result | New subsection in §III-B, and a results subsection |
| Ascon protects telemetry (§III-G framing) | Ascon protects the **model weights** in both directions; the cloud hop is **plain TLS** by deliberate decision | **Rewrite §III-G.** This is the architecture's central security claim and the draft describes the wrong channel. |
| K = 3 by assumption A1 / objective O3 | K = 3 is a **hardware budget constraint**; the architecture was measured to K = 50 | State as constraint; add the sweep as a result |
| Evaluation reports five baselines (§III-I) | Correct, and all five now have numbers; plus a sixth axis (client count) the draft never anticipated | Extend, do not replace |
| Gap G6 "arises only when a trained detector is wired to a simulated device stream" | Still true, and now with a measured runtime: 13,253 readings, 4,239 malicious, 0 on the cloud path | Keep the framing, add the measurement |

### The one thing the draft got right that we nearly broke

The draft's insistence that the malicious handler **hold no reference to the cloud client**, with
"an automated test asserts that no encrypted payload is emitted during a malicious-only run", is
exactly what the implementation does, and it is why the property survived a year of refactoring.
The draft's §III-A paragraph on this should be **promoted**, not rewritten.

---

## 2. What is missing — the paper's actual contributions

None of this existed when the draft was written. All of it is measured and committed.

| Contribution | Evidence | Target section |
| --- | --- | --- |
| **Ascon-sealed bidirectional weight channel** — per-client, per-direction keys; AD binding client, round, direction, schema | 120 frames, 0 rejected; 271,883 B per client per round | §III-G (rewritten), §VI-D |
| **Per-client federation gain** | +0.1338 / +0.0789 / +0.0516, every seed, both corpora; a class-starved client 0.6145 → 0.8145 | §VI-B — **the headline** |
| **Client-count scalability sweep** | 21 points, K ∈ {3…50}; macro-F1 −0.079, rarest class −0.256, 0 empty clients throughout | §VI-C |
| **Dataset selection as a measured decision** | three corpora characterised, two rejected; `Variance` 0…1 vs 0…1.13×10⁸; window sizes differing **7.33×** | §III-B, §VI-E |
| **Edge-IIoTset on its native schema** | 0.9822 ± 0.0043, and why it is still excluded | §VI-E |
| **Software-in-the-loop validation** (see §3) | 10 containers, all exit 0; G1 under 4,239 malicious readings | §VII |
| **A vacuous-pass finding about safety-property testing** | G1 "passed" on 1 malicious reading; a 10 %-leak system passes that test 90 % of the time | §VII, and it is a methodological contribution |
| **Cross-corpus transfer as a replication** | our 0.0853 / 0.0763 against the literature's 56 % decay and 30 pp loss | §VI-F, with citations |
| Hardware demonstration | pending the hardware run | §VII |

---

## 3. Renaming — the two things the user flagged

### 3.1 "Twin test" → **software-in-the-loop (SIL) validation**

"Twin" is our internal word and means nothing to a reviewer. The established engineering term for
*the real software running against simulated inputs, before the target hardware exists* is
**software-in-the-loop**, and it sits in a recognised progression:

| Stage | Name | Status |
| --- | --- | --- |
| Real node software, simulated sensors, containers at the target architecture | **Software-in-the-loop (SIL)** | **done** |
| Real node software on the real Raspberry Pis and ESP32s | **Hardware-in-the-loop (HIL)** / on-device deployment | pending |

The paper should define SIL precisely when first used — *the production node software, unmodified,
executing as separate OS processes on a real network at the target CPU architecture, with sensor
inputs supplied by a contract-equivalent simulator* — so no reader assumes more than was done. The
sentence "container timings are not Raspberry Pi timings" belongs in the same paragraph.

### 3.2 `G1`…`G6` → named gaps, written out

A reviewer cannot be asked to carry six opaque labels. Each gap gets a descriptive name, is stated
in full where it is introduced, and is referred to by name thereafter.

| Was | Name to use | Full statement for its first appearance |
| --- | --- | --- |
| G1 | **the verdict-routing gap** | Intrusion detection and payload protection are studied as separate problems, so no prior work specifies what the detector's verdict should do to the data path. |
| G2 | **the unbounded-privacy-claim gap** | Federated IDS work asserts privacy because raw records are not transmitted, without bounding what the transmitted parameters leak. |
| G3 | **the partition-specification gap** | Client partitions are under-specified and usually IID, so reported federated results are not reproducible and understate heterogeneity. |
| G4 | **the missing-lower-bound gap** | Without a local-only baseline, a federated result has no lower bound and the benefit of federating is unquantified. |
| G5 | **the sequence-construction gap** | Window ordering, label homogeneity and labelling rules are left unstated, so sequence models are not comparable across papers. |
| G6 | **the feature-provenance gap** | A detector trained on flow-derived features cannot consume an application-layer payload, and runtime demonstrations do not say which features the model actually consumed. |

**Our own additions to the gap list**, both discovered by measurement and both worth stating as
gaps rather than as incidents:

- **the vacuous-verification gap** — safety properties are reported as satisfied without reporting
  how many opportunities to violate them occurred, so a passing assertion can carry no information
- **the cross-corpus semantics gap** — corpora are combined on column *names*, so a shared schema
  can hide incompatible quantities; ours differed by a factor of 2.4×10¹¹ on `IAT`

---

## 4. Equations to carry forward, and the ones to add

The draft's numbering must be preserved where the equation survives, because the code cites it
(`Eq. (21)` appears in `federated/aggregation.py`, `Eq. (27)` in the AD layout, and so on).

| Eq. | What it is | Status |
| --- | --- | --- |
| (5) | verdict → data path selection | keep, central to the verdict-routing gap |
| (12) | sequence count from contiguous runs | keep |
| (18) | class weights from training counts | keep |
| (19) | GRU parameter count = 33,800 | keep, and it is asserted by a test |
| (20) | Dirichlet client partition | keep |
| (21) | FedAvg weighting by sequence count | keep — the common bug it prevents is worth a sentence |
| (22) | bytes per round | keep; measured 815,136 B against 811,200 predicted (0.5 % safetensors framing) |
| (23)–(24) | Chan's parallel scaler combination | keep; measured equal to pooled to ~5×10⁻¹² at every K |
| (27) | telemetry AD layout | keep |
| (31) | false positive rate | keep |
| **new** | **weight-channel AD layout** ⟨client_id, round, direction, schema_version⟩ | **add** — this is the central security construction and has no equation |
| **new** | **per-packet normalisation** for cross-corpus count features | **add** if §VI-E reports it |
| **new** | **pigeonhole bound** on client count from rarest-class block count | **add** — predicts 42, and the sweep's second descent begins there |

---

## 5. Figures and algorithms

### Figures (every measurement gets one)

| # | Figure | Kind | Data |
| --- | --- | --- | --- |
| 1 | System architecture, crypto named per hop | diagram | `architecture-diagram.md` |
| 2 | Two planes and the feature-provenance boundary | diagram | same |
| 3 | Four-stage feature selection funnel, 39 → 16 | diagram | `phase2_feature_selection.json` |
| 4 | Confusion matrices, centralised vs federated | heatmap pair | manifests |
| 5 | Per-class F1, centralised vs federated vs local-only | grouped bar | manifests |
| 6 | **Federation bracket** — local-only / federated / centralised | bar + error bars | Phase 4 |
| 7 | **Per-client gain** | bar, per client per corpus | federation-gain manifest |
| 8 | **Client-count sweep: macro-F1 and rarest-class F1 vs K** | twin-axis line | `k_sweep_results.json` |
| 9 | **Communication cost vs K**, with the per-round bytes | line, log y | same |
| 10 | Convergence curves at several K | multi-line | `per_round_macro_f1` |
| 11 | Dirichlet partition histograms at α = 0.5 | stacked bar | `block_histograms` |
| 12 | **Cross-corpus feature range overlap** — the semantics finding | horizontal bar | our measurement |
| 13 | SIL runtime: readings → verdict → path, with counts | Sankey or flow | runtime manifests |

### Algorithms (pseudocode)

| # | Algorithm | Why it earns a float |
| --- | --- | --- |
| 1 | Leakage-controlled preparation: subsample → **dedup → split** → window | the ordering *is* the control |
| 2 | Four-stage feature selection | reproducibility |
| 3 | Sealed federated round, both directions | the central security construction |
| 4 | Federated scaler via Chan's parallel formula | shows no raw data leaves |
| 5 | Verdict routing with disjoint paths | shows the structural guarantee |

---

## 6. Citation audit — to be done before submission

The user's requirement: every citation's link must open, and the cited paper must actually contain
the claim we attribute to it.

| Claim we make | Source | Verified? |
| --- | --- | --- |
| Ascon is the NIST lightweight AEAD standard | NIST SP 800-232 | ⏳ fetch and confirm the designation |
| Ascon design and security | Dobraunig, Eichlseder, Mendel, Schläffer, *Ascon v1.2: Lightweight Authenticated Encryption and Hashing*, J. Cryptology 34(3), 2021, DOI 10.1007/s00145-021-09398-9 | ⏳ confirm authors, venue, year |
| Ascon won CAESAR as primary lightweight choice | CAESAR final portfolio | ⏳ |
| Cross-dataset decay 56.28 % on standardised features | [arXiv 2205.04112](https://arxiv.org/abs/2205.04112) | ✅ figure read from the paper |
| Up to 30 pp loss; combined-FL recovers ≈90 % | [Nature Sci Rep](https://www.nature.com/articles/s41598-025-32567-w) | ✅ |
| Edge-IIoTset placeholder artefact; AgriEdge; leave-one-device-out 0.9988 → 0.5083 | [arXiv 2608.15761](https://arxiv.org/abs/2608.15761) | ✅ abstract read |
| Standardised NetFlow feature sets | [arXiv 2101.11315](https://arxiv.org/abs/2101.11315), [arXiv 2011.09144](https://arxiv.org/abs/2011.09144) | ⏳ |
| DANN for cross-domain NIDS | [DI-NIDS, arXiv 2210.08252](https://arxiv.org/abs/2210.08252) | ✅ abstract read |
| CICIoT2023 corpus | Neto et al., *Sensors* 23(13), 2023 | ⏳ |
| The draft's 20 reviewed studies (L1–L5, S13, Threads A–D) | — | ⏳ **every one must be re-checked**; the draft's citations carry broken `[URL 🔗](#page-0)` placeholders throughout |

> ⚠️ The draft's inline citations are all `[URL 🔗](#page-0)` — a conversion artefact. **No citation
> in the existing draft can be trusted as-is.** The reference list must be rebuilt from scratch.

### Comparisons must be complete

Where we compare against a method, we report it on the same protocol, not a number quoted from
elsewhere. Already satisfied for the within-paper baselines: random forest **0.6855**, MLP
**0.6070**, centralised GRU **0.8297**, all at W ∈ {1,16} over 3 seeds on the same split. Any
comparison to an *external* paper's number must state that the protocols differ.

---

## 7. Order of work

1. **Rebuild the reference list** — the draft has no usable citations
2. Rewrite §I-G (contributions) and §III-G (Ascon on the weight channel) — the two places the draft
   is substantively wrong
3. Add the Ascon primer section the user asked for, from the Journal of Cryptology paper and SP 800-232
4. Rename gaps throughout; introduce SIL/HIL
5. Write §VI results subsections against the figure list
6. Generate the 13 figures from the manifests
7. Write the 5 algorithm floats
8. §VIII limitations from `paper-claims-inventory.md` §B
9. Citation audit pass — open every link, confirm every attributed claim

---

## 8. Carry-over checklist from the submitted review paper

Added 2026-10-02, after the user supplied the submitted paper. **`main.tex` currently carries a
minority of it.** This is the precise gap.

### Done

| Item | State |
| --- | --- |
| Author block (4 authors, VIT Chennai) | ✅ in `main.tex` |
| References [1]–[25] | ✅ merged into `references.bib`, plus 12 added = **37 entries** |
| Eq. (5) routing, (12) sequences, (18) class weights, (19) params, (20) Dirichlet, (21) FedAvg, (22) bytes, (23)–(24) Chan | ✅ 9 of the original equations |
| Weight-channel AD layout, pigeonhole bound | ✅ 2 new equations |
| Algorithm 1 (federated round) | ✅ rewritten as the *sealed* round |
| Figures | ✅ 11, all generated from manifests |

### Owed — equations (21 of the original 31 are missing)

| Eq. | What | Belongs in |
| --- | --- | --- |
| (1) | window matrix $X_i$ | §IV formal statement |
| (2) | binary projection $b(\hat y)$ | §IV |
| (3), (4) | federated objective $F(\theta)$, $F_k(\theta)$ | §IV |
| (6) | imbalance ratio IR $\approx 5751$ | §IV-A |
| (7) | per-class cap $n^{\text{sub}}_c$ | §IV-A |
| (8) | **leakage inflation** $\Delta \approx \delta\rho(1-\text{Acc})$ | §IV-A — this is the argument for dedup-before-split and must not be dropped |
| (9) | the ordering dedup $\to$ split $\to$ fit $\to$ transform | §IV-A |
| (10) | mutual information $I(X_j;Y)$ | §IV-B |
| (11) | reciprocal rank fusion | §IV-B |
| (13)–(16) | GRU gates | §IV-C |
| (17) | softmax over LayerNorm | §IV-C |
| (25), (26) | $\mathrm{Enc}$, and $\mathrm{Dec} \in \{p, \bot\}$ | §IV-E |
| (27) | telemetry AD layout | §IV-E |
| (28) | nonce birthday bound | §IV-E |
| (29) | wire overhead $|p| + 32$ | §IV-E |
| (30) | macro-F1 | §V |
| (31) | FPR | §V |

### Owed — tables (5 of the original 6 are missing)

| Table | What | Why it matters |
| --- | --- | --- |
| I | Attack families and their **agricultural consequence** | The only place the paper connects a class label to a farm outcome |
| II | Assumptions A1–A7 with status | Separates simulation artefacts from facts; A3 is the provenance boundary |
| III | Selection parameters P1–P6 | Makes the review auditable |
| V | **The 20 reviewed studies S1–S20** | The literature review *is* this table |
| VI | Risks R1–R7 with controls | R3 dedup leakage, R5 Ascon variant, R6 nonce reuse |

### Owed — prose

| Section | Content in the submitted paper |
| --- | --- |
| Deployment environment | Constraints **C1–C4**: resource, backhaul, physical exposure, data ownership |
| Threat model | **Three channels**, each with its own adversary |
| Why the three sub-problems are coupled | detection at the edge / collaborative learning / classification protects nothing |
| Objectives | **O1–O5**, each with a completion criterion |
| Scope | in-scope and out-of-scope lists |
| Review protocol | sources, search strings, screening funnel (Fig. 1) |
| Threads A–D | the four literature threads, with limitations L1–L5 |
| Patent landscape | four filings and what remains unclaimed |
| Ethics | no human subjects, replayed records only, keys out of version control |
| Communication argument | 0.77 MiB per round against ~276 MB to pool — factor ~18 |

### Two things in the submitted paper that measurement has since contradicted

- The communication comparison quotes **0.77 MiB per round**; we measured **815,136 B**
  (0.78 MiB) at $K=3$, which confirms it — but the sweep shows it reaching **13.59 MB** at
  $K=50$, so the "factor of 18 cheaper" claim is specific to small $K$ and must be stated that way.
- "We claim no new algorithm; the contribution is the composition" is no longer accurate; see §1.
